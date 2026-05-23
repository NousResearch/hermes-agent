"""Unified Codex credential resolver.

Auth Stores — Ownership Contract
================================
There are two Codex credential stores on disk. They MUST stay
separate. Do NOT copy one to the other and do NOT have Hermes write
back to the Codex CLI's store.

- ``~/.codex/auth.json`` — **owned by the Codex CLI** (and the VS
  Code extension that shares its identity). Codex CLI refreshes
  tokens here. Hermes is allowed to **read** it — borrow access
  tokens until they expire. Hermes is NEVER allowed to write to it.
  Reason: refresh-token race — if Hermes refreshes and writes back
  while the Codex CLI is also refreshing, one of them ends up holding
  a revoked token.

- ``~/.hermes/auth.json`` — **owned by Hermes**. Hermes' own auth
  subsystem (``hermes auth login codex``) writes here. Hermes
  refreshes its own tokens here.

This resolver respects the contract: read both, write neither
(except via Hermes' own refresh flow into ``~/.hermes/auth.json``).
The borrow path returns the live access token only — the refresh
token from the Codex CLI store is intentionally not exposed past the
borrow point so no code path can accidentally feed it into Hermes'
refresh-and-save machinery.

Future maintainers: do NOT "promote" the borrowed token into the
Hermes store automatically. The user crosses that boundary
explicitly via ``hermes auth login codex``, which is the only place
that imports from the Codex CLI store and writes to the Hermes
store.

Resolution chain
----------------
1. ``HERMES_CODEX_ACCESS_TOKEN`` env override — developer escape
   hatch. No refresh, no expiry check, no disk I/O. Skips both
   stores entirely.
2. ``~/.hermes/auth.json`` via :func:`_read_codex_tokens` under
   :func:`_auth_store_lock`. Refresh-when-expiring is gated by
   ``refresh_if_expiring`` and always serialised under the lock so
   concurrent callers do not race the single-use refresh token.
3. ``~/.codex/auth.json`` read-only borrow via
   :func:`_import_codex_cli_tokens` when (a) the Hermes store yielded
   no tokens at all, (b) ``allow_codex_cli_fallback`` is True, and
   (c) the borrowed access token is not expired (the importer
   already rejects expired tokens).

Refresh failures from the Hermes store (e.g.,
``refresh_token_reused``, ``invalid_grant``) are surfaced loudly to
the caller — they do NOT fall through to the borrow path. The user
needs the actionable re-auth guidance; silently borrowing would
mask a broken Hermes session.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

# Module-level reference rather than ``from ... import`` so tests that
# monkeypatch ``hermes_cli.auth.<symbol>`` propagate into this resolver
# at call time. Resolver-local helpers (``_codex_access_token_is_expiring``
# etc.) that aren't typically patched are imported by name below.
from hermes_cli import auth as _hermes_auth
from hermes_cli.auth import (
    AUTH_LOCK_TIMEOUT_SECONDS,
    AuthError,
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    DEFAULT_CODEX_BASE_URL,
    _auth_store_lock,
    _codex_access_token_is_expiring,
)

ENV_CODEX_ACCESS_TOKEN = "HERMES_CODEX_ACCESS_TOKEN"
ENV_CODEX_BASE_URL = "HERMES_CODEX_BASE_URL"
ENV_CODEX_REFRESH_TIMEOUT_SECONDS = "HERMES_CODEX_REFRESH_TIMEOUT_SECONDS"

# Error codes that indicate the Hermes auth store has no usable tokens
# at all (as opposed to refresh-time failures, which should surface).
_HERMES_STORE_EMPTY_CODES = frozenset(
    {
        "codex_auth_missing",
        "codex_auth_invalid_shape",
        "codex_auth_missing_access_token",
        "codex_auth_missing_refresh_token",
    }
)

# Double-checked-locking window for ``force_refresh``. When concurrent
# callers contend on ``_auth_store_lock``, a second caller that arrives
# within this many seconds of the previous refresh's persisted timestamp
# returns the freshly-stored token instead of spending the (single-use)
# refresh token a second time. Keeps the lock honest under concurrent
# 401-retry storms.
_FORCE_REFRESH_DCL_WINDOW_SECONDS = 5.0


CodexCredentialSource = Literal[
    "env",
    "hermes-auth-store",
    "codex-cli-borrow",
]


@dataclass(frozen=True)
class CodexCredentials:
    """Resolved Codex runtime credentials.

    The dict produced by :meth:`as_runtime_dict` matches the legacy
    ``resolve_codex_runtime_credentials`` return shape so the new
    resolver drops into existing call sites without changes.
    """

    access_token: str
    base_url: str
    source: CodexCredentialSource
    last_refresh: Optional[str]
    account_id: Optional[str]
    auth_mode: Literal["chatgpt"] = "chatgpt"

    def as_runtime_dict(self) -> Dict[str, Any]:
        return {
            "provider": "openai-codex",
            "base_url": self.base_url,
            "api_key": self.access_token,
            "source": self.source,
            "last_refresh": self.last_refresh,
            "auth_mode": self.auth_mode,
        }


def _resolve_base_url() -> str:
    override = os.getenv(ENV_CODEX_BASE_URL, "").strip().rstrip("/")
    return override or DEFAULT_CODEX_BASE_URL


def _extract_account_id(tokens: Any) -> Optional[str]:
    if not isinstance(tokens, dict):
        return None
    candidate = tokens.get("account_id")
    if isinstance(candidate, str):
        cleaned = candidate.strip()
        return cleaned or None
    return None


def _last_refresh_within_dcl_window(last_refresh: Any) -> bool:
    """True if ``last_refresh`` is recent enough that concurrent
    ``force_refresh`` callers should skip their own HTTP refresh.

    See ``_FORCE_REFRESH_DCL_WINDOW_SECONDS`` for the rationale.
    """
    if not isinstance(last_refresh, str):
        return False
    text = last_refresh.strip()
    if not text:
        return False
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta = (datetime.now(timezone.utc) - ts).total_seconds()
    return 0 <= delta <= _FORCE_REFRESH_DCL_WINDOW_SECONDS


def _resolve_from_hermes_store(
    *,
    force_refresh: bool,
    refresh_if_expiring: bool,
    refresh_skew_seconds: int,
) -> CodexCredentials:
    """Read the Hermes Codex store, optionally refreshing under the lock."""
    data = _hermes_auth._read_codex_tokens()
    tokens = dict(data["tokens"])
    access_token = str(tokens.get("access_token", "") or "").strip()
    refresh_timeout_seconds = float(
        os.getenv(ENV_CODEX_REFRESH_TIMEOUT_SECONDS, "20")
    )

    should_refresh = bool(force_refresh)
    if (not should_refresh) and refresh_if_expiring:
        should_refresh = _codex_access_token_is_expiring(
            access_token, refresh_skew_seconds
        )

    if should_refresh:
        lock_timeout = max(
            float(AUTH_LOCK_TIMEOUT_SECONDS),
            refresh_timeout_seconds + 5.0,
        )
        with _auth_store_lock(timeout_seconds=lock_timeout):
            # Re-read under the lock so a concurrent refresher's result is
            # observed before we decide to spend the single-use refresh
            # token ourselves.
            data = _hermes_auth._read_codex_tokens(_lock=False)
            tokens = dict(data["tokens"])
            access_token = str(tokens.get("access_token", "") or "").strip()

            # Under-lock recheck. ``force_refresh`` respects a short DCL
            # window so concurrent force-refresh callers serialize into a
            # single HTTP refresh — the second caller observes the just-
            # written tokens and returns them instead of spending the
            # (single-use) refresh token again.
            do_refresh = False
            if force_refresh:
                do_refresh = not _last_refresh_within_dcl_window(
                    data.get("last_refresh")
                )
            elif refresh_if_expiring:
                do_refresh = _codex_access_token_is_expiring(
                    access_token, refresh_skew_seconds
                )

            if do_refresh:
                tokens = _hermes_auth._refresh_codex_auth_tokens(
                    tokens, refresh_timeout_seconds
                )
                access_token = str(
                    tokens.get("access_token", "") or ""
                ).strip()

    return CodexCredentials(
        access_token=access_token,
        base_url=_resolve_base_url(),
        source="hermes-auth-store",
        last_refresh=data.get("last_refresh"),
        account_id=_extract_account_id(tokens),
    )


def _resolve_from_codex_cli_borrow() -> Optional[CodexCredentials]:
    """Borrow a live access token from the Codex CLI store, or return None.

    ``_import_codex_cli_tokens`` already rejects expired tokens and
    never writes back to ``~/.codex/auth.json``. A non-None return
    implies a usable access token. We deliberately do NOT persist the
    borrowed token into the Hermes auth store — see the ownership
    contract at the top of this module.
    """
    borrowed = _hermes_auth._import_codex_cli_tokens()
    if not borrowed:
        return None
    access_token = str(borrowed.get("access_token", "") or "").strip()
    if not access_token:
        return None
    return CodexCredentials(
        access_token=access_token,
        base_url=_resolve_base_url(),
        source="codex-cli-borrow",
        last_refresh=None,
        account_id=_extract_account_id(borrowed),
    )


def resolve_codex_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    allow_codex_cli_fallback: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
) -> CodexCredentials:
    """Resolve Codex OAuth credentials via the unified fallback chain.

    See the module docstring for the full ownership contract and
    chain rationale. Raises :class:`AuthError` when no source yields
    a usable access token.
    """
    env_token = os.environ.get(ENV_CODEX_ACCESS_TOKEN, "").strip()
    if env_token:
        return CodexCredentials(
            access_token=env_token,
            base_url=_resolve_base_url(),
            source="env",
            last_refresh=None,
            account_id=None,
        )

    try:
        return _resolve_from_hermes_store(
            force_refresh=force_refresh,
            refresh_if_expiring=refresh_if_expiring,
            refresh_skew_seconds=refresh_skew_seconds,
        )
    except AuthError as exc:
        if not allow_codex_cli_fallback:
            raise
        if exc.code not in _HERMES_STORE_EMPTY_CODES:
            # Refresh-time failures (refresh_token_reused, invalid_grant,
            # transport errors, etc.) must surface so the user sees the
            # actionable re-auth guidance instead of a silent borrow.
            raise
        borrowed = _resolve_from_codex_cli_borrow()
        if borrowed is None:
            raise
        return borrowed
