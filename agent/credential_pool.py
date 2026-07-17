"""Persistent multi-credential pool for same-provider failover."""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import uuid
import re
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import OPENROUTER_BASE_URL
from hermes_cli.config import load_env
from agent.secret_scope import get_secret as _get_secret
from agent.credential_persistence import (
    is_borrowed_credential_source,
    sanitize_borrowed_credential_payload,
)
import hermes_cli.auth as auth_mod
from hermes_cli.auth import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    PROVIDER_REGISTRY,
    _auth_store_lock,
    _codex_access_token_is_expiring,
    _decode_jwt_claims,
    _load_auth_store,
    _load_provider_state,
    _resolve_kimi_base_url,
    _resolve_zai_base_url,
    _save_auth_store,
    _save_provider_state,
    _store_provider_state,
    read_credential_pool,
    write_credential_pool,
)

logger = logging.getLogger(__name__)


def _load_config_safe() -> Optional[dict]:
    """Load config.yaml, returning None on any error."""
    try:
        from hermes_cli.config import load_config

        return load_config()
    except Exception:
        return None


# --- Status and type constants ---

STATUS_OK = "ok"
STATUS_EXHAUSTED = "exhausted"
# Terminal failure — the credential will never recover on its own.  Used for
# upstream-permanent OAuth states like ``token_invalidated`` / ``token_revoked``
# where retrying after a TTL cooldown is guaranteed to fail.  ``DEAD`` entries
# are excluded from rotation unconditionally and only clear when an explicit
# write-side sync (e.g. ``_save_codex_tokens`` after a fresh device-code
# login) rewrites the tokens.
STATUS_DEAD = "dead"

# OAuth error reasons that indicate the credential is permanently invalid
# server-side and cannot be recovered by retry/refresh.  Sourced from
# OpenAI Codex Responses API, Anthropic, xAI, and Google OAuth spec.
_TERMINAL_AUTH_REASONS = frozenset({
    "token_invalidated",   # OpenAI Codex: "Your authentication token has been invalidated."
    "token_revoked",        # OAuth 2.0 RFC 7009: token explicitly revoked
    "invalid_token",        # RFC 6750: bearer token is malformed/expired/revoked
    "invalid_grant",        # RFC 6749: refresh_token rejected during refresh
    "unauthorized_client",  # RFC 6749: client no longer authorized
    "refresh_token_reused", # Single-use refresh token consumed by another process
})

# How long a DEAD manual credential is preserved before being pruned.
# Manual entries (``manual:*``) are independent credentials with no singleton
# to re-seed from, so pruning them after a quiet window cleans up dead state
# without losing recoverability — the user always has the option to re-add
# via ``hermes auth add``.
#
# Singleton-seeded entries (``device_code``, ``claude_code``)
# are NOT pruned because ``_seed_from_singletons`` would just re-create them
# on the next ``load_pool()`` with the same stale singleton tokens, defeating
# the cleanup.  They remain in the pool marked DEAD until an explicit re-auth
# write-side sync (``_save_codex_tokens`` etc.) clears the status.
DEAD_MANUAL_PRUNE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

AUTH_TYPE_OAUTH = "oauth"
AUTH_TYPE_API_KEY = "api_key"

SOURCE_MANUAL = "manual"
SOURCE_MANUAL_DEVICE_CODE = f"{SOURCE_MANUAL}:device_code"

STRATEGY_FILL_FIRST = "fill_first"
STRATEGY_ROUND_ROBIN = "round_robin"
STRATEGY_RANDOM = "random"
STRATEGY_LEAST_USED = "least_used"
SUPPORTED_POOL_STRATEGIES = {
    STRATEGY_FILL_FIRST,
    STRATEGY_ROUND_ROBIN,
    STRATEGY_RANDOM,
    STRATEGY_LEAST_USED,
}

# Cooldown before retrying an exhausted credential.
# Transient 401 auth failures cool down briefly so single-key setups can recover.
# 429 (rate-limited), 402 (billing/quota), and other failures cool down after 1 hour.
# Provider-supplied reset_at timestamps override these defaults.
EXHAUSTED_TTL_401_SECONDS = 5 * 60           # 5 minutes
EXHAUSTED_TTL_429_SECONDS = 60 * 60          # 1 hour
EXHAUSTED_TTL_DEFAULT_SECONDS = 60 * 60      # 1 hour

# Pool key prefix for custom OpenAI-compatible endpoints.
# Custom endpoints all share provider='custom' but are keyed by their
# custom_providers name: 'custom:<normalized_name>'.
CUSTOM_POOL_PREFIX = "custom:"


# Fields that are only round-tripped through JSON — never used for logic as attributes.
_EXTRA_KEYS = frozenset({
    "token_type", "scope", "client_id", "portal_base_url", "obtained_at",
    "expires_in", "agent_key_id", "agent_key_expires_in", "agent_key_reused",
    "agent_key_obtained_at", "tls", "secret_source", "secret_fingerprint",
})


def _normalize_pool_auth_type(provider: str, token: Any, auth_type: Any) -> str:
    """Infer pool auth metadata for token formats with one unambiguous meaning."""
    if (
        provider == "anthropic"
        and isinstance(token, str)
        and token.startswith("sk-ant-oat")
    ):
        return AUTH_TYPE_OAUTH
    return str(auth_type or AUTH_TYPE_API_KEY)


@dataclass
class PooledCredential:
    provider: str
    id: str
    label: str
    auth_type: str
    priority: int
    source: str
    access_token: str
    refresh_token: Optional[str] = None
    last_status: Optional[str] = None
    last_status_at: Optional[float] = None
    last_error_code: Optional[int] = None
    last_error_reason: Optional[str] = None
    last_error_message: Optional[str] = None
    last_error_reset_at: Optional[float] = None
    base_url: Optional[str] = None
    expires_at: Optional[str] = None
    expires_at_ms: Optional[int] = None
    last_refresh: Optional[str] = None
    inference_base_url: Optional[str] = None
    agent_key: Optional[str] = None
    agent_key_expires_at: Optional[str] = None
    request_count: int = 0
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
        self.auth_type = _normalize_pool_auth_type(
            self.provider,
            self.access_token,
            self.auth_type,
        )

    def __getattr__(self, name: str):
        if name in _EXTRA_KEYS:
            return self.extra.get(name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")

    @classmethod
    def from_dict(cls, provider: str, payload: Dict[str, Any]) -> "PooledCredential":
        field_names = {f.name for f in fields(cls) if f.name != "provider"}
        data = {k: payload.get(k) for k in field_names if k in payload}
        # Rehydrated last_status_at may be an ISO string from to_dict() — normalize to float epoch
        if "last_status_at" in data and isinstance(data["last_status_at"], str):
            data["last_status_at"] = _parse_absolute_timestamp(data["last_status_at"])
        extra = {k: payload[k] for k in _EXTRA_KEYS if k in payload and payload[k] is not None}
        data["extra"] = extra
        data.setdefault("id", uuid.uuid4().hex[:6])
        data.setdefault("label", payload.get("source", provider))
        data.setdefault("auth_type", AUTH_TYPE_API_KEY)
        data.setdefault("priority", 0)
        data.setdefault("source", SOURCE_MANUAL)
        data.setdefault("access_token", "")
        return cls(provider=provider, **data)

    def to_dict(self) -> Dict[str, Any]:
        _ALWAYS_EMIT = {
            "last_status",
            "last_status_at",
            "last_error_code",
            "last_error_reason",
            "last_error_message",
            "last_error_reset_at",
        }
        result: Dict[str, Any] = {}
        for field_def in fields(self):
            if field_def.name in {"provider", "extra"}:
                continue
            value = getattr(self, field_def.name)
            if value is not None or field_def.name in _ALWAYS_EMIT:
                result[field_def.name] = value
        for k, v in self.extra.items():
            if v is not None:
                result[k] = v
        return sanitize_borrowed_credential_payload(result, self.provider)

    @property
    def runtime_api_key(self) -> str:
        if self.provider == "nous":
            # Nous stores the runtime inference credential in agent_key for
            # compatibility. It must be a NAS invoke JWT.
            for token, expires_at in (
                (self.agent_key, self.agent_key_expires_at),
                (self.access_token, self.expires_at),
            ):
                if (
                    isinstance(token, str)
                    and token.strip()
                    and auth_mod._nous_invoke_jwt_is_usable(
                        token,
                        scope=getattr(self, "scope", None),
                        expires_at=expires_at,
                    )
                ):
                    return token.strip()
            return ""
        return str(self.access_token or "")

    @property
    def runtime_base_url(self) -> Optional[str]:
        if self.provider == "nous":
            return self.inference_base_url or self.base_url
        return self.base_url


def label_from_token(token: str, fallback: str) -> str:
    claims = _decode_jwt_claims(token)
    for key in ("email", "preferred_username", "upn"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _next_priority(entries: List[PooledCredential]) -> int:
    return max((entry.priority for entry in entries), default=-1) + 1


def _is_manual_source(source: str) -> bool:
    normalized = (source or "").strip().lower()
    return normalized == SOURCE_MANUAL or normalized.startswith(f"{SOURCE_MANUAL}:")


def _exhausted_ttl(error_code: Optional[int]) -> int:
    """Return cooldown seconds based on the HTTP status that caused exhaustion."""
    if error_code == 401:
        return EXHAUSTED_TTL_401_SECONDS
    if error_code == 429:
        return EXHAUSTED_TTL_429_SECONDS
    return EXHAUSTED_TTL_DEFAULT_SECONDS


def _parse_absolute_timestamp(value: Any) -> Optional[float]:
    """Best-effort parse for provider reset timestamps.

    Accepts epoch seconds, epoch milliseconds, and ISO-8601 strings.
    Returns seconds since epoch.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except ValueError:
            numeric = None
        if numeric is not None:
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _extract_retry_delay_seconds(message: str) -> Optional[float]:
    if not message:
        return None
    delay_match = re.search(r"quotaResetDelay[:\s\"]+(\d+(?:\.\d+)?)(ms|s)", message, re.IGNORECASE)
    if delay_match:
        value = float(delay_match.group(1))
        return value / 1000.0 if delay_match.group(2).lower() == "ms" else value
    sec_match = re.search(r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)", message, re.IGNORECASE)
    if sec_match:
        return float(sec_match.group(1))
    # "Resets in 4hr 5min" format used by OpenCode Go weekly usage limits
    hr_min_match = re.search(r"resets?\s+in\s+(\d+)\s*hr\s+(\d+)\s*min", message, re.IGNORECASE)
    if hr_min_match:
        return int(hr_min_match.group(1)) * 3600 + int(hr_min_match.group(2)) * 60
    hr_only_match = re.search(r"resets?\s+in\s+(\d+)\s*hr\b", message, re.IGNORECASE)
    if hr_only_match:
        return int(hr_only_match.group(1)) * 3600
    min_only_match = re.search(r"resets?\s+in\s+(\d+)\s*min\b", message, re.IGNORECASE)
    if min_only_match:
        return int(min_only_match.group(1)) * 60
    return None


def _normalize_error_context(error_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(error_context, dict):
        return {}
    normalized: Dict[str, Any] = {}
    reason = error_context.get("reason")
    if isinstance(reason, str) and reason.strip():
        normalized["reason"] = reason.strip()
    message = error_context.get("message")
    if isinstance(message, str) and message.strip():
        normalized["message"] = message.strip()
    reset_at = (
        error_context.get("reset_at")
        or error_context.get("resets_at")
        or error_context.get("retry_until")
    )
    parsed_reset_at = _parse_absolute_timestamp(reset_at)
    if parsed_reset_at is None and isinstance(message, str):
        retry_delay_seconds = _extract_retry_delay_seconds(message)
        if retry_delay_seconds is not None:
            parsed_reset_at = time.time() + retry_delay_seconds
    if parsed_reset_at is not None:
        normalized["reset_at"] = parsed_reset_at
    return normalized


def _exhausted_until(entry: PooledCredential) -> Optional[float]:
    if entry.last_status != STATUS_EXHAUSTED:
        return None
    reset_at = _parse_absolute_timestamp(getattr(entry, "last_error_reset_at", None))
    if reset_at is not None:
        return reset_at
    if entry.last_status_at:
        return entry.last_status_at + _exhausted_ttl(entry.last_error_code)
    return None


def _normalize_custom_pool_name(name: str) -> str:
    """Normalize a custom provider name for use as a pool key suffix."""
    return name.strip().lower().replace(" ", "-")


def _iter_custom_providers(config: Optional[dict] = None):
    """Yield (normalized_name, entry_dict) for each valid custom_providers entry."""
    if config is None:
        config = _load_config_safe()
    if config is None:
        return
    custom_providers = config.get("custom_providers")
    if not isinstance(custom_providers, list):
        # Fall back to the v12+ providers dict via the compatibility layer
        try:
            from hermes_cli.config import get_compatible_custom_providers

            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            return
    if not custom_providers:
        return
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        yield _normalize_custom_pool_name(name), entry


def get_custom_provider_pool_key(base_url: Optional[str], provider_name: Optional[str] = None) -> Optional[str]:
    """Look up the custom_providers list in config.yaml and return 'custom:<name>' for a matching base_url.

    When provider_name is given, prefer matching by name first (solving the case where
    multiple custom providers share the same base_url but have different API keys).
    Falls back to base_url matching when no name match is found.

    Returns None if no match is found.
    """
    if not base_url:
        return None
    normalized_url = base_url.strip().rstrip("/")

    # When a provider name is given, try to match by name first.
    # This fixes the P1 bug where two custom providers sharing the same
    # base_url always resolve to the first one's credentials.
    if provider_name:
        normalized_name = _normalize_custom_pool_name(provider_name)
        for norm_name, entry in _iter_custom_providers():
            if norm_name == normalized_name:
                return f"{CUSTOM_POOL_PREFIX}{norm_name}"

    # Fall back to base_url matching (original behavior)
    for norm_name, entry in _iter_custom_providers():
        entry_url = str(entry.get("base_url") or "").strip().rstrip("/")
        if entry_url and entry_url == normalized_url:
            return f"{CUSTOM_POOL_PREFIX}{norm_name}"
    return None


def list_custom_pool_providers() -> List[str]:
    """Return all 'custom:*' pool keys that have entries in auth.json."""
    pool_data = read_credential_pool(None)
    return sorted(
        key for key in pool_data
        if key.startswith(CUSTOM_POOL_PREFIX)
        and isinstance(pool_data.get(key), list)
        and pool_data[key]
    )


def _get_custom_provider_config(pool_key: str) -> Optional[Dict[str, Any]]:
    """Return the custom_providers config entry matching a pool key like 'custom:together.ai'."""
    if not pool_key.startswith(CUSTOM_POOL_PREFIX):
        return None
    suffix = pool_key[len(CUSTOM_POOL_PREFIX):]
    for norm_name, entry in _iter_custom_providers():
        if norm_name == suffix:
            return entry
    return None


def get_pool_strategy(provider: str) -> str:
    """Return the configured selection strategy for a provider."""
    config = _load_config_safe()
    if config is None:
        return STRATEGY_FILL_FIRST

    strategies = config.get("credential_pool_strategies")
    if not isinstance(strategies, dict):
        return STRATEGY_FILL_FIRST

    strategy = str(strategies.get(provider, "") or "").strip().lower()
    if strategy in SUPPORTED_POOL_STRATEGIES:
        return strategy
    return STRATEGY_FILL_FIRST


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
) -> bool:
    """Return whether a pool belongs to the requested runtime provider.

    Named custom endpoints intentionally use two identities: the live agent is
    ``custom`` while its pool is keyed ``custom:<name>``. Accept that pair only
    when the runtime base URL resolves to the exact same custom pool key.
    Empty string identities fail closed. Legacy pool adapters without a
    ``provider`` attribute remain compatible; production pools are scoped.
    """
    raw_pool_provider = getattr(pool_or_provider, "provider", None)
    if raw_pool_provider is None:
        if isinstance(pool_or_provider, str):
            raw_pool_provider = pool_or_provider
        else:
            # Backward compatibility for lightweight/unscoped pool adapters.
            # Production CredentialPool instances always carry ``provider``;
            # old plugins and tests may expose only select()/has_credentials().
            return True
    pool_provider = str(raw_pool_provider or "").strip().lower()
    provider_norm = str(provider or "").strip().lower()
    if not pool_provider or not provider_norm:
        return False
    if pool_provider == provider_norm:
        return True
    if provider_norm != "custom" or not pool_provider.startswith(CUSTOM_POOL_PREFIX):
        return False
    try:
        matched_pool = get_custom_provider_pool_key(base_url or "")
    except Exception:
        return False
    return str(matched_pool or "").strip().lower() == pool_provider


DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL = 1


def _write_through_provider_state_to_global_root(
    provider_id: str, state: Dict[str, Any]
) -> None:
    """Persist a rotated OAuth ``state`` into the global-root auth.json.

    Best-effort write-through for the multi-profile rotation hazard
    (#48415 / #43589): nous, openai-codex, and xai-oauth rotate the
    refresh_token on refresh, so when a profile pool refresh rotates a grant
    it resolved from the root fallback, the rotated chain must land back in
    root. Otherwise root keeps a now-revoked refresh token and every other
    profile reading the stale root grant dies with ``refresh_token_reused`` /
    ``invalid_grant`` once its access token expires.

    Only updates ``providers.<provider_id>`` in the root store; never touches
    the profile store (the caller already saved that). Swallows all errors — a
    failed write-through degrades to the pre-existing behavior (root stale), it
    must never break the profile's own successful save. Mirrors
    ``hermes_cli.auth._write_through_xai_oauth_to_global_root`` (which covers
    the non-pool xAI refresh path) for the credential-pool refresh path.
    """
    try:
        global_path = auth_mod._global_auth_file_path()
    except Exception:
        return
    if global_path is None:
        # Classic mode (profile == root); the profile save already hit root.
        return
    # Seat belt: under pytest, refuse to write the real user's
    # ~/.hermes/auth.json even when HERMES_HOME points at a profile path
    # (mirrors the read-side guard in _load_global_auth_store). Uses the
    # unmodified HOME env, not Path.home() which fixtures may monkeypatch.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        real_home_env = os.environ.get("HOME", "")
        if real_home_env:
            real_root = Path(real_home_env) / ".hermes" / "auth.json"
            try:
                if global_path.resolve(strict=False) == real_root.resolve(strict=False):
                    return
            except Exception:
                return
    try:
        auth_mod._persist_provider_state_to_store(
            provider_id,
            state,
            global_path,
            set_active=False,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug(
            "%s pool refresh: write-through to global root failed: %s",
            provider_id,
            exc,
        )


class CredentialPool:
    def __init__(self, provider: str, entries: List[PooledCredential]):
        self.provider = provider
        self._entries = sorted(entries, key=lambda entry: entry.priority)
        self._current_id: Optional[str] = None
        self._strategy = get_pool_strategy(provider)
        self._lock = threading.Lock()
        self._active_leases: Dict[str, int] = {}
        self._max_concurrent = DEFAULT_MAX_CONCURRENT_PER_CREDENTIAL

    def has_credentials(self) -> bool:
        return bool(self._entries)

    def has_available(self) -> bool:
        """True if at least one entry is not currently in exhaustion cooldown."""
        return bool(self._available_entries())

    def entries(self) -> List[PooledCredential]:
        return list(self._entries)

    def current(self) -> Optional[PooledCredential]:
        if not self._current_id:
            return None
        return next((entry for entry in self._entries if entry.id == self._current_id), None)

    def _replace_entry(self, old: PooledCredential, new: PooledCredential) -> None:
        """Swap an entry in-place by id, preserving sort order."""
        for idx, entry in enumerate(self._entries):
            if entry.id == old.id:
                self._entries[idx] = new
                return

    def _persist(self, *, removed_ids: Optional[List[str]] = None) -> None:
        write_credential_pool(
            self.provider,
            [entry.to_dict() for entry in self._entries],
            removed_ids=removed_ids,
        )

    def _is_terminal_auth_failure(
        self,
        status_code: Optional[int],
        normalized_error: Dict[str, Any],
    ) -> bool:
        """Detect upstream-permanent OAuth failures that won't recover on TTL.

        Only fires for 401 responses whose error code/reason matches a known
        terminal OAuth state (token_invalidated, token_revoked, invalid_grant,
        etc.).  Distinguishes permanent failures from transient ones like
        token_expired (refreshable) or generic 401 without a specific reason
        (could be a server-side glitch worth retrying).

        Returns False for non-401 status codes — 429 rate limits and 402
        billing failures are transient by nature and should keep TTL semantics.
        """
        if status_code != 401:
            return False
        reason = normalized_error.get("reason")
        if not isinstance(reason, str):
            return False
        return reason.strip().lower() in _TERMINAL_AUTH_REASONS

    def _mark_exhausted(
        self,
        entry: PooledCredential,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
    ) -> PooledCredential:
        normalized_error = _normalize_error_context(error_context)
        # Permanent OAuth failures (token_invalidated, token_revoked, etc.)
        # transition to STATUS_DEAD instead of STATUS_EXHAUSTED.  Without this,
        # a revoked credential gets a 1-hour TTL cooldown and then re-enters
        # rotation, failing immediately every hour until the user manually
        # removes it (issue #32849).  DEAD entries are excluded from rotation
        # unconditionally and only clear via an explicit re-auth write-side
        # sync (``_save_codex_tokens`` after a fresh device-code login).
        if self._is_terminal_auth_failure(status_code, normalized_error):
            terminal_status = STATUS_DEAD
        else:
            terminal_status = STATUS_EXHAUSTED
        updated = replace(
            entry,
            last_status=terminal_status,
            last_status_at=time.time(),
            last_error_code=status_code,
            last_error_reason=normalized_error.get("reason"),
            last_error_message=normalized_error.get("message"),
            last_error_reset_at=normalized_error.get("reset_at"),
        )
        self._replace_entry(entry, updated)
        self._persist()
        return updated

    def _sync_anthropic_entry_from_credentials_file(self, entry: PooledCredential) -> PooledCredential:
        """Sync a claude_code pool entry from ~/.claude/.credentials.json if tokens differ.

        OAuth refresh tokens are single-use. When something external (e.g.
        Claude Code CLI, or another profile's pool) refreshes the token, it
        writes the new pair to ~/.claude/.credentials.json. The pool entry's
        refresh token becomes stale. This method detects that and syncs.
        """
        if self.provider != "anthropic" or entry.source != "claude_code":
            return entry
        try:
            from agent.anthropic_adapter import read_claude_code_credentials
            creds = read_claude_code_credentials()
            if not creds:
                return entry
            file_refresh = creds.get("refreshToken", "")
            file_access = creds.get("accessToken", "")
            file_expires = creds.get("expiresAt", 0)
            # Sync when either token changed.  Access tokens can be re-issued
            # without a new refresh token (silent re-issue path), so checking
            # only refresh_token misses that case and leaves a stale
            # access_token in the pool → 401 on every request until the pool
            # entry's exhausted TTL expires.
            entry_access = entry.access_token or ""
            entry_refresh = entry.refresh_token or ""
            if (file_access or file_refresh) and (
                (file_access and file_access != entry_access)
                or (file_refresh and file_refresh != entry_refresh)
            ):
                logger.debug(
                    "Pool entry %s: syncing tokens from credentials file (tokens changed)",
                    entry.id,
                )
                updated = replace(
                    entry,
                    access_token=file_access or entry.access_token,
                    refresh_token=file_refresh or entry.refresh_token,
                    expires_at_ms=file_expires or entry.expires_at_ms,
                    last_status=None,
                    last_status_at=None,
                    last_error_code=None,
                    last_error_reason=None,
                    last_error_message=None,
                    last_error_reset_at=None,
                )
                self._replace_entry(entry, updated)
                self._persist()
                return updated
        except Exception as exc:
            logger.debug("Failed to sync from credentials file: %s", exc)
        return entry

    def _sync_codex_entry_from_auth_store(self, entry: PooledCredential) -> PooledCredential:
        """Sync a Codex device_code pool entry from auth.json if tokens differ.

        When a Codex OAuth access token expires (or the ChatGPT account hits
        its 5h/weekly quota), the pool entry gets marked ``STATUS_EXHAUSTED``
        with a ``last_error_reset_at`` that can be many hours in the future.
        Meanwhile the user may run ``hermes model`` / ``hermes auth`` which
        performs a fresh device-code login and writes new tokens to
        ``auth.json`` under ``_auth_store_lock``.  Without this sync the pool
        entry stays frozen until ``last_error_reset_at`` elapses — even
        though fresh credentials are sitting on disk — and every request
        fails with "no available entries (all exhausted or empty)".

        Mirrors the Nous/Anthropic resync paths above.  Only applies to
        device_code-sourced entries; env/API-key-sourced entries have no
        auth.json shadow to sync from.
        """
        if self.provider != "openai-codex" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "openai-codex")
            if not isinstance(state, dict):
                return entry
            tokens = state.get("tokens")
            if not isinstance(tokens, dict):
                return entry
            store_access = tokens.get("access_token", "")
            store_refresh = tokens.get("refresh_token", "")
            # Adopt auth.json tokens when either side differs.  Codex refresh
            # tokens are single-use too, so a fresh refresh_token from
            # another process means our entry's pair is consumed/stale.
            entry_access = entry.access_token or ""
            entry_refresh = entry.refresh_token or ""
            if store_access and (
                store_access != entry_access
                or (store_refresh and store_refresh != entry_refresh)
            ):
                logger.debug(
                    "Pool entry %s: syncing Codex tokens from auth.json "
                    "(refreshed by another process)",
                    entry.id,
                )
                field_updates: Dict[str, Any] = {
                    "access_token": store_access,
                    "refresh_token": store_refresh or entry.refresh_token,
                    "last_status": None,
                    "last_status_at": None,
                    "last_error_code": None,
                    "last_error_reason": None,
                    "last_error_message": None,
                    "last_error_reset_at": None,
                }
                if state.get("last_refresh"):
                    field_updates["last_refresh"] = state["last_refresh"]
                updated = replace(entry, **field_updates)
                self._replace_entry(entry, updated)
                self._persist()
                return updated
        except Exception as exc:
            logger.debug("Failed to sync Codex entry from auth.json: %s", exc)
        return entry

    def _sync_xai_oauth_entry_from_auth_store(self, entry: PooledCredential) -> PooledCredential:
        """Sync an xAI OAuth pool entry from auth.json if tokens differ.

        xAI OAuth refresh tokens are single-use.  When another Hermes process
        (or another profile sharing the same auth.json) refreshes the token,
        it writes the new pair to ``providers["xai-oauth"]["tokens"]`` under
        ``_auth_store_lock``.  Without this resync, our in-memory pool entry
        keeps the consumed refresh_token and the next ``_refresh_entry`` call
        would replay it and get a ``refresh_token_reused``-style 4xx.

        Only applies to entries seeded from the singleton (``device_code``);
        manually added entries are independent credentials with their own
        refresh-token lifecycle.
        """
        if self.provider != "xai-oauth" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "xai-oauth")
            if not isinstance(state, dict):
                return entry
            tokens = state.get("tokens")
            if not isinstance(tokens, dict):
                return entry
            store_access = tokens.get("access_token", "")
            store_refresh = tokens.get("refresh_token", "")
            entry_access = entry.access_token or ""
            entry_refresh = entry.refresh_token or ""
            if store_access and (
                store_access != entry_access
                or (store_refresh and store_refresh != entry_refresh)
            ):
                logger.debug(
                    "Pool entry %s: syncing xAI OAuth tokens from auth.json "
                    "(refreshed by another process)",
                    entry.id,
                )
                field_updates: Dict[str, Any] = {
                    "access_token": store_access,
                    "refresh_token": store_refresh or entry.refresh_token,
                    "last_status": None,
                    "last_status_at": None,
                    "last_error_code": None,
                    "last_error_reason": None,
                    "last_error_message": None,
                    "last_error_reset_at": None,
                }
                if state.get("last_refresh"):
                    field_updates["last_refresh"] = state["last_refresh"]
                updated = replace(entry, **field_updates)
                self._replace_entry(entry, updated)
                self._persist()
                return updated
        except Exception as exc:
            logger.debug("Failed to sync xAI OAuth entry from auth.json: %s", exc)
        return entry

    def _sync_xai_oauth_entry_from_pool_store(
        self, entry: PooledCredential
    ) -> PooledCredential:
        """Adopt a token pair rotated by another pool instance.

        Direct xAI integrations load a fresh ``CredentialPool`` for each
        request. Their in-memory locks therefore cannot protect xAI's
        single-use refresh token across concurrent requests or processes.
        This helper is called while the shared auth-store lock is held and
        re-reads the exact persisted row before a refresh POST is attempted.
        """
        if self.provider != "xai-oauth":
            return entry
        try:
            persisted = next(
                (
                    payload
                    for payload in read_credential_pool(self.provider)
                    if isinstance(payload, dict) and payload.get("id") == entry.id
                ),
                None,
            )
            if not isinstance(persisted, dict):
                return entry
            stored = PooledCredential.from_dict(self.provider, persisted)
            if (
                stored.access_token != entry.access_token
                or stored.refresh_token != entry.refresh_token
            ):
                logger.debug(
                    "Pool entry %s: adopting xAI OAuth tokens rotated by another pool instance",
                    entry.id,
                )
                self._replace_entry(entry, stored)
                return stored
        except Exception as exc:
            logger.debug("Failed to sync xAI OAuth entry from credential pool: %s", exc)
        return entry

    def _sync_nous_entry_from_auth_store(self, entry: PooledCredential) -> PooledCredential:
        """Sync a Nous pool entry from auth.json if tokens differ.

        Nous OAuth refresh tokens are single-use.  When another process
        (e.g. a concurrent cron) refreshes the token via
        ``resolve_nous_runtime_credentials``, it writes fresh tokens to
        auth.json under ``_auth_store_lock``.  The pool entry's tokens
        become stale.  This method detects that and adopts the newer pair,
        avoiding a "refresh token reuse" revocation on the Nous Portal.
        """
        if self.provider != "nous" or entry.source != "device_code":
            return entry
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                state = _load_provider_state(auth_store, "nous")
            if not state:
                return entry
            store_refresh = state.get("refresh_token", "")
            store_access = state.get("access_token", "")
            comparable_updates = {
                "access_token": store_access,
                "refresh_token": store_refresh,
                "expires_at": state.get("expires_at"),
                "agent_key": state.get("agent_key"),
                "agent_key_expires_at": state.get("agent_key_expires_at"),
                "inference_base_url": state.get("inference_base_url"),
            }
            should_sync = any(
                value not in (None, "") and getattr(entry, key, None) != value
                for key, value in comparable_updates.items()
            )
            if should_sync:
                logger.debug(
                    "Pool entry %s: syncing Nous state from auth.json",
                    entry.id,
                )
                field_updates: Dict[str, Any] = {
                    "last_status": None,
                    "last_status_at": None,
                    "last_error_code": None,
                    "last_error_reason": None,
                    "last_error_message": None,
                    "last_error_reset_at": None,
                }
                if store_access:
                    field_updates["access_token"] = store_access
                if store_refresh:
                    field_updates["refresh_token"] = store_refresh
                if state.get("expires_at"):
                    field_updates["expires_at"] = state["expires_at"]
                if state.get("agent_key"):
                    field_updates["agent_key"] = state["agent_key"]
                if state.get("agent_key_expires_at"):
                    field_updates["agent_key_expires_at"] = state["agent_key_expires_at"]
                if state.get("inference_base_url"):
                    field_updates["inference_base_url"] = state["inference_base_url"]
                extra_updates = dict(entry.extra)
                for extra_key in ("obtained_at", "expires_in", "agent_key_id",
                                  "agent_key_expires_in", "agent_key_reused",
                                  "agent_key_obtained_at"):
                    val = state.get(extra_key)
                    if val is not None:
                        extra_updates[extra_key] = val
                updated = replace(entry, extra=extra_updates, **field_updates)
                self._replace_entry(entry, updated)
                self._persist()
                return updated
        except Exception as exc:
            logger.debug("Failed to sync Nous entry from auth.json: %s", exc)
        return entry

    def _sync_device_code_entry_to_auth_store(self, entry: PooledCredential) -> None:
        """Write refreshed pool entry tokens back to auth.json providers.

        After a pool-level refresh, the pool entry has fresh tokens but
        auth.json's ``providers.<id>`` still holds the pre-refresh state.
        On the next ``load_pool()``, ``_seed_from_singletons()`` reads that
        stale state and can overwrite the fresh pool entry — potentially
        re-seeding a consumed single-use refresh token.

        Applies to any OAuth provider whose singleton lives in auth.json
        (currently Nous, OpenAI Codex, and xAI Grok OAuth).

        ``set_active=False`` on every write: a pool sync-back is a
        token-rotation side effect, not the user choosing a provider.
        Using ``_save_provider_state`` (which sets ``active_provider``)
        here would mean every Nous/Codex/xAI refresh in a multi-provider
        setup silently flips the ``active_provider`` flag — the next
        ``hermes`` invocation that defaults to the active provider
        (e.g. setup wizard, ``hermes auth status``) would land on
        whatever provider happened to refresh last, not whatever the
        user actually chose.
        """
        # Only sync entries that were seeded *from* a singleton.  Manually
        # added pool entries (source="manual:*") are independent credentials
        # and must not write back to the singleton.  All singleton-seeded
        # device-code sources (nous, openai-codex, xAI) use ``device_code``.
        if entry.source != "device_code":
            return
        try:
            with _auth_store_lock():
                auth_store = _load_auth_store()
                # Decide BEFORE writing whether this profile is reading the
                # grant from the global root (no own providers.<id> block) vs.
                # genuinely shadowing it. A pool refresh rotates single-use
                # OAuth refresh tokens, so a profile that resolved the grant
                # from root MUST write the rotated chain back to root too —
                # otherwise root keeps a revoked refresh token and every other
                # profile reading the stale root grant dies with
                # refresh_token_reused / invalid_grant once its access token
                # expires. This mirrors the xAI write-through in
                # hermes_cli.auth._save_xai_oauth_tokens (#43589); the pool
                # refresh path is the Codex/xAI analog reported in #48415.
                _wt_provider_id = {
                    "nous": "nous",
                    "openai-codex": "openai-codex",
                    "xai-oauth": "xai-oauth",
                }.get(self.provider)
                write_through_to_root = bool(_wt_provider_id) and not (
                    isinstance(auth_store.get("providers"), dict)
                    and isinstance(
                        auth_store["providers"].get(_wt_provider_id), dict
                    )
                )
                if self.provider == "nous":
                    state = _load_provider_state(auth_store, "nous")
                    if state is None:
                        return
                    state["access_token"] = entry.access_token
                    if entry.refresh_token:
                        state["refresh_token"] = entry.refresh_token
                    if entry.expires_at:
                        state["expires_at"] = entry.expires_at
                    if entry.agent_key:
                        state["agent_key"] = entry.agent_key
                    if entry.agent_key_expires_at:
                        state["agent_key_expires_at"] = entry.agent_key_expires_at
                    for extra_key in ("obtained_at", "expires_in", "agent_key_id",
                                      "agent_key_expires_in", "agent_key_reused",
                                      "agent_key_obtained_at"):
                        val = entry.extra.get(extra_key)
                        if val is not None:
                            state[extra_key] = val
                    if entry.inference_base_url:
                        state["inference_base_url"] = entry.inference_base_url
                    _store_provider_state(auth_store, "nous", state, set_active=False)

                elif self.provider == "openai-codex":
                    state = _load_provider_state(auth_store, "openai-codex")
                    if not isinstance(state, dict):
                        return
                    tokens = state.get("tokens")
                    if not isinstance(tokens, dict):
                        return
                    tokens["access_token"] = entry.access_token
                    if entry.refresh_token:
                        tokens["refresh_token"] = entry.refresh_token
                    if entry.last_refresh:
                        state["last_refresh"] = entry.last_refresh
                    _store_provider_state(auth_store, "openai-codex", state, set_active=False)

                elif self.provider == "xai-oauth":
                    state = _load_provider_state(auth_store, "xai-oauth")
                    if not isinstance(state, dict):
                        return
                    tokens = state.get("tokens")
                    if not isinstance(tokens, dict):
                        return
                    tokens["access_token"] = entry.access_token
                    if entry.refresh_token:
                        tokens["refresh_token"] = entry.refresh_token
                    if entry.last_refresh:
                        state["last_refresh"] = entry.last_refresh
                    _store_provider_state(auth_store, "xai-oauth", state, set_active=False)

                else:
                    return

                _save_auth_store(auth_store)
                if write_through_to_root and _wt_provider_id:
                    _write_through_provider_state_to_global_root(
                        _wt_provider_id, state
                    )
        except Exception as exc:
            logger.debug("Failed to sync %s pool entry back to auth store: %s", self.provider, exc)

    def _refresh_entry(self, entry: PooledCredential, *, force: bool) -> Optional[PooledCredential]:
        if entry.auth_type != AUTH_TYPE_OAUTH or not entry.refresh_token:
            if force:
                self._mark_exhausted(entry, None)
            return None

        # Codex and xAI OAuth refresh tokens are single-use.  The
        # sync→POST→write-back sequence below must run atomically across Hermes
        # processes: otherwise two processes can both adopt the same on-disk
        # token, both POST it, and the loser gets ``refresh_token_reused``.
        # Serialize the whole sequence through the shared cross-process
        # auth-store flock (the same lock and extended-timeout pattern used by
        # resolve_codex_runtime_credentials()).  When a waiter finally acquires
        # the lock, the in-lock re-sync below picks up the rotated token the
        # winner persisted and skips the POST.
        if self.provider in ("openai-codex", "xai-oauth"):
            sync_entry = (
                self._sync_codex_entry_from_auth_store
                if self.provider == "openai-codex"
                else self._sync_xai_oauth_entry_from_pool_store
            )
            with _auth_store_lock(
                timeout_seconds=self._single_use_refresh_lock_timeout()
            ):
                synced = sync_entry(entry)
                if self.provider == "openai-codex":
                    if synced is not entry:
                        entry = synced
                        if not force and not self._entry_needs_refresh(entry):
                            return entry
                    return self._refresh_entry_impl(entry, force=force)
                if (
                    synced.access_token != entry.access_token
                    or synced.refresh_token != entry.refresh_token
                ):
                    return synced
                return self._refresh_entry_impl(synced, force=force)
        return self._refresh_entry_impl(entry, force=force)

    def _single_use_refresh_lock_timeout(self) -> float:
        """Lock timeout for single-use-refresh-token providers.

        Covers the configured refresh POST timeout plus a margin so a slow
        token endpoint cannot make the flock give up before the refresh
        resolves.  Reads the provider's ``HERMES_*_REFRESH_TIMEOUT_SECONDS``
        override.
        """
        env_var = (
            "HERMES_CODEX_REFRESH_TIMEOUT_SECONDS"
            if self.provider == "openai-codex"
            else "HERMES_XAI_REFRESH_TIMEOUT_SECONDS"
        )
        refresh_timeout_seconds = auth_mod.env_float(env_var, 20)
        return max(
            float(auth_mod.AUTH_LOCK_TIMEOUT_SECONDS),
            float(refresh_timeout_seconds) + 5.0,
        )

    def _refresh_entry_impl(
        self, entry: PooledCredential, *, force: bool
    ) -> Optional[PooledCredential]:
        try:
            if self.provider == "anthropic":
                from agent.anthropic_adapter import refresh_anthropic_oauth_pure

                refreshed = refresh_anthropic_oauth_pure(
                    entry.refresh_token,
                    use_json=entry.source.endswith("hermes_pkce"),
                )
                updated = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    expires_at_ms=refreshed["expires_at_ms"],
                )
                # Keep ~/.claude/.credentials.json in sync so that the
                # fallback path (resolve_anthropic_token) and other profiles
                # see the latest tokens.
                if entry.source == "claude_code":
                    try:
                        from agent.anthropic_adapter import _write_claude_code_credentials
                        _write_claude_code_credentials(
                            refreshed["access_token"],
                            refreshed["refresh_token"],
                            refreshed["expires_at_ms"],
                        )
                    except Exception as wexc:
                        logger.debug("Failed to write refreshed token to credentials file: %s", wexc)
            elif self.provider == "openai-codex":
                # Adopt fresher tokens from auth.json before spending the
                # refresh_token — single-use tokens consumed by another Hermes
                # process sharing the same auth.json singleton would otherwise
                # trigger ``refresh_token_reused`` on the next POST.
                synced = self._sync_codex_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                refreshed = auth_mod.refresh_codex_oauth_pure(
                    entry.access_token,
                    entry.refresh_token,
                )
                updated = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    last_refresh=refreshed.get("last_refresh"),
                )
            elif self.provider == "xai-oauth":
                # Adopt fresher tokens from auth.json before spending the
                # refresh_token — single-use tokens consumed by another
                # process (or another profile sharing the singleton) would
                # otherwise trigger ``refresh_token_reused`` on the next
                # POST.  Only meaningful for singleton-seeded entries.
                synced = self._sync_xai_oauth_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                refreshed = auth_mod.refresh_xai_oauth_pure(
                    entry.access_token,
                    entry.refresh_token,
                )
                updated = replace(
                    entry,
                    access_token=refreshed["access_token"],
                    refresh_token=refreshed["refresh_token"],
                    last_refresh=refreshed.get("last_refresh"),
                )
            elif self.provider == "nous":
                synced = self._sync_nous_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                auth_mod.resolve_nous_runtime_credentials(
                    force_refresh=force,
                )
                updated = self._sync_nous_entry_from_auth_store(entry)
            else:
                return entry
        except Exception as exc:
            logger.debug("Credential refresh failed for %s/%s: %s", self.provider, entry.id, exc)
            # For anthropic claude_code entries: the refresh token may have been
            # consumed by another process. Check if ~/.claude/.credentials.json
            # has a newer token pair and retry once.
            if self.provider == "anthropic" and entry.source == "claude_code":
                synced = self._sync_anthropic_entry_from_credentials_file(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug("Retrying refresh with synced token from credentials file")
                    try:
                        from agent.anthropic_adapter import refresh_anthropic_oauth_pure
                        refreshed = refresh_anthropic_oauth_pure(
                            synced.refresh_token,
                            use_json=synced.source.endswith("hermes_pkce"),
                        )
                        updated = replace(
                            synced,
                            access_token=refreshed["access_token"],
                            refresh_token=refreshed["refresh_token"],
                            expires_at_ms=refreshed["expires_at_ms"],
                            last_status=STATUS_OK,
                            last_status_at=None,
                            last_error_code=None,
                        )
                        self._replace_entry(synced, updated)
                        self._persist()
                        try:
                            from agent.anthropic_adapter import _write_claude_code_credentials
                            _write_claude_code_credentials(
                                refreshed["access_token"],
                                refreshed["refresh_token"],
                                refreshed["expires_at_ms"],
                            )
                        except Exception as wexc:
                            logger.debug("Failed to write refreshed token to credentials file (retry path): %s", wexc)
                        return updated
                    except Exception as retry_exc:
                        logger.debug("Retry refresh also failed: %s", retry_exc)
                elif not self._entry_needs_refresh(synced):
                    # Credentials file had a valid (non-expired) token — use it directly
                    logger.debug("Credentials file has valid token, using without refresh")
                    return synced
            # For xai-oauth: same race as nous — another process may have
            # consumed the refresh token between our proactive sync and the
            # HTTP call.  Re-check auth.json and adopt the fresh tokens if
            # they have rotated since.  Only meaningful for singleton-seeded
            # (device_code) entries; manual entries don't share
            # state with the singleton.
            if self.provider == "xai-oauth":
                synced = self._sync_xai_oauth_entry_from_auth_store(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug(
                        "xAI OAuth refresh failed but auth.json has newer tokens — adopting"
                    )
                    updated = replace(
                        synced,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(synced, updated)
                    self._persist()
                    return updated
                # Terminal error: auth.json has no newer tokens — the stored
                # refresh_token is dead.  Clear it from auth.json so the next
                # session does not re-seed the same revoked credentials, and
                # remove all singleton-seeded xAI entries from the in-memory
                # pool. Mirrors the Nous quarantine path above.
                if auth_mod._is_terminal_xai_oauth_refresh_error(exc):
                    logger.debug(
                        "xAI OAuth refresh token is terminally invalid; clearing local token state"
                    )
                    try:
                        with _auth_store_lock():
                            auth_store = _load_auth_store()
                            state = _load_provider_state(auth_store, "xai-oauth") or {}
                            if isinstance(state, dict):
                                tokens = state.get("tokens") or {}
                                if isinstance(tokens, dict):
                                    store_refresh = str(tokens.get("refresh_token") or "").strip()
                                    entry_refresh = str(entry.refresh_token or "").strip()
                                    if not store_refresh or store_refresh == entry_refresh:
                                        tokens.pop("access_token", None)
                                        tokens.pop("refresh_token", None)
                                        state["tokens"] = tokens
                                        state["last_auth_error"] = {
                                            "provider": "xai-oauth",
                                            "code": getattr(exc, "code", "unknown"),
                                            "message": str(exc),
                                            "reason": "credential_pool_refresh_failure",
                                            "relogin_required": True,
                                            "at": datetime.now(timezone.utc).isoformat(),
                                        }
                                        _save_provider_state(auth_store, "xai-oauth", state)
                                        _save_auth_store(auth_store)
                    except Exception as clear_exc:
                        logger.debug(
                            "Failed to clear terminal xAI OAuth state: %s", clear_exc
                        )
                    removed_ids = [
                        item.id for item in self._entries
                        if item.source == "device_code"
                    ]
                    self._entries = [
                        item for item in self._entries
                        if item.source != "device_code"
                    ]
                    if self._current_id == entry.id:
                        self._current_id = None
                    self._persist(removed_ids=removed_ids)
                    return None
            # For openai-codex: same race as xAI/nous — another Hermes process
            # may have consumed the refresh token between our proactive sync
            # and the HTTP call.  Re-check auth.json and adopt the fresh tokens
            # if they have rotated since.
            if self.provider == "openai-codex":
                synced = self._sync_codex_entry_from_auth_store(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug(
                        "Codex OAuth refresh failed but auth.json has newer tokens — adopting"
                    )
                    updated = replace(
                        synced,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(synced, updated)
                    self._persist()
                    return updated
                # Terminal error: auth.json has no newer tokens — the stored
                # refresh_token is dead.  Clear it from auth.json so the next
                # session does not re-seed the same revoked credentials, and
                # remove all singleton-seeded (device_code) entries from the
                # in-memory pool.  Mirrors the xAI and Nous quarantine paths.
                if auth_mod._is_terminal_codex_oauth_refresh_error(exc):
                    logger.debug(
                        "Codex OAuth refresh token is terminally invalid; clearing local token state"
                    )
                    try:
                        with _auth_store_lock():
                            auth_store = _load_auth_store()
                            state = _load_provider_state(auth_store, "openai-codex") or {}
                            if isinstance(state, dict):
                                tokens = state.get("tokens") or {}
                                if isinstance(tokens, dict):
                                    store_refresh = str(tokens.get("refresh_token") or "").strip()
                                    entry_refresh = str(entry.refresh_token or "").strip()
                                    if not store_refresh or store_refresh == entry_refresh:
                                        tokens.pop("access_token", None)
                                        tokens.pop("refresh_token", None)
                                        state["tokens"] = tokens
                                        state["last_auth_error"] = {
                                            "provider": "openai-codex",
                                            "code": getattr(exc, "code", "unknown"),
                                            "message": str(exc),
                                            "reason": "credential_pool_refresh_failure",
                                            "relogin_required": True,
                                            "at": datetime.now(timezone.utc).isoformat(),
                                        }
                                        _save_provider_state(auth_store, "openai-codex", state)
                                        _save_auth_store(auth_store)
                    except Exception as clear_exc:
                        logger.debug(
                            "Failed to clear terminal Codex OAuth state: %s", clear_exc
                        )
                    removed_ids = [
                        item.id for item in self._entries
                        if item.source == "device_code"
                    ]
                    self._entries = [
                        item for item in self._entries
                        if item.source != "device_code"
                    ]
                    if self._current_id == entry.id:
                        self._current_id = None
                    self._persist(removed_ids=removed_ids)
                    return None
            # For nous: another process may have consumed the refresh token
            # between our proactive sync and the HTTP call.  Re-sync from
            # auth.json and adopt the fresh tokens if available.
            if self.provider == "nous":
                synced = self._sync_nous_entry_from_auth_store(entry)
                if synced.refresh_token != entry.refresh_token:
                    logger.debug("Nous refresh failed but auth.json has newer tokens — adopting")
                    updated = replace(
                        synced,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(synced, updated)
                    self._persist()
                    self._sync_device_code_entry_to_auth_store(updated)
                    return updated
                if auth_mod._is_terminal_nous_refresh_error(exc):
                    logger.debug("Nous refresh token is terminally invalid; clearing local token state")
                    try:
                        with _auth_store_lock():
                            auth_store = _load_auth_store()
                            state = _load_provider_state(auth_store, "nous") or {
                                "client_id": entry.client_id,
                                "portal_base_url": entry.portal_base_url,
                                "inference_base_url": entry.inference_base_url,
                                "token_type": entry.token_type,
                                "scope": entry.scope,
                                "tls": entry.tls,
                            }
                            store_refresh = str(state.get("refresh_token") or "").strip()
                            entry_refresh = str(entry.refresh_token or "").strip()
                            if not store_refresh or store_refresh == entry_refresh:
                                auth_mod._quarantine_nous_oauth_state(
                                    state,
                                    exc,
                                    reason="credential_pool_refresh_failure",
                                )
                                auth_mod._quarantine_nous_pool_entries(
                                    auth_store,
                                    exc,
                                    reason="credential_pool_refresh_failure",
                                )
                                _save_provider_state(auth_store, "nous", state)
                                _save_auth_store(auth_store)
                    except Exception as clear_exc:
                        logger.debug("Failed to clear terminal Nous OAuth state: %s", clear_exc)

                    singleton_sources = {
                        auth_mod.NOUS_DEVICE_CODE_SOURCE,
                        f"manual:{auth_mod.NOUS_DEVICE_CODE_SOURCE}",
                    }
                    removed_ids = [
                        item.id for item in self._entries
                        if item.source in singleton_sources
                    ]
                    self._entries = [
                        item for item in self._entries
                        if item.source not in singleton_sources
                    ]
                    if self._current_id == entry.id:
                        self._current_id = None
                    self._persist(removed_ids=removed_ids)
                    return None
            self._mark_exhausted(entry, None)
            return None

        updated = replace(
            updated,
            last_status=STATUS_OK,
            last_status_at=None,
            last_error_code=None,
            last_error_reason=None,
            last_error_message=None,
            last_error_reset_at=None,
        )
        self._replace_entry(entry, updated)
        self._persist()
        # Sync refreshed tokens back to auth.json providers so that
        # _seed_from_singletons() on the next load_pool() sees fresh state
        # instead of re-seeding stale/consumed tokens.
        self._sync_device_code_entry_to_auth_store(updated)
        return updated

    def _entry_needs_refresh(self, entry: PooledCredential) -> bool:
        if entry.auth_type != AUTH_TYPE_OAUTH:
            return False
        if self.provider == "anthropic":
            if entry.expires_at_ms is None:
                return False
            return int(entry.expires_at_ms) <= int(time.time() * 1000) + 120_000
        if self.provider == "openai-codex":
            return _codex_access_token_is_expiring(
                entry.access_token,
                CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
            )
        if self.provider == "xai-oauth":
            return auth_mod._xai_access_token_is_expiring(
                entry.access_token,
                auth_mod._xai_proactive_refresh_skew_seconds(entry.access_token),
            )
        if self.provider == "nous":
            # Nous refresh can require network access and should happen when
            # runtime credentials are actually resolved, not merely when the pool
            # is enumerated for listing, migration, or selection.
            return False
        return False

    def select(self) -> Optional[PooledCredential]:
        with self._lock:
            return self._select_unlocked()

    def _available_entries(self, *, clear_expired: bool = False, refresh: bool = False) -> List[PooledCredential]:
        """Return entries not currently in exhaustion cooldown.

        When *clear_expired* is True, entries whose cooldown has elapsed are
        reset to STATUS_OK and persisted.  When *refresh* is True, entries
        that need a token refresh are refreshed (skipped on failure).
        """
        now = time.time()
        cleared_any = False
        entries_to_prune: List[str] = []
        available: List[PooledCredential] = []
        for entry in self._entries:
            # Borrowed credentials persist as metadata-only references and are
            # hydrated from their live source on load.  A stale duplicate row
            # can remain unhydrated; never lease or select it as an empty key.
            if entry.auth_type == AUTH_TYPE_API_KEY and not entry.runtime_api_key:
                continue
            # For anthropic claude_code entries, sync from the credentials file
            # before any status/refresh checks. This picks up tokens refreshed
            # by other processes (Claude Code CLI, other Hermes profiles).
            if (self.provider == "anthropic" and entry.source == "claude_code"
                    and entry.last_status in {STATUS_EXHAUSTED, STATUS_DEAD}):
                synced = self._sync_anthropic_entry_from_credentials_file(entry)
                if synced is not entry:
                    entry = synced
                    cleared_any = True
            # For nous entries, sync from auth.json before status checks.
            # Another process may have successfully refreshed via
            # resolve_nous_runtime_credentials(), making this entry's
            # exhausted status stale.
            if (self.provider == "nous"
                    and entry.source == "device_code"
                    and entry.last_status in {STATUS_EXHAUSTED, STATUS_DEAD}):
                synced = self._sync_nous_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                    cleared_any = True
            # For openai-codex entries, same pattern: the user may have
            # re-authed via `hermes model` / `hermes auth` after a 429/401,
            # leaving fresh tokens on disk while the pool entry is still
            # frozen behind last_error_reset_at (can be hours in the
            # future for ChatGPT weekly windows).
            if (self.provider == "openai-codex"
                    and entry.source == "device_code"
                    and entry.last_status in {STATUS_EXHAUSTED, STATUS_DEAD}):
                synced = self._sync_codex_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                    cleared_any = True
            # For xai-oauth singleton-seeded entries, identical pattern:
            # an entry frozen as exhausted may simply be holding stale
            # tokens that another process (or a fresh `hermes model` ->
            # xAI Grok OAuth login) has since rotated in auth.json.
            if (self.provider == "xai-oauth"
                    and entry.source == "device_code"
                    and entry.last_status in {STATUS_EXHAUSTED, STATUS_DEAD}):
                synced = self._sync_xai_oauth_entry_from_auth_store(entry)
                if synced is not entry:
                    entry = synced
                    cleared_any = True
            if entry.last_status == STATUS_DEAD:
                # Manual DEAD credentials get pruned after a 24h quiet window
                # so the pool doesn't accumulate dead entries forever.  The
                # user can always re-add via ``hermes auth add``.  Singleton-
                # seeded DEAD entries are kept so the audit trail (label,
                # last_error_reason, timestamps) stays visible — pruning them
                # would just be undone by ``_seed_from_singletons`` on the
                # next load anyway.
                if _is_manual_source(entry.source):
                    dead_at = entry.last_status_at or 0
                    if dead_at and now - dead_at > DEAD_MANUAL_PRUNE_TTL_SECONDS:
                        _label = entry.label or entry.id[:8]
                        logger.warning(
                            "credential pool: pruning DEAD manual entry %s "
                            "(reason=%s, age=%.1fh) — re-add via `hermes auth add %s`",
                            _label,
                            entry.last_error_reason or "unknown",
                            (now - dead_at) / 3600.0,
                            self.provider,
                        )
                        # Mark for removal after the loop completes; we can't
                        # mutate self._entries while iterating.
                        entries_to_prune.append(entry.id)
                        cleared_any = True
                # Permanently failed credentials never re-enter rotation via
                # TTL.  They only clear when a write-side re-auth sync rewrites
                # the tokens (e.g. ``_save_codex_tokens`` after a fresh
                # device-code login).  The auth.json-sync paths below handle
                # the re-auth case for OAuth singletons.
                continue
            if entry.last_status == STATUS_EXHAUSTED:
                exhausted_until = _exhausted_until(entry)
                if exhausted_until is not None and now < exhausted_until:
                    continue
                if clear_expired:
                    cleared = replace(
                        entry,
                        last_status=STATUS_OK,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                    self._replace_entry(entry, cleared)
                    entry = cleared
                    cleared_any = True
            if refresh and self._entry_needs_refresh(entry):
                refreshed = self._refresh_entry(entry, force=False)
                if refreshed is None:
                    continue
                entry = refreshed
            available.append(entry)
        if entries_to_prune:
            pruned_ids = set(entries_to_prune)
            self._entries = [e for e in self._entries if e.id not in pruned_ids]
        if cleared_any:
            self._persist(removed_ids=entries_to_prune)
        return available

    def _select_unlocked(self, *, refresh: bool = True) -> Optional[PooledCredential]:
        available = self._available_entries(clear_expired=True, refresh=refresh)
        if not available:
            self._current_id = None
            logger.info("credential pool: no available entries (all exhausted or empty)")
            return None

        if self._strategy == STRATEGY_RANDOM:
            entry = random.choice(available)
            self._current_id = entry.id
            return entry

        if self._strategy == STRATEGY_LEAST_USED and len(available) > 1:
            entry = min(available, key=lambda e: e.request_count)
            # Increment usage counter so subsequent selections distribute load
            updated = replace(entry, request_count=entry.request_count + 1)
            self._replace_entry(entry, updated)
            self._current_id = entry.id
            return updated

        if self._strategy == STRATEGY_ROUND_ROBIN and len(available) > 1:
            entry = available[0]
            rotated = [candidate for candidate in self._entries if candidate.id != entry.id]
            rotated.append(replace(entry, priority=len(self._entries) - 1))
            self._entries = [replace(candidate, priority=idx) for idx, candidate in enumerate(rotated)]
            self._persist()
            self._current_id = entry.id
            return self.current() or entry

        entry = available[0]
        self._current_id = entry.id
        return entry

    def peek(self) -> Optional[PooledCredential]:
        current = self.current()
        if current is not None:
            return current
        available = self._available_entries()
        return available[0] if available else None

    def mark_exhausted_and_rotate(
        self,
        *,
        status_code: Optional[int],
        error_context: Optional[Dict[str, Any]] = None,
        api_key_hint: Optional[str] = None,
    ) -> Optional[PooledCredential]:
        with self._lock:
            entry = None
            if api_key_hint:
                # Prefer the specific entry whose API key matches the one that
                # actually failed.  When this pool was freshly loaded from disk
                # (another process already rotated), current() is None and
                # _select_unlocked() would return the NEXT key — the wrong one.
                entry = next(
                    (e for e in self._entries if e.runtime_api_key == api_key_hint),
                    None,
                )
            if entry is None:
                entry = self.current() or self._select_unlocked()
            if entry is None:
                return None
            _label = entry.label or entry.id[:8]
            self._mark_exhausted(entry, status_code, error_context)
            # Re-read the updated entry to log the correct terminal state.
            updated_entry = next(
                (e for e in self._entries if e.id == entry.id), entry,
            )
            if updated_entry.last_status == STATUS_DEAD:
                logger.warning(
                    "credential pool: marking %s DEAD (status=%s, reason=%s) — "
                    "permanently failed, will NOT re-enter rotation until re-auth",
                    _label, status_code, updated_entry.last_error_reason or "unknown",
                )
            else:
                logger.info(
                    "credential pool: marking %s exhausted (status=%s), rotating",
                    _label, status_code,
                )
            self._current_id = None
            next_entry = self._select_unlocked()
            if next_entry:
                _next_label = next_entry.label or next_entry.id[:8]
                logger.info("credential pool: rotated to %s", _next_label)
            return next_entry

    def acquire_lease(self, credential_id: Optional[str] = None) -> Optional[str]:
        """Acquire a soft lease on a credential.

        If a specific credential_id is provided, lease that entry directly.
        Otherwise prefer the least-leased available credential, using priority as
        a stable tie-breaker. When every credential is already at the soft cap,
        still return the least-leased one instead of blocking.
        """
        with self._lock:
            if credential_id:
                self._active_leases[credential_id] = self._active_leases.get(credential_id, 0) + 1
                self._current_id = credential_id
                return credential_id

            available = self._available_entries(clear_expired=True, refresh=True)
            if not available:
                return None

            below_cap = [
                entry for entry in available
                if self._active_leases.get(entry.id, 0) < self._max_concurrent
            ]
            candidates = below_cap if below_cap else available
            chosen = min(
                candidates,
                key=lambda entry: (self._active_leases.get(entry.id, 0), entry.priority),
            )
            self._active_leases[chosen.id] = self._active_leases.get(chosen.id, 0) + 1
            self._current_id = chosen.id
            return chosen.id

    def release_lease(self, credential_id: str) -> None:
        """Release a previously acquired credential lease."""
        with self._lock:
            count = self._active_leases.get(credential_id, 0)
            if count <= 1:
                self._active_leases.pop(credential_id, None)
            else:
                self._active_leases[credential_id] = count - 1

    def try_refresh_current(self) -> Optional[PooledCredential]:
        with self._lock:
            return self._try_refresh_current_unlocked()

    def try_refresh_matching(
        self, api_key_hint: Optional[str] = None
    ) -> Optional[PooledCredential]:
        """Force-refresh the entry that supplied ``api_key_hint``.

        Direct provider integrations may reload the pool after a request has
        already failed, so they cannot rely on ``current_id`` identifying the
        issuing credential. With no hint, select an entry without first doing
        the normal proactive refresh; the forced refresh below must consume a
        rotating refresh token exactly once.
        """
        with self._lock:
            entry = None
            if api_key_hint:
                entry = next(
                    (
                        candidate
                        for candidate in self._entries
                        if candidate.runtime_api_key == api_key_hint
                    ),
                    None,
                )
            else:
                entry = self.current() or self._select_unlocked(refresh=False)
            if entry is None:
                return None
            self._current_id = entry.id
            return self._try_refresh_current_unlocked()

    def _try_refresh_current_unlocked(self) -> Optional[PooledCredential]:
        entry = self.current()
        if entry is None:
            return None
        refreshed = self._refresh_entry(entry, force=True)
        if refreshed is not None:
            self._current_id = refreshed.id
        return refreshed

    def reset_statuses(self) -> int:
        count = 0
        new_entries = []
        for entry in self._entries:
            if entry.last_status or entry.last_status_at or entry.last_error_code:
                new_entries.append(
                    replace(
                        entry,
                        last_status=None,
                        last_status_at=None,
                        last_error_code=None,
                        last_error_reason=None,
                        last_error_message=None,
                        last_error_reset_at=None,
                    )
                )
                count += 1
            else:
                new_entries.append(entry)
        if count:
            self._entries = new_entries
            self._persist()
        return count

    def remove_index(self, index: int) -> Optional[PooledCredential]:
        if index < 1 or index > len(self._entries):
            return None
        removed = self._entries.pop(index - 1)
        self._entries = [
            replace(entry, priority=new_priority)
            for new_priority, entry in enumerate(self._entries)
        ]
        write_credential_pool(
            self.provider,
            [entry.to_dict() for entry in self._entries],
            removed_ids=[removed.id],
        )
        if self._current_id == removed.id:
            self._current_id = None
        return removed

    def resolve_target(self, target: Any) -> Tuple[Optional[int], Optional[PooledCredential], Optional[str]]:
        raw = str(target or "").strip()
        if not raw:
            return None, None, "No credential target provided."

        for idx, entry in enumerate(self._entries, start=1):
            if entry.id == raw:
                return idx, entry, None

        label_matches = [
            (idx, entry)
            for idx, entry in enumerate(self._entries, start=1)
            if entry.label.strip().lower() == raw.lower()
        ]
        if len(label_matches) == 1:
            return label_matches[0][0], label_matches[0][1], None
        if len(label_matches) > 1:
            return None, None, f'Ambiguous credential label "{raw}". Use the numeric index or entry id instead.'
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(self._entries):
                return index, self._entries[index - 1], None
            return None, None, f"No credential #{index}."
        return None, None, f'No credential matching "{raw}".'

    def add_entry(self, entry: PooledCredential) -> PooledCredential:
        entry = replace(entry, priority=_next_priority(self._entries))
        self._entries.append(entry)
        self._persist()
        return entry


def _upsert_entry(entries: List[PooledCredential], provider: str, source: str, payload: Dict[str, Any]) -> bool:
    matching_indices = []
    for idx, entry in enumerate(entries):
        if entry.source == source:
            matching_indices.append(idx)

    existing_idx = matching_indices[0] if matching_indices else None
    duplicate_indices = set(matching_indices[1:])
    if duplicate_indices:
        entries[:] = [entry for idx, entry in enumerate(entries) if idx not in duplicate_indices]

    if existing_idx is None:
        payload.setdefault("id", uuid.uuid4().hex[:6])
        payload.setdefault("priority", _next_priority(entries))
        payload.setdefault("label", payload.get("label") or source)
        entries.append(PooledCredential.from_dict(provider, payload))
        return True

    existing = entries[existing_idx]
    field_updates = {}
    extra_updates = {}
    _field_names = {f.name for f in fields(existing)}
    for key, value in payload.items():
        if key in {"id", "priority"} or value is None:
            continue
        if key == "label" and existing.label:
            continue
        if key in _field_names:
            if getattr(existing, key) != value:
                field_updates[key] = value
        elif key in _EXTRA_KEYS:
            if existing.extra.get(key) != value:
                extra_updates[key] = value
    if field_updates or extra_updates:
        if extra_updates:
            field_updates["extra"] = {**existing.extra, **extra_updates}
        updated = replace(existing, **field_updates)
        entries[existing_idx] = updated
        # Runtime-only borrowed secret updates should refresh the in-memory
        # entry without forcing auth.json churn when the disk-safe payload is
        # unchanged (for example env keys with the same fingerprint).
        return bool(duplicate_indices) or existing.to_dict() != updated.to_dict()
    return bool(duplicate_indices)


def _normalize_pool_priorities(provider: str, entries: List[PooledCredential]) -> bool:
    if provider != "anthropic":
        return False

    source_rank = {
        "env:ANTHROPIC_TOKEN": 0,
        "env:CLAUDE_CODE_OAUTH_TOKEN": 1,
        "hermes_pkce": 2,
        "claude_code": 3,
        "env:ANTHROPIC_API_KEY": 4,
    }
    manual_entries = sorted(
        (entry for entry in entries if _is_manual_source(entry.source)),
        key=lambda entry: entry.priority,
    )
    seeded_entries = sorted(
        (entry for entry in entries if not _is_manual_source(entry.source)),
        key=lambda entry: (
            source_rank.get(entry.source, len(source_rank)),
            entry.priority,
            entry.label,
        ),
    )

    ordered = [*manual_entries, *seeded_entries]
    id_to_idx = {entry.id: idx for idx, entry in enumerate(entries)}
    changed = False
    for new_priority, entry in enumerate(ordered):
        if entry.priority != new_priority:
            entries[id_to_idx[entry.id]] = replace(entry, priority=new_priority)
            changed = True
    return changed


def _seed_from_singletons(provider: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    changed = False
    active_sources: Set[str] = set()
    auth_store = _load_auth_store()

    # Shared suppression gate — used at every upsert site so
    # `hermes auth remove <provider> <N>` is stable across all source types.
    try:
        from hermes_cli.auth import is_source_suppressed as _is_suppressed
    except ImportError:
        def _is_suppressed(_p, _s):  # type: ignore[misc]
            return False

    if provider == "anthropic":
        # Only auto-discover external credentials (Claude Code, Hermes PKCE)
        # when the user has explicitly configured anthropic as their provider.
        # Without this gate, auxiliary client fallback chains silently read
        # ~/.claude/.credentials.json without user consent.  See PR #4210.
        try:
            from hermes_cli.auth import is_provider_explicitly_configured
            if not is_provider_explicitly_configured("anthropic"):
                return changed, active_sources
        except ImportError:
            pass

        # API-key vs OAuth is a user-visible choice at `hermes setup` ("Claude
        # Pro/Max subscription" vs "Anthropic API key").  The signal that the
        # user picked the API-key path is: ANTHROPIC_API_KEY set in the env,
        # AND no OAuth env vars set — `save_anthropic_api_key()` writes the
        # API key and zeros ANTHROPIC_TOKEN; `save_anthropic_oauth_token()`
        # does the inverse.  When that signal is present we MUST NOT seed
        # autodiscovered OAuth tokens (~/.claude/.credentials.json from the
        # Claude Code CLI, hermes_pkce creds from a previous OAuth login)
        # into the anthropic pool — otherwise rotation on a 401/429 silently
        # flips the session onto an OAuth credential, which forces the Claude
        # Code identity injection, `mcp_` tool-name rewrite, and claude-cli
        # User-Agent header (`agent/anthropic_adapter.py:2128`).  Users who
        # explicitly opted into the API-key path are explicitly opting OUT of
        # that masquerade.  Prefer ~/.hermes/.env over os.environ for the
        # same reason `_seed_from_env` does — that's the authoritative file
        # that `hermes setup` writes.
        _env_file = load_env()

        def _env_val(key: str) -> str:
            return (_env_file.get(key) or _get_secret(key, "") or "").strip()

        anthropic_api_key = _env_val("ANTHROPIC_API_KEY")
        anthropic_oauth_env = (
            _env_val("ANTHROPIC_TOKEN") or _env_val("CLAUDE_CODE_OAUTH_TOKEN")
        )
        api_key_path_explicit = bool(anthropic_api_key and not anthropic_oauth_env)

        if api_key_path_explicit:
            # Prune any stale autodiscovered OAuth entries that may have been
            # seeded into the on-disk pool during a previous OAuth session.
            # Without this, switching OAuth -> API key at setup leaves the
            # OAuth entries dormant in auth.json forever and rotation on a
            # transient 401 could revive them.
            retained = [
                entry for entry in entries
                if entry.source not in {"hermes_pkce", "claude_code"}
            ]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True
            return changed, active_sources

        from agent.anthropic_adapter import read_claude_code_credentials, read_hermes_oauth_credentials

        for source_name, creds in (
            ("hermes_pkce", read_hermes_oauth_credentials()),
            ("claude_code", read_claude_code_credentials()),
        ):
            if creds and creds.get("accessToken"):
                if _is_suppressed(provider, source_name):
                    continue
                active_sources.add(source_name)
                changed |= _upsert_entry(
                    entries,
                    provider,
                    source_name,
                    {
                        "source": source_name,
                        "auth_type": AUTH_TYPE_OAUTH,
                        "access_token": creds.get("accessToken", ""),
                        "refresh_token": creds.get("refreshToken"),
                        "expires_at_ms": creds.get("expiresAt"),
                        "label": label_from_token(creds.get("accessToken", ""), source_name),
                    },
                )

    elif provider == "nous":
        state = _load_provider_state(auth_store, "nous")
        has_runtime_material = bool(
            isinstance(state, dict)
            and (
                str(state.get("access_token") or "").strip()
                or str(state.get("agent_key") or "").strip()
            )
        )
        if state and not has_runtime_material:
            retained = [
                entry for entry in entries
                if entry.source not in {"device_code", "manual:device_code"}
            ]
            if len(retained) != len(entries):
                entries[:] = retained
                changed = True
        if state and has_runtime_material and not _is_suppressed(provider, "device_code"):
            active_sources.add("device_code")
            # Prefer a user-supplied label embedded in the singleton state
            # (set by persist_nous_credentials(label=...) when the user ran
            # `hermes auth add nous --label <name>`).  Fall back to the
            # auto-derived token fingerprint for logins that didn't supply one.
            custom_label = str(state.get("label") or "").strip()
            seeded_label = custom_label or label_from_token(
                state.get("access_token", ""), "device_code"
            )
            changed |= _upsert_entry(
                entries,
                provider,
                "device_code",
                {
                    "source": "device_code",
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": state.get("access_token", ""),
                    "refresh_token": state.get("refresh_token"),
                    "expires_at": state.get("expires_at"),
                    "token_type": state.get("token_type"),
                    "scope": state.get("scope"),
                    "client_id": state.get("client_id"),
                    "portal_base_url": state.get("portal_base_url"),
                    "inference_base_url": state.get("inference_base_url"),
                    "agent_key": state.get("agent_key"),
                    "agent_key_expires_at": state.get("agent_key_expires_at"),
                    # Carry the refresh timestamps into the pool so
                    # freshness-sensitive consumers (self-heal hooks, pool
                    # pruning by age) can distinguish just-refreshed credentials
                    # from stale ones.  Without these, fresh device_code
                    # entries get obtained_at=None and look older than they
                    # are (#15099).
                    "obtained_at": state.get("obtained_at"),
                    "expires_in": state.get("expires_in"),
                    "agent_key_id": state.get("agent_key_id"),
                    "agent_key_expires_in": state.get("agent_key_expires_in"),
                    "agent_key_reused": state.get("agent_key_reused"),
                    "agent_key_obtained_at": state.get("agent_key_obtained_at"),
                    "tls": state.get("tls") if isinstance(state.get("tls"), dict) else None,
                    "label": seeded_label,
                },
            )

    elif provider == "copilot":
        # Copilot tokens are resolved dynamically via `gh auth token` or
        # env vars (COPILOT_GITHUB_TOKEN / GH_TOKEN).  They don't live in
        # the auth store or credential pool, so we resolve them here.
        try:
            from hermes_cli.copilot_auth import resolve_copilot_token, get_copilot_api_token
            token, source = resolve_copilot_token()
            if token:
                api_token, enterprise_base_url = get_copilot_api_token(token)
                source_name = "gh_cli" if "gh" in source.lower() else f"env:{source}"
                if not _is_suppressed(provider, source_name):
                    active_sources.add(source_name)
                    pconfig = PROVIDER_REGISTRY.get(provider)
                    # Use enterprise base URL from token exchange if available,
                    # otherwise fall back to the provider's default.
                    effective_base_url = enterprise_base_url or (
                        pconfig.inference_base_url if pconfig else ""
                    )
                    changed |= _upsert_entry(
                        entries,
                        provider,
                        source_name,
                        {
                            "source": source_name,
                            "auth_type": AUTH_TYPE_API_KEY,
                            "access_token": api_token,
                            "base_url": effective_base_url,
                            "label": source,
                        },
                    )
        except Exception as exc:
            logger.debug("Copilot token seed failed: %s", exc)

    elif provider == "qwen-oauth":
        # Qwen OAuth tokens live in ~/.qwen/oauth_creds.json, written by
        # the Qwen CLI (`qwen auth qwen-oauth`).  They aren't in the
        # Hermes auth store or env vars, so resolve them here.
        # Use refresh_if_expiring=False to avoid network calls during
        # pool loading / provider discovery.
        try:
            from hermes_cli.auth import resolve_qwen_runtime_credentials
            creds = resolve_qwen_runtime_credentials(refresh_if_expiring=False)
            token = creds.get("api_key", "")
            if token:
                source_name = creds.get("source", "qwen-cli")
                if not _is_suppressed(provider, source_name):
                    active_sources.add(source_name)
                    changed |= _upsert_entry(
                        entries,
                        provider,
                        source_name,
                        {
                            "source": source_name,
                            "auth_type": AUTH_TYPE_OAUTH,
                            "access_token": token,
                            "expires_at_ms": creds.get("expires_at_ms"),
                            "base_url": creds.get("base_url", ""),
                            "label": creds.get("auth_file", source_name),
                        },
                    )
        except Exception as exc:
            logger.debug("Qwen OAuth token seed failed: %s", exc)

    elif provider == "minimax-oauth":
        # MiniMax OAuth tokens live in ~/.hermes/auth.json providers.minimax-oauth.
        # Seed the pool so `/auth list` reflects the logged-in state and the
        # standard `hermes auth remove minimax-oauth <N>` flow works.
        # Use refresh_if_expiring=False equivalent: resolve_minimax_oauth_runtime_credentials
        # always refreshes on expiry, so instead read raw state here to avoid
        # surprise network calls during provider discovery.
        try:
            from hermes_cli.auth import get_provider_auth_state
            state = get_provider_auth_state("minimax-oauth")
            if state and state.get("access_token"):
                source_name = "oauth"
                if not _is_suppressed(provider, source_name):
                    active_sources.add(source_name)
                    expires_at_ms = None
                    try:
                        from datetime import datetime as _dt
                        raw = state.get("expires_at", "")
                        if raw:
                            expires_at_ms = int(_dt.fromisoformat(raw).timestamp() * 1000)
                    except Exception:
                        expires_at_ms = None
                    base_url = str(state.get("inference_base_url", "") or "").rstrip("/")
                    changed |= _upsert_entry(
                        entries,
                        provider,
                        source_name,
                        {
                            "source": source_name,
                            "auth_type": AUTH_TYPE_OAUTH,
                            "access_token": state["access_token"],
                            "refresh_token": state.get("refresh_token"),
                            "expires_at_ms": expires_at_ms,
                            "base_url": base_url,
                            "label": state.get("label", "") or label_from_token(
                                state.get("access_token", ""), source_name
                            ),
                        },
                    )
        except Exception as exc:
            logger.debug("MiniMax OAuth token seed failed: %s", exc)

    elif provider == "openai-codex":
        # Respect user suppression — `hermes auth remove openai-codex` marks
        # the device_code source as suppressed so it won't be re-seeded from
        # the Hermes auth store.  Without this gate the removal is instantly
        # undone on the next load_pool() call.
        if _is_suppressed(provider, "device_code"):
            return changed, active_sources

        state = _load_provider_state(auth_store, "openai-codex")
        tokens = state.get("tokens") if isinstance(state, dict) else None
        # Hermes owns its own Codex auth state — we do NOT auto-import from
        # ~/.codex/auth.json at pool-load time.  OAuth refresh tokens are
        # single-use, so sharing them with Codex CLI / VS Code causes
        # refresh_token_reused race failures.  Users who want to adopt
        # existing Codex CLI credentials get a one-time, explicit prompt
        # via `hermes auth openai-codex`.
        if isinstance(tokens, dict) and tokens.get("access_token"):
            active_sources.add("device_code")
            custom_label = str(state.get("label") or "").strip()
            changed |= _upsert_entry(
                entries,
                provider,
                "device_code",
                {
                    "source": "device_code",
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": tokens.get("access_token", ""),
                    "refresh_token": tokens.get("refresh_token"),
                    "base_url": "https://chatgpt.com/backend-api/codex",
                    "last_refresh": state.get("last_refresh"),
                    "label": custom_label or label_from_token(tokens.get("access_token", ""), "device_code"),
                },
            )

    elif provider == "xai-oauth":
        # When the user logs in via ``hermes model`` -> xAI Grok OAuth,
        # tokens are written to the auth.json singleton
        # (``providers["xai-oauth"]``).  Surface them in the pool too so
        # ``hermes auth list`` reflects the logged-in state and so the pool
        # is the single source of truth for refresh during runtime resolution.
        state = _load_provider_state(auth_store, "xai-oauth")
        tokens = state.get("tokens") if isinstance(state, dict) else None
        if isinstance(tokens, dict) and tokens.get("access_token"):
            # Device code is the only supported xAI OAuth flow; the singleton is
            # always surfaced as ``device_code`` (consistent with nous/codex).
            source = "device_code"
            if _is_suppressed(provider, source):
                return changed, active_sources
            active_sources.add(source)
            from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

            base_url = DEFAULT_XAI_OAUTH_BASE_URL
            changed |= _upsert_entry(
                entries,
                provider,
                source,
                {
                    "source": source,
                    "auth_type": AUTH_TYPE_OAUTH,
                    "access_token": tokens.get("access_token", ""),
                    "refresh_token": tokens.get("refresh_token"),
                    "base_url": base_url,
                    "last_refresh": state.get("last_refresh"),
                    "label": label_from_token(tokens.get("access_token", ""), source),
                },
            )

    return changed, active_sources


def _seed_from_env(provider: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    changed = False
    active_sources: Set[str] = set()

    # Prefer ~/.hermes/.env over os.environ — the user's config file is the
    # authoritative source for Hermes credentials. Stale env vars from parent
    # processes (Codex CLI, test scripts, etc.) should not override deliberate
    # changes to the .env file.
    def _get_env_prefer_dotenv(key: str) -> str:
        env_file = load_env()
        raw = env_file.get(key, "").strip()
        env_val = os.environ.get(key, "").strip()
        # If .env contains an unresolved op:// reference, prefer the
        # already-resolved value from os.environ (set by
        # load_hermes_dotenv() -> apply_onepassword_secrets()).  The raw
        # "op://Vault/Item/field" string would otherwise win and every
        # provider auth attempt would receive a URL instead of a key.  This
        # happens during a partial migration, or when the user wrote op://
        # references straight into .env rather than the secrets.onepassword
        # config block.  For every non-op:// value the original
        # .env-takes-precedence behaviour is preserved unchanged.
        if raw.startswith("op://") and env_val:
            return env_val
        return raw or _get_secret(key, "") or env_val

    # Honour user suppression — `hermes auth remove <provider> <N>` for an
    # env-seeded credential marks the env:<VAR> source as suppressed so it
    # won't be re-seeded from the user's shell environment or ~/.hermes/.env.
    # Without this gate the removal is silently undone on the next
    # load_pool() call whenever the var is still exported by the shell.
    try:
        from hermes_cli.auth import is_source_suppressed as _is_source_suppressed
    except ImportError:
        def _is_source_suppressed(_p, _s):  # type: ignore[misc]
            return False

    def _secret_source_for_env(env_var: str) -> Optional[str]:
        try:
            from hermes_cli.env_loader import get_secret_source
            source_label = get_secret_source(env_var)
        except Exception:
            source_label = None
        return str(source_label).strip() if source_label else None

    def _env_payload(
        *,
        source: str,
        env_var: str,
        token: str,
        base_url: str,
        auth_type: str = AUTH_TYPE_API_KEY,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": source,
            "auth_type": auth_type,
            "access_token": token,
            "base_url": base_url,
            "label": env_var,
        }
        secret_source = _secret_source_for_env(env_var)
        if secret_source:
            payload["secret_source"] = secret_source
        return payload

    if provider == "openrouter":
        # Prefer ~/.hermes/.env over os.environ
        token = _get_env_prefer_dotenv("OPENROUTER_API_KEY")
        if token:
            source = "env:OPENROUTER_API_KEY"
            if _is_source_suppressed(provider, source):
                return changed, active_sources
            active_sources.add(source)
            changed |= _upsert_entry(
                entries,
                provider,
                source,
                _env_payload(
                    source=source,
                    env_var="OPENROUTER_API_KEY",
                    token=token,
                    base_url=OPENROUTER_BASE_URL,
                ),
            )
        return changed, active_sources

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig or pconfig.auth_type != AUTH_TYPE_API_KEY:
        return changed, active_sources

    env_url = ""
    if pconfig.base_url_env_var:
        env_url = _get_env_prefer_dotenv(pconfig.base_url_env_var).rstrip("/")

    env_vars = list(pconfig.api_key_env_vars)
    if provider == "anthropic":
        env_vars = [
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        ]

    for env_var in env_vars:
        # Prefer ~/.hermes/.env over os.environ
        token = _get_env_prefer_dotenv(env_var)
        if not token:
            continue
        source = f"env:{env_var}"
        if _is_source_suppressed(provider, source):
            continue
        active_sources.add(source)
        base_url = env_url or pconfig.inference_base_url
        if provider == "kimi-coding":
            base_url = _resolve_kimi_base_url(token, pconfig.inference_base_url, env_url)
        elif provider == "zai":
            base_url = _resolve_zai_base_url(token, pconfig.inference_base_url, env_url)
        changed |= _upsert_entry(
            entries,
            provider,
            source,
            _env_payload(
                source=source,
                env_var=env_var,
                token=token,
                base_url=base_url,
            ),
        )
    return changed, active_sources


def _prune_stale_seeded_entries(
    entries: List[PooledCredential],
    active_sources: Set[str],
    *,
    prune_env_sources: bool = True,
) -> bool:
    def _is_prunable(entry: PooledCredential) -> bool:
        # ``env:*`` entries are persisted references that get re-hydrated from
        # the environment on every load. A process that merely lacks the env
        # var this call must NOT delete the on-disk entry for every other
        # process — that destructive read is the bug behind #9331. Only prune
        # an env source when ``prune_env_sources`` is explicitly requested
        # (e.g. an `hermes auth` command that confirmed the source is gone).
        if entry.source.startswith("env:"):
            return prune_env_sources
        # File-backed singletons (device-code OAuth, claude_code) and Hermes
        # PKCE should disappear from the pool when their backing file is gone.
        return (
            is_borrowed_credential_source(entry.source, entry.provider)
            or entry.source == "hermes_pkce"
        )

    retained = [
        entry
        for entry in entries
        if _is_manual_source(entry.source)
        or entry.source in active_sources
        or not _is_prunable(entry)
    ]
    if len(retained) == len(entries):
        return False
    entries[:] = retained
    return True


def _seed_custom_pool(pool_key: str, entries: List[PooledCredential]) -> Tuple[bool, Set[str]]:
    """Seed a custom endpoint pool from custom_providers config and model config."""
    changed = False
    active_sources: Set[str] = set()

    # Shared suppression gate — same pattern as _seed_from_env/_seed_from_singletons.
    try:
        from hermes_cli.auth import is_source_suppressed as _is_suppressed
    except ImportError:
        def _is_suppressed(_p, _s):  # type: ignore[misc]
            return False

    # Seed from the custom_providers config entry's api_key field
    cp_config = _get_custom_provider_config(pool_key)
    if cp_config:
        api_key = str(cp_config.get("api_key") or "").strip()
        base_url = str(cp_config.get("base_url") or "").strip().rstrip("/")
        name = str(cp_config.get("name") or "").strip()
        if api_key:
            source = f"config:{name}"
            if not _is_suppressed(pool_key, source):
                active_sources.add(source)
                changed |= _upsert_entry(
                    entries,
                    pool_key,
                    source,
                    {
                        "source": source,
                        "auth_type": AUTH_TYPE_API_KEY,
                        "access_token": api_key,
                        "base_url": base_url,
                        "label": name or source,
                    },
                )

    # Seed from model.api_key if model.provider=='custom' and model.base_url matches
    try:
        config = _load_config_safe()
        model_cfg = config.get("model") if config else None
        if isinstance(model_cfg, dict):
            model_provider = str(model_cfg.get("provider") or "").strip().lower()
            model_base_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            model_api_key = ""
            for k in ("api_key", "api"):
                v = model_cfg.get(k)
                if isinstance(v, str) and v.strip():
                    model_api_key = v.strip()
                    break
            if model_provider == "custom" and model_base_url and model_api_key:
                # Check if this model's base_url matches our custom provider
                matched_key = get_custom_provider_pool_key(model_base_url)
                if matched_key == pool_key:
                    source = "model_config"
                    if not _is_suppressed(pool_key, source):
                        active_sources.add(source)
                        changed |= _upsert_entry(
                            entries,
                            pool_key,
                            source,
                            {
                                "source": source,
                                "auth_type": AUTH_TYPE_API_KEY,
                                "access_token": model_api_key,
                                "base_url": model_base_url,
                                "label": "model_config",
                            },
                        )
    except Exception:
        pass

    return changed, active_sources


def load_pool(provider: str) -> CredentialPool:
    provider = (provider or "").strip().lower()
    raw_entries = read_credential_pool(provider)
    disk_ids = {
        entry.get("id")
        for entry in raw_entries
        if isinstance(entry, dict) and entry.get("id")
    }
    raw_needs_sanitization = any(
        isinstance(payload, dict)
        and sanitize_borrowed_credential_payload(payload, provider) != payload
        for payload in raw_entries
    )
    entries = [PooledCredential.from_dict(provider, payload) for payload in raw_entries]
    raw_needs_auth_normalization = any(
        isinstance(payload, dict)
        and _normalize_pool_auth_type(
            provider,
            payload.get("access_token"),
            payload.get("auth_type", AUTH_TYPE_API_KEY),
        ) != payload.get("auth_type", AUTH_TYPE_API_KEY)
        for payload in raw_entries
    )
    if raw_needs_auth_normalization:
        # A profile may be reading this provider from the global-root fallback.
        # Keep that fallback read-only: only the store that owns these rows may
        # rewrite them. Loading the default/root profile will heal global rows.
        active_pool = _load_auth_store().get("credential_pool")
        active_entries = active_pool.get(provider) if isinstance(active_pool, dict) else None
        raw_needs_auth_normalization = bool(active_entries)

    if provider.startswith(CUSTOM_POOL_PREFIX):
        # Custom endpoint pool — seed from custom_providers config and model config
        custom_changed, custom_sources = _seed_custom_pool(provider, entries)
        changed = raw_needs_sanitization or raw_needs_auth_normalization or custom_changed
        changed |= _prune_stale_seeded_entries(entries, custom_sources)
    else:
        singleton_changed, singleton_sources = _seed_from_singletons(provider, entries)
        env_changed, env_sources = _seed_from_env(provider, entries)
        changed = (
            raw_needs_sanitization
            or raw_needs_auth_normalization
            or singleton_changed
            or env_changed
        )
        # ``load_pool()`` is a non-destructive read for env-seeded entries: a
        # process missing a provider env var must not delete the persisted
        # pool entry for every other process (#9331). File-backed singletons
        # still prune when their backing file is gone.
        changed |= _prune_stale_seeded_entries(
            entries,
            singleton_sources | env_sources,
            prune_env_sources=False,
        )
        changed |= _normalize_pool_priorities(provider, entries)

    if changed:
        new_ids = {entry.id for entry in entries}
        write_credential_pool(
            provider,
            [entry.to_dict() for entry in sorted(entries, key=lambda item: item.priority)],
            removed_ids=disk_ids - new_ids,
        )
    return CredentialPool(provider, entries)
