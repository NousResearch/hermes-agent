"""Canonical codex authentication mechanics; no CLI dependencies."""

from __future__ import annotations

import logging

import hashlib

import json

import os

import threading

import time


from pathlib import Path

from typing import Any, Dict, Iterator, List, Optional, Tuple

from auth.token_validation import _decode_jwt_claims

from auth.errors import AuthError

from auth.constants import (
    CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    DEFAULT_CODEX_BASE_URL,
    _codex_err,
    httpx,
)

from auth.store import AUTH_LOCK_TIMEOUT_SECONDS

from utils import env_float

logger = logging.getLogger("hermes_cli.auth")

_MISSING_ACCESS_TOKEN_MSG = (
    "Codex auth is missing access_token. Run `{relogin}` to re-authenticate."
)

_MISSING_REFRESH_TOKEN_MSG = (
    "Codex auth is missing refresh_token. Run `{relogin}` to re-authenticate."
)

_NO_CREDENTIALS_MSG = "No Codex credentials stored. Run `{relogin}` to authenticate."


def _codex_relogin_command() -> str:
    from agent.turn_failure_copy import oauth_relogin_command

    return oauth_relogin_command("openai-codex")


def _parse_retry_after_seconds(headers: Any) -> Optional[int]:
    """Best-effort parse of a ``Retry-After`` header into whole seconds."""
    from agent.retry_utils import parse_retry_after_seconds

    seconds = parse_retry_after_seconds(headers)
    return None if seconds is None else int(seconds)


def _stripped(value: Any) -> str:
    return str(value or "").strip()


def _clear_pool_entry_status(entry: Dict[str, Any]) -> None:
    """Reset a pool entry's cooldown / last-error metadata to healthy."""
    from auth.pool_persistence import _POOL_STATUS_FIELDS

    for status_field in _POOL_STATUS_FIELDS:
        entry[status_field] = None


def _codex_access_token_is_expiring(access_token: Any, skew_seconds: int) -> bool:
    exp = _decode_jwt_claims(access_token).get("exp")
    return isinstance(exp, (int, float)) and float(exp) <= (
        time.time() + max(0, int(skew_seconds))
    )


def _codex_base_url() -> str:
    return (
        os.getenv("HERMES_CODEX_BASE_URL", "").strip().rstrip("/")
        or DEFAULT_CODEX_BASE_URL
    )


def _codex_runtime_result(
    api_key: str,
    *,
    source: str,
    last_refresh: Optional[str],
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "provider": "openai-codex",
        "base_url": base_url or _codex_base_url(),
        "api_key": api_key,
        "source": source,
        "last_refresh": last_refresh,
        "auth_mode": "chatgpt",
    }


def _load_auth_store_maybe_locked(lock: bool) -> Dict[str, Any]:
    """Load the auth store, taking the cross-process lock unless the caller already holds it."""
    from auth.store import _auth_store_lock, _load_auth_store

    if lock:
        with _auth_store_lock():
            return _load_auth_store()
    return _load_auth_store()


def _read_codex_tokens(*, _lock: bool = True) -> Dict[str, Any]:
    """Read Codex OAuth tokens from Hermes auth store (~/.hermes/auth.json)."""
    from auth.provider_state import _load_provider_state
    from auth.token_validation import _nonempty_str

    auth_store = _load_auth_store_maybe_locked(_lock)
    state = _load_provider_state(auth_store, "openai-codex")
    if not state:
        raise _codex_err(
            _NO_CREDENTIALS_MSG.format(relogin=_codex_relogin_command()),
            "codex_auth_missing",
            relogin=True,
        )
    tokens = state.get("tokens")
    if not isinstance(tokens, dict):
        raise _codex_err(
            f"Codex auth state is missing tokens. Run `{_codex_relogin_command()}` to re-authenticate.",
            "codex_auth_invalid_shape",
            relogin=True,
        )
    if not _nonempty_str(tokens.get("access_token")):
        raise _codex_err(
            _MISSING_ACCESS_TOKEN_MSG.format(relogin=_codex_relogin_command()),
            "codex_auth_missing_access_token",
            relogin=True,
        )
    if not _nonempty_str(tokens.get("refresh_token")):
        raise _codex_err(
            _MISSING_REFRESH_TOKEN_MSG.format(relogin=_codex_relogin_command()),
            "codex_auth_missing_refresh_token",
            relogin=True,
        )
    return {"tokens": tokens, "last_refresh": state.get("last_refresh")}


def _sync_codex_pool_entries(
    auth_store: Dict[str, Any],
    tokens: Dict[str, str],
    last_refresh: Optional[str],
    previous_singleton_tokens: Optional[Dict[str, str]] = None,
) -> None:
    """Mirror a fresh Codex re-auth into the credential_pool OAuth entries.

    ``device_code`` (the singleton-seeded entry from ``hermes setup`` / the model picker) is always
    synced. ``manual:device_code`` (``hermes auth add openai-codex``) is synced only when its
    access_token equals the PREVIOUS singleton token — a legacy alias of the singleton; an entry
    with its own token material is an independent account and must be left alone. ``manual:api_key``
    and any other source are independent credentials and are never overwritten by a re-auth.

    See #33000, #39236.
    The original #33538 fix refreshed every ``manual:device_code`` entry unconditionally. That worked when
    ``manual:device_code`` only meant "legacy alias of the singleton", but the same source string is now
    also produced by independent-account additions, and the broad sync silently clobbered distinct accounts
    with the latest-authenticated token pair. The access_token-match check distinguishes the two cases
    without changing the source-string contract.
    """
    from auth.providers.codex_quota import _codex_pool_dicts

    access_token = tokens.get("access_token")
    if not access_token:
        return
    refresh_token = tokens.get("refresh_token")
    entries = _pool_entries(auth_store, "openai-codex")
    if entries is None:
        return
    # None/empty prev_at → no manual entry can be an alias (right default for a first-ever save).
    prev_at = (previous_singleton_tokens or {}).get("access_token") or None
    for entry in _codex_pool_dicts(entries):
        source = entry.get("source")
        is_alias = source == "manual:device_code" and bool(
            prev_at and entry.get("access_token") == prev_at
        )
        if not (source == "device_code" or is_alias):
            continue
        entry["access_token"] = access_token
        if refresh_token:
            entry["refresh_token"] = refresh_token
        if last_refresh:
            entry["last_refresh"] = last_refresh
        _clear_pool_entry_status(entry)


def _save_codex_tokens(
    tokens: Dict[str, str],
    last_refresh: str = None,
    label: str = None,
    *,
    set_active: bool = True,
    write_through: bool = False,
) -> None:
    """Save Codex OAuth tokens to the auth store the grant was resolved FROM.

    Codex refresh tokens are single-use with rotation-family reuse detection. A profile without its
    own ``providers.openai-codex`` block reads root's grant via the fallback, so a refresh under that
    profile must rotate ROOT's chain — singleton AND ``credential_pool`` entries — or root keeps the
    consumed refresh token, the next process replays it and OpenAI revokes the whole family
    (#87503). Root-only write-back: a profile copy would shadow root and disable the write-through
    on the next refresh (#74339). Mirrors the xAI source-aware save.

    Only a token REFRESH passes ``write_through=True``: a fresh login or import under a profile
    is the profile's own grant and must not overwrite the root account it was borrowing.
    ``set_active=False`` stores credentials for a side tool (image gen) without making Codex the
    active inference provider.
    """
    from auth.store import (
        _auth_file_path,
        _load_auth_store,
        _same_path,
        _save_auth_store,
    )
    from auth.provider_state import _provider_state_transaction, _store_provider_state
    from auth.oauth import _utc_now_z

    if last_refresh is None:
        last_refresh = _utc_now_z()
    with _provider_state_transaction("openai-codex") as (
        auth_store,
        state,
        source_path,
    ):
        state = dict(state) if state else {}
        # Capture the previous singleton tokens BEFORE overwriting: the pool sync uses them to
        # tell legacy singleton-aliases (refresh) from independent ``auth add`` accounts (keep).
        previous_singleton_tokens = (
            state.get("tokens") if isinstance(state.get("tokens"), dict) else None
        )
        state.update(tokens=tokens, last_refresh=last_refresh, auth_mode="chatgpt")
        if label and str(label).strip():
            state["label"] = str(label).strip()
        target_store, target_path = auth_store, None
        if (
            write_through
            and source_path is not None
            and not _same_path(source_path, _auth_file_path())
        ):
            # Root-borrowed grant: the transaction already holds root's lock, so write the rotated
            # chain into ROOT's store (never set_active — a refresh is not a provider choice).
            target_store, target_path, set_active = (
                _load_auth_store(source_path),
                source_path,
                False,
            )
        _store_provider_state(
            target_store, "openai-codex", state, set_active=set_active
        )
        _sync_codex_pool_entries(
            target_store,
            tokens,
            last_refresh,
            previous_singleton_tokens=previous_singleton_tokens,
        )
        _save_auth_store(target_store, target_path=target_path)


def _recover_codex_tokens_from_cli(
    reason: str, observed_access_token: Optional[str] = None, *, environment
) -> Optional[Dict[str, str]]:
    """Adopt a valid Codex CLI token pair into Hermes auth, if available.

    Automatic adoption only; the interactive import offer in ``_login_openai_codex`` asks first and is
    not subject to ``auth.adopt_external_logins``.

    ``observed_access_token`` is the singleton access_token (or None) the caller saw when it decided
    the credential needs repair. Recovery repairs THAT credential and nothing else (#73667): a Codex
    Desktop/CLI login into another ChatGPT workspace must not replace it silently, and a concurrent
    explicit re-auth must not be overwritten, so the save is a compare-and-swap under the store lock.
    """
    environment.require_current_scope()
    pass
    from auth.credential_pool import _codex_principal_identity
    from auth.source_policy import adopt_external_logins_enabled
    from auth.providers.codex import _import_codex_cli_tokens, _save_codex_tokens
    from auth.provider_state import _provider_state_transaction

    if not adopt_external_logins_enabled(environment=environment):
        return None
    imported = _import_codex_cli_tokens()
    # Require BOTH tokens before adopting: persisting a payload without a usable refresh_token
    # would only break the next refresh cycle.
    if not (
        imported
        and _stripped(imported.get("access_token"))
        and _stripped(imported.get("refresh_token"))
    ):
        return None
    observed = _stripped(observed_access_token) or None
    with _provider_state_transaction("openai-codex") as (_store, state, _source):
        stored = (state or {}).get("tokens")
        stored = stored if isinstance(stored, dict) else {}
        if (_stripped(stored.get("access_token")) or None) != observed:
            logger.info(
                "Codex CLI recovery skipped (%s): the credential was re-authenticated meanwhile.",
                reason,
            )
            return None
        known = _codex_principal_identity(observed)
        if known and _codex_principal_identity(imported["access_token"]) not in (
            None,
            known,
        ):
            logger.warning(
                "Codex CLI recovery refused (%s): the Codex CLI login belongs to a different ChatGPT "
                "workspace than the Hermes credential. Run `%s` to re-authenticate it.",
                reason,
                _codex_relogin_command(),
            )
            return None
        logger.info("Codex auth recovered from Codex CLI auth.json (%s).", reason)
        _save_codex_tokens(imported)  # nested: the per-path lock is reentrant
    return dict(imported)


def _refresh_payload_access_token(
    response: "httpx.Response",
    *,
    provider: str,
    invalid_json: Tuple[str, str],
    invalid_response: Optional[Tuple[str, str]],
    missing_access: Tuple[str, str],
    relogin_required: bool = True,
    invalid_json_relogin: Optional[bool] = None,
    strict_str: bool = True,
) -> Tuple[Dict[str, Any], str]:
    """Parse a 200 token-refresh response; return ``(payload, stripped access_token)``.

    Each ``(message, code)`` pair keeps the provider's historical wording; ``{exc}`` in
    *invalid_json*'s message is formatted with the JSON error. *strict_str* rejects non-string
    access tokens; otherwise they are ``str()``-coerced.
    """

    def _err(message: str, code: str, relogin: bool = relogin_required) -> AuthError:
        return AuthError(
            message, provider=provider, code=code, relogin_required=relogin
        )

    try:
        payload = response.json()
    except Exception as exc:
        relogin = (
            relogin_required if invalid_json_relogin is None else invalid_json_relogin
        )
        raise _err(invalid_json[0].format(exc=exc), invalid_json[1], relogin) from exc
    if not isinstance(payload, dict):
        if invalid_response is not None:
            raise _err(*invalid_response)
        payload = {}
    access = payload.get("access_token")
    if strict_str:
        access = access.strip() if isinstance(access, str) else ""
    else:
        access = _stripped(access)
    if not access:
        raise _err(*missing_access)
    return payload, access


_SSL_TROUBLE_MARKERS = ("[SSL:", "_ssl.c", "UNEXPECTED_EOF")


def _capped_byte_stream_class() -> type:
    """Build the capped stream subclass on first use.

    The base class is ``httpx.SyncByteStream``; naming it at module scope would resolve the
    lazy ``httpx`` proxy at import time and put httpx back on the interactive-CLI startup path
    (see ``auth_constants``).
    """
    from auth.providers.codex_http import _CODEX_AUTH_BODY_MAX_BYTES

    cached = getattr(_capped_byte_stream_class, "cls", None)
    if cached is not None:
        return cached

    class _CappedByteStream(httpx.SyncByteStream):
        """Body stream that raises once more than ``_CODEX_AUTH_BODY_MAX_BYTES`` came off the wire.

        httpx type-checks ``response.stream`` against ``SyncByteStream``, so the cap has to be a
        stream subclass rather than a bare generator.
        """

        def __init__(self, response: "httpx.Response") -> None:
            self._response, self._raw = response, response.stream

        def __iter__(self) -> Iterator[bytes]:
            total = 0
            for chunk in self._raw:  # type: ignore[union-attr]  # sync client only
                total += len(chunk)
                if total > _CODEX_AUTH_BODY_MAX_BYTES:
                    self.close()
                    raise _codex_err(
                        f"Codex auth response from {self._response.url.host} exceeded "
                        f"{_CODEX_AUTH_BODY_MAX_BYTES // 1024} KiB; refusing to parse it.",
                        "codex_auth_response_too_large",
                        relogin=False,
                    )
                yield chunk

        def close(self) -> None:
            self._raw.close()  # type: ignore[union-attr]

    _capped_byte_stream_class.cls = _CappedByteStream  # type: ignore[attr-defined]
    return _CappedByteStream


def _codex_refresh_failure_error(response: "httpx.Response") -> AuthError:
    """Decode a non-200 Codex token-refresh response into a shaped AuthError."""
    from auth.token_validation import _nonempty_str

    code = "codex_refresh_failed"
    message = f"Codex token refresh failed with status {response.status_code}."
    try:
        err = response.json()
        if isinstance(err, dict):
            err_obj = err.get("error")
            # OpenAI shape: {"error": {"code": "...", "message": "...", "type": "..."}}
            if isinstance(err_obj, dict):
                nested_code = err_obj.get("code") or err_obj.get("type")
                if _nonempty_str(nested_code):
                    code = nested_code.strip()
                nested_msg = err_obj.get("message")
                if _nonempty_str(nested_msg):
                    message = f"Codex token refresh failed: {nested_msg.strip()}"
            # OAuth spec shape: {"error": "code_str", "error_description": "..."}
            elif _nonempty_str(err_obj):
                code = err_obj.strip()
                err_desc = err.get("error_description") or err.get("message")
                if _nonempty_str(err_desc):
                    message = f"Codex token refresh failed: {err_desc.strip()}"
    except Exception:
        pass
    if code == "refresh_token_reused":
        message = (
            "Codex refresh token was already consumed by another client "
            "(e.g. Codex CLI or VS Code extension). "
            "Run `codex` in your terminal to generate fresh tokens, "
            f"then run `{_codex_relogin_command()}` to re-authenticate."
        )
    # A 401/403 from the token endpoint always means the refresh token is invalid/expired —
    # force relogin even if the body error code wasn't one of the known strings.
    relogin_required = code in {
        "invalid_grant",
        "invalid_token",
        "invalid_request",
        "refresh_token_reused",
    } or response.status_code in {401, 403}
    return _codex_err(message, code, relogin=relogin_required)


def refresh_codex_oauth_pure(
    access_token: str, refresh_token: str, *, timeout_seconds: float = 20.0, environment
) -> Dict[str, Any]:
    """Refresh Codex OAuth tokens without mutating Hermes auth state."""
    from auth.providers.codex_http import _codex_http_client
    from auth.providers.codex_quota import _codex_quota_exhausted_error

    environment.require_current_scope()
    from auth.token_validation import _nonempty_str
    from auth.oauth import _utc_now_z

    del (
        access_token
    )  # Access token is only used by callers to decide whether to refresh.
    if not _nonempty_str(refresh_token):
        raise _codex_err(
            _MISSING_REFRESH_TOKEN_MSG.format(relogin=_codex_relogin_command()),
            "codex_auth_missing_refresh_token",
            relogin=True,
        )
    with _codex_http_client(
        timeout=httpx.Timeout(max(5.0, float(timeout_seconds))),
        headers={
            "Accept": "application/json",
            "User-Agent": environment.oauth_user_agent(),
        },
    ) as client:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
            },
        )
    if response.status_code == 429:
        # Quota exhaustion on the token endpoint: the refresh token is still valid and re-auth
        # cannot lift a quota cap, so classify distinctly from auth failures ("retry later").
        raise _codex_quota_exhausted_error(
            _parse_retry_after_seconds(getattr(response, "headers", None))
        )
    if response.status_code != 200:
        raise _codex_refresh_failure_error(response)
    refresh_payload, refreshed_access = _refresh_payload_access_token(
        response,
        provider="openai-codex",
        invalid_response=None,
        invalid_json=(
            "Codex token refresh returned invalid JSON.",
            "codex_refresh_invalid_json",
        ),
        missing_access=(
            "Codex token refresh response was missing access_token.",
            "codex_refresh_missing_access_token",
        ),
    )
    updated = {
        "access_token": refreshed_access,
        "refresh_token": refresh_token.strip(),
        "last_refresh": _utc_now_z(),
    }
    next_refresh = refresh_payload.get("refresh_token")
    if _nonempty_str(next_refresh):
        updated["refresh_token"] = next_refresh.strip()
    return updated


def _refresh_codex_auth_tokens(
    tokens: Dict[str, str], timeout_seconds: float, *, environment
) -> Dict[str, str]:
    """Refresh Codex access token using the refresh token.

    The whole re-read -> endpoint POST -> write-back runs inside the SOURCE store's transaction:
    two profiles borrowing the same root grant otherwise both submit the same single-use refresh
    token (each holds only its own profile lock) and OpenAI revokes the family. A waiter that
    finds root already rotated by its peer adopts the stored pair instead of replaying the
    consumed token. Both locks wait out a full endpoint timeout so the waiter adopts, not times out.
    """
    environment.require_current_scope()
    from auth.provider_state import _provider_state_transaction
    from auth.providers.codex import _save_codex_tokens, refresh_codex_oauth_pure

    lock_timeout = max(float(AUTH_LOCK_TIMEOUT_SECONDS), float(timeout_seconds) + 5.0)
    with _provider_state_transaction("openai-codex", lock_timeout) as (
        _store,
        state,
        _source,
    ):
        stored = (state or {}).get("tokens")
        stored = stored if isinstance(stored, dict) else {}
        stored_at, stored_rt = (
            _stripped(stored.get("access_token")),
            _stripped(stored.get("refresh_token")),
        )
        if (
            stored_at
            and stored_rt
            and stored_rt != _stripped(tokens.get("refresh_token"))
        ):
            logger.info(
                "Codex refresh token already rotated by a peer — adopting the stored pair."
            )
            return {**tokens, "access_token": stored_at, "refresh_token": stored_rt}
        try:
            refreshed = refresh_codex_oauth_pure(
                str(tokens.get("access_token", "") or ""),
                str(tokens.get("refresh_token", "") or ""),
                timeout_seconds=timeout_seconds,
                environment=environment,
            )
        except AuthError as exc:
            # Self-heal cross-store rotation: refresh_tokens are single-use, so when the Codex CLI
            # (or another Hermes process) rotates the shared token this frozen copy fails with a
            # relogin-required error (invalid_grant / refresh_token_reused / 401). Adopt the
            # canonical fresh token from ~/.codex/auth.json before surfacing a hard 401. Transient
            # failures (429 quota) keep relogin_required=False — the stored token is still valid —
            # re-raise.
            if not getattr(exc, "relogin_required", False):
                raise
            imported = _recover_codex_tokens_from_cli(
                f"refresh_token rejected: {getattr(exc, 'code', None) or 'auth_error'}",
                observed_access_token=stored_at or None,
                environment=environment,
            )
            if not imported:
                raise
            return imported
        updated_tokens = {
            **tokens,
            "access_token": refreshed["access_token"],
            "refresh_token": refreshed["refresh_token"],
        }
        # Nested transaction: the per-path lock is reentrant, and it re-reads under the held locks.
        _save_codex_tokens(updated_tokens, write_through=True)
    return updated_tokens


def _import_codex_cli_tokens() -> Optional[Dict[str, str]]:
    """Read ~/.codex/auth.json (Codex CLI file) tokens if valid and not expired; never writes."""
    from auth.providers.codex import _codex_access_token_is_expiring

    codex_home = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        tokens = json.loads(auth_path.read_text(encoding="utf-8-sig")).get("tokens")
        if not (
            isinstance(tokens, dict)
            and tokens.get("access_token")
            and tokens.get("refresh_token")
        ):
            return None
        # Importing stale tokens that can't be refreshed would leave the user with
        # "Login successful!" but no working credentials.
        if _codex_access_token_is_expiring(tokens["access_token"], 0):
            logger.debug(
                "Codex CLI tokens at %s are expired — skipping import.", auth_path
            )
            return None
        return dict(tokens)
    except Exception:
        return None


def resolve_codex_runtime_credentials(
    *,
    force_refresh: bool = False,
    refresh_if_expiring: bool = True,
    refresh_skew_seconds: int = CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    read_only: bool = False,
    environment,
) -> Dict[str, Any]:
    """Resolve runtime credentials from Hermes's own Codex token store.

    ``read_only=True`` (status / doctor / pickers) reports the stored state as-is: no Codex CLI
    adoption, no token refresh, no auth-store write — and it wins over ``force_refresh``. A
    diagnostic that silently imports another program's rotating refresh token or spends one is a
    mutation the user never asked for (#68004).

    Falls back to the credential pool when the singleton (``providers.openai-codex.tokens``) has no
    usable access_token but the pool (``credential_pool.openai-codex``) does.

    This closes the divergence between the chat path (singleton-only via this function) and the auxiliary
    path (pool-first via ``auxiliary_client._resolve_codex_credential_and_base``). Without this fallback, a user whose tokens live only
    in the pool — for example after a manual pool seed, a partial re-auth, or pool-only restoration from a
    backup — gets a bare HTTP 401 ``Missing Authentication header`` from the wire instead of a usable
    credential. See issue #32992.
    """
    from auth.providers.codex_quota import (
        _codex_pool_rate_limit_status,
        _codex_quota_exhausted_error,
        _pool_codex_credential,
        _probe_codex_pool_entry_quota_restored,
        clear_codex_pool_quota_cooldowns,
    )

    environment.require_current_scope()
    pass
    from auth.store import _auth_store_lock
    from auth.providers.codex import _codex_access_token_is_expiring, _read_codex_tokens

    read_error: Optional[AuthError] = None
    data = None
    observed: Optional[str] = None
    try:
        if read_only:
            # A read-only report takes no store lock: ``_save_auth_store`` replaces auth.json
            # atomically, and materialising ``auth.lock`` is itself a write a diagnostic must not
            # make. No recovery follows a read-only read, so no observed token is needed.
            data = _read_codex_tokens(_lock=False)
        else:
            with _auth_store_lock():
                # Observe the singleton in the same locked snapshot the read validates, so recovery
                # can compare-and-swap against exactly the credential it is repairing (#73667).
                from auth.store import _load_auth_store
                from auth.provider_state import _load_provider_state

                raw = (
                    _load_provider_state(_load_auth_store(), "openai-codex") or {}
                ).get("tokens")
                observed = raw.get("access_token") if isinstance(raw, dict) else None
                data = _read_codex_tokens(_lock=False)
    except AuthError as exc:
        read_error = exc
        if (
            not read_only
            and exc.relogin_required
            and exc.code
            in {
                "codex_auth_missing_access_token",
                "codex_auth_missing_refresh_token",
                "codex_auth_invalid_shape",
            }
        ):
            imported = _recover_codex_tokens_from_cli(
                str(exc.code or "auth_error"),
                observed_access_token=observed,
                environment=environment,
            )
            if imported:
                data = {
                    "tokens": imported,
                    "last_refresh": imported.get("last_refresh"),
                }
    if data is None:
        pool_token, pool_base = _pool_codex_credential()
        if pool_token and force_refresh and not read_only:
            # Pool-only setup: a forced refresh must rotate the pool entry, not resend its token.
            from auth.credential_pool import load_pool

            refreshed = load_pool(
                "openai-codex", environment=environment
            ).try_refresh_matching(api_key_hint=pool_token)
            pool_token = refreshed.runtime_api_key if refreshed is not None else ""
        if pool_token:
            # Report the host this row routes to, not the ambient default: a pooled gateway key
            # paired with chatgpt.com leaks to every consumer of this result (#121486).
            return _codex_runtime_result(
                pool_token,
                source="credential_pool",
                last_refresh=None,
                base_url=environment.provider_hooks("openai-codex").route_base_url(
                    pool_base
                ),
            )
        pool_rate_limit = _codex_pool_rate_limit_status()
        if pool_rate_limit:
            # Before surfacing the persisted cooldown, ask the usage endpoint whether the quota
            # reset early (banked reset redeemed, plan upgraded): ``last_error_reset_at`` can be
            # days in the future while the account is already usable again. Never from a
            # read-only caller: the picker fingerprint resolves on every cache-only read.
            if not read_only and _probe_codex_pool_entry_quota_restored(
                pool_rate_limit, environment=environment
            ):
                logger.info(
                    "Codex quota restored upstream — clearing stale pool cooldown(s)."
                )
                clear_codex_pool_quota_cooldowns()
                pool_token, pool_base = _pool_codex_credential()
                if pool_token:
                    return _codex_runtime_result(
                        pool_token,
                        source="credential_pool",
                        last_refresh=None,
                        base_url=environment.provider_hooks(
                            "openai-codex"
                        ).route_base_url(pool_base),
                    )
            reset_at = pool_rate_limit.get("reset_at")
            in_future = isinstance(reset_at, (int, float)) and reset_at > time.time()
            raise _codex_quota_exhausted_error(
                int(reset_at - time.time()) if in_future else None
            )
        if read_error is not None:
            raise read_error
        raise _codex_err(
            _NO_CREDENTIALS_MSG.format(relogin=_codex_relogin_command()),
            "codex_auth_missing",
            relogin=True,
        )
    tokens = dict(data["tokens"])
    access_token = _stripped(tokens.get("access_token"))
    refresh_timeout_seconds = env_float("HERMES_CODEX_REFRESH_TIMEOUT_SECONDS", 20)

    def _should_refresh(token: str) -> bool:
        if read_only:
            return False
        return bool(force_refresh) or (
            refresh_if_expiring
            and _codex_access_token_is_expiring(token, refresh_skew_seconds)
        )

    if _should_refresh(access_token):
        # Re-read under lock to avoid racing with other Hermes processes
        lock_timeout = max(
            float(AUTH_LOCK_TIMEOUT_SECONDS), refresh_timeout_seconds + 5.0
        )
        with _auth_store_lock(timeout_seconds=lock_timeout):
            data = _read_codex_tokens(_lock=False)
            tokens = dict(data["tokens"])
            if _should_refresh(_stripped(tokens.get("access_token"))):
                tokens = _refresh_codex_auth_tokens(
                    tokens, refresh_timeout_seconds, environment=environment
                )
            access_token = _stripped(tokens.get("access_token"))
    return _codex_runtime_result(
        access_token, source="hermes-auth-store", last_refresh=data.get("last_refresh")
    )


def _entry_is_rate_limit_exhausted(entry: Dict[str, Any]) -> bool:
    """Pool entry frozen by a 429/quota stop (as opposed to an auth failure)."""
    from auth.providers.codex_quota import _is_codex_rate_limit_shaped

    return entry.get("last_status") == "exhausted" and _is_codex_rate_limit_shaped(
        entry.get("last_error_code"),
        entry.get("last_error_reason"),
        entry.get("last_error_message"),
    )


_codex_quota_probe_lock = threading.Lock()


def _codex_quota_probe_cache_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _pool_entries(auth_store: Dict[str, Any], provider_id: str) -> Optional[List[Any]]:
    """``auth_store["credential_pool"][provider_id]`` when it is a list, else None."""
    pool = auth_store.get("credential_pool")
    entries = pool.get(provider_id) if isinstance(pool, dict) else None
    return entries if isinstance(entries, list) else None
