"""Auth-failure tracking and 401-retry wrapper for HonchoSessionManager."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from plugins.memory.honcho.oauth import redact_tokens as _redact_tokens

logger = logging.getLogger("plugins.memory.honcho.session")


class HonchoAuthError(RuntimeError):
    """Auth failure that survived a forced refresh and one retry. Raised, not swallowed, so
    callers can tell a rejected credential from an empty result."""


# Matched narrowly: a false positive spends a token rotation, and a lost rotation revokes the grant.
_AUTH_ERROR_MARKERS = (
    "invalid or expired access token",
    "authentication failed",
    "unauthorized",
)

# A 401 in text counts only with HTTP context ("HTTP 401", "status 401"), never as a bare number.
_HTTP_401_RE = re.compile(r"\b(?:http|status(?:[ _]code)?\s*[:=]?)\s*401\b")


def _is_auth_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 401:
        return True
    # The transport reported a concrete non-auth status; trust it over text.
    if isinstance(status, int) and status not in (0, 401):
        return False
    text = str(exc).lower()
    if _HTTP_401_RE.search(text):
        return True
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


_REAUTH_REQUIRED_MESSAGE = (
    "Honcho OAuth grant is revoked and cannot be refreshed; "
    "re-authenticate with 'hermes honcho setup'."
)


def _auth_error_message(exc: BaseException) -> str:
    return (f"Honcho rejected our credentials and a forced token refresh did not recover: {_redact_tokens(str(exc))}. "
            "Re-authenticate with 'hermes honcho setup'.")


class SessionAuthMixin:
    """Auth state + ``_authed_call`` for HonchoSessionManager (state lives in __init__)."""

    def _record_auth_failure(self, exc: BaseException) -> None:
        detail = _redact_tokens(str(exc))
        if self._auth_failure is None:
            logger.error("Honcho authentication failed and token refresh did not recover; "
                         "memory sync and recall are paused until the user re-authenticates: %s", detail)
        self._auth_failure = detail

    def _clear_auth_failure(self) -> None:
        if self._auth_failure is not None:
            logger.info("Honcho authentication recovered; memory sync and recall resumed")
            self._auth_failure = None
            self._auth_notice_emitted = False

    def pop_auth_notice(self) -> str | None:
        """Return the pending auth-failure message once; later calls return None."""
        if self._auth_failure is None or self._auth_notice_emitted:
            return None
        self._auth_notice_emitted = True
        return self._auth_failure

    def _bound_config_path(self) -> Path:
        """Config path for OAuth checks, bound to this manager's profile: background threads can't
        see the ContextVar-backed ambient profile, so the bound path keeps them on THIS profile's
        honcho.json; ambient resolution is only the fallback for configless managers (tests)."""
        from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

        return self._config.bound_config_path() if isinstance(self._config, HonchoClientConfig) else resolve_config_path()

    def _reauth_required(self) -> bool:
        """True when the grant is dead and only a new login can fix it (no network call)."""
        try:
            from plugins.memory.honcho import oauth

            # Fast path: runs before every SDK call, so skip path resolution when nothing is dead.
            host = getattr(self._config, "host", "") or ""
            return bool(oauth.any_dead_grants() and host and oauth.reauth_required(self._bound_config_path(), host))
        except Exception:
            return False

    def _force_reauth(self) -> bool:
        """Rotate the token after a 401 and rebind the client. False for a static API key, a dead
        grant, or a failed exchange."""
        try:
            from plugins.memory.honcho import oauth
            from plugins.memory.honcho.client import reset_honcho_client

            host = getattr(self._config, "host", "") or ""
            if not host:
                return False
            token = oauth.force_refresh_token(self._bound_config_path(), host)
            if not token:
                return False
            if not oauth.apply_token_to_client(self.honcho, token):
                # SDK shape changed: rebuild the client and drop objects holding the old transport.
                reset_honcho_client()
                with self._cache_lock:
                    self._client_generation += 1
                    self._peers_cache.clear()
                    self._sessions_cache.clear()
            return True
        except Exception:
            logger.warning("Honcho post-401 token refresh failed", exc_info=True)
            return False

    def _authed_call(self, op_name: str, operation: Callable[[], Any]) -> Any:
        """Run an authenticated SDK operation, forcing one token refresh on a 401. ``operation``
        must re-resolve peer/session objects itself: a failed in-place refresh rebuilds the
        client, orphaning objects captured earlier."""
        if self._reauth_required():
            exc = HonchoAuthError(_REAUTH_REQUIRED_MESSAGE)
            self._record_auth_failure(exc)
            raise exc
        try:
            result = operation()
        except HonchoAuthError:
            raise
        except Exception as e:
            if not _is_auth_error(e):
                raise
            logger.warning("Honcho %s hit an auth error; forcing token refresh and retrying once: %s",
                           op_name, _redact_tokens(str(e)))
            if not self._force_reauth():
                self._record_auth_failure(e)
                raise HonchoAuthError(_auth_error_message(e)) from e
            try:
                result = operation()
            except Exception as retry_exc:
                if not _is_auth_error(retry_exc):
                    raise
                self._record_auth_failure(retry_exc)
                raise HonchoAuthError(_auth_error_message(retry_exc)) from retry_exc
        self._clear_auth_failure()
        return result
