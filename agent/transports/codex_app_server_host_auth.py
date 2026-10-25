"""Selected-pool-row authentication binding for the native Codex App Server session."""

from __future__ import annotations

import contextvars
import logging
from typing import Any

from agent.credential_pool import PooledCredential, _codex_principal_identity
from agent.transports.codex_app_server import CodexAppServerError


_AUTH_ERROR = "Codex App Server host authentication failed"
_REFRESH_ERROR = "Codex App Server credential refresh unavailable"
logger = logging.getLogger(__name__)


class CodexAppServerHostAuthError(CodexAppServerError):
    """Safe, fixed-message startup failure from native host authentication."""

    def __init__(self, message: str = _AUTH_ERROR) -> None:
        super().__init__(code=-32001, message=message)


class CodexAppServerHostAuth:
    """Bind a Codex session to one exact Hermes-owned OAuth pool row.

    Tokens cross only the native JSON-RPC channel. Refresh stays in the existing
    serialized CredentialPool API, under the profile context captured at bind time.
    """

    def __init__(self, pool: Any, entry: PooledCredential) -> None:
        self._pool = pool
        entry_id = str(getattr(entry, "id", "") or "")
        identity = self._entry_identity(entry)
        access_token = str(getattr(entry, "access_token", "") or "")
        self._profile_context = contextvars.copy_context()
        self._invalid = False
        self._runtime_owner: Any = None
        if not entry_id or identity is None or not access_token:
            raise self._auth_error()
        self._entry_id, self._identity, self._access_token = entry_id, identity, access_token

    @classmethod
    def from_agent(cls, agent: Any) -> CodexAppServerHostAuth:
        """Resolve only the runtime's selected pool id; never consult singleton auth or Codex CLI state."""
        try:
            pool = getattr(agent, "_credential_pool", None)
            selected_id = getattr(agent, "_credential_pool_entry_id", None)
            if pool is None or not isinstance(selected_id, str) or not selected_id:
                raise cls._auth_error()
            entry = next((row for row in pool.entries() if row.id == selected_id), None)
            if entry is None or entry.provider != "openai-codex" or entry.auth_type != "oauth":
                raise cls._auth_error()
            if str(getattr(agent, "api_key", "") or "") != str(entry.runtime_api_key or entry.access_token or ""):
                raise cls._auth_error()
            binding = cls.from_selected_credential(pool=pool, entry=entry)
            binding._runtime_owner = agent
            return binding
        except Exception:  # health: allow BLE001 -- provider-backed getters may expose secrets in their exception text.
            raise cls._auth_error() from None

    @classmethod
    def from_selected_credential(cls, *, pool: Any, entry: PooledCredential) -> CodexAppServerHostAuth:
        """Create a reusable binding from a caller-selected row without running a credential resolver."""
        try:
            selected = next((row for row in pool.entries() if row.id == getattr(entry, "id", None)), None)
            if (selected is None or selected.provider != "openai-codex" or selected.auth_type != "oauth"
                    or selected.access_token != getattr(entry, "access_token", None)):
                raise cls._auth_error()
            return cls(pool, selected)
        except Exception:  # health: allow BLE001 -- selected pool getters can expose secret-bearing errors.
            raise cls._auth_error() from None

    @staticmethod
    def _entry_identity(entry: PooledCredential) -> tuple[str, str] | None:
        if getattr(entry, "provider", None) != "openai-codex" or getattr(entry, "auth_type", None) != "oauth":
            return None
        return _codex_principal_identity(getattr(entry, "access_token", None))

    @staticmethod
    def _auth_error() -> CodexAppServerHostAuthError:
        return CodexAppServerHostAuthError()

    @staticmethod
    def _refresh_error() -> CodexAppServerHostAuthError:
        return CodexAppServerHostAuthError(_REFRESH_ERROR)

    @property
    def credential_id(self) -> str:
        return self._entry_id

    @property
    def ready(self) -> bool:
        return not self._invalid and bool(self._access_token and self._identity)

    def validate_current(self) -> None:
        """Fail closed if the captured row or its runtime owner no longer selects this principal."""
        try:
            if not self.ready:
                raise self._auth_error()
            owner = self._runtime_owner
            if owner is not None and (
                getattr(owner, "_credential_pool", None) is not self._pool
                or getattr(owner, "_credential_pool_entry_id", None) != self._entry_id
                or getattr(owner, "provider", None) != "openai-codex"
            ):
                raise self._auth_error()
            current = next((row for row in self._pool.entries() if row.id == self._entry_id), None)
            if (current is None or current.provider != "openai-codex" or current.auth_type != "oauth"
                    or self._entry_identity(current) != self._identity
                    or current.access_token != self._access_token):
                raise self._auth_error()
            if owner is not None and str(getattr(owner, "api_key", "") or "") != str(
                current.runtime_api_key or current.access_token or "",
            ):
                raise self._auth_error()
        except Exception:  # health: allow BLE001 -- pool/owner getters may expose secret-bearing errors.
            raise self._auth_error() from None

    def _owner_still_selects(self, credential_id: str) -> bool:
        owner = self._runtime_owner
        return owner is None or (
            getattr(owner, "_credential_pool", None) is self._pool
            and getattr(owner, "_credential_pool_entry_id", None) == credential_id
            and getattr(owner, "provider", None) == "openai-codex"
        )

    def __repr__(self) -> str:
        return "CodexAppServerHostAuth(selected_pool_row_bound=True, token=<redacted>)"

    def clear(self) -> None:
        """Drop token and identity references when the owning native session closes."""
        self._access_token = ""
        self._identity = None
        self._invalid = True

    def initialize(self, client: Any, *, client_name: str, client_title: str, client_version: str) -> dict:
        """Initialize experimental auth RPCs, then bind and verify the selected identity."""
        identity, token = self._identity, self._access_token
        if identity is None or not token:
            raise self._auth_error()
        try:
            self.validate_current()
            response = client.initialize(
                client_name=client_name, client_title=client_title, client_version=client_version,
                capabilities={"experimentalApi": True},
            )
            self._login_and_verify(client, token, identity)
            return response
        except CodexAppServerError:
            raise self._auth_error() from None
        except Exception:  # health: allow BLE001 -- upstream auth errors may contain token material; expose only a fixed message.
            raise self._auth_error() from None

    def bind_entry(self, client: Any, entry: PooledCredential) -> bool:
        """Authenticate an explicitly rotated row on the same native connection."""
        identity = self._entry_identity(entry)
        token = str(getattr(entry, "access_token", "") or "")
        if not entry.id or identity is None or not token:
            return False
        try:
            selected = next((row for row in self._pool.entries() if row.id == entry.id), None)
            if (selected is None or selected.provider != "openai-codex" or selected.auth_type != "oauth"
                    or selected.access_token != token or self._entry_identity(selected) != identity):
                return False
            self._login_and_verify(client, token, identity)
        except Exception:  # health: allow BLE001 -- fail closed and do not surface auth-provider response details.
            return False
        self._entry_id, self._identity, self._access_token = selected.id, identity, token
        self._invalid = False
        return True

    @staticmethod
    def _login_and_verify(client: Any, token: str, identity: tuple[str, str]) -> None:
        account_id, _subject = identity
        login = client.request("account/login/start", {
            "type": "chatgptAuthTokens", "accessToken": token, "chatgptAccountId": account_id,
        }, timeout=10)
        if not isinstance(login, dict) or login.get("type") != "chatgptAuthTokens":
            raise CodexAppServerHostAuth._auth_error()
        observed = client.request("account/read", {}, timeout=10)
        account = observed.get("account") if isinstance(observed, dict) else None
        if (not isinstance(account, dict) or account.get("type") != "chatgpt"
                or observed.get("requiresOpenaiAuth") is not True):
            raise CodexAppServerHostAuth._auth_error()

    def handle_server_request(self, client: Any, request: dict[str, Any]) -> bool:
        """Handle native refresh requests inside the existing session pump; return whether handled."""
        if request.get("method") != "account/chatgptAuthTokens/refresh":
            return False
        request_id = request.get("id")
        params = request.get("params")
        try:
            self._refresh_and_reply(client, request_id, params)
        except Exception:  # health: allow BLE001 -- server request errors are returned with a constant redacted message.
            self._invalid = True
            try:
                client.respond_error(request_id, code=-32001, message=_REFRESH_ERROR)
            except Exception:  # health: allow BLE001 -- the transport may already be closed; no safe response is possible.
                logger.debug("Failed to send redacted Codex credential refresh error")
        return True

    def _refresh_and_reply(self, client: Any, request_id: Any, params: Any) -> None:
        identity = self._identity
        self.validate_current()
        previous_account_id = params.get("previousAccountId") if isinstance(params, dict) else None
        if (self._invalid or not isinstance(params, dict) or params.get("reason") != "unauthorized"
                or identity is None or (previous_account_id is not None and previous_account_id != identity[0])):
            raise self._refresh_error()
        old_identity, old_id = identity, self._entry_id

        def refresh_selected() -> PooledCredential | None:
            return self._pool.try_refresh_matching(credential_id=old_id, expected_principal=old_identity)

        try:
            refreshed = self._profile_context.copy().run(refresh_selected)
        except Exception:  # health: allow BLE001 -- pool refresh details can contain secret-backed provider errors.
            raise self._refresh_error() from None
        if (refreshed is None or refreshed.id != old_id or refreshed.provider != "openai-codex"
                or self._entry_identity(refreshed) != old_identity or not refreshed.access_token):
            raise self._refresh_error()
        if not self._owner_still_selects(old_id):
            raise self._refresh_error()
        self._access_token = refreshed.access_token
        self._update_runtime_owner(refreshed)
        response = {"accessToken": refreshed.access_token, "chatgptAccountId": old_identity[0]}
        client.respond(request_id, response)

    def _update_runtime_owner(self, entry: PooledCredential) -> None:
        """Keep the owning Hermes agent aligned with the pool after an async refresh."""
        owner = self._runtime_owner
        if owner is None:
            return
        if not self._owner_still_selects(entry.id):
            raise self._refresh_error()
        owner._credential_pool_entry_id = entry.id
        owner.api_key = entry.runtime_api_key or entry.access_token
        owner.base_url = entry.runtime_base_url or entry.base_url or owner.base_url

    def rotate_and_bind(self, client: Any, entry: PooledCredential) -> bool:
        """Bind a new, explicitly selected account for the one permitted pre-tool quota retry."""
        if self._invalid or entry.id == self._entry_id:
            return False
        return self.bind_entry(client, entry)
