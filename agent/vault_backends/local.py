"""Local Fernet vault as a login backend (the always-on default)."""

from __future__ import annotations

from typing import Dict, List, Optional

from agent.vault_backends.base import LoginBackend
from agent.vault_store import VaultItemMeta


def _store():
    # Late import: tests and callers patch ``agent.vault_store.get_vault_store``; binding it here
    # at call time keeps that facade name the single seam.
    from agent.vault_store import get_vault_store
    return get_vault_store()


class LocalLoginBackend(LoginBackend):
    name = "local"
    display_name = "Hermes vault"
    prefix = "vault_"

    def capabilities(self):
        return super().capabilities() | {"create_login", "update_login", "remove"}

    def list_items(self) -> List[VaultItemMeta]:
        return _store().list_items()

    def get_meta(self, handle: str) -> Optional[VaultItemMeta]:
        return _store().get_meta(handle)

    def resolve_password(self, handle: str) -> str:
        return str(_store().resolve_secret(handle).get("password") or "")

    def resolve_otp(self, handle: str) -> Optional[str]:
        from agent.vault_store import totp_now
        seed = str(_store().resolve_secret(handle).get("otp_secret") or "")
        return totp_now(seed) if seed else None

    def resolve_secret(self, handle: str) -> Dict[str, str]:
        return {k: str(v) for k, v in _store().resolve_secret(handle).items()}

    def create_login(self, *, label: str, origin: str, identifier_type: str,
                     identifier: str, password: str, otp_secret: Optional[str] = None) -> VaultItemMeta:
        secret = {
            "identifier_type": identifier_type,
            "identifier": identifier,
            "password": password,
        }
        if otp_secret is not None:
            secret["otp_secret"] = otp_secret
        return _store().add_item("login", label, secret, origin=origin)

    def update_login(self, handle: str, *, label: str, origin: str, identifier_type: str,
                     identifier: str, password: str, otp_secret: Optional[str] = None) -> VaultItemMeta:
        return _store().update_login(handle, label=label, origin=origin,
                                     identifier_type=identifier_type, identifier=identifier,
                                     password=password, otp_secret=otp_secret)

    def remove_item(self, handle: str) -> bool:
        return _store().remove_item(handle)
