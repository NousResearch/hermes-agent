"""Unified routing for model-blind browser login credentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from agent.vault_backends.base import LoginBackend, LoginSaveResult, UnlockRequired, enabled_backends
from agent.vault_store import LOGIN_IDENTIFIER_TYPES, VaultError, normalize_origin, scrub_secret_from_text


@dataclass(frozen=True)
class LoginMigrationItem:
    handle: str
    label: str
    origin: str
    identifier: str
    status: str
    error: Optional[str] = None


@dataclass(frozen=True)
class LoginMigrationResult:
    source: str
    target: str
    execute: bool
    items: List[LoginMigrationItem]

    @property
    def imported(self) -> int:
        return sum(item.status == "imported" for item in self.items)

    @property
    def skipped(self) -> int:
        return sum(item.status == "skipped_existing" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.items)


class CredentialBroker:
    """Select a configured write backend while preserving all enabled read backends."""

    def __init__(self, backends: Optional[Iterable[LoginBackend]] = None,
                 config: Optional[Dict] = None):
        self._backends = list(backends) if backends is not None else enabled_backends()
        if config is None:
            from agent.vault_backends.base import vault_config
            config = vault_config()
        self._config = config

    def write_backend(self) -> LoginBackend:
        name = str(self._config.get("write_backend") or "local").strip().lower()
        backend = next((item for item in self._backends if item.name == name), None)
        if backend is None:
            raise VaultError(
                f"credential write backend {name!r} is unavailable; enable it or set vault.write_backend to local"
            )
        if "create_login" not in backend.capabilities():
            raise VaultError(f"credential backend {name!r} is read-only")
        return backend

    def save_login(self, *, label: str, origin: str, identifier_type: str,
                   identifier: str, password: str, otp_secret: Optional[str] = None) -> LoginSaveResult:
        origin = normalize_origin(origin)
        identifier = (identifier or "").strip()
        if identifier_type not in LOGIN_IDENTIFIER_TYPES:
            raise VaultError(f"identifier_type must be one of {LOGIN_IDENTIFIER_TYPES}")
        if not identifier or not password:
            raise VaultError("login items require identifier and password")

        backend = self.write_backend()
        existing = next((item for item in backend.list_items()
                         if item.kind == "login" and item.origin == origin
                         and item.identifier == identifier), None)
        kwargs = {
            "label": label,
            "origin": origin,
            "identifier_type": identifier_type,
            "identifier": identifier,
            "password": password,
            "otp_secret": otp_secret,
        }
        if existing is not None:
            if "update_login" not in backend.capabilities():
                raise VaultError(f"credential backend {backend.name!r} cannot update an existing login")
            return LoginSaveResult(backend.update_login(existing.id, **kwargs), "updated")
        return LoginSaveResult(backend.create_login(**kwargs), "created")

    def remove_item(self, handle: str) -> bool:
        backend = self.backend_for_handle(handle)
        if backend is None:
            return False
        if "remove" not in backend.capabilities():
            raise VaultError(f"credential backend {backend.name!r} cannot remove items")
        return backend.remove_item(handle)

    def backend_for_handle(self, handle: str) -> Optional[LoginBackend]:
        return next((item for item in self._backends if item.owns(handle)), None)

    def migrate_local_logins(self, *, execute: bool = False) -> LoginMigrationResult:
        source = next((item for item in self._backends if item.name == "local"), None)
        if source is None:
            raise VaultError("local credential backend is unavailable")
        target = self.write_backend()
        if target.name == "local":
            raise VaultError("set an external credential write backend before migrating local logins")

        local_items = [item for item in source.list_items()
                       if item.kind == "login" and item.origin and item.identifier]
        if not local_items:
            return LoginMigrationResult("local", target.name, execute, [])
        if target.needs_unlock and not target.is_unlocked():
            raise UnlockRequired(target)

        existing = {(item.origin, item.identifier) for item in target.list_items()
                    if item.kind == "login" and item.origin and item.identifier}
        results: List[LoginMigrationItem] = []
        for meta in local_items:
            key = (str(meta.origin), str(meta.identifier))
            base = {"handle": meta.id, "label": meta.label, "origin": key[0], "identifier": key[1]}
            if key in existing:
                results.append(LoginMigrationItem(**base, status="skipped_existing"))
                continue
            if not execute:
                results.append(LoginMigrationItem(**base, status="would_import"))
                continue

            secret = source.resolve_secret(meta.id)
            created = None
            password = ""
            otp_secret = None
            try:
                password = str(secret.get("password") or "")
                otp_secret = str(secret.get("otp_secret") or "") or None
                created = target.create_login(
                    label=meta.label,
                    origin=key[0],
                    identifier_type=str(meta.identifier_type or "username"),
                    identifier=key[1],
                    password=password,
                    otp_secret=otp_secret,
                )
                if target.resolve_password(created.id) != password:
                    raise VaultError("target password verification failed")
                existing.add(key)
                results.append(LoginMigrationItem(**base, status="imported"))
            except Exception as exc:
                if created is not None and "remove" in target.capabilities():
                    try:
                        target.remove_item(created.id)
                    except Exception:
                        pass
                error = scrub_secret_from_text(str(exc), secret)[:200]
                results.append(LoginMigrationItem(**base, status="failed", error=error))
            finally:
                secret.clear()
                password = ""
                otp_secret = None

        return LoginMigrationResult("local", target.name, execute, results)


def get_credential_broker() -> CredentialBroker:
    return CredentialBroker()
