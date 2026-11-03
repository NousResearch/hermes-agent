"""Credential-vault dashboard routes.

Exposes the local encrypted vault (``agent/vault_store.py``) and the external
password-manager sources to the dashboard, mirroring the Desktop's ``vault.*``
JSON-RPC surface. Every route is profile-scoped via ``?profile=``.

The vault's contract is that the agent (and therefore this API) never sees a stored
password: reads return metadata only, and every error path is scrubbed before it can
carry a secret back to the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import config_scoped_to_thread, destructive_profile

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()

_require_token = late("_require_token")


class VaultAddBody(BaseModel):
    kind: str
    label: str
    origin: Optional[str] = None
    secret: dict[str, Any]


class VaultUnlockBody(BaseModel):
    name: str
    password: str


class VaultSourceSetBody(BaseModel):
    enabled: bool


class VaultLockBody(BaseModel):
    name: Optional[str] = None


@router.get("/api/vault/items")
async def list_vault_items(request: Request, profile: Optional[str] = None):
    _require_token(request)

    def _run():
        from agent.vault_backends import enabled_backends

        items: list[dict[str, Any]] = []
        locked: list[str] = []
        for backend in enabled_backends():
            if backend.needs_unlock and not backend.is_unlocked():
                locked.append(backend.display_name)
                continue
            for meta in backend.list_items():
                row = meta.to_dict()
                row["backend"] = backend.name
                items.append(row)
        return {"items": items, "locked": locked}

    return await config_scoped_to_thread(profile, _run)


@router.get("/api/vault/sources")
async def list_vault_sources(request: Request, profile: Optional[str] = None):
    _require_token(request)

    def _run():
        from agent.vault_backends import enabled_backends
        from agent.vault_backends.base import external_backend_classes, is_installed

        enabled = {b.name: b for b in enabled_backends()}
        rows: list[dict[str, Any]] = [
            {
                "name": "local",
                "display_name": "Hermes vault",
                "enabled": True,
                "needs_unlock": False,
                "unlocked": True,
                "installed": True,
            }
        ]
        for cls in external_backend_classes():
            live = enabled.get(cls.name)
            rows.append(
                {
                    "name": cls.name,
                    "display_name": cls.display_name,
                    "enabled": live is not None,
                    "needs_unlock": True,
                    "unlocked": bool(live and live.is_unlocked()),
                    "installed": is_installed(cls.name),
                }
            )
        return {"sources": rows}

    return await config_scoped_to_thread(profile, _run)


@router.post("/api/vault/items")
async def add_vault_item(
    body: VaultAddBody, request: Request, profile: Optional[str] = None
):
    _require_token(request)

    def _run():
        from agent.vault_store import VaultError, get_vault_store, scrub_secret_from_text

        secret = body.secret or {}
        try:
            meta = get_vault_store().add_item(
                kind=body.kind,
                label=body.label,
                origin=body.origin,
                secret=secret,
            )
        except VaultError as exc:  # the store's own refusal: safe to show, still scrubbed
            raise HTTPException(
                status_code=400, detail=scrub_secret_from_text(str(exc), secret)
            )
        return {"id": meta.id}

    return await config_scoped_to_thread(profile, _run)


@router.delete("/api/vault/items/{item_id}")
async def remove_vault_item(
    item_id: str, request: Request, profile: Optional[str] = None
):
    _require_token(request)
    profile = destructive_profile(profile, "DELETE /api/vault/items/{id}")

    def _run():
        from agent.vault_store import get_vault_store

        return {"removed": get_vault_store().remove_item(item_id)}

    return await config_scoped_to_thread(profile, _run)


@router.put("/api/vault/sources/{name}")
async def set_vault_source(
    name: str, body: VaultSourceSetBody, request: Request, profile: Optional[str] = None
):
    _require_token(request)
    profile = destructive_profile(profile, "PUT /api/vault/sources/{name}")

    def _run():
        from agent.vault_backends.base import external_backend_classes
        from agent.vault_backends.unlock import lock
        from hermes_cli.config import _ensure_dict, load_config, save_config

        if name not in {cls.name for cls in external_backend_classes()}:
            raise HTTPException(status_code=400, detail=f"unknown vault source: {name}")
        cfg = load_config()
        section = _ensure_dict(_ensure_dict(cfg, "vault"), name)
        if body.enabled:
            section.pop("enabled", None)
        else:
            section["enabled"] = False
            lock(name)
        save_config(cfg)
        return {"name": name, "enabled": body.enabled}

    return await config_scoped_to_thread(profile, _run)


@router.post("/api/vault/unlock")
async def unlock_vault_source(
    body: VaultUnlockBody, request: Request, profile: Optional[str] = None
):
    _require_token(request)

    def _run():
        from agent.vault_backends import enabled_backends

        backend = next(
            (b for b in enabled_backends() if b.name == body.name and b.needs_unlock),
            None,
        )
        if backend is None:
            raise HTTPException(
                status_code=400, detail=f"{body.name} is not an enabled password manager"
            )
        if not body.password:
            raise HTTPException(status_code=400, detail="master password is required")
        password = body.password
        try:
            backend.unlock(password)  # type: ignore[attr-defined]
        # health: allow BLE001 -- backend error text may echo the master password
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc).replace(password, "[REDACTED]"),
            )
        finally:
            del password
        return {"name": body.name, "unlocked": True}

    return await config_scoped_to_thread(profile, _run)


@router.post("/api/vault/lock")
async def lock_vault_sources(
    body: VaultLockBody, request: Request, profile: Optional[str] = None
):
    _require_token(request)

    def _run():
        from agent.vault_backends.unlock import lock

        lock(body.name if body.name else None)
        return {"locked": True}

    return await config_scoped_to_thread(profile, _run)
