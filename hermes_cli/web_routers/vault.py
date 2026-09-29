"""Vault memory for the desktop "Mapa wiedzy": ``/api/vault/*``.

Thin REST over :mod:`agent.vault_notes`. The vault is the user's own Obsidian-compatible
folder; every handler runs under ``scoped_to_thread`` so it reads the active profile's
``.env`` (``OBSIDIAN_VAULT_PATH``) off the event loop.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hermes_cli.web_routers._common import log as _log
from hermes_cli.web_routers._common import scoped_to_thread

router = APIRouter()


class VaultNoteRef(BaseModel):
    id: str
    profile: Optional[str] = None


class VaultNoteEdit(VaultNoteRef):
    content: str


class VaultNoteCreate(BaseModel):
    title: str
    folder: str = ""
    content: Optional[str] = None
    profile: Optional[str] = None


async def _run(profile: Optional[str], fn):
    try:
        return await scoped_to_thread(profile, fn)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="note not found")
    except FileExistsError:
        raise HTTPException(status_code=409, detail="note already exists")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError:
        _log.exception("vault I/O failed")
        raise HTTPException(status_code=500, detail="vault I/O failed")


@router.get("/api/vault/graph")
async def get_vault_graph(profile: Optional[str] = None):
    from agent.vault_notes import build_vault_graph
    return await _run(profile, build_vault_graph)


@router.get("/api/vault/note")
async def get_vault_note(id: str, profile: Optional[str] = None):
    from agent.vault_notes import read_note
    return await _run(profile, lambda: read_note(id))


@router.put("/api/vault/note")
async def put_vault_note(body: VaultNoteEdit):
    from agent.vault_notes import write_note
    return await _run(body.profile, lambda: write_note(body.id, body.content))


@router.post("/api/vault/note")
async def post_vault_note(body: VaultNoteCreate):
    from agent.vault_notes import create_note
    return await _run(body.profile, lambda: create_note(body.title, body.folder, body.content))


@router.delete("/api/vault/note")
async def delete_vault_note(body: VaultNoteRef):
    from agent.vault_notes import delete_note
    return await _run(body.profile, lambda: delete_note(body.id))
