"""Vault dashboard plugin — backend API routes.

Mounted at /api/plugins/vault/ by the dashboard plugin system.
Wraps the workstation.vault.VaultManager local-first Markdown engine.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from workstation.vault import VaultManager

_log = logging.getLogger(__name__)

router = APIRouter()
_vault_manager: Optional[VaultManager] = None


def _get_vault() -> VaultManager:
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = VaultManager()
    return _vault_manager


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------

class WriteNoteRequest(BaseModel):
    title: str = Field(..., description="Note title or filename")
    content: str = Field(..., description="Markdown note content")
    tags: Optional[List[str]] = Field(default=None, description="Tags list")
    frontmatter: Optional[Dict[str, Any]] = Field(default=None, description="Frontmatter metadata")
    subfolder: Optional[str] = Field(default=None, description="Subfolder within vault")


class AppendNoteRequest(BaseModel):
    content: str = Field(..., description="Content to append to note")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/notes")
def list_notes():
    """List all notes with summaries, tags, and link counts."""
    mgr = _get_vault()
    notes = mgr.list_notes()
    return {"notes": notes, "count": len(notes)}


@router.get("/notes/{title:path}")
def get_note(title: str):
    """Get single note content, metadata, forward links, and backlinks."""
    mgr = _get_vault()
    note = mgr.get_note(title)
    if not note:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note '{title}' not found")
    return {"note": note}


@router.post("/notes")
def create_or_update_note(payload: WriteNoteRequest):
    """Create or overwrite a Markdown note."""
    mgr = _get_vault()
    try:
        note = mgr.write_note(
            title=payload.title,
            content=payload.content,
            tags=payload.tags,
            frontmatter=payload.frontmatter,
            subfolder=payload.subfolder,
        )
        return {"status": "ok", "note": note}
    except Exception as e:
        _log.exception("Failed to write note")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/notes/{title:path}/append")
def append_to_note(title: str, payload: AppendNoteRequest):
    """Append text to an existing note."""
    mgr = _get_vault()
    try:
        note = mgr.append_note(title=title, append_content=payload.content)
        return {"status": "ok", "note": note}
    except Exception as e:
        _log.exception("Failed to append to note")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/notes/{title:path}")
def delete_note(title: str):
    """Delete a note from vault and index."""
    mgr = _get_vault()
    deleted = mgr.delete_note(title)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Note '{title}' not found")
    return {"status": "ok", "deleted": title}


@router.get("/search")
def search_notes(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)):
    """Search notes by keyword across titles, tags, and content."""
    mgr = _get_vault()
    results = mgr.search(query=q, limit=limit)
    return {"results": results, "count": len(results)}


@router.get("/graph")
def get_graph():
    """Get full knowledge graph nodes and wikilink edges for D3 visualization."""
    mgr = _get_vault()
    return {"graph": mgr.get_graph()}


@router.get("/suggest")
def suggest_wikilinks(prefix: str = Query("", description="Wikilink prefix to match")):
    """Get autocomplete suggestions for [[wikilinks]]."""
    mgr = _get_vault()
    return {"suggestions": mgr.suggest_wikilinks(prefix=prefix)}


@router.post("/rescan")
def rescan_vault():
    """Force re-scan vault directory from disk."""
    mgr = _get_vault()
    res = mgr.scan()
    return {"status": "ok", **res}
