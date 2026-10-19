"""Hermes Docs dashboard plugin — backend API routes.

Mounted at /api/plugins/hermes-docs/ by the dashboard plugin system.

Provides:
  - Workspace registry (create, list, remove) backed by
    ~/.hermes/docs-workspaces/registry.json
  - Per-workspace file tree and file read/write (with preview-before-write)
  - Per-workspace comment (annotation) CRUD
  - Per-workspace preferences (drawer pin state, theme override)
  - Side-chat endpoint: routes to local docs agent when the ``docs`` profile
    is present; falls back to a deterministic stub otherwise.
  - Status endpoint consumed by the Dashboard launcher surface

Storage is profile-aware: every path goes through get_hermes_home() so
that multi-profile setups each get isolated state.  Workspace source
folders are never touched unless the user explicitly writes a file there.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from anyio import to_thread
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hermes_constants import get_hermes_home

# docs_profile is co-located in the same dashboard package.  When loaded via
# the plugin system (which sets __package__ and submodule_search_locations)
# the relative import resolves normally.  When loaded dynamically by tests
# via importlib without a package context, we fall back to loading the sibling
# file directly by path.
import importlib.util as _ilu

try:
    from .docs_profile import get_docs_profile_status, bootstrap_docs_profile  # type: ignore[import]
except ImportError:
    _dp_path = Path(__file__).parent / "docs_profile.py"
    _dp_name = f"{__name__}_docs_profile"
    if _dp_name not in sys.modules:
        _dp_spec = _ilu.spec_from_file_location(_dp_name, _dp_path)
        _dp_mod = _ilu.module_from_spec(_dp_spec)
        sys.modules[_dp_name] = _dp_mod
        _dp_spec.loader.exec_module(_dp_mod)
    else:
        _dp_mod = sys.modules[_dp_name]
    get_docs_profile_status = _dp_mod.get_docs_profile_status  # type: ignore[attr-defined]
    bootstrap_docs_profile = _dp_mod.bootstrap_docs_profile  # type: ignore[attr-defined]

# kordoc_helper lives alongside plugin_api.py — use an explicit importlib
# load so this module works whether invoked as part of a package or loaded
# directly (as tests do) via spec_from_file_location.
_kordoc_spec = _ilu.spec_from_file_location(
    "hermes_docs_kordoc_helper",
    Path(__file__).parent / "kordoc_helper.py",
)
kordoc_helper = _ilu.module_from_spec(_kordoc_spec)
_kordoc_spec.loader.exec_module(kordoc_helper)  # type: ignore[union-attr]

# codex_auth_helper lives alongside plugin_api.py — same explicit-load pattern
# so the module resolves correctly as a standalone file in tests.
_codex_auth_spec = _ilu.spec_from_file_location(
    "hermes_docs_codex_auth_helper",
    Path(__file__).parent / "codex_auth_helper.py",
)
codex_auth_helper = _ilu.module_from_spec(_codex_auth_spec)
_codex_auth_spec.loader.exec_module(codex_auth_helper)  # type: ignore[union-attr]

log = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Docs agent broker — injectable override for tests
# ---------------------------------------------------------------------------

# Tests can replace _broker_override with a callable(user_msg, context) -> str
# to exercise the broker-available path without launching a real AIAgent.
# Production code leaves this as None and the real broker path is taken.
_broker_override = None


def _docs_profile_home() -> Path | None:
    """Return the Hermes home directory for the ``docs`` profile if it exists.

    Looks for ``~/.hermes/profiles/docs/config.yaml`` — the standard profile
    layout produced by ``hermes profile create docs``.  Returns None when the
    profile does not exist so the caller can fall back gracefully.
    """
    hermes_home = get_hermes_home()
    # get_hermes_home() may already be a profile dir (e.g. ~/.hermes/profiles/docs)
    # or the default ~/.hermes.  We compute the profiles root from the Hermes home.
    # The display_hermes_home helper resolves the *root* (parent of profiles/),
    # but we replicate the simple heuristic here to avoid importing more of the
    # config layer into the plugin.
    if hermes_home.parent.name == "profiles":
        # We're already inside a profile; root is grandparent
        hermes_root = hermes_home.parent.parent
    else:
        hermes_root = hermes_home
    candidate = hermes_root / "profiles" / "docs"
    if (candidate / "config.yaml").exists():
        return candidate
    return None


def _call_docs_agent(user_msg: str, context: dict) -> str:
    """Invoke the local docs agent profile and return its response text.

    ``context`` is a dict with optional keys:
      - ``workspace``: workspace name (str)
      - ``document``: active document rel-path (str | None)
      - ``selection``: selected text (str | None)

    Raises RuntimeError if the agent call fails so the caller can fall back.
    """
    if _broker_override is not None:
        return _broker_override(user_msg, context)

    docs_home = _docs_profile_home()
    if docs_home is None:
        raise RuntimeError("docs profile not found")

    parts = [f"Workspace: {context.get('workspace', '(unnamed)')}"]
    if context.get("document"):
        parts.append(f"Document: {context['document']}")
    if context.get("selection"):
        parts.append(f"Selected text:\n> {context['selection'][:500]}")
    system_context = "\n".join(parts)

    full_prompt = (
        f"[Hermes Docs Side Chat]\n{system_context}\n\n"
        f"User: {user_msg}"
    )

    hermes_cmd = Path(sys.executable).with_name("hermes")
    executable = str(hermes_cmd) if hermes_cmd.exists() else shutil.which("hermes")
    if not executable:
        raise RuntimeError("hermes executable not found")

    completed = subprocess.run(
        [executable, "-p", "docs", "chat", "-q", full_prompt],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"docs agent exited with status {completed.returncode}: {error[:500]}"
        )

    response = completed.stdout.strip()
    if not response:
        raise RuntimeError("docs agent returned an empty response")
    return response

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _docs_home() -> Path:
    """Profile-aware root for all Docs metadata.  Never use ~/.hermes directly."""
    return get_hermes_home() / "docs-workspaces"


def _registry_path() -> Path:
    return _docs_home() / "registry.json"


def _workspace_meta_dir(workspace_id: str) -> Path:
    return _docs_home() / workspace_id


def _load_registry() -> list[dict]:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_registry(workspaces: list[dict]) -> None:
    _docs_home().mkdir(parents=True, exist_ok=True)
    _registry_path().write_text(
        json.dumps(workspaces, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _get_workspace(workspace_id: str) -> dict | None:
    return next((w for w in _load_registry() if w["id"] == workspace_id), None)


def _init_workspace_dirs(workspace_id: str) -> None:
    base = _workspace_meta_dir(workspace_id)
    for subdir in ("sessions", "annotations", "conversions"):
        (base / subdir).mkdir(parents=True, exist_ok=True)
    state_file = base / "document-state.json"
    if not state_file.exists():
        state_file.write_text(
            json.dumps({"open_files": [], "active_file": None}, indent=2),
            encoding="utf-8",
        )
    pref_file = base / "preferences.json"
    if not pref_file.exists():
        pref_file.write_text(
            json.dumps({"drawer_pinned": False, "theme": "system"}, indent=2),
            encoding="utf-8",
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_relative_target(base: Path, rel: str) -> Path:
    """Resolve a relative path inside base and raise 403 if it escapes."""
    resolved_base = base.resolve()
    target = (resolved_base / rel).resolve()
    try:
        target.relative_to(resolved_base)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    return target


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class WorkspaceCreate(BaseModel):
    name: str
    path: str


class FileWrite(BaseModel):
    content: str
    preview: bool = True


class CommentCreate(BaseModel):
    document: str
    text: str
    anchor_start: int
    anchor_end: int
    anchor_text: str


class CommentPatch(BaseModel):
    resolved: bool


class SideChatMessage(BaseModel):
    content: str
    document: Optional[str] = None
    selection: Optional[str] = None


# ---------------------------------------------------------------------------
# Status — consumed by the Dashboard launcher tab
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_status():
    workspaces = _load_registry()
    recent = [
        {"id": w["id"], "name": w["name"], "path": w["path"]}
        for w in workspaces[:3]
    ]
    return {
        "available": True,
        "workspace_count": len(workspaces),
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# Docs persona profile status and bootstrap
# ---------------------------------------------------------------------------


@router.get("/profile/status")
async def get_profile_status():
    """Report whether the docs persona profile is installed.

    Returns a JSON object with::

        {
            "installed": bool,
            "profile_dir": str | null,
            "has_soul": bool,
            "has_config": bool,
        }
    """
    return get_docs_profile_status()


@router.post("/profile/bootstrap", status_code=200)
async def bootstrap_profile():
    """Create a minimal docs profile if one does not already exist.

    Idempotent.  Returns::

        {
            "status": "created" | "already_exists",
            "profile_dir": str,
            "created_files": list[str],
        }
    """
    return bootstrap_docs_profile()


# ---------------------------------------------------------------------------
# Workspace registry
# ---------------------------------------------------------------------------


@router.get("/workspaces")
async def list_workspaces():
    workspaces = _load_registry()
    result = []
    for ws in workspaces:
        ws = dict(ws)
        ws["folder_exists"] = Path(ws["path"]).exists()
        result.append(ws)
    return result


@router.post("/workspaces", status_code=201)
async def create_workspace(body: WorkspaceCreate):
    folder = Path(body.path)
    if not folder.exists():
        raise HTTPException(
            status_code=400, detail=f"Folder does not exist: {body.path}"
        )
    if not folder.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Path is not a folder: {body.path}"
        )

    workspaces = _load_registry()
    resolved = str(folder.resolve())
    if any(w["path"] == resolved for w in workspaces):
        raise HTTPException(status_code=409, detail="Workspace already registered")

    ws_id = str(uuid.uuid4())
    now = _now_iso()
    workspace = {
        "id": ws_id,
        "name": body.name.strip() or folder.name,
        "path": resolved,
        "created_at": now,
        "last_opened": now,
    }
    workspaces.append(workspace)
    _save_registry(workspaces)
    _init_workspace_dirs(ws_id)
    return workspace


@router.delete("/workspaces/{workspace_id}")
async def remove_workspace(workspace_id: str):
    workspaces = _load_registry()
    before = len(workspaces)
    workspaces = [w for w in workspaces if w["id"] != workspace_id]
    if len(workspaces) == before:
        raise HTTPException(status_code=404, detail="Workspace not found")
    _save_registry(workspaces)
    return {"ok": True}


@router.post("/workspaces/{workspace_id}/open")
async def open_workspace(workspace_id: str):
    """Update last_opened timestamp when the user switches to a workspace."""
    workspaces = _load_registry()
    for ws in workspaces:
        if ws["id"] == workspace_id:
            ws["last_opened"] = _now_iso()
            _save_registry(workspaces)
            return ws
    raise HTTPException(status_code=404, detail="Workspace not found")


# ---------------------------------------------------------------------------
# File tree and file I/O
# ---------------------------------------------------------------------------


@router.get("/workspaces/{workspace_id}/files")
async def list_files(workspace_id: str, rel: str = ""):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    base = Path(ws["path"])
    target = _safe_relative_target(base, rel) if rel else base.resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    entries = []
    try:
        items = sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        for item in items:
            if item.name.startswith("."):
                continue
            entry = {
                "name": item.name,
                "rel": str(item.relative_to(base)),
                "is_dir": item.is_dir(),
            }
            if item.is_file():
                stat = item.stat()
                entry["size"] = stat.st_size
                entry["modified"] = datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat()
            entries.append(entry)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied") from exc
    return entries


@router.get("/workspaces/{workspace_id}/file")
async def read_file(workspace_id: str, rel: str):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    base = Path(ws["path"])
    target = _safe_relative_target(base, rel)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        content = target.read_text(encoding="utf-8")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied") from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="File is not valid UTF-8 text"
        ) from exc
    return {"rel": rel, "content": content}


@router.put("/workspaces/{workspace_id}/file")
async def write_file(workspace_id: str, rel: str, body: FileWrite):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    base = Path(ws["path"])
    target = _safe_relative_target(base, rel)

    if body.preview:
        existing = ""
        if target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except Exception:
                pass
        return {"preview": True, "rel": rel, "current": existing, "proposed": body.content}

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.content, encoding="utf-8")
    return {"ok": True, "rel": rel}


# ---------------------------------------------------------------------------
# Comments / annotations
# ---------------------------------------------------------------------------


@router.get("/workspaces/{workspace_id}/comments")
async def list_comments(workspace_id: str, document: Optional[str] = None):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ann_dir = _workspace_meta_dir(workspace_id) / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    comments = []
    for f in ann_dir.glob("*.json"):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
            if document is None or c.get("document") == document:
                comments.append(c)
        except Exception:
            pass
    comments.sort(key=lambda c: c.get("created_at", ""))
    return comments


@router.post("/workspaces/{workspace_id}/comments", status_code=201)
async def create_comment(workspace_id: str, body: CommentCreate):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ann_dir = _workspace_meta_dir(workspace_id) / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    c_id = str(uuid.uuid4())
    comment = {
        "id": c_id,
        "workspace_id": workspace_id,
        "document": body.document,
        "text": body.text,
        "anchor_start": body.anchor_start,
        "anchor_end": body.anchor_end,
        "anchor_text": body.anchor_text,
        "resolved": False,
        "created_at": _now_iso(),
        "resolved_at": None,
    }
    (ann_dir / f"{c_id}.json").write_text(
        json.dumps(comment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return comment


@router.patch("/workspaces/{workspace_id}/comments/{comment_id}")
async def update_comment(workspace_id: str, comment_id: str, body: CommentPatch):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ann_dir = _workspace_meta_dir(workspace_id) / "annotations"
    cf = ann_dir / f"{comment_id}.json"
    if not cf.exists():
        raise HTTPException(status_code=404, detail="Comment not found")
    comment = json.loads(cf.read_text(encoding="utf-8"))
    comment["resolved"] = body.resolved
    comment["resolved_at"] = _now_iso() if body.resolved else None
    cf.write_text(json.dumps(comment, indent=2, ensure_ascii=False), encoding="utf-8")
    return comment


# ---------------------------------------------------------------------------
# Workspace preferences
# ---------------------------------------------------------------------------


@router.get("/workspaces/{workspace_id}/preferences")
async def get_preferences(workspace_id: str):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    pref_file = _workspace_meta_dir(workspace_id) / "preferences.json"
    if not pref_file.exists():
        return {"drawer_pinned": False, "theme": "system"}
    return json.loads(pref_file.read_text(encoding="utf-8"))


@router.put("/workspaces/{workspace_id}/preferences")
async def update_preferences(workspace_id: str, request: Request):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    body = await request.json()
    pref_file = _workspace_meta_dir(workspace_id) / "preferences.json"
    existing: dict = {}
    if pref_file.exists():
        try:
            existing = json.loads(pref_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(body)
    _workspace_meta_dir(workspace_id).mkdir(parents=True, exist_ok=True)
    pref_file.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return existing


# ---------------------------------------------------------------------------
# Side-chat — routes to docs agent profile; falls back to stub when unavailable
# ---------------------------------------------------------------------------


@router.post("/workspaces/{workspace_id}/sidechat")
async def sidechat(workspace_id: str, body: SideChatMessage):
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Persist the exchange as a session entry for future recall
    sessions_dir = _workspace_meta_dir(workspace_id) / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": str(uuid.uuid4()),
        "workspace_id": workspace_id,
        "user": body.content,
        "document": body.document,
        "selection": body.selection,
        "created_at": _now_iso(),
    }
    entry_path = sessions_dir / f"{entry['id']}.json"

    broker_context = {
        "workspace": ws["name"],
        "document": body.document,
        "selection": body.selection,
    }

    # Attempt to route to the local docs agent profile.
    brokered = False
    try:
        response_text = await to_thread.run_sync(
            _call_docs_agent, body.content, broker_context
        )
        brokered = True
    except Exception as exc:
        log.debug("docs agent broker unavailable (%s); using stub response", exc)
        response_text = (
            "Docs persona not available yet. "
            "Create a 'docs' profile in Hermes settings to enable real brainstorming assistance.\n\n"
            f"You asked: {body.content[:200]}"
        )
        if body.selection:
            response_text += f"\n\nSelected text:\n> {body.selection[:300]}"

    # Persist assistant response alongside the user message
    entry["assistant"] = response_text
    entry["brokered"] = brokered
    entry_path.write_text(
        json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "role": "assistant",
        "content": response_text,
        "brokered": brokered,
        "context": {
            "workspace": ws["name"],
            "document": body.document,
        },
    }


# ---------------------------------------------------------------------------
# Kordoc detection and conversion preview
# ---------------------------------------------------------------------------


class ConversionPreviewRequest(BaseModel):
    rel: str
    target_format: str = "markdown"  # "markdown" | "json"


@router.get("/kordoc/status")
async def kordoc_status():
    """Report whether kordoc is available on this machine.

    Returns a JSON object:
      ``available``  (bool)  — deterministic availability flag
      ``version``    (str|null) — reported version string, or null
      ``detail``     (str)   — human-readable status message
    """
    return await to_thread.run_sync(kordoc_helper.detect_kordoc)


@router.post("/workspaces/{workspace_id}/kordoc/preview")
async def kordoc_preview(workspace_id: str, body: ConversionPreviewRequest):
    """Return a conversion preview for a workspace file.

    - Accepts a workspace-relative file path and a target format.
    - Blocks path traversal (HTTP 403).
    - Returns a structured stub when kordoc is unavailable (HTTP 200 with
      ``available: false``) — never raises 5xx for an availability gap.
    - Does not mutate the source document.
    """
    ws = _get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    workspace_base = Path(ws["path"])

    result = await to_thread.run_sync(
        kordoc_helper.preview_conversion,
        workspace_base,
        body.rel,
        body.target_format,
    )
    return result


# ---------------------------------------------------------------------------
# Codex auth readiness status
# ---------------------------------------------------------------------------


@router.get("/auth/codex/status")
async def get_codex_auth_status():
    """Report whether the local Codex / OpenAI auth broker is configured.

    Returns a JSON object with::

        {
            "provider_id":  "openai-codex",
            "configured":   bool,   // True when valid credentials are present
            "available":    bool,   // alias for configured
            "cli_command":  str,    // "hermes auth add openai-codex"
            "token_exposed": false, // always False — no secrets are returned
            "detail":       str,    // human-readable status line
            "next_action":  str | null,  // what to do next; null when ready
        }

    This endpoint is **read-only**.  It never initiates OAuth or any network
    call.  Raw tokens are never included in the response.
    """
    return await to_thread.run_sync(codex_auth_helper.get_codex_status)
