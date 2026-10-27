"""The Webapp's door to this host's desktop plugins (``<root>/desktop-plugins/<name>/plugin.js``).

Native Desktop reads that folder from its own disk (``desktopPluginsRoot`` in Electron's
``fs-ipc.ts``). The Webapp's renderer runs in a browser whose only disk is this server, so the
browser bridge asks here instead, and the same runtime loader scans, loads and polls it.

The routes answer only what the loader reads: the root, its plugin folders, each folder's
``plugin.js`` and ``.hermes-package.json``, and a change stamp per ``plugin.js`` for hot reload. Containment is checked on the path as written, then
symlinks are followed as Electron's ``readDir`` does, so ``desktop-plugins/foo -> ~/src/foo`` loads
here too. They never go through ``/api/fs``, which follows the profile's SSH workspace and truncates
at the preview cap. Webapp only (``SurfacePolicy.desktop_plugins``; 404 elsewhere).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException

from hermes_constants import get_default_hermes_root

router = APIRouter()

DESKTOP_PLUGINS_DIR = "desktop-plugins"
#: The files the loader reads in a plugin folder: its entry and the unified-package marker.
_READABLE_FILES = frozenset({"plugin.js", ".hermes-package.json"})
#: Electron's ``PLUGIN_SOURCE_MAX_BYTES``: a plugin loads whole or not at all, never truncated.
PLUGIN_SOURCE_MAX_BYTES = 16 * 1024 * 1024


def desktop_plugins_root() -> Path:
    """App-level, like Electron's: the Hermes root's folder, whichever profile the request names.
    Created on demand as Electron does, and resolved, so it prefixes every path the routes return."""
    root = get_default_hermes_root() / DESKTOP_PLUGINS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _inside_root(raw: str) -> tuple[Path, tuple[str, ...]]:
    """*raw* and its parts below the root, judged as written: no ``..``, at most a folder and a file.
    Links are not resolved here; a linked plugin folder is followed like native Desktop follows it.
    The link buys nothing the tab cannot already reach through ``/api/fs`` or the host terminal."""
    root = desktop_plugins_root()
    try:
        parts = Path(raw).relative_to(root).parts
    except ValueError:
        raise HTTPException(status_code=404, detail="Not a desktop plugin path") from None
    if len(parts) > 2 or ".." in parts:
        raise HTTPException(status_code=404, detail="Not a desktop plugin path")
    return root.joinpath(*parts), parts


def _list(raw: str) -> dict:
    target, parts = _inside_root(raw)
    if len(parts) > 1 or not target.is_dir():
        raise HTTPException(status_code=404, detail="Not a desktop plugin folder")
    entries = [
        {"name": child.name, "path": str(target / child.name), "isDirectory": child.is_dir()}
        for child in sorted(target.iterdir(), key=lambda child: child.name)
    ]
    return {"entries": entries}


def _read(raw: str) -> dict:
    target, parts = _inside_root(raw)
    if len(parts) != 2 or parts[1] not in _READABLE_FILES or not target.is_file():
        raise HTTPException(status_code=404, detail="Not a desktop plugin file")
    size = target.stat().st_size
    if size > PLUGIN_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Plugin source exceeds 16 MiB")
    text = target.read_text(encoding="utf-8-sig", errors="replace")
    return {"byteSize": size, "path": str(target), "text": text, "truncated": False}


def _stamps() -> dict:
    """A change stamp for each folder's ``plugin.js``, keyed by the path ``_list`` hands out. The
    browser has no file watcher, so it polls this one listing to hot-reload an edited entry the way
    Electron's ``watchPreviewFile`` does. Size joins the mtime so a same-tick rewrite still shows."""
    root = desktop_plugins_root()
    stamps = {}
    for folder in root.iterdir():
        entry = folder / "plugin.js"
        try:
            stat = entry.stat()
        except OSError:
            continue
        stamps[str(entry)] = f"{stat.st_mtime_ns}:{stat.st_size}"
    return {"stamps": stamps}


@router.get("/api/desktop-plugins/root")
async def desktop_plugins_root_route() -> dict:
    return {"root": str(await asyncio.to_thread(desktop_plugins_root))}


@router.get("/api/desktop-plugins/list")
async def list_desktop_plugins(path: str) -> dict:
    return await asyncio.to_thread(_list, path)


@router.get("/api/desktop-plugins/read")
async def read_desktop_plugin_file(path: str) -> dict:
    return await asyncio.to_thread(_read, path)


@router.get("/api/desktop-plugins/stamps")
async def desktop_plugin_stamps() -> dict:
    return await asyncio.to_thread(_stamps)
