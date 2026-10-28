"""The Webapp's door to this host's desktop plugins (``<root>/desktop-plugins/<name>/plugin.js``).

Native Desktop reads that folder from its own disk (``desktopPluginsRoot`` in Electron's
``fs-ipc.ts``). The Webapp's renderer runs in a browser whose only disk is this server, so the
browser bridge asks here instead, and the same runtime loader scans, loads and polls it.

The routes answer only what the loader reads: the root, its plugin folders, and each folder's
``plugin.js`` and ``.hermes-package.json``. Containment is checked on the path as written, then
symlinks are followed as Electron's ``readDir`` does, so ``desktop-plugins/foo -> ~/src/foo`` loads
here too. They never go through ``/api/fs``, which follows the profile's SSH workspace and truncates
at the preview cap. Webapp only (``SurfacePolicy.desktop_plugins``; 404 elsewhere).
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

from fastapi import APIRouter, HTTPException

from hermes_constants import get_default_hermes_root

router = APIRouter()

DESKTOP_PLUGINS_DIR = "desktop-plugins"
#: The files the loader reads in a plugin folder: its entry and the unified-package marker.
_READABLE_FILES = frozenset({"plugin.js", ".hermes-package.json"})
#: Electron's ``PLUGIN_SOURCE_MAX_BYTES``: a plugin loads whole or not at all, never truncated.
PLUGIN_SOURCE_MAX_BYTES = 16 * 1024 * 1024
#: Never follow a final link where the OS can refuse one, and never block on a FIFO.
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


def desktop_plugins_root() -> Path:
    """App-level, like Electron's: the Hermes root's folder, whichever profile the request names.
    Created on demand as Electron does, and resolved, so it prefixes every path the routes return."""
    root = get_default_hermes_root() / DESKTOP_PLUGINS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _inside_root(raw: str) -> tuple[Path, tuple[str, ...]]:
    """*raw* and its parts below the root, judged as written: no ``..``, at most a folder and a file.
    Links are not resolved here; a linked plugin folder is followed like native Desktop follows it.
    The files inside it are not: :func:`_read` refuses a linked entry file, which could name any
    file on the host, ``.env`` included, past the sensitive-file guard ``/api/fs`` applies."""
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
    from hermes_cli.web_routers.files import _is_sensitive_path

    target, parts = _inside_root(raw)
    not_found = HTTPException(status_code=404, detail="Not a desktop plugin file")
    if len(parts) != 2 or parts[1] not in _READABLE_FILES:
        raise not_found
    # Follow a linked plugin folder, never a linked file in it. The file actually opened must be the
    # regular file at the entry's real, non-sensitive path, checked again after the open, so swapping
    # the entry or its folder for a link mid-read fails closed, with or without O_NOFOLLOW.
    try:
        entry = target.parent.resolve(strict=True) / parts[1]
        if _is_sensitive_path(entry) or not stat.S_ISREG(os.lstat(entry).st_mode):
            raise not_found
        fd = os.open(entry, _OPEN_FLAGS)
    except OSError:
        raise not_found from None
    with os.fdopen(fd, "rb") as handle:
        st = os.fstat(handle.fileno())
        try:
            here = os.lstat(entry)
            unchanged = entry.resolve(strict=True) == entry
        except OSError:
            raise not_found from None
        if not (unchanged and stat.S_ISREG(st.st_mode) and (st.st_dev, st.st_ino) == (here.st_dev, here.st_ino)):
            raise not_found
        if st.st_size > PLUGIN_SOURCE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Plugin source exceeds 16 MiB")
        data = handle.read(PLUGIN_SOURCE_MAX_BYTES + 1)
    if len(data) > PLUGIN_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Plugin source exceeds 16 MiB")
    text = data.decode("utf-8-sig", errors="replace")
    return {"byteSize": st.st_size, "path": str(target), "text": text, "truncated": False}


@router.get("/api/desktop-plugins/root")
async def desktop_plugins_root_route() -> dict:
    return {"root": str(await asyncio.to_thread(desktop_plugins_root))}


@router.get("/api/desktop-plugins/list")
async def list_desktop_plugins(path: str) -> dict:
    return await asyncio.to_thread(_list, path)


@router.get("/api/desktop-plugins/read")
async def read_desktop_plugin_file(path: str) -> dict:
    return await asyncio.to_thread(_read, path)
