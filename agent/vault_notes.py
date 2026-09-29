"""Obsidian-style vault as Agent Czesiek's browsable memory.

The vault is plain markdown the user owns (see ``apps/desktop/electron/vault-seed.ts``).
This module turns it into a graph for the desktop "Mapa wiedzy" (notes = nodes,
``[[wikilinks]]`` = edges) and offers safe read/write/create/delete on note files.

Every path that comes from the client is a vault-relative POSIX path and is resolved
through :func:`_resolve_note`, which refuses anything outside the vault, anything that is
not ``.md``, and hidden/tooling directories (``.obsidian``, ``.git``, ``.trash``).
Deleting moves the note to ``.trash/`` (Obsidian's own convention) so it is recoverable.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

VAULT_ENV_KEY = "OBSIDIAN_VAULT_PATH"
VAULT_DIR_NAME = "Czesiek Vault"

_SKIP_DIRS = {".obsidian", ".git", ".trash", "node_modules", ".archive"}
_MAX_NOTES = 2000
_MAX_NOTE_BYTES = 512 * 1024
# `![[file]]` embeds a file rather than linking a note, so it is not an edge.
_WIKILINK = re.compile(r"(?<!!)\[\[([^\]\n|#]+)(?:#[^\]\n|]*)?(?:\|[^\]\n]*)?\]\]")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
_TAG = re.compile(r"(?<![\w/])#([A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż][\w/-]*)")


def _env_file_value(key: str) -> Optional[str]:
    try:
        text = (get_hermes_home() / ".env").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == key:
            return value.strip().strip("'\"") or None
    return None


def resolve_vault_path() -> Path:
    """Configured vault (process env, then the profile ``.env``), else ``~/Documents/Czesiek Vault``."""
    raw = os.environ.get(VAULT_ENV_KEY) or _env_file_value(VAULT_ENV_KEY)
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "Documents" / VAULT_DIR_NAME


def vault_status() -> dict[str, Any]:
    path = resolve_vault_path()
    return {"path": str(path), "exists": path.is_dir()}


def _iter_note_paths(root: Path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.lower().endswith(".md") and not name.startswith("."):
                yield Path(dirpath) / name
                count += 1
                if count >= _MAX_NOTES:
                    return


def _resolve_note(rel: str, *, must_exist: bool = True) -> Path:
    root = resolve_vault_path().resolve()
    if not rel or not rel.lower().endswith(".md"):
        raise ValueError("note path must end with .md")
    candidate = (root / rel).resolve()
    try:
        parts = candidate.relative_to(root).parts
    except ValueError as exc:
        raise ValueError("note path escapes the vault") from exc
    if any(p in _SKIP_DIRS or p.startswith(".") for p in parts):
        raise ValueError("note path points into a hidden vault folder")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(rel)
    return candidate


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_text(path: Path) -> str:
    with path.open("rb") as handle:
        return handle.read(_MAX_NOTE_BYTES).decode("utf-8", errors="replace")


def _title(rel: str, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()[:120]
    return Path(rel).stem.replace("_", " ")


# Parsed notes keyed by absolute path and validated by (mtime, size), so a poll of an unchanged vault
# costs one stat per note instead of a read and three regex passes each.
_PARSE_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
_PARSE_LOCK = threading.Lock()


def _parse_note(rel: str, path: Path) -> dict[str, Any]:
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    with _PARSE_LOCK:
        hit = _PARSE_CACHE.get(str(path))
    if hit and hit[0] == key:
        return hit[1]
    text = _read_text(path)
    prose = _CODE.sub(" ", text)  # links and tags inside code are examples, not references
    parsed = {
        "label": _title(rel, text), "timestamp": int(stat.st_mtime), "size": len(text),
        "tags": sorted(set(_TAG.findall(prose)))[:8], "excerpt": " ".join(text.split())[:160],
        "links": _WIKILINK.findall(prose),
    }
    with _PARSE_LOCK:
        _PARSE_CACHE[str(path)] = (key, parsed)
    return parsed


def build_vault_graph() -> dict[str, Any]:
    """Notes as nodes, resolved wikilinks as undirected edges."""
    root = resolve_vault_path()
    status = {"path": str(root), "exists": root.is_dir()}
    if not status["exists"]:
        return {"vault": status, "nodes": [], "edges": []}
    root = root.resolve()

    notes: list[dict[str, Any]] = []
    by_stem: dict[str, str] = {}
    by_rel: dict[str, str] = {}
    raw_links: dict[str, list[str]] = {}
    seen: set[str] = set()
    for path in _iter_note_paths(root):
        rel = _rel(root, path)
        try:
            parsed = _parse_note(rel, path)
        except OSError:
            continue
        seen.add(str(path))
        folder = rel.rsplit("/", 1)[0] if "/" in rel else ""
        notes.append({
            "id": rel, "label": parsed["label"], "folder": folder, "timestamp": parsed["timestamp"],
            "size": parsed["size"], "tags": parsed["tags"], "excerpt": parsed["excerpt"],
        })
        by_stem.setdefault(path.stem.lower(), rel)
        by_rel[rel[:-3].lower()] = rel
        raw_links[rel] = parsed["links"]
    with _PARSE_LOCK:  # forget notes that are gone, so the cache cannot outgrow the vault
        for stale in [k for k in _PARSE_CACHE if k.startswith(str(root)) and k not in seen]:
            del _PARSE_CACHE[stale]

    edges: set[tuple[str, str]] = set()
    for rel, targets in raw_links.items():
        for target in targets:
            key = target.strip().lower().removesuffix(".md")
            hit = by_rel.get(key) or by_stem.get(key.rsplit("/", 1)[-1])
            if hit and hit != rel:
                edges.add((min(rel, hit), max(rel, hit)))

    degree: dict[str, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    for note in notes:
        note["links"] = degree.get(note["id"], 0)
    return {
        "vault": status,
        "nodes": notes,
        "edges": [{"source": a, "target": b} for a, b in sorted(edges)],
    }


def read_note(rel: str) -> dict[str, Any]:
    path = _resolve_note(rel)
    text = _read_text(path)
    return {"ok": True, "id": rel, "label": _title(rel, text), "content": text}


def write_note(rel: str, content: str) -> dict[str, Any]:
    """Overwrite an existing note atomically (temp file + rename)."""
    path = _resolve_note(rel)
    if len(content.encode("utf-8")) > _MAX_NOTE_BYTES:
        raise ValueError("note is too large")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return {"ok": True, "id": rel}


def _safe_filename(title: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|#^\[\]]+', " ", title).strip(" .")
    return " ".join(cleaned.split())[:80]


def create_note(
    title: str,
    folder: str = "",
    content: Optional[str] = None,
    *,
    conflict: str = "error",
    dedupe_key: Optional[str] = None,
) -> dict[str, Any]:
    """Create ``<folder>/<title>.md``.

    ``conflict="error"`` refuses to overwrite (``FileExistsError``). ``conflict="suffix"``
    picks ``<title> (2).md``, ``(3)`` ... instead. ``dedupe_key`` (e.g. a URL) makes the call
    idempotent: an existing note in that name family whose text contains the key IS the note,
    so it is returned with ``existed: True`` rather than duplicated — while a different note
    that merely sanitises to the same file name gets its own suffixed file.
    """
    if conflict not in ("error", "suffix"):
        raise ValueError("conflict must be 'error' or 'suffix'")
    name = _safe_filename(title)
    if not name:
        raise ValueError("note title is empty")
    folder = folder.strip("/ ")
    prefix = f"{folder}/" if folder else ""
    body = content if content is not None else f"# {name}\n\n"

    n = 1
    while True:
        stem = name if n == 1 else f"{name} ({n})"
        rel = f"{prefix}{stem}.md"
        path = _resolve_note(rel, must_exist=False)
        if not path.exists():
            break
        if dedupe_key and dedupe_key in _read_text(path):
            return {"ok": True, "id": rel, "existed": True}
        if conflict == "error" and not dedupe_key:
            raise FileExistsError(rel)
        n += 1
        if n > 200:
            raise FileExistsError(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return {"ok": True, "id": rel, "existed": False}


def delete_note(rel: str) -> dict[str, Any]:
    """Move the note into ``.trash/`` so Obsidian (and the user) can restore it."""
    path = _resolve_note(rel)
    root = resolve_vault_path().resolve()
    trash = root / ".trash"
    trash.mkdir(exist_ok=True)
    target = trash / path.name
    n = 1
    while target.exists():
        n += 1
        target = trash / f"{path.stem} ({n}){path.suffix}"
    shutil.move(str(path), str(target))
    return {"ok": True, "id": rel}
