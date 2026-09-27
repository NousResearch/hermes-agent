"""Per-thread working directories on the matter host.

Layout under ``LITCO_MATTER_HOME`` (default ``~/matter``)::

    shared/                 channel threads (the whole case team)
      deliverables/
    users/<userId>/         dm threads (one lawyer's private work)
      deliverables/
    inbox/<turnId>/         attachments fetched for one turn

Directories are created on first use.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DELIVERABLES = "deliverables"
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]")


def matter_home(env: Optional[Dict[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    raw = env.get("LITCO_MATTER_HOME") or "~/matter"
    return Path(os.path.expanduser(raw)).resolve()


def safe_segment(value: str) -> str:
    """One path segment from an id: no separators, no dot-dot, bounded length."""
    cleaned = _SAFE_SEGMENT.sub("_", str(value or "")).strip("._") or "_"
    return cleaned[:128]


def thread_home(home: Path, kind: str, user_id: Optional[str]) -> Path:
    """Working directory for a thread: ``shared/`` for channel, ``users/<u>/`` for dm."""
    if kind == "dm":
        if not user_id:
            raise ValueError("a dm thread needs a userId")
        path = home / "users" / safe_segment(user_id)
    else:
        path = home / "shared"
    (path / DELIVERABLES).mkdir(parents=True, exist_ok=True)
    return path


def inbox_dir(home: Path, turn_id: str) -> Path:
    path = home / "inbox" / safe_segment(turn_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_deliverables(thread_dir: Path) -> Dict[str, Tuple[float, int]]:
    """``{abs path: (mtime, size)}`` for every file under the thread's deliverables folder."""
    root = thread_dir / DELIVERABLES
    out: Dict[str, Tuple[float, int]] = {}
    if not root.is_dir():
        return out
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.name.startswith("."):
                st = path.stat()
                out[str(path)] = (st.st_mtime, st.st_size)
        except OSError:
            continue
    return out


def encode_file_id(rel_path: str) -> str:
    return "dlv_" + base64.urlsafe_b64encode(rel_path.encode("utf-8")).rstrip(b"=").decode("ascii")


def decode_file_id(file_id: str) -> Optional[str]:
    if not file_id.startswith("dlv_"):
        return None
    body = file_id[4:]
    try:
        return base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode("utf-8")
    except Exception:
        return None


def changed_deliverables(home: Path, thread_dir: Path, before: Dict[str, Tuple[float, int]]) -> List[dict]:
    """Deliverables created or modified since ``before``, as ``final.deliverables`` entries.

    ``path`` is relative to the matter home; ``fileId`` encodes that path and can be fetched
    back through ``GET /deliverables/{fileId}``.
    """
    after = snapshot_deliverables(thread_dir)
    items = []
    for abs_path, stamp in sorted(after.items()):
        if before.get(abs_path) == stamp:
            continue
        rel = os.path.relpath(abs_path, home)
        mime = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"
        items.append({"fileId": encode_file_id(rel), "filename": os.path.basename(abs_path), "mime": mime,
                      "path": rel})
    return items


def resolve_deliverable(home: Path, file_id: str) -> Optional[Path]:
    """Map a deliverable id back to a file, refusing anything outside a deliverables folder."""
    rel = decode_file_id(file_id)
    if not rel:
        return None
    candidate = (home / rel).resolve()
    try:
        parts = candidate.relative_to(home).parts
    except ValueError:
        return None
    ok = (len(parts) >= 3 and parts[0] == "shared" and parts[1] == DELIVERABLES) or (
        len(parts) >= 4 and parts[0] == "users" and parts[2] == DELIVERABLES)
    if not ok or not candidate.is_file():
        return None
    return candidate
