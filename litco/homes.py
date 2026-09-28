"""Per-thread working directories on the matter host.

Layout under ``LITCO_MATTER_HOME`` (default ``~/matter``)::

    shared/                 channel threads (the whole case team)
      deliverables/
    users/<userId>/         dm threads (one lawyer's private work)
      deliverables/
    inbox/<turnId>/         attachments fetched for one turn

Directories are created on first use.

What counts as a deliverable. A turn delivers (``final.deliverables``) exactly:

* every file the agent registered with ``litco_deliver_local`` during the turn (copied into the
  thread's ``deliverables/`` folder if it lived elsewhere), whatever its type; and
* every other file created or changed under ``deliverables/`` whose extension is on
  :data:`DELIVERABLE_EXTENSIONS`, unless it is scratch: a dotfile or a file in a dot-folder,
  ``*.spec.json``, ``*.tmp``, or an Office lock file (``~$*``).
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DELIVERABLES = "deliverables"
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]")
# Files under deliverables/ that ship without registration. Anything else must be registered.
DELIVERABLE_EXTENSIONS = frozenset({".docx", ".xlsx", ".pptx", ".pdf", ".md", ".txt", ".csv", ".png", ".jpg",
                                    ".jpeg"})


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


def is_scratch(path: Path, root: Path) -> bool:
    """A working file that never ships unregistered: dotfiles and dot-folders, ``*.spec.json``,
    ``*.tmp``, Office lock files (``~$*``)."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = (path.name,)
    if any(part.startswith(".") for part in parts):
        return True
    name = path.name.lower()
    return name.startswith("~$") or name.endswith(".spec.json") or name.endswith(".tmp")


def auto_deliverable(path: Path, root: Path) -> bool:
    """True when an unregistered file under ``deliverables/`` ships: allow-listed type, not scratch."""
    return path.suffix.lower() in DELIVERABLE_EXTENSIONS and not is_scratch(path, root)


# ---------------------------------------------------------------------------
# Explicit registration (the ``litco_deliver_local`` tool)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegisteredDeliverable:
    path: Path                       # absolute, inside a thread's deliverables/ folder
    filename: str
    deliverable_class: Optional[str] = None


_REGISTRY: Dict[str, Dict[str, RegisteredDeliverable]] = {}
_REGISTRY_LOCK = threading.Lock()


def register_deliverable(turn_id: str, thread_dir: Path, source: Path, *, name: Optional[str] = None,
                         deliverable_class: Optional[str] = None) -> RegisteredDeliverable:
    """Mark ``source`` as a file this turn delivers to its thread.

    A file outside the thread's ``deliverables/`` folder (or registered under a different name)
    is copied there first, so ``GET /deliverables/{fileId}`` can serve it. Registering the same
    target twice keeps the latest call.
    """
    if not turn_id:
        raise ValueError("no turn is active, so there is no thread to deliver to")
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist; write the file first")
    root = (Path(thread_dir) / DELIVERABLES).resolve()
    root.mkdir(parents=True, exist_ok=True)
    filename = safe_segment(name or source.name)
    try:
        source.relative_to(root)
        in_place = name is None or filename == source.name
    except ValueError:
        in_place = False
    target = source if in_place else root / filename
    if target != source:
        shutil.copy2(source, target)
    entry = RegisteredDeliverable(path=target, filename=target.name,
                                  deliverable_class=(deliverable_class or None))
    with _REGISTRY_LOCK:
        _REGISTRY.setdefault(turn_id, {})[str(target)] = entry
    return entry


def pop_registered(turn_id: str) -> List[RegisteredDeliverable]:
    """Registrations for a turn, removed from the registry (the server calls this once per turn)."""
    with _REGISTRY_LOCK:
        return list(_REGISTRY.pop(turn_id, {}).values())


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


def _entry(home: Path, abs_path: str, deliverable_class: Optional[str] = None) -> dict:
    rel = os.path.relpath(abs_path, home)
    mime = mimetypes.guess_type(abs_path)[0] or "application/octet-stream"
    item = {"fileId": encode_file_id(rel), "filename": os.path.basename(abs_path), "mime": mime, "path": rel}
    if deliverable_class:
        item["deliverableClass"] = deliverable_class
    return item


def changed_deliverables(home: Path, thread_dir: Path, before: Dict[str, Tuple[float, int]],
                         registered: Optional[List[RegisteredDeliverable]] = None) -> List[dict]:
    """The turn's ``final.deliverables`` entries.

    Registered files come first (in registration order), then unregistered files created or
    modified under ``deliverables/`` since ``before`` that pass :func:`auto_deliverable`.
    ``path`` is relative to the matter home; ``fileId`` encodes that path and can be fetched back
    through ``GET /deliverables/{fileId}``. ``deliverableClass`` is present when the agent gave one.
    """
    root = thread_dir / DELIVERABLES
    items: List[dict] = []
    seen = set()
    for reg in registered or []:
        abs_path = str(reg.path)
        if abs_path in seen or resolve_deliverable(home, encode_file_id(os.path.relpath(abs_path, home))) is None:
            continue
        seen.add(abs_path)
        items.append(_entry(home, abs_path, reg.deliverable_class))
    after = snapshot_deliverables(thread_dir)
    for abs_path, stamp in sorted(after.items()):
        if abs_path in seen or before.get(abs_path) == stamp or not auto_deliverable(Path(abs_path), root):
            continue
        items.append(_entry(home, abs_path))
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
