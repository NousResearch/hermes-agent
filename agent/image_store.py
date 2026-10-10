"""Keep user-attached images across turns that are rebuilt from the session DB.

The session DB stores user rows as text: ``session_persistence._durable_content`` turns each image
part into a ``[screenshot]`` line. Any turn whose history is reloaded from the DB (the API server
builds a fresh agent per request) therefore sent the model a placeholder instead of the image the
user asked about one message earlier. This module closes that gap without putting base64 in the DB:

* **write** — when a user row is flushed, its ``data:`` images are written once to
  ``<HERMES_HOME>/image_store/<sha256>.<ext>`` and the row's ``message_uid`` gets a reference file
  ``image_store/refs/<message_uid>.json`` (session id, row timestamp, ordered refs). The DB row and
  every transcript API stay text-only.
* **read** — when a request is built for a model that takes images natively, the
  ``vision.replay_recent_images`` most recent images of the conversation (default 3) go back on the
  wire copy as ``image_url`` parts at their ``[screenshot]`` positions; every other marker reads
  ``OLDER_IMAGE_TEXT``. The result depends only on the history, so a row's bytes change only when a
  newer image pushes it out of the window, and the prompt-cache prefix otherwise stays stable.
* **retention** — each session keeps references to its newest ``replay_recent_images`` images only;
  deleting or pruning a session drops its references; the gateway's state.db housekeeping drops
  references whose message is no longer active (compaction, a crash between write and insert).
  A file is deleted once no reference points to it. ``replay_recent_images: 0`` stores nothing and
  the next sweep removes what was stored.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_REPLAY_RECENT_IMAGES = 3
SCREENSHOT_MARKER = "[screenshot]"  # what session_persistence._durable_content writes per image part
OLDER_IMAGE_TEXT = "[image sent earlier, no longer attached]"
_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}  # the parts _durable_content turns into markers
_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)
_EXT = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp",
        "image/heic": "heic", "image/heif": "heif", "image/avif": "avif", "image/bmp": "bmp"}
_UID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_FILE_RE = re.compile(r"^[0-9a-f]{64}\.[a-z0-9]{1,5}$")
_TMP_MAX_AGE_S = 3600
_SIDECAR_VERSION = 1

Pair = tuple[dict[str, Any], dict[str, Any]]


def resolve_replay_recent_images() -> int:
    """``vision.replay_recent_images`` (>= 0; 0 disables the store); the default on a bad value."""
    try:
        from hermes_cli.config import cfg_get, load_config
        raw = cfg_get(load_config(), "vision", "replay_recent_images", default=DEFAULT_REPLAY_RECENT_IMAGES)
    except Exception:
        logger.warning("image_store: could not read vision.replay_recent_images", exc_info=True)
        return DEFAULT_REPLAY_RECENT_IMAGES
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return DEFAULT_REPLAY_RECENT_IMAGES
    try:
        return max(int(raw), 0)
    except ValueError:
        return DEFAULT_REPLAY_RECENT_IMAGES


def store_root(home: Optional[Path] = None) -> Path:
    if home is None:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    return Path(home) / "image_store"


def _db_store_root(db: Any) -> Path:
    """The store next to ``db``'s state.db: its directory is the owning profile's home."""
    return store_root(Path(db.db_path).parent)


def content_has_images(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES for p in content)


def _part_url(part: dict[str, Any]) -> tuple[str, Optional[str]]:
    """``(url, detail)`` of an OpenAI/Responses-style image part ('' when absent)."""
    ref = part.get("image_url")
    if isinstance(ref, dict):
        return str(ref.get("url") or ""), ref.get("detail") if isinstance(ref.get("detail"), str) else None
    if isinstance(ref, str):
        return ref, part.get("detail") if isinstance(part.get("detail"), str) else None
    return "", None


def _write_private(path: Path, data: bytes) -> None:
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(path.parent)  # a late flush must never recreate a deleted named profile
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _store_part(root: Path, part: dict[str, Any]) -> dict[str, Any]:
    """The ref for one image part; ``{"pruned": True}`` holds the marker position of one not replayed.

    Only bytes we hold are replayed: a remote URL is re-fetched by the provider on every request, so
    an expired signed URL or a deleted file would turn every later turn into a 400."""
    url, detail = _part_url(part)
    m = _DATA_URL_RE.match(url)
    if not m:
        return {"pruned": True}
    try:
        raw = base64.b64decode(m.group(2), validate=False)
    except (binascii.Error, ValueError):
        return {"pruned": True}
    if not raw:
        return {"pruned": True}
    mime = m.group(1).lower()
    name = f"{hashlib.sha256(raw).hexdigest()}.{_EXT.get(mime, 'bin')}"
    if not (root / name).exists():
        _write_private(root / name, raw)
    ref: dict[str, Any] = {"file": name, "mime": mime}
    if detail:
        ref["detail"] = detail
    return ref


# --- reference files ---------------------------------------------------------------------------

def _sidecar_path(root: Path, uid: str) -> Path:
    return root / "refs" / f"{uid}.json"


def _read_sidecar(path: Path) -> Optional[dict[str, Any]]:
    """``{"session_id", "ts", "refs"}``, or ``None`` when missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("refs"), list):
        return None
    data["refs"] = [r for r in data["refs"] if isinstance(r, dict)]
    return data


def _write_sidecar(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps({"v": _SIDECAR_VERSION, "session_id": data.get("session_id"), "ts": data.get("ts", 0.0),
                          "refs": data["refs"]}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if not path.exists() or path.read_bytes() != payload:
        _write_private(path, payload)


def _iter_sidecars(root: Path) -> Iterable[tuple[str, Path, dict[str, Any]]]:
    refs_dir = root / "refs"
    if not refs_dir.is_dir():
        return
    for path in sorted(refs_dir.glob("*.json")):
        data = _read_sidecar(path) if _UID_RE.match(path.stem) else None
        if data is not None:
            yield path.stem, path, data


def _live_ref_count(data: dict[str, Any]) -> int:
    return sum(1 for r in data["refs"] if not r.get("pruned"))


def _enforce_window(entries: list[tuple[Path, dict[str, Any]]], keep: int) -> int:
    """``entries`` = one session's reference files, NEWEST FIRST. Keep the ``keep`` newest images;
    older refs become ``{"pruned": true}`` (positions must survive) and a file left with no live ref
    is deleted. Returns the number of refs pruned."""
    budget, pruned = keep, 0
    for path, data in entries:
        changed = False
        for ref in reversed(data["refs"]):  # the latest images of a message first
            if ref.get("pruned"):
                continue
            if budget > 0:
                budget -= 1
                continue
            ref.clear()
            ref["pruned"] = True
            pruned += 1
            changed = True
        if not _live_ref_count(data):
            path.unlink(missing_ok=True)
        elif changed:
            _write_sidecar(path, data)
    return pruned


def remove_orphan_files(root: Path, now: Optional[float] = None) -> int:
    """Delete stored images no live ref points to (plus stale temp files); returns the count."""
    if not root.is_dir():
        return 0
    keep = {str(r["file"]) for _, _, data in _iter_sidecars(root) for r in data["refs"]
            if not r.get("pruned") and r.get("file")}
    now = time.time() if now is None else now
    removed = 0
    for directory in (root, root / "refs"):
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            name = path.name
            stale_tmp = name.startswith(".") and name.endswith(".tmp") and now - path.stat().st_mtime > _TMP_MAX_AGE_S
            orphan = directory == root and _FILE_RE.match(name) is not None and name not in keep
            if path.is_file() and (orphan or stale_tmp):
                path.unlink(missing_ok=True)
                removed += 1
    return removed


# --- write path --------------------------------------------------------------------------------

def persist_message_images(msg: dict[str, Any], session_id: Optional[str] = None, home: Optional[Path] = None,
                           ts: Optional[float] = None, keep: Optional[int] = None) -> int:
    """Store the images of a live user message and its reference file, then apply the session's
    window; returns the number of images stored. ``ts`` (the row's timestamp) orders the window, so
    a late re-flush of an old message never counts as the newest image.

    Stamps the message's ``message_uid`` (minted once per logical message, as the DB insert would)
    so the row and the reference file share the key. Never raises: a failure only means the image
    is not replayed on later turns."""
    content = msg.get("content")
    if msg.get("role") != "user" or not content_has_images(content):
        return 0
    keep = resolve_replay_recent_images() if keep is None else keep
    if keep <= 0:
        return 0
    try:
        from agent.message_metadata import stamp_message_uid

        uid = stamp_message_uid(msg)
        if not _UID_RE.match(uid):
            return 0
        root = store_root(home)
        sidecar = _sidecar_path(root, uid)
        if sidecar.exists():  # re-flush of a message already stored
            return _live_ref_count(_read_sidecar(sidecar) or {"refs": []})
        # One ref per image part, in order: refs map to the row's markers by position.
        refs = [_store_part(root, p) for p in content if isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES]
        if not _live_ref_count({"refs": refs}):
            return 0
        stamp = float(ts) if isinstance(ts, (int, float)) else time.time()
        _write_sidecar(sidecar, {"session_id": session_id, "ts": stamp, "refs": refs})
        if session_id:
            same = [(path, data) for _, path, data in _iter_sidecars(root) if data.get("session_id") == session_id]
            same.sort(key=lambda e: e[1].get("ts") or 0.0, reverse=True)
            if _enforce_window(same, keep):
                remove_orphan_files(root)
        return _live_ref_count({"refs": refs})
    except Exception:
        logger.warning("image_store: could not store the images of a user message", exc_info=True)
        return 0


# --- cleanup against the DB --------------------------------------------------------------------

def _active_rows_by_uid(db: Any, uids: list[str]) -> dict[str, tuple[str, int]]:
    """``uid -> (session_id, newest active row id)`` for the uids still ACTIVE in the DB (rows archived
    by compaction never reach the model again)."""
    found: dict[str, tuple[str, int]] = {}
    with db._read_ctx() as conn:
        for i in range(0, len(uids), 500):
            chunk = uids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for uid, sid, rid in conn.execute(
                f"SELECT message_uid, session_id, MAX(id) FROM messages WHERE active = 1 "
                f"AND message_uid IN ({marks}) GROUP BY message_uid", chunk,
            ).fetchall():
                found[uid] = (sid, rid)
    return found


def gc_store(db: Any, keep: Optional[int] = None) -> dict[str, int]:
    """Drop refs whose message is no longer active in ``db``, re-apply the per-session window in DB
    order, then delete unreferenced files."""
    root = _db_store_root(db)
    keep = resolve_replay_recent_images() if keep is None else keep
    stats = {"refs_removed": 0, "refs_pruned": 0, "files_removed": 0}
    sidecars = list(_iter_sidecars(root))
    alive = _active_rows_by_uid(db, [uid for uid, _, _ in sidecars]) if sidecars else {}
    by_session: dict[str, list[tuple[int, Path, dict[str, Any]]]] = {}
    for uid, path, data in sidecars:
        if uid not in alive:
            path.unlink(missing_ok=True)
            stats["refs_removed"] += 1
            continue
        sid, rid = alive[uid]
        by_session.setdefault(sid, []).append((rid, path, data))
    for entries in by_session.values():
        entries.sort(key=lambda e: e[0], reverse=True)
        stats["refs_pruned"] += _enforce_window([(p, d) for _, p, d in entries], keep)
    stats["files_removed"] = remove_orphan_files(root)
    return stats


def purge_deleted_sessions(db: Any, session_ids: list[str]) -> int:
    """After sessions are deleted from ``db``: drop the refs saved for them whose message is gone
    (a compression continuation may still own the same message uid), then unreferenced files.
    Never raises: the next sweep retries."""
    try:
        root = _db_store_root(db)
        wanted = set(session_ids)
        candidates = [(uid, path) for uid, path, data in _iter_sidecars(root) if data.get("session_id") in wanted]
        if not candidates:
            return 0
        alive = _active_rows_by_uid(db, [uid for uid, _ in candidates])
        removed = 0
        for uid, path in candidates:
            if uid not in alive:
                path.unlink(missing_ok=True)
                removed += 1
        files = remove_orphan_files(root)
        logger.info("image_store: session delete dropped %d ref(s), %d file(s)", removed, files)
        return removed
    except Exception:
        logger.warning("image_store: cleanup after a session delete failed", exc_info=True)
        return 0


def sweep_store() -> None:
    """Housekeeping entry point (the caller binds the owning profile's scope). Never raises."""
    from hermes_state_registry import acquire, release_or_close

    try:
        db = acquire()
        try:
            stats = gc_store(db)
        finally:
            release_or_close(db)
    except Exception:
        logger.warning("image_store: sweep failed", exc_info=True)
        return
    if any(stats.values()):
        logger.info("image_store: sweep %s", stats)


# --- read path ---------------------------------------------------------------------------------

def load_refs(uid: Any, home: Optional[Path] = None) -> list[dict[str, Any]]:
    if not isinstance(uid, str) or not _UID_RE.match(uid):
        return []
    data = _read_sidecar(_sidecar_path(store_root(home), uid))
    return data["refs"] if data else []


def _ref_to_part(ref: dict[str, Any], root: Path) -> Optional[dict[str, Any]]:
    if ref.get("pruned"):
        return None
    if ref.get("file"):
        name = str(ref["file"])
        if not _FILE_RE.match(name):
            return None
        try:
            raw = (root / name).read_bytes()
        except OSError:
            return None
        url = f"data:{ref.get('mime') or 'image/jpeg'};base64,{base64.b64encode(raw).decode('ascii')}"
    else:
        return None
    image_url: dict[str, Any] = {"url": url}
    if isinstance(ref.get("detail"), str):
        image_url["detail"] = ref["detail"]
    return {"type": "image_url", "image_url": image_url}


def _split_on_markers(text: str) -> list[Optional[str]]:
    """Text -> segments: str chunks and ``None`` for each ``[screenshot]`` line (the inverse of
    ``_durable_content``'s ``"\\n".join``)."""
    segments: list[Optional[str]] = []
    buf: list[str] = []
    for line in text.split("\n"):
        if line == SCREENSHOT_MARKER:
            if buf:
                segments.append("\n".join(buf))
                buf = []
            segments.append(None)
        else:
            buf.append(line)
    if buf:
        segments.append("\n".join(buf))
    return segments


def _rebuild(text: str, refs: list[dict[str, Any]], keep: list[bool], root: Path) -> Any:
    """Content for one stored row: kept refs become image parts, every other marker ``OLDER_IMAGE_TEXT``."""
    segments = _split_on_markers(text)
    n_markers = sum(1 for s in segments if s is None)
    segments += [None] * max(0, len(refs) - n_markers)  # refs whose marker the text lost go last
    parts: list[dict[str, Any]] = []
    i = 0
    for seg in segments:
        if seg is not None:
            if seg.strip():
                parts.append({"type": "text", "text": seg})
            continue
        part = _ref_to_part(refs[i], root) if i < len(refs) and keep[i] else None
        i += 1
        parts.append(part if part is not None else {"type": "text", "text": OLDER_IMAGE_TEXT})
    if not any(p["type"] == "image_url" for p in parts):
        return "\n".join(p["text"] for p in parts)  # nothing re-attached: keep a plain string
    return parts


def rehydrate_wire_images(pairs: list[Pair], keep_recent: int, home: Optional[Path] = None) -> int:
    """Re-attach stored images on the wire copies. ``pairs`` = ``(source_msg, api_msg)`` in
    conversation order; only ``api_msg`` is modified. Live images (list content still holding image
    parts) count toward the window but are left alone. Returns the number of images re-attached."""
    from agent.message_metadata import message_uid_or_none

    root = store_root(home)
    plan: list[tuple[dict[str, Any], list[dict[str, Any]], list[bool]]] = []
    budget = max(0, int(keep_recent))
    for src, api_msg in reversed(pairs):
        if src.get("role") != "user":
            continue
        content = api_msg.get("content")
        if content_has_images(content):
            budget -= sum(1 for p in content if isinstance(p, dict) and p.get("type") in _IMAGE_PART_TYPES)
            continue
        if not isinstance(content, str) or SCREENSHOT_MARKER not in content:
            continue
        refs = load_refs(message_uid_or_none(src), home)
        keep = [False] * len(refs)
        for j in range(len(refs) - 1, -1, -1):  # the latest images of a row first
            if budget > 0 and not refs[j].get("pruned"):
                keep[j] = True
                budget -= 1
        plan.append((api_msg, refs, keep))
    attached = 0
    for api_msg, refs, keep in plan:
        rebuilt = _rebuild(api_msg["content"], refs, keep, root)
        api_msg["content"] = rebuilt
        if isinstance(rebuilt, list):
            attached += sum(1 for p in rebuilt if p.get("type") == "image_url")
    return attached


def rehydrate_for_agent(agent: Any, pairs: list[Pair]) -> None:
    """``build_api_messages`` hook: replay stored images only when this turn sends images natively
    (``agent.image_input_mode`` / model vision support) and the window is not disabled."""
    try:
        keep = resolve_replay_recent_images()
        if keep <= 0:
            return
        from agent.image_routing import decide_image_input_mode
        from hermes_cli.config import load_config
        provider = (getattr(agent, "provider", "") or "").strip()
        model = (getattr(agent, "model", "") or "").strip()
        if decide_image_input_mode(provider, model, load_config()) != "native":
            return
        attached = rehydrate_wire_images(pairs, keep)
        if attached:
            logger.info("image_store: re-attached %d stored image(s) to the request", attached)
    except Exception:
        logger.warning("image_store: re-attaching stored images failed", exc_info=True)
