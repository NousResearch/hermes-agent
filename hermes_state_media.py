"""Lossless image replay, separate from the text column used by search/display/rewind.

References belong to the row, not its message identity: rewriting stripped content
clears them. Flush rows keep their text-only display/export; direct structured DB
writers retain their existing rich display via the sidecar's display flag.
The store is next to state.db (the owning profile), never in the swept inbound caches.
No sandbox mount is needed: references are read by SessionDB on the host and only
rehydrated data URLs reach providers, never host paths or agent-visible artifacts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes_state")
_IMAGE_TYPES = {"image", "image_url", "input_image"}
MEDIA_GC_GRACE_SECONDS = 3600


def sweep_transcript_media(db) -> int:
    """Reclaim only old, unreferenced regular files in this DB directory's store.

    A file may precede its committed row (even in another process). Capture the
    cutoff BEFORE the read snapshot and re-stat each candidate after reading refs;
    both new writes and dedupe hits renew that one-hour publication window.
    """
    root = _media_root(db.db_path)
    if root.is_symlink() or root.parent.is_symlink() or not root.is_dir():
        return 0
    cutoff = time.time() - MEDIA_GC_GRACE_SECONDS
    with db._read_ctx() as conn:
        conn.execute("BEGIN")
        try:
            referenced = set()
            for row in conn.execute("SELECT media_content FROM messages WHERE media_content IS NOT NULL"):
                for ref in json.loads(row[0])["images"]:
                    referenced.update(ref[key] for key in ("sha256", "encoded_sha256") if key in ref)
        finally:
            conn.rollback()
    removed = 0
    for path in root.iterdir():
        if path.name in referenced or not (re.fullmatch(r"[0-9a-f]{64}", path.name) or path.name.startswith(".write-")):
            continue
        try:
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:  # another sweeper won
            continue
    return removed


def try_sweep_transcript_media(db, *, enabled: bool = True) -> None:
    """Post-commit housekeeping must never turn a committed write into a retry."""
    if not enabled:
        return
    try:
        sweep_transcript_media(db)
    except Exception:
        logger.debug("Cannot sweep transcript media", exc_info=True)


def clear_inactive_media(conn, *, enabled: bool = True) -> None:
    """Retired rows are audit/display only; run AFTER any in-txn tail clones."""
    if enabled:
        conn.execute("UPDATE messages SET media_content = NULL WHERE active = 0 AND media_content IS NOT NULL")


def has_images(content: Any) -> bool:
    parts = content.get("content") if isinstance(content, dict) and content.get("_multimodal") else content
    return isinstance(parts, list) and any(isinstance(p, dict) and p.get("type") in _IMAGE_TYPES for p in parts)


def _media_root(db_path) -> Path:
    return Path(db_path).parent / "cache" / "transcript_media"


def _put(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = root / digest
    if not target.is_symlink() and target.is_file():
        try:
            os.utime(target, follow_symlinks=False)
            return digest
        except FileNotFoundError:  # a sweep won before we renewed the lease
            pass
    fd, name = tempfile.mkstemp(prefix=".write-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return digest


def _get(root: Path, digest: str) -> bytes:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid transcript media digest")
    path = root / digest
    # A dangling link or a non-regular file must not block replay or escape the store.
    if path.is_symlink() or not path.is_file():
        raise OSError("transcript media is not a regular file")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("transcript media checksum mismatch")
    return data


def _externalize(value: Any, root: Path, refs: list, path: tuple = ()) -> Any:
    if isinstance(value, dict):
        # Anthropic image.source uses bare base64, unlike OpenAI data URLs.
        if value.get("type") == "base64" and isinstance(value.get("data"), str):
            return {k: _externalize_payload(v, "", root, refs, (*path, k)) if k == "data"
                    else _externalize(v, root, refs, (*path, k)) for k, v in value.items()}
        return {k: _externalize(v, root, refs, (*path, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_externalize(v, root, refs, (*path, i)) for i, v in enumerate(value)]
    if not isinstance(value, str) or not value.startswith("data:image/") or ";base64," not in value:
        return value
    prefix, payload = value.split(",", 1)
    return _externalize_payload(payload, prefix + ",", root, refs, path)


def _externalize_payload(payload: str, prefix: str, root: Path, refs: list, path: tuple) -> None:
    ref = {"path": path, "prefix": prefix}
    try:
        data = base64.b64decode(payload)
    except ValueError:
        # Persistence is lossless even for malformed provider payloads; validation is
        # the send path's job. Never put the raw payload in SQLite on this fallback.
        data = None
    if data is not None:
        ref["sha256"] = _put(root, data)
    # Preserve even noncanonical base64 (whitespace/padding), not just decoded pixels.
    if data is None or base64.b64encode(data).decode("ascii") != payload:
        ref["encoded_sha256"] = _put(root, payload.encode("utf-8"))
    refs.append(ref)
    return None


def prepare_media_content(db_path, content: Any, media_content: Any = None, *, display: bool = True) -> tuple[Any, str | None]:
    """Return the text projection and a reference-only sidecar for any DB writer."""
    from agent.session_persistence import _durable_content

    original = content if has_images(content) else media_content
    if isinstance(original, str):
        # Raw get_messages -> replace/import: retain references only for their own projection.
        try:
            stored = json.loads(original)
            if stored["text"] != content:
                return content, None
            root = _media_root(db_path)
            for ref in stored["images"]:
                for key in ("sha256", "encoded_sha256"):
                    if key in ref:
                        _put(root, _get(root, ref[key]))
            return content, original
        except (OSError, ValueError, KeyError, TypeError):
            return content, None
    if not has_images(original):
        return content, None
    # Direct structured writers displayed only text parts (e.g. the SQL timeline);
    # only the agent flush historically inserted [screenshot] placeholders.
    from agent.message_content import flatten_message_text
    text = flatten_message_text(original) if display and isinstance(original, list) else _durable_content(original)
    try:
        refs: list = []
        skeleton = _externalize(original, _media_root(db_path), refs)
        return text, json.dumps({"version": 1, "text": text, "content": skeleton, "images": refs, "display": display})
    except (OSError, ValueError, TypeError):
        logger.debug("Cannot persist transcript media; keeping text projection", exc_info=True)
        return text, None


def restore_media_content(db_path, sidecar: Any, fallback: Any, *, model: bool = True) -> Any:
    """All-or-nothing replay: an unavailable image retains the old text-only behavior."""
    if not sidecar:
        return fallback
    try:
        stored = json.loads(sidecar)
        if stored["version"] != 1:
            raise ValueError("unknown transcript media version")
        if not model and not stored.get("display"):
            return fallback
        content = stored["content"]
        root = _media_root(db_path)
        for ref in stored["images"]:
            if "encoded_sha256" in ref:
                payload = _get(root, ref["encoded_sha256"]).decode("utf-8")
            else:
                payload = base64.b64encode(_get(root, ref["sha256"])).decode("ascii")
            node = content
            for key in ref["path"][:-1]:
                node = node[key]
            node[ref["path"][-1]] = ref["prefix"] + payload
        return content
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        logger.debug("Cannot restore transcript media; keeping text projection", exc_info=True)
        return fallback
