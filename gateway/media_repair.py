"""Repair and durable-stage ``MEDIA:`` paths in final assistant responses.

Two related jobs share this module so every delivery surface (messaging
gateway, TUI/desktop, cron) applies the same path hygiene:

1. **computer_use repair** — recover model-mangled screenshot paths when the
   model rewrites a Windows path into a POSIX-looking one inside an explicit
   ``MEDIA:`` directive.
2. **ephemeral staging** — copy ``MEDIA:`` targets that live under volatile
   locations (``/tmp``, system temp, etc.) into
   ``$HERMES_HOME/cache/*/chat-media/`` and rewrite the tags. Desktop remote
   mode fetches those paths via ``/api/fs/read-data-url``; if the agent left
   files only in ``/tmp`` and the OS reaped them, the UI shows
   "Couldn't fetch … from the gateway (missing, unreadable, or too large)."

Staging is fail-open and size-capped. Paths already under the Hermes cache
are left alone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Absolute-path prefix accepted for canonical capture paths: Windows drive
# letter, POSIX root, or UNC share. Kept as a pattern string so the summary
# regex below and the compiled prefix check stay in sync.
_ABS_PATH_PREFIX_PATTERN = r"(?:[A-Za-z]:[/\\]|/|\\\\)"
_ABS_PATH_PREFIX_RE = re.compile(r"^" + _ABS_PATH_PREFIX_PATTERN)

_COMPUTER_USE_CAPTURE_BASENAME_RE = re.compile(
    r"^computer_use_[0-9a-f]{32}\.(?:png|jpe?g)$",
    re.IGNORECASE,
)
_COMPUTER_USE_CAPTURE_SUMMARY_RE = re.compile(
    r"\(shareable screenshot saved to "
    r"(?P<path>" + _ABS_PATH_PREFIX_PATTERN + r"[^\r\n]*?"
    r"computer_use_[0-9a-f]{32}\.(?:png|jpe?g))\)",
    re.IGNORECASE,
)


def tool_name_by_call_id(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map assistant tool-call ids to tool names for the given messages."""
    mapping: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id") or call.get("call_id")
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            if call_id and name:
                mapping[str(call_id)] = name
    return mapping


def _computer_use_capture_basename(path: Any) -> str:
    """Return a canonical capture basename for either path separator style."""
    value = str(path or "").strip().strip("`\"'")
    basename = re.split(r"[/\\]", value)[-1]
    if _COMPUTER_USE_CAPTURE_BASENAME_RE.fullmatch(basename):
        return basename.lower()
    return ""


def _iter_computer_use_capture_paths(content: Any) -> Iterator[str]:
    """Yield persisted screenshot paths from computer_use result content.

    The tool can return JSON, a multimodal content list, or a text fallback.
    The latter two retain the canonical path in the human-readable summary
    even though the multimodal envelope's ``meta`` dictionary is not stored in
    the tool message.
    """
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            # JSON-looking content: parse first, never regex-scan the raw
            # text. JSON escaping doubles backslashes, so a summary-regex hit
            # on the raw string would yield ``C:\\\\Users\\\\...`` — a path
            # that exists nowhere. Fail closed on unparseable JSON (tool
            # output truncation) rather than repair to a corrupted path.
            try:
                payload = json.loads(stripped)
            except Exception:
                return
            if isinstance(payload, (dict, list)):
                yield from _iter_computer_use_capture_paths(payload)
            return
        for match in _COMPUTER_USE_CAPTURE_SUMMARY_RE.finditer(content):
            yield match.group("path").strip()
        return

    if isinstance(content, list):
        for part in content:
            yield from _iter_computer_use_capture_paths(part)
        return

    if not isinstance(content, dict):
        return

    screenshot_path = content.get("screenshot_path")
    if isinstance(screenshot_path, str):
        yield screenshot_path
    meta = content.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("screenshot_path"), str):
        yield meta["screenshot_path"]
    # Producer shapes (tools/computer_use/tool.py::_capture_response):
    # "content"/"text" — multimodal envelope parts; "text_summary"/"summary" —
    # the human-readable summary carrying the "(shareable screenshot saved
    # to ...)" line.
    for field in ("content", "text", "text_summary", "summary"):
        nested = content.get(field)
        if isinstance(nested, (str, dict, list)):
            yield from _iter_computer_use_capture_paths(nested)


def repair_explicit_computer_use_media_paths(
    response: str,
    messages: List[Dict[str, Any]],
    history_offset: int = 0,
) -> str:
    """Recover model-mangled paths for explicitly requested screenshots.

    Repair only an already-explicit ``MEDIA:`` directive whose unique
    generated basename case-insensitively matches a canonical screenshot
    path from this turn. This does not auto-attach ordinary computer-use
    captures, and normal media path validation still runs after the repair.

    Fail-open: the repair is cosmetic, so an unexpected error returns the
    response unchanged rather than aborting delivery.
    """
    try:
        return _repair_explicit_computer_use_media_paths_inner(
            response, messages, history_offset
        )
    except Exception:
        logger.debug("computer_use media path repair failed", exc_info=True)
        return response


def _repair_explicit_computer_use_media_paths_inner(
    response: str,
    messages: List[Dict[str, Any]],
    history_offset: int = 0,
) -> str:
    if "MEDIA:" not in response:
        return response

    if history_offset and len(messages) >= history_offset:
        turn_messages = messages[history_offset:]
    elif history_offset:
        # Compression can invalidate the original slice boundary. Recover the
        # current turn from its last user message; fail closed if none
        # remains. (Deliberately narrower than the scan-everything fallback
        # in gateway/run.py::_collect_auto_append_media_tags — that helper
        # decides whether to ATTACH, this one only rewrites paths the model
        # already explicitly emitted.)
        last_user = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            None,
        )
        turn_messages = messages[last_user:] if last_user is not None else []
    else:
        turn_messages = messages

    call_id_names = tool_name_by_call_id(turn_messages)

    canonical_by_basename: Dict[str, str] = {}
    for msg in turn_messages:
        if msg.get("role") not in {"tool", "function"}:
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        tool_name = str(
            msg.get("name")
            or msg.get("tool_name")
            or call_id_names.get(call_id)
            or ""
        )
        if tool_name != "computer_use":
            continue
        for path in _iter_computer_use_capture_paths(msg.get("content")):
            basename = _computer_use_capture_basename(path)
            if basename and _ABS_PATH_PREFIX_RE.match(path):
                canonical_by_basename[basename] = path

    if not canonical_by_basename:
        return response

    # Lazy on purpose: keeps `import gateway.media_repair` cheap for
    # standalone cron processes that may never hit a MEDIA: response.
    # No import cycle either way (base.py imports neither this module
    # nor gateway.run at module level).
    from gateway.platforms.base import BasePlatformAdapter

    media_files, _ = BasePlatformAdapter.extract_media(response)
    repaired = response
    for emitted_path, _is_voice in media_files:
        canonical = canonical_by_basename.get(
            _computer_use_capture_basename(emitted_path)
        )
        if canonical and emitted_path != canonical:
            repaired = repaired.replace(emitted_path, canonical)
    return repaired


# ---------------------------------------------------------------------------
# Ephemeral MEDIA staging for durable chat display
# ---------------------------------------------------------------------------

# Cap staged copies so a runaway MEDIA dump cannot fill the Hermes home.
# Matches the desktop /api/fs/read-data-url ceiling for inline image display.
_CHAT_MEDIA_STAGE_MAX_BYTES = 16 * 1024 * 1024

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
_AUDIO_EXTS = {".mp3", ".m2a", ".wav", ".ogg", ".opus", ".m4a", ".flac"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

_EPHEMERAL_PREFIXES = (
    "/tmp/",
    "/var/tmp/",
    "/private/tmp/",  # macOS
)


def _chat_media_bucket(ext: str) -> str:
    e = (ext or "").lower()
    if e in _IMAGE_EXTS:
        return "images"
    if e in _AUDIO_EXTS:
        return "audio"
    if e in _VIDEO_EXTS:
        return "videos"
    return "documents"


def _is_ephemeral_media_path(path: Path) -> bool:
    """True when a path is expected to vanish (tmp dirs, OS temp roots)."""
    try:
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    s = str(resolved)
    posix = s.replace("\\", "/")
    lower = posix.lower()
    for prefix in _EPHEMERAL_PREFIXES:
        base = prefix.rstrip("/").lower()
        if lower == base or lower.startswith(base + "/"):
            return True
    try:
        tmp = Path(tempfile.gettempdir()).expanduser().resolve(strict=False)
        tmp_s = str(tmp).replace("\\", "/").lower()
        if lower == tmp_s or lower.startswith(tmp_s.rstrip("/") + "/"):
            return True
    except (OSError, RuntimeError, ValueError):
        pass
    # macOS per-user temp: /var/folders/.../T/
    if "/var/folders/" in lower and "/t/" in lower:
        return True
    return False


def _already_in_hermes_cache(path: Path) -> bool:
    """Skip restaging files already under $HERMES_HOME cache trees."""
    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home().expanduser().resolve(strict=False)
        resolved = path.expanduser().resolve(strict=False)
        try:
            resolved.relative_to(home / "cache")
            return True
        except ValueError:
            pass
        for legacy in ("image_cache", "audio_cache", "video_cache", "document_cache", "media"):
            try:
                resolved.relative_to(home / legacy)
                return True
            except ValueError:
                continue
    except Exception:
        return False
    return False


def _stage_one_media_file(src: Path) -> Optional[Path]:
    """Copy *src* into $HERMES_HOME/cache/<bucket>/chat-media/ and return dest."""
    try:
        if not src.is_file():
            return None
        st = src.stat()
        if st.st_size <= 0 or st.st_size > _CHAT_MEDIA_STAGE_MAX_BYTES:
            return None
        if _already_in_hermes_cache(src):
            return src
        if not _is_ephemeral_media_path(src):
            return None

        from hermes_constants import get_hermes_home

        ext = src.suffix.lower() or ".bin"
        bucket = _chat_media_bucket(ext)
        dest_dir = get_hermes_home() / "cache" / bucket / "chat-media"
        dest_dir.mkdir(parents=True, exist_ok=True)

        digest = hashlib.sha1(
            str(src.resolve(strict=False)).encode("utf-8", "replace")
        ).hexdigest()[:10]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem)[:80] or "media"
        dest = dest_dir / f"{stem}-{digest}{ext}"
        if dest.exists() and dest.stat().st_size == st.st_size:
            return dest
        tmp = dest.with_name(f".{dest.name}.tmp-{os.getpid()}")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dest)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        return dest
    except Exception:
        logger.debug("chat media stage failed for %s", src, exc_info=True)
        return None


def stage_ephemeral_chat_media_paths(response: str) -> str:
    """Copy ephemeral ``MEDIA:`` targets into durable Hermes cache and rewrite tags.

    Fail-open: any error leaves the response unchanged. Non-ephemeral paths
    (project trees, already-cached artifacts) are left alone.
    """
    if not response or "MEDIA:" not in response:
        return response
    try:
        from gateway.platforms.base import (
            MEDIA_TAG_CLEANUP_RE,
            BasePlatformAdapter,
            _normalize_media_tag_path,
        )
    except Exception:
        logger.debug("chat media stage import failed", exc_info=True)
        return response

    try:
        scan = BasePlatformAdapter._mask_protected_spans(response)
        scan = BasePlatformAdapter._mask_json_string_media(scan)
    except Exception:
        scan = response

    replacements: list[Tuple[str, str]] = []
    seen: set[str] = set()
    try:
        for match in MEDIA_TAG_CLEANUP_RE.finditer(scan):
            raw = match.group("path")
            path_str = _normalize_media_tag_path(raw)
            if not path_str or path_str in seen:
                continue
            seen.add(path_str)
            try:
                src = Path(path_str).expanduser()
                if not src.is_absolute():
                    continue
            except (OSError, RuntimeError, ValueError):
                continue
            dest = _stage_one_media_file(src)
            if dest is None:
                continue
            dest_s = str(dest)
            if dest_s != path_str:
                replacements.append((path_str, dest_s))
    except Exception:
        logger.debug("chat media stage scan failed", exc_info=True)
        return response

    if not replacements:
        return response

    repaired = response
    for old, new in sorted(replacements, key=lambda pair: len(pair[0]), reverse=True):
        repaired = repaired.replace(old, new)
    return repaired


def finalize_chat_media_paths(
    response: str,
    messages: Optional[List[Dict[str, Any]]] = None,
    history_offset: int = 0,
) -> str:
    """Apply computer_use path repair then durable ephemeral MEDIA staging.

    Shared entry point for gateway turn / background / cron / TUI so no surface
    forgets one of the two steps.
    """
    if not isinstance(response, str) or not response:
        return response
    if messages is not None and "MEDIA:" in response:
        response = repair_explicit_computer_use_media_paths(
            response, messages, history_offset=history_offset
        )
    return stage_ephemeral_chat_media_paths(response)
