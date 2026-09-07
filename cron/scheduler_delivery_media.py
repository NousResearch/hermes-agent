"""Native attachment sends for cron's live-adapter lane, routed by file type."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Audio routing is centralized in gateway.platforms.base.should_send_media_as_audio().
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter, chat_id: str, media_files: list, metadata: dict | None, loop, job: dict, platform=None,
    *, force_document: bool = False,
) -> list[tuple[str, bool, str]]:
    """Send MEDIA files as native attachments (routed by extension, as in
    _process_message_background). ``force_document`` (``[[as_document]]``) sends images as
    documents, unrecompressed, as the gateway does.

    Returns ``(path, is_voice, error)`` for fallback-eligible failures only. A timeout counts
    only when the send never started; an in-flight upload is left running.
    """
    from cron.scheduler_delivery import _note_target_error
    from cron import scheduler_script as _script
    from gateway.platforms.base import (
        BasePlatformAdapter, should_send_media_as_audio, validate_media_delivery_path)
    from agent.async_utils import safe_schedule_threadsafe
    job_ref = {"id": job.get("id", "?")}
    failures: list[tuple[str, bool, str]] = []
    requested = [(str(p), v) for p, v in (media_files or [])]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Report paths the safety filter dropped (missing file, denied prefix, strict-mode miss).
    kept = {p for p, _ in media_files}
    for raw_path, is_voice in requested:
        try:
            dropped = validate_media_delivery_path(raw_path) not in kept
        except Exception:
            dropped = True
        if dropped:
            failures.append(
                (raw_path, is_voice, f"attachment dropped by media path policy: {raw_path}"))

    route_platform = platform if platform is not None else getattr(adapter, "platform", None)
    for media_path, is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            if should_send_media_as_audio(route_platform, ext, is_voice=is_voice):
                method, path_kw = "send_voice", "audio_path"
            elif ext in _VIDEO_EXTS:
                method, path_kw = "send_video", "video_path"
            elif ext in _IMAGE_EXTS and not force_document:
                method, path_kw = "send_image_file", "image_path"
            else:
                method, path_kw = "send_document", "file_path"
            extra = {"is_voice": is_voice} if method == "send_voice" else {}
            lock = threading.Lock()
            state = {"started": False, "abandoned": False}

            async def _send_once(
                method=method, path_kw=path_kw, extra=extra, media_path=media_path,
                lock=lock, state=state,
            ):
                with lock:
                    if state["abandoned"]:
                        return None
                    state["started"] = True
                return await getattr(adapter, method)(
                    chat_id=chat_id, metadata=metadata, **{path_kw: media_path}, **extra)

            future = safe_schedule_threadsafe(_send_once(), loop)
            if future is None:
                msg = f"cannot send media {media_path}: gateway loop unavailable"
                _note_target_error(job_ref, msg, [])
                failures.append((media_path, is_voice, msg))
                return failures
            try:
                result = future.result(timeout=_script._media_send_timeout_for(media_path))
            except TimeoutError:
                with lock:
                    state["abandoned"] = not state["started"]
                    in_flight = state["started"]
                if in_flight:
                    logger.warning(
                        "Job '%s': media send for %s timed out after dispatch; "
                        "skipping fallback to avoid duplicate delivery",
                        job_ref["id"], media_path)
                    continue
                future.cancel()
                raise
            raw = getattr(result, "raw_response", None)
            if result and getattr(result, "success", False) and isinstance(raw, dict) and raw.get("ambiguous"):
                job["_media_ack_lost"] = True
            elif result and not getattr(result, "success", True):
                msg = (
                    f"media send failed for {media_path}: "
                    f"{getattr(result, 'error', 'unknown')}")
                _note_target_error(job_ref, msg, [])
                failures.append((media_path, is_voice, msg))
        except Exception as e:
            msg = f"failed to send media {media_path}: {str(e) or type(e).__name__}"
            _note_target_error(job_ref, msg, [])
            failures.append((media_path, is_voice, msg))
    return failures