"""Native attachment sends for cron's live-adapter lane, routed by file type."""

from __future__ import annotations

from pathlib import Path

# Audio routing is centralized in gateway.platforms.base.should_send_media_as_audio().
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter, chat_id: str, media_files: list, metadata: dict | None, loop, job: dict, platform=None,
    *, force_document: bool = False,
) -> list:
    """Send MEDIA files as native attachments (routed by extension, as in
    _process_message_background). ``force_document`` (``[[as_document]]``) sends images as
    documents, unrecompressed, as the gateway does. Returns per-file error strings so a dropped
    attachment surfaces in run status, not just the gateway log."""
    from cron.scheduler_delivery import _note_target_error
    from cron import scheduler_script as _script
    from gateway.platforms.base import (
        BasePlatformAdapter, should_send_media_as_audio, validate_media_delivery_path)
    from agent.async_utils import safe_schedule_threadsafe
    job_ref = {"id": job.get("id", "?")}
    errors: list = []
    requested = [(str(p), v) for p, v in (media_files or [])]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Report paths the safety filter dropped (missing file, denied prefix, strict-mode miss).
    kept = {p for p, _ in media_files}
    for raw_path, _v in requested:
        try:
            dropped = validate_media_delivery_path(raw_path) not in kept
        except Exception:
            dropped = True
        if dropped:
            errors.append(f"attachment dropped by media path policy: {raw_path}")

    route_platform = platform if platform is not None else getattr(adapter, "platform", None)
    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            if should_send_media_as_audio(route_platform, ext, is_voice=_is_voice):
                method, path_kw = "send_voice", "audio_path"
            elif ext in _VIDEO_EXTS:
                method, path_kw = "send_video", "video_path"
            elif ext in _IMAGE_EXTS and not force_document:
                method, path_kw = "send_image_file", "image_path"
            else:
                method, path_kw = "send_document", "file_path"
            # The voice sender decides bubble vs music-file from ``is_voice`` (Telegram
            # transcodes non-Opus only when it is set), matching the gateway dispatch
            # in BasePlatformAdapter._send_one.
            extra = {"is_voice": _is_voice} if method == "send_voice" else {}
            coro = getattr(adapter, method)(
                chat_id=chat_id, metadata=metadata, **{path_kw: media_path}, **extra)
            future = safe_schedule_threadsafe(coro, loop)
            if future is None:
                _note_target_error(
                    job_ref, f"cannot send media {media_path}: gateway loop unavailable", errors)
                return errors
            try:
                # Large attachments can exceed 30s: _get_media_send_timeout(), widened by file size.
                result = future.result(timeout=_script._media_send_timeout_for(media_path))
            except TimeoutError:
                future.cancel()
                raise
            raw = getattr(result, "raw_response", None)
            if result and getattr(result, "success", False) and isinstance(raw, dict) and raw.get("ambiguous"):
                # Sent but the ack was lost: possibly delivered, so never an error (a re-run
                # would duplicate it); _live_send_media records the target as unverified.
                job["_media_ack_lost"] = True
            elif result and not getattr(result, "success", True):
                _note_target_error(
                    job_ref,
                    f"media send failed for {media_path}: {getattr(result, 'error', 'unknown')}",
                    errors,
                )
        except Exception as e:
            # TimeoutError etc. have an empty str(); fall back to the class name.
            _note_target_error(
                job_ref, f"failed to send media {media_path}: {str(e) or type(e).__name__}", errors)
    return errors
