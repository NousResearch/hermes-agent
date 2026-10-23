"""Shared rejection feedback for normal, streamed and background attachments."""

import logging

from gateway.session import build_session_key

logger = logging.getLogger(__name__)
MEDIA_REJECTION_NOTICE = "[A MEDIA attachment was rejected by the delivery policy or was unavailable; it was not attached.]"


async def extract_media_with_feedback(runner, adapter, response, source, session_key="", *, notify=False, metadata=None, feedback_session_id=None):
    """Keep validation unchanged; append a path-free receipt without rewriting history."""
    from gateway.platforms.base import BasePlatformAdapter

    dropped = []
    session_key = session_key or build_session_key(source)
    media_files, response = adapter.extract_media(response)
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files, session_key=session_key, dropped=dropped)
    if not dropped:
        return media_files, response
    if runner is not None:
        try:
            store = runner.async_session_store
            if feedback_session_id is None:
                session = await store.get_or_create_session(source)
                feedback_session_id = session.session_id
            await store.append_to_transcript(feedback_session_id, {"role": "user", "content": MEDIA_REJECTION_NOTICE})
        except Exception:
            logger.warning("Could not persist MEDIA rejection feedback", exc_info=True)
    if notify:
        try:
            await adapter.emit_warning(source.chat_id, MEDIA_REJECTION_NOTICE, metadata=metadata)
        except Exception:
            logger.warning("Could not send MEDIA rejection feedback", exc_info=True)
    return media_files, (response or "") + "\n" + MEDIA_REJECTION_NOTICE
