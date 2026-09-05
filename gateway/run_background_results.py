"""Direct-outbound delivery of finished claude-agent-sdk background results for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO. The completion-queue
drain in ``gateway.run_notifications`` hands ``sdk_background_result`` events here. The old lane
wrapped the agent's own answer in a synthetic empty-id delegation and asked the model to relay it;
the model recognized its own text, refused, and the answer never left the box (2026-08-06).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Optional

from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio
from hermes_constants import get_hermes_home

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Media kind -> (adapter send method, path kwarg). Voice is a platform predicate, so it is decided
# first; the extension table settles the rest, with documents as the default.
_MEDIA_SENDERS = {
    "voice": ("send_voice", "audio_path"),
    "video": ("send_video", "video_path"),
    "image": ("send_image_file", "image_path"),
    "document": ("send_document", "file_path"),
}
_MEDIA_KIND_BY_EXT = {
    **dict.fromkeys((".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"), "video"),
    **dict.fromkeys((".png", ".jpg", ".jpeg", ".gif", ".webp"), "image"),
}


def _media_kind(platform, media_path: str, is_voice: bool) -> str:
    ext = os.path.splitext(media_path)[1].lower()
    if should_send_media_as_audio(platform, ext, is_voice):
        return "voice"
    return _MEDIA_KIND_BY_EXT.get(ext, "document")


class GatewayBackgroundResultsMixin:
    """Direct-outbound delivery, transcript projection and orphan-file durability of
    ``sdk_background_result`` events for GatewayRunner."""

    def _persist_orphaned_result_file(
        self, payloads: list, *, session_id: Optional[str], reason: str,
    ) -> Optional[str]:
        """Last-resort durability for a finished background/delegation payload that can reach neither
        its transcript nor its recipient (terminal parent drop, projection failure): a copy under
        ``<hermes_home>/orphaned-results/`` so a gateway restart can never erase the only copy —
        nothing finished may become unrecoverable again. Returns the path, or None; never raises
        (persistence trouble must not block the drop disposition or the delivery attempt)."""
        try:
            out_dir = get_hermes_home() / "orphaned-results"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            safe_session = re.sub(r"[^A-Za-z0-9._-]", "_", str(session_id or "unknown"))[:80]
            path = out_dir / f"{stamp}-{safe_session}-{uuid.uuid4().hex[:8]}.json"
            path.write_text(
                json.dumps({
                    "persisted_at": time.time(),
                    "session_id": session_id,
                    "reason": reason,
                    "payloads": [str(p) for p in payloads],
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.warning("orphaned result persisted to %s (%s, %d payload(s))", path, reason, len(payloads))
            return str(path)
        except Exception:
            logger.warning(
                "orphaned-result file persistence failed (%s) — payload remains at risk", reason, exc_info=True,
            )
            return None

    async def _project_result_payloads(
        self, payloads: list, *, session_id: str, unavailable_reason: str, failure_reason: str,
        display_metadata: Optional[dict] = None, lane: str = "sdk_background_result",
    ) -> None:
        """Save the text FIRST: project every payload into the hermes transcript as an assistant row
        marked ``display_kind="sdk_background_result"`` (the continuity digest must never re-present the
        agent's own delivered answer; the 2026-08-06 report was FTS-invisible and one session retire away
        from unrecoverable), falling back to an orphaned-results file when the db is unavailable or an
        append fails. Projection trouble never blocks delivery."""
        db = getattr(self, "_session_db", None)
        if db is None or not session_id:
            # Queue-only payloads die with a gateway restart — give them a durable copy first.
            logger.warning(
                "%s not projected into transcript (session_db=%s, parent_session_id=%r) "
                "— persisting to orphaned-results", lane, "ok" if db is not None else None, session_id,
            )
            self._persist_orphaned_result_file(payloads, session_id=session_id or None, reason=unavailable_reason)
            return
        unprojected = []
        for payload in payloads:
            try:
                await db.append_message(
                    session_id=session_id, role="assistant", content=payload,
                    display_kind="sdk_background_result", display_metadata=dict(display_metadata or {}),
                )
            except Exception:
                unprojected.append(payload)
                logger.warning(
                    "%s transcript projection failed for session %s — continuing with delivery",
                    lane, session_id, exc_info=True,
                )
        if unprojected:
            self._persist_orphaned_result_file(unprojected, session_id=session_id, reason=failure_reason)

    async def _deliver_sdk_background_result(self, evt: dict) -> Optional[bool]:
        """Send a finished claude-agent-sdk background result DIRECTLY on the platform outbound lane,
        each payload as its own agent message, in order — never re-injected into the agent session.

        ``True`` = every payload sent. ``False`` = retry (the caller requeues; already-sent payloads are
        trimmed off the event first so the retry delivers only the remainder). ``None`` = empty event.
        A missing route fails SAFE with ``False``: the payload is a finished result the user is waiting
        on and must never be dropped silently. Every payload is projected into the transcript before any
        routing or send, and each send is a delivery-ledger obligation; projection and ledger trouble
        never block the send."""
        payloads = [p for p in (evt.get("payloads") or []) if isinstance(p, str) and p.strip()]
        if not payloads:
            logger.warning("sdk_background_result event carried no payloads — dropping")
            return None
        if not evt.get("_projected"):
            # Once per event lifetime — a requeued retry (trimmed or unroutable) must not write duplicate rows.
            evt["_projected"] = True
            await self._project_result_payloads(
                payloads, session_id=str(evt.get("parent_session_id") or "").strip(),
                unavailable_reason="projection_unavailable", failure_reason="projection_failed",
                display_metadata={"completed_at": evt.get("completed_at")},
            )
        source = self._build_process_event_source(evt)
        adapter = platform_name = None
        if source is not None:
            platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
            adapter = self._adapter_by_platform_value(platform_name)
        from gateway.wake import adapter_supports_push
        if source is None or adapter is None or not adapter_supports_push(adapter):
            # Non-push adapters (api_server) deliver by running a wake turn — re-injection, the exact
            # mechanism this lane exists to avoid — so they have no deliverable route here either.
            if not evt.get("_route_warned"):
                evt["_route_warned"] = True
                logger.warning(
                    "sdk_background_result has no deliverable route (session_key=%r, source=%s, adapter=%s) — requeued",
                    evt.get("session_key"), "resolved" if source is not None else None,
                    type(adapter).__name__ if adapter is not None else None,
                )
            else:
                logger.debug("sdk_background_result still unroutable (session_key=%r)", evt.get("session_key"))
            return False
        metadata = self._thread_metadata_for_source(source)
        from gateway.delivery_ledger import (
            compute_obligation_id, ledger_enabled, mark_attempting, mark_delivered, mark_failed, record_obligation,
        )
        try:
            ledger_on = await asyncio.to_thread(ledger_enabled)
        except Exception:
            ledger_on = False
        session_key = str(evt.get("session_key") or "")
        obligation_ref = f"sdk_bg:{evt.get('completed_at') or ''}"
        for idx, payload in enumerate(payloads):
            obligation_id = None
            if ledger_on:
                try:
                    obligation_id = compute_obligation_id(session_key, obligation_ref, payload)
                    await asyncio.to_thread(
                        record_obligation, obligation_id=obligation_id, session_key=session_key,
                        platform=platform_name, chat_id=source.chat_id, thread_id=source.thread_id, content=payload,
                    )
                    await asyncio.to_thread(mark_attempting, obligation_id)
                except Exception:
                    logger.debug("sdk_background_result ledger record failed", exc_info=True)
                    obligation_id = None
            try:
                await self._send_background_result_payload(adapter, source, metadata, payload)
            except Exception as e:
                if obligation_id:
                    try:
                        await asyncio.to_thread(mark_failed, obligation_id, str(e))
                    except Exception:
                        logger.debug("sdk_background_result mark_failed failed", exc_info=True)
                evt["payloads"] = payloads[idx:]
                logger.warning(
                    "sdk_background_result send failed at payload %d/%d (%s) — remainder requeued",
                    idx + 1, len(payloads), e,
                )
                return False
            if obligation_id:
                try:
                    await asyncio.to_thread(mark_delivered, obligation_id)
                except Exception:
                    logger.debug("sdk_background_result mark_delivered failed", exc_info=True)
        logger.info(
            "sdk_background_result delivered: %d payload(s) to %s chat=%s", len(payloads), platform_name, source.chat_id,
        )
        return True

    @staticmethod
    async def _send_background_result_payload(adapter, source, metadata, payload: str) -> None:
        """One payload: text, then inline images, then media files dispatched by kind."""
        media_files, text_content = adapter.extract_media(payload)
        media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
        images, text_content = adapter.extract_images(text_content)
        if text_content:
            await adapter.send(chat_id=source.chat_id, content=text_content, metadata=metadata)
        for image_url, alt_text in images or []:
            await adapter.send_image(chat_id=source.chat_id, image_url=image_url, caption=alt_text, metadata=metadata)
        for media_path, is_voice in media_files or []:
            method, path_kwarg = _MEDIA_SENDERS[_media_kind(source.platform, media_path, is_voice)]
            await getattr(adapter, method)(chat_id=source.chat_id, metadata=metadata, **{path_kwarg: media_path})
