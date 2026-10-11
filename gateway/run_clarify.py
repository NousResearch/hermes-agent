"""Clarification reply lifecycle and copy shared by gateway turn paths."""
from __future__ import annotations

import logging
from typing import Optional

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.platforms.typing_scope import set_typing_paused
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


def clarify_expired_notice() -> str:
    """Native prompt whose wait ended without a click."""
    return t("gateway.clarify.expired")


class GatewayClarifyReplyMixin:
    async def _hm_clarify_reply(
        self, event: MessageEvent, source: SessionSource, _quick_key: str
    ) -> Optional[str]:
        """Intercept a reply to a pending clarify prompt; None when the message falls through.
        Free text answers open-ended/"Other" prompts; "2" answers a multi-choice one. Resolved/retained
        replies return "" so adapters don't double-post — the agent produces the next user-facing message."""
        try:
            from tools import clarify_gateway as _clarify_mod
            _pending_clarify = _clarify_mod.get_pending_for_session(_quick_key, include_choice_prompts=True)
        except Exception:
            logger.debug("Unable to inspect pending clarification", exc_info=True)
            return None
        if _pending_clarify is None:
            return None
        _clarify_has_audio = bool(self._pending_event_audio_paths(event))
        _raw_clarify_reply = await self._prepare_clarify_reply_text(event)

        def _retain(why: str) -> str:
            logger.info(
                "Gateway retained pending clarify after %s (session=%s, id=%s)",
                why, _quick_key, _pending_clarify.clarify_id,
            )
            return ""

        if _clarify_has_audio and not _raw_clarify_reply:
            return _retain("voice transcription produced no usable text")
        # Slash commands: the user wanted a command, not to answer the clarify. Leave it pending so
        # they can retry; on timeout the agent unblocks with an empty response.
        if not _raw_clarify_reply or _raw_clarify_reply.startswith("/"):
            return None
        _text_outcome = _clarify_mod.attempt_text_response_for_session(_quick_key, _raw_clarify_reply)
        if _text_outcome == _clarify_mod.TEXT_RESOLVED:
            logger.info(
                "Gateway intercepted clarify text response (session=%s, id=%s)",
                _quick_key, _pending_clarify.clarify_id,
            )
            # The clarify callback pauses the platform typing/status indicator while waiting so
            # Slack users can type; the active agent resumes now, so re-enable its indicator.
            _clarify_adapter = self._delivery_adapter_for(source)
            if _clarify_adapter:
                try:
                    set_typing_paused(_clarify_adapter, source.chat_id, paused=False, metadata={"thread_id": source.thread_id})
                except Exception:
                    logger.debug("Failed to resume typing after clarify response", exc_info=True)
                # A typed answer to a native card (numeric pick, or text after "Other") never
                # reaches the click handler, so the card would keep its buttons forever.
                if callable(getattr(type(_clarify_adapter), "retire_clarify_card", None)):
                    try:
                        await _clarify_adapter.retire_clarify_card(
                            _pending_clarify.clarify_id,
                            t("gateway.clarify.answered",
                              response=_raw_clarify_reply if _pending_clarify.response == _clarify_mod.SKIPPED
                              else _pending_clarify.response or _raw_clarify_reply))
                    except Exception:
                        logger.debug("Failed to retire clarify card after typed answer", exc_info=True)
            return ""
        if _text_outcome == _clarify_mod.TEXT_REJECTED_SELECTION:
            # Selection-shaped but invalid (out-of-range number, bad comma-list): keep the clarify
            # armed for retry — don't cancel, don't treat as an unrelated follow-up.
            return _retain("invalid selection attempt")
        if _text_outcome == _clarify_mod.TEXT_REJECTED_PROSE:
            # Native-choice prompts reject unmatched prose so it continues through normal busy
            # routing. Release this clarify first: redirect() degrades to steer() while tools
            # execute, and that steer cannot drain until the clarify tool returns.
            if _clarify_mod.resolve_gateway_clarify(_pending_clarify.clarify_id, _clarify_mod.CANCELLED):
                # Adapters with a persistent native card (Slack Block Kit) retire it now, before the
                # prose is routed, so its buttons stop advertising a dead answer path. The pop inside
                # retire_clarify_card runs before its first await, so the agent thread's own expiry
                # notice (scheduled once the wait unblocks) finds nothing and stays a no-op.
                _clarify_adapter = self._delivery_adapter_for(source)
                # Class lookup: a MagicMock adapter must not fabricate the method.
                if callable(getattr(type(_clarify_adapter), "retire_clarify_card", None)):
                    try:
                        await _clarify_adapter.retire_clarify_card(
                            _pending_clarify.clarify_id,
                            t("gateway.clarify.cancelled"))
                    except Exception:
                        logger.debug("Failed to retire clarify card after prose cancellation", exc_info=True)
        return None

