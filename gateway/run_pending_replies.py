"""Existing pending-reply handlers, extracted without changing capture policy."""

import logging

logger = logging.getLogger("gateway.run")


async def clarify_reply(self, event, source, _quick_key):
    """Resolve only according to the existing model-tool clarify policy."""
    try:
        from tools import clarify_gateway as _clarify_mod
        _pending_clarify = _clarify_mod.get_pending_for_session(_quick_key, include_choice_prompts=True)
    except Exception:
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
    # Commands retain control semantics and do not answer a model's question.
    if not _raw_clarify_reply or _raw_clarify_reply.startswith("/"):
        return None
    _text_outcome = _clarify_mod.attempt_text_response_for_session(_quick_key, _raw_clarify_reply)
    if _text_outcome == _clarify_mod.TEXT_RESOLVED:
        logger.info(
            "Gateway intercepted clarify text response (session=%s, id=%s)",
            _quick_key, _pending_clarify.clarify_id,
        )
        _clarify_adapter = self._adapter_for_source(source)
        if _clarify_adapter:
            try:
                _clarify_adapter.resume_typing_for_chat(source.chat_id)
            except Exception:
                logger.debug("Failed to resume typing after clarify response", exc_info=True)
        return ""
    if _text_outcome == _clarify_mod.TEXT_REJECTED_SELECTION:
        return _retain("invalid selection attempt")
    if _text_outcome == _clarify_mod.TEXT_REJECTED_PROSE:
        # Release the rejected prompt before the ordinary busy path can steer the agent.
        _clarify_mod.resolve_gateway_clarify(_pending_clarify.clarify_id, "")
    return None


async def pending_reply_intercepts(self, event, source, quick_key):
    if not event.allow_gateway_control:
        return None
    reply = self._hm_update_prompt_reply(event, quick_key)
    if reply is None:
        reply = await self._hm_clarify_reply(event, source, quick_key)
    if reply is None:
        reply = await self._hm_slash_confirm_reply(event, quick_key)
    return reply
