"""Gateway conversation middleware: opt-in native plugins at the chat boundary.

Factories registered as ``gateway_conversation`` return adapter-local objects.
Objects implement bind(dispatch), receive(event), send(..., send=next_send),
processing(event, phase, session_id, outcome), typing(chat_id), and close().
The host owns control-command bypass and native delivery; plugins own policy.
"""
from __future__ import annotations

import functools
import logging
import contextvars
import inspect
from dataclasses import dataclass, asdict

_active_conversation = contextvars.ContextVar("gateway_conversation", default=None)
_native_delivery = contextvars.ContextVar("gateway_native_delivery", default=None)
_standalone_delivery = contextvars.ContextVar("gateway_standalone_delivery", default=False)


@dataclass(frozen=True)
class ConversationKey:
    platform: str
    chat_id: str
    scope_id: str = ""
    thread_id: str = ""

    @classmethod
    def from_source(cls, source):
        return cls(getattr(source.platform, "value", str(source.platform)), str(source.chat_id),
                   str(getattr(source, "scope_id", "") or ""), str(getattr(source, "thread_id", "") or ""))

    def metadata(self):
        return {"conversation": asdict(self)}

logger = logging.getLogger(__name__)


class ConversationMiddleware:
    def __init__(self, adapter):
        from hermes_cli.plugins import invoke_middleware
        self.adapter = adapter
        self.handlers = invoke_middleware("gateway_conversation", adapter=adapter)
        self.message_contexts = {}
        dispatch = adapter._handle_message_after_conversation
        for handler in reversed(self.handlers):
            handler.bind(dispatch)
            dispatch = handler.receive
        self._receive = dispatch

    @property
    def delays_messages(self):
        return any(handler.delays_messages for handler in self.handlers)

    @property
    def buffers_output(self):
        return any(handler.buffers_output for handler in self.handlers)

    async def receive(self, event):
        key = ConversationKey.from_source(event.source) if event.source else None
        token = _active_conversation.set(key)
        try:
            if event.internal or event.is_command():
                return await self.adapter._handle_message_after_conversation(event)
            return await self._receive(event)
        finally:
            _active_conversation.reset(token)

    async def send(self, chat_id, content, reply_to=None, metadata=None, *, send):
        current = send
        for handler in reversed(self.handlers):
            current = functools.partial(handler.send, send=current)
        return await current(chat_id, content, reply_to, metadata)

    def processing(self, event, phase, session_id, outcome=None):
        for handler in self.handlers:
            try:
                handler.processing(event, phase, session_id, outcome)
            except Exception:
                logger.warning("Conversation middleware lifecycle callback failed", exc_info=True)

    def typing(self, chat_id):
        for handler in self.handlers:
            handler.typing(chat_id)

    def output_suppressed(self, event, content, reason):
        """Notify policy plugins when the host suppresses output before send()."""
        for handler in self.handlers:
            callback = getattr(handler, "output_suppressed", None)
            if callback is None:
                continue
            try:
                callback(event, content, reason)
            except Exception:
                logger.warning("Conversation middleware suppression callback failed", exc_info=True)

    async def close(self):
        for handler in self.handlers:
            try:
                await handler.close()
            except Exception:
                logger.warning("Conversation middleware shutdown failed", exc_info=True)


def conversation_send(send):
    return conversation_output(send)


def conversation_output(send):
    """Intercept the normalized adapter contract, independent of transport SDKs."""
    if getattr(send, "_conversation_wrapped", False):
        return send
    signature = inspect.signature(send)
    kind = send.__name__
    @functools.wraps(send)
    async def wrapped(self, *args, **kwargs):
        if _native_delivery.get() is self or _standalone_delivery.get():
            return await send(self, *args, **kwargs)
        middleware = self.conversation_middleware()
        if not middleware.handlers:
            return await send(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        text_key = next((key for key in ("content", "caption", "text") if isinstance(values.get(key), str)), None)
        if text_key is None:
            return await send(self, *args, **kwargs)
        from gateway.platforms.base import SendResult
        if kind == "send_draft" and middleware.buffers_output:
            # No provisional content may escape before classification of the final reply.
            return SendResult(success=True)
        chat_id = values.get("chat_id")
        metadata = dict(values.get("metadata") or {})
        active = _active_conversation.get()
        platform = getattr(self.platform, "value", str(self.platform))
        if "conversation" not in metadata:
            if kind == "edit_message":
                metadata.update(middleware.message_contexts.get((str(chat_id), str(values.get("message_id"))), {}))
            if "conversation" not in metadata and active and active.platform == platform and active.chat_id == str(chat_id):
                metadata.update(active.metadata())
        async def deliver(target, text, reply_to=None, delivery_metadata=None):
            values[text_key] = text
            if "metadata" in signature.parameters:
                values["metadata"] = delivery_metadata
            token = _native_delivery.set(self)
            try:
                return await send(*bound.args, **bound.kwargs)
            finally:
                _native_delivery.reset(token)
        if text_key == "caption":
            # Filtering a caption must not silently discard the attached document.
            captured = []
            async def capture(target, text, reply_to=None, delivery_metadata=None):
                captured.append(text)
                return SendResult(success=True)
            await middleware.send(chat_id, values[text_key], metadata=metadata, send=capture)
            return await deliver(chat_id, captured[0] if captured else "", delivery_metadata=metadata)
        result = await middleware.send(chat_id, values[text_key], values.get("reply_to"), metadata, send=deliver)
        if getattr(result, "success", False) and getattr(result, "message_id", None):
            middleware.message_contexts[(str(chat_id), str(result.message_id))] = metadata
            if len(middleware.message_contexts) > 2048:
                middleware.message_contexts.pop(next(iter(middleware.message_contexts)))
        return result
    wrapped._conversation_wrapped = True
    return wrapped


def conversation_standalone(send):
    """Apply the same middleware before standalone tool/cron chunking and upload."""
    @functools.wraps(send)
    async def wrapped(platform, pconfig, chat_id, message, thread_id=None, media_files=None, force_document=False, args=None):
        if _standalone_delivery.get():
            return await send(platform, pconfig, chat_id, message, thread_id, media_files, force_document, args)
        from types import SimpleNamespace
        from gateway.platforms.base import SendResult
        async def unused_dispatch(event):
            return None
        adapter = SimpleNamespace(platform=platform, config=pconfig, _handle_message_after_conversation=unused_dispatch)
        middleware = ConversationMiddleware(adapter)
        if not middleware.handlers:
            return await send(platform, pconfig, chat_id, message, thread_id, media_files, force_document, args)
        approved = []
        async def capture(target, text, reply_to=None, metadata=None):
            approved.append(text)
            return SendResult(success=True)
        key = ConversationKey(getattr(platform, "value", str(platform)), str(chat_id), thread_id=str(thread_id or ""))
        try:
            await middleware.send(chat_id, message or "", metadata=key.metadata(), send=capture)
            if not approved and not media_files:
                return {"success": True, "suppressed": True}
            token = _standalone_delivery.set(True)
            try:
                approved_text = approved[0] if approved else ""
                approved_args = dict(args, message=approved_text) if args is not None else None
                return await send(platform, pconfig, chat_id, approved_text, thread_id, media_files, force_document, approved_args)
            finally:
                _standalone_delivery.reset(token)
        finally:
            await middleware.close()
    return wrapped
