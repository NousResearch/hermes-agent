"""``pre_send_message``: let a plugin see, rewrite or drop any message before an adapter sends it (#134330).

Every outbound text in the gateway (final replies, cron and kanban notices, approval prompts,
interim and status messages, shutdown notices) ends in some adapter's ``send``. A plugin that
needs one policy for everything that reaches a person (a daily budget, quiet hours, dropping a
repeated notice, never sending internal text) otherwise has to monkeypatch every adapter class.

``BasePlatformAdapter.__init_subclass__`` wraps each concrete ``send`` with :func:`guard_send`, so
the hook fires once per outbound message on every platform, built-in or plugin-provided. Nested
``send`` calls (a subclass calling ``super().send``, a split payload sent chunk by chunk through
the same adapter) fire it only for the outermost call.

Not fired for streamed previews (``metadata["expect_edits"]``): a partial can't be judged, and
the stream's final edit goes through ``edit_message``. Media sends (``send_image`` and friends)
are out of scope for v1.

Callback kwargs: ``platform``, ``chat_id``, ``text``, ``kind`` (``"interim"`` for mid-turn
status/advisory sends, else ``"message"``), ``reply_to``, ``metadata`` (a copy). Return:

- ``None`` / anything else: send unchanged;
- ``{"action": "rewrite", "text": "..."}``: send this text instead (first rewrite wins);
- ``{"action": "drop", "reason": "..."}``: don't send; the caller gets a successful
  ``SendResult`` with no ``message_id`` and ``raw_response={"pre_send": "dropped"}``, so no
  retry or plain-text fallback runs. A drop beats any rewrite.

Fail-open: a callback that raises or times out is logged and the message goes out unchanged.
"""

from __future__ import annotations

import contextvars
import functools
import logging
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

PRE_SEND_HOOK = "pre_send_message"

# True while an outer guarded send is in flight on this task: inner sends skip the hook.
_IN_GUARDED_SEND: contextvars.ContextVar[bool] = contextvars.ContextVar("_in_guarded_send", default=False)


def _kind(metadata: Any) -> Optional[str]:
    """Lane name for the hook, or ``None`` when the send is a streamed preview (hook skipped)."""
    if not isinstance(metadata, dict):
        return "message"
    if metadata.get("expect_edits"):
        return None
    return "interim" if metadata.get("_interim_send") else "message"


async def run_pre_send_hook(
    platform: str, chat_id: Any, text: str, *, reply_to: Optional[str], metadata: Any, kind: str,
) -> Tuple[str, Optional[str]]:
    """Return ``("send", text)`` (possibly rewritten) or ``("drop", reason)``."""
    try:
        from hermes_cli.lifecycle import ainvoke_hook, has_hook
        if not has_hook(PRE_SEND_HOOK):
            return "send", text
        results = await ainvoke_hook(
            PRE_SEND_HOOK, platform=platform, chat_id=str(chat_id), text=text, kind=kind,
            reply_to=reply_to, metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )
    except Exception:
        logger.warning("%s failed; sending unchanged", PRE_SEND_HOOK, exc_info=True)
        return "send", text
    rewrite: Optional[str] = None
    for result in results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action == "drop":
            return "drop", str(result.get("reason") or "")
        if action == "rewrite" and rewrite is None and isinstance(result.get("text"), str):
            rewrite = result["text"]
    return "send", text if rewrite is None else rewrite


def _arg(args: tuple, kwargs: dict, index: int, name: str) -> Any:
    """``send``'s argument by keyword or by its position in ``(chat_id, content, reply_to, metadata)``."""
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def guard_send(send: Callable[..., Any], dropped_result: Callable[[], Any]) -> Callable[..., Any]:
    """Wrap an adapter class's ``send`` so :data:`PRE_SEND_HOOK` runs once per outbound message."""
    if getattr(send, "__pre_send_guarded__", False):
        return send

    @functools.wraps(send)
    async def guarded_send(self, *args, **kwargs):
        content = _arg(args, kwargs, 1, "content")
        metadata = _arg(args, kwargs, 3, "metadata")
        kind = _kind(metadata)
        if _IN_GUARDED_SEND.get() or kind is None or not isinstance(content, str):
            return await send(self, *args, **kwargs)
        chat_id = _arg(args, kwargs, 0, "chat_id")
        platform = getattr(getattr(self, "platform", None), "value", None) or getattr(self, "name", "") or ""
        action, value = await run_pre_send_hook(
            str(platform), chat_id, content, reply_to=_arg(args, kwargs, 2, "reply_to"), metadata=metadata,
            kind=kind)
        if action == "drop":
            logger.info("[%s] %s dropped a %s send to %s: %s", platform, PRE_SEND_HOOK, kind, chat_id, value)
            return dropped_result()
        if "content" in kwargs:
            kwargs["content"] = value
        else:
            args = args[:1] + (value,) + args[2:]
        token = _IN_GUARDED_SEND.set(True)
        try:
            return await send(self, *args, **kwargs)
        finally:
            _IN_GUARDED_SEND.reset(token)

    guarded_send.__pre_send_guarded__ = True  # type: ignore[attr-defined]
    return guarded_send
