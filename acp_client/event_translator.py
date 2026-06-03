"""Translate inbound ACP ``session_update`` notifications to Hermes events.

Reversed mirror of ``acp_adapter/events.py``.  On the server side Hermes emits
``session_update`` notifications *to* an editor; here Hermes is the client and
*receives* them from an external agent.  Each inbound update is normalised into
a small plain dict (a ``progress.jsonl`` line in Phase 2) and, for message
chunks, appended to the session's mirror history (design §2.3, §2.6 — the
SessionDB row stays the source of truth, this history is derived/mirror state).

The translator is intentionally pure-Python and synchronous: it does not touch
the network, the event loop, or the SessionDB directly.  Callers decide what to
do with the normalised event and the accumulated text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _content_text(content: Any) -> str:
    """Best-effort extraction of text from an ACP content block (or list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(_content_text(c) for c in content)
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    # tool content wraps a content block under ``.content``.
    inner = getattr(content, "content", None)
    if inner is not None and inner is not content:
        return _content_text(inner)
    return ""


def _update_kind(update: Any) -> str:
    """Return the ACP discriminator (``session_update``) for *update*.

    Falls back to a snake-cased class name when the attribute is absent (e.g.
    a hand-built stub), so the translator stays robust against SDK drift.
    """
    kind = getattr(update, "session_update", None)
    if isinstance(kind, str) and kind:
        return kind
    name = type(update).__name__
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


@dataclass
class EventTranslator:
    """Stateful translator for one outbound session's inbound updates.

    Args:
        on_event: Optional sink called with each normalised event dict
            (the Phase-2 ``progress.jsonl`` writer plugs in here).
    """

    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    history: List[Dict[str, str]] = field(default_factory=list)
    message_text: str = ""
    thought_text: str = ""

    def translate(self, update: Any) -> Dict[str, Any]:
        """Normalise one inbound ``session_update`` payload.

        Returns a plain dict with at least ``{"type": <kind>}``.  Side effects:
        agent message chunks accumulate into ``message_text`` and, on a
        complete turn (see :meth:`finalize_message`), into ``history``.
        """
        kind = _update_kind(update)
        event: Dict[str, Any] = {"type": kind}

        if kind == "agent_message_chunk":
            text = _content_text(getattr(update, "content", None))
            self.message_text += text
            event["text"] = text
        elif kind == "agent_thought_chunk":
            text = _content_text(getattr(update, "content", None))
            self.thought_text += text
            event["text"] = text
        elif kind in ("tool_call", "tool_call_update"):
            event.update(
                tool_call_id=getattr(update, "tool_call_id", None),
                title=getattr(update, "title", None),
                tool_kind=getattr(update, "kind", None),
                status=getattr(update, "status", None),
            )
        elif kind == "plan":
            entries = getattr(update, "entries", None) or []
            event["entries"] = [
                {
                    "content": _content_text(getattr(e, "content", None))
                    or getattr(e, "content", ""),
                    "status": getattr(e, "status", None),
                    "priority": getattr(e, "priority", None),
                }
                for e in entries
            ]
        elif kind == "usage":
            event["usage"] = self._dump(update)
        else:
            event["raw"] = self._dump(update)

        self._emit(event)
        return event

    def finalize_message(self) -> Optional[Dict[str, str]]:
        """Flush the accumulated assistant message into ``history``.

        Call this when a prompt turn completes (``stop_reason`` received).
        Returns the appended history row, or ``None`` if nothing accumulated.
        """
        text = self.message_text
        self.message_text = ""
        self.thought_text = ""
        if not text:
            return None
        row = {"role": "assistant", "content": text}
        self.history.append(row)
        return row

    def record_user_prompt(self, text: str) -> Dict[str, str]:
        """Append the outbound user prompt to mirror history."""
        row = {"role": "user", "content": text}
        self.history.append(row)
        return row

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _dump(obj: Any) -> Any:
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            try:
                return dump(mode="json", exclude_none=True)
            except Exception:
                try:
                    return dump()
                except Exception:
                    return str(obj)
        return str(obj)

    def _emit(self, event: Dict[str, Any]) -> None:
        if callable(self.on_event):
            try:
                self.on_event(event)
            except Exception:
                logger.debug("event sink failed for %s", event.get("type"), exc_info=True)
