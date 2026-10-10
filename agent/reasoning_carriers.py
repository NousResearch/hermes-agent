"""Provider-native assistant reasoning carriers outside the standard reasoning fields.

* Gemini text-turn ``thought_signature`` (message-level ``extra_content.google`` on the OpenAI-compat
  and Vertex endpoints; the non-functionCall ``thoughtSignature`` part on the native API). Replaying it
  makes the server rehydrate the turn's thoughts (live: 31 -> 237/250 prompt tokens).
* GitHub Copilot ``/chat/completions`` ``reasoning_text`` / ``reasoning_opaque`` (top-level assistant
  keys, replayed by the Copilot editor clients) and the Gemini-3 ``function.thought_signature``.

Capture stores them top-level on the live assistant dict AND as ONE private ``reasoning_details`` record,
so they persist in the existing column and survive gateway/resume replay. Its type ends in ``.native_assistant``, which no profile declares, so
the chat-completions transport never forwards the record itself; ``shape_wire_carriers`` lifts each
field back onto the wire copy only for the route that reads it and strips it everywhere else.
"""

from __future__ import annotations

from typing import Any, Optional

CARRIER_TYPE = "hermes.native_assistant"
_MESSAGE_KEYS = ("extra_content", "reasoning_opaque", "reasoning_text")


def field(obj: Any, name: str) -> Any:
    """``obj.<name>``, else the same key from pydantic ``model_extra`` (unknown SDK fields park there)."""
    value = getattr(obj, name, None)
    for side in ("model_extra", "provider_data"):  # pydantic unknown fields / normalized responses
        if value is None and isinstance(getattr(obj, side, None), dict):
            value = getattr(obj, side).get(name)
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else value


def thought_signature(extra: Any) -> Optional[str]:
    """Gemini signature inside an ``extra_content`` dict (``google.thought_signature`` or flat)."""
    if not isinstance(extra, dict):
        return None
    google = extra.get("google")
    sig = google.get("thought_signature") if isinstance(google, dict) else extra.get("thought_signature")
    return sig if isinstance(sig, str) and sig.strip() else None


def carrier_record(assistant_message: Any) -> Optional[dict]:
    """The private ``reasoning_details`` record for one response message, or None when it has no carrier."""
    record: dict[str, Any] = {}
    if sig := thought_signature(field(assistant_message, "extra_content")):
        record["extra_content"] = {"google": {"thought_signature": sig}}
    for key in ("reasoning_opaque", "reasoning_text"):
        value = field(assistant_message, key)
        if isinstance(value, str) and value:
            record[key] = value
    return {"type": CARRIER_TYPE, **record} if record else None


class StreamCarriers:
    """Accumulates the carriers from streamed deltas: the latest signature / opaque blob (Claude on
    Copilot sends a new opaque before each tool call; the editor clients replay only the latest) and
    the concatenated ``reasoning_text``."""

    def __init__(self) -> None:
        self.signature: Optional[str] = None
        self.opaque: Optional[str] = None
        self.text_parts: list[str] = []

    def feed(self, delta: Any, reasoning: Optional[str]) -> Optional[str]:
        """Absorb one delta; returns ``reasoning``, else its ``reasoning_text`` fragment (Copilot's readable reasoning)."""
        self.signature = thought_signature(field(delta, "extra_content")) or self.signature
        opaque = field(delta, "reasoning_opaque")
        self.opaque = opaque if isinstance(opaque, str) and opaque else self.opaque
        text = field(delta, "reasoning_text")
        if not (isinstance(text, str) and text):
            return reasoning
        self.text_parts.append(text)
        return text if reasoning is None else reasoning

    def apply(self, response: Any) -> None:
        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        if message is None:
            return
        if self.signature:
            message.extra_content = {"google": {"thought_signature": self.signature}}
        if self.opaque:
            message.reasoning_opaque = self.opaque
        if self.text_parts:
            message.reasoning_text = "".join(self.text_parts)


def _route_reads(model: Any, base_url: Any) -> tuple[str, ...]:
    """Message-level carrier keys this route consumes."""
    from agent.transports.chat_completions import _model_consumes_thought_signature
    from utils import base_url_host_matches

    # Same gate as tool-call extra_content, so the two signature carriers can never disagree.
    gemini = ("extra_content",) if _model_consumes_thought_signature(model) else ()
    copilot = ("reasoning_opaque", "reasoning_text") if base_url_host_matches(str(base_url or ""), "githubcopilot.com") else ()
    return gemini + copilot


def _strip_function_signature(tool_calls: list) -> list:
    return [
        {**tc, "function": {k: v for k, v in tc["function"].items() if k != "thought_signature"}}
        if isinstance(tc, dict) and isinstance(tc.get("function"), dict) and "thought_signature" in tc["function"]
        else tc
        for tc in tool_calls
    ]


def _shape(msg: Any, reads: tuple[str, ...], copilot: bool) -> Any:
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return msg
    details = msg.get("reasoning_details")
    details = details if isinstance(details, list) else []
    record = next((d for d in details if isinstance(d, dict) and d.get("type") == CARRIER_TYPE), None)
    tool_calls = msg.get("tool_calls")
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    fn_sig = not copilot and any(
        isinstance(tc, dict) and isinstance(tc.get("function"), dict) and "thought_signature" in tc["function"]
        for tc in tool_calls)
    if record is None and not fn_sig and not any(k in msg for k in _MESSAGE_KEYS):
        return msg
    out = {k: v for k, v in msg.items() if k not in _MESSAGE_KEYS}
    # Live history carries the keys top-level; a resumed row only has the persisted record.
    values = {**(record or {}), **{k: msg[k] for k in _MESSAGE_KEYS if k in msg}}
    if record is not None:
        rest = [d for d in details if d is not record]
        if rest:
            out["reasoning_details"] = rest
        else:
            out.pop("reasoning_details", None)
    for key in reads:
        # Copilot clients send reasoning_text only next to the opaque blob it belongs to.
        if values.get(key) and (key != "reasoning_text" or values.get("reasoning_opaque")):
            out[key] = values[key]
    if fn_sig:
        out["tool_calls"] = _strip_function_signature(tool_calls)
    return out


def shape_wire_carriers(messages: list, *, model: Any, base_url: Any) -> list:
    """Wire copy of ``messages`` with each carrier on the route that reads it and nowhere else.
    Never mutates the input; unchanged messages are returned by identity."""
    from utils import base_url_host_matches

    reads = _route_reads(model, base_url)
    copilot = base_url_host_matches(str(base_url or ""), "githubcopilot.com")
    shaped = [_shape(m, reads, copilot) for m in messages]
    return messages if all(a is b for a, b in zip(shaped, messages)) else shaped
