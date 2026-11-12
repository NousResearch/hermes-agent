"""An Anthropic MoA aggregator's requests, kept append-only so its signed thinking stays valid.

Anthropic binds each signed thinking block to everything sent before it ("preserved thinking"):
replayed behind a changed prefix, the block is dropped without an error or answered with a 400.
MoA attaches the reference guidance as a trailing user message, so each later request lost the
guidance that preceded the aggregator's earlier replies. ``AggregatorWire`` puts every guidance
message back where it was first sent and attaches a fan-out's guidance only once. Where the
transcript itself changed (compression, a rebuilt history, a fresh facade after an eviction or a
restart), the signed thinking behind the change is dropped for good: Anthropic accepts removed
thinking blocks, and thinking produced afterwards replays again.
"""

import hashlib
import json
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


def aggregator_reasoning_view(messages: list[dict[str, Any]], agg_runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """The conversation's assistant reasoning shaped for the aggregator's own wire (copies; input untouched).

    The agent hands MoA every reasoning carrier (``moa`` in ``_REASONING_CARRIER_ROUTES``). An Anthropic
    aggregator gets the native route's replay: signed thinking from ``reasoning_details`` / the ordered
    sidecar and never the readable text, which conversion would otherwise demote to assistant text — a
    running log of the model's own reasoning, refused by Anthropic's classifier as reasoning extraction.
    Any other aggregator gets the shape it always had: the readable text carriers only.
    """
    from agent.message_sanitization import apply_reasoning_content_policy, reasoning_replay_route

    anthropic = agg_runtime.get("api_mode") == "anthropic_messages"
    route = reasoning_replay_route(
        "anthropic_messages", agg_runtime.get("provider"), agg_runtime.get("model"), agg_runtime.get("base_url"),
    ) if anthropic else None
    view = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            shaped = dict(message)
            if route is not None:
                apply_reasoning_content_policy(message, shaped, route.pad, route.carriers)
            else:
                shaped.pop("reasoning_details", None)
                shaped.pop("anthropic_content_blocks", None)
            message = shaped
        view.append(message)
    return view


def replays_signed_thinking(agg_runtime: dict[str, Any]) -> bool:
    """Whether the aggregator's route sends signed thinking back (Anthropic and its cloud platforms, MiniMax)."""
    from agent.anthropic_thinking_replay import tracks_rejected_thinking

    return tracks_rejected_thinking(SimpleNamespace(**{k: agg_runtime.get(k) for k in ("api_mode", "base_url", "model")}))


def _signed(message: Any) -> set[str]:
    from agent.anthropic_thinking_replay import _CARRIERS, _fingerprint

    if not isinstance(message, dict):
        return set()
    return {fp for key in _CARRIERS for block in (message.get(key) if isinstance(message.get(key), list) else ())
            if (fp := _fingerprint(block))}


def without_thinking(messages: list[Any], rejected: set[str]) -> list[Any]:
    """Request copy without the signed blocks in ``rejected`` (``strip_rejected_thinking``)."""
    from agent.anthropic_thinking_replay import strip_rejected_thinking

    out = []
    for message in messages:
        if _signed(message) & rejected:
            message = dict(message)
            strip_rejected_thinking(message, rejected)
        out.append(message)
    return out


def _wire_key(message: Any) -> str:
    """What the wire carries of ``message`` before any thinking: thinking is not part of the prefix, and
    cache markers move freely."""
    if not isinstance(message, dict):
        return repr(message)
    content = message.get("content")
    if isinstance(content, list):
        content = [{k: v for k, v in part.items() if k != "cache_control"} if isinstance(part, dict) else part
                   for part in content]
    calls = []
    for call in message.get("tool_calls") or ():
        function = (call.get("function") if isinstance(call, dict) else getattr(call, "function", None)) or {}
        get = function.get if isinstance(function, dict) else lambda key: getattr(function, key, None)
        calls.append([call.get("id") if isinstance(call, dict) else getattr(call, "id", None),
                      get("name"), get("arguments")])
    wire = [message.get("role"), content, calls, message.get("tool_call_id"), message.get("name")]
    return hashlib.sha256(json.dumps(wire, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class AggregatorWire:
    """The aggregator's last request, so the next one can extend it instead of rewriting it."""

    def __init__(self) -> None:
        # One entry per message of the last request: (wire key, fan-out key, guidance message). A guidance
        # message has no wire key; a conversation message has no fan-out key.
        self._sent: list[tuple[str | None, Any, dict[str, Any] | None]] = []
        self._pending: list[tuple[str | None, Any, dict[str, Any] | None]] = []
        self._produced: set[str] = set()  # signed blocks this aggregator returned
        self.rejected: set[str] = set()  # signed blocks never to be sent again

    def request(
        self, history: list[dict[str, Any]], guidance: Any, guidance_key: Any,
    ) -> tuple[list[dict[str, Any]], Any]:
        """``history`` with every earlier guidance message back in place and the thinking that cannot replay
        removed, plus the guidance to attach at the end (None when this fan-out's is already in place)."""
        out: list[dict[str, Any]] = []
        entries: list[tuple[str | None, Any, dict[str, Any] | None]] = []
        changed_at, position = None, 0
        for message in history:
            if changed_at is None:
                position = self._replay_guidance(position, out, entries)
            key = _wire_key(message)
            if changed_at is None and position < len(self._sent):
                if key == self._sent[position][0]:
                    position += 1
                else:
                    changed_at = len(out)
            out.append(message)
            entries.append((key, None, None))
        if changed_at is None:
            self._replay_guidance(position, out, entries)
        else:
            behind = set().union(*(_signed(message) for message in out[changed_at:]))
            if behind - self.rejected:
                logger.info("MoA aggregator transcript changed at message %d since the last request; dropping the "
                            "%d signed thinking block(s) from there on", changed_at, len(behind - self.rejected))
            self.rejected |= behind
        # Thinking this aggregator did not return (a fresh facade after an eviction or a restart, another model's
        # turn) was signed behind a prefix it cannot rebuild.
        unknown = set().union(*(_signed(message) for message in out)) - self._produced - self.rejected
        if unknown:
            logger.info("MoA aggregator: not replaying %d signed thinking block(s) it did not produce", len(unknown))
        self.rejected |= unknown
        if guidance and (guidance_key is None or all(fanout != guidance_key for _, fanout, _ in entries)):
            entries.append((None, guidance_key, {"role": "user", "content": str(guidance)}))
        else:
            guidance = None
        self._pending = entries
        return without_thinking(out, self.rejected), guidance

    def _replay_guidance(self, position: int, out: list, entries: list) -> int:
        while position < len(self._sent) and self._sent[position][2] is not None:
            out.append(dict(self._sent[position][2]))
            entries.append(self._sent[position])
            position += 1
        return position

    def sent(self, response: Any) -> None:
        """The request built by ``request()`` succeeded with ``response``: remember both."""
        self._sent, self._pending = self._pending, []
        message = getattr((getattr(response, "choices", None) or [None])[0], "message", None)
        self._produced |= _signed({key: getattr(message, key, None)
                                   for key in ("reasoning_details", "anthropic_content_blocks")})

    def send_with_recovery(self, send: Any, label: str, *, messages: list[dict[str, Any]]) -> Any:
        """``send``, retried once without the request's signed thinking when Anthropic rejects it.

        A replayed block the API no longer accepts (another model's signature, an order the sidecar could not
        keep) 400s every later iteration too, so it is dropped from this session's aggregator requests, as the
        native route's ``strip_rejected_thinking`` does; thinking produced later still replays.
        """
        try:
            return send(messages=messages)
        except Exception as exc:
            from agent.error_classifier import FailoverReason, classify_api_error

            signed = set().union(*(_signed(message) for message in messages))
            if not signed or classify_api_error(exc).reason != FailoverReason.thinking_signature:
                raise
            self.rejected |= signed
            logger.warning("MoA aggregator %s rejected replayed thinking; dropping the %d signed block(s) it was sent "
                           "for the rest of the session and retrying once: %.200s", label, len(signed), exc)
            return send(messages=without_thinking(messages, self.rejected))
