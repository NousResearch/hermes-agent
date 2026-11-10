"""Provider-native assistant reasoning carriers outside the standard reasoning fields.

* Gemini text-turn ``thought_signature`` (message-level ``extra_content.google`` on the OpenAI-compat
  and Vertex endpoints; the non-functionCall ``thoughtSignature`` part on the native API). Replaying it
  makes the server rehydrate the turn's thoughts (live: 31 -> 237/250 prompt tokens).
* Top-level assistant keys a provider profile declares (``ProviderProfile.message_carrier_opaque`` /
  ``message_carrier_text``: an opaque reasoning blob and its readable text), plus the tool-call
  ``function.thought_signature`` some gateways sign Gemini 3 calls with.

Capture stores them top-level on the live assistant dict AND as ONE private ``reasoning_details`` record,
so they persist in the existing column and survive gateway/resume replay. Its type ends in ``.native_assistant``, which no profile declares, so
the chat-completions transport never forwards the record itself; ``shape_wire_carriers`` lifts each
field back onto the wire copy only for the route that reads it (the endpoint profile's
``replay_message_carriers``) and strips it everywhere else, including keys of records whose declaring
plugin is no longer installed.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

CARRIER_TYPE = "hermes.native_assistant"
_GEMINI_KEY = "extra_content"


def declared_carrier_keys() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(opaque keys, text keys)`` declared by the registered profiles, in registration order. Call once per
    message/stream/request, never per delta."""
    from providers import list_providers

    profiles = list_providers()
    opaque = (p.message_carrier_opaque for p in profiles if p.message_carrier_opaque)
    text = (p.message_carrier_text for p in profiles if p.message_carrier_text)
    return tuple(dict.fromkeys(opaque)), tuple(dict.fromkeys(text))


def _message_keys() -> tuple[str, ...]:
    opaque, text = declared_carrier_keys()
    return (_GEMINI_KEY, *dict.fromkeys(opaque + text))


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
    if sig := thought_signature(field(assistant_message, _GEMINI_KEY)):
        record[_GEMINI_KEY] = {"google": {"thought_signature": sig}}
    for key in _message_keys()[1:]:
        value = field(assistant_message, key)
        if isinstance(value, str) and value:
            record[key] = value
    return {"type": CARRIER_TYPE, **record} if record else None


class StreamCarriers:
    """Accumulates the carriers from streamed deltas: the latest signature / opaque blob per declared key (a
    provider may send a new opaque before each tool call and expect only the latest back) and the concatenated
    text of each declared text key. Keys are resolved once per stream."""

    def __init__(self) -> None:
        self.opaque_keys, self.text_keys = declared_carrier_keys()
        self.signature: Optional[str] = None
        self.opaque: dict[str, str] = {}
        self.text_parts: dict[str, list[str]] = {}

    def feed(self, delta: Any, reasoning: Optional[str]) -> Optional[str]:
        """Absorb one delta; returns ``reasoning``, else its first declared text-carrier fragment (readable reasoning)."""
        self.signature = thought_signature(field(delta, _GEMINI_KEY)) or self.signature
        for key in self.opaque_keys:
            opaque = field(delta, key)
            if isinstance(opaque, str) and opaque:
                self.opaque[key] = opaque
        for key in self.text_keys:
            text = field(delta, key)
            if isinstance(text, str) and text:
                self.text_parts.setdefault(key, []).append(text)
                reasoning = text if reasoning is None else reasoning
        return reasoning

    def apply(self, response: Any) -> None:
        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        if message is None:
            return
        if self.signature:
            message.extra_content = {"google": {"thought_signature": self.signature}}
        for key, value in self.opaque.items():
            setattr(message, key, value)
        for key, parts in self.text_parts.items():
            setattr(message, key, "".join(parts))


class _WireRoute(NamedTuple):
    model: Any
    base_url: str
    gemini: bool  # the route consumes the Gemini text-turn signature
    profile: Any  # profile serving the host (``providers.profile_for_host``) or None
    keep_fn_sig: bool  # replayed tool calls keep ``function.thought_signature``
    keys: tuple[str, ...]  # every message-level carrier key core knows about
    scope: Any  # who remembers a failed hook (``provider_route_hooks.ask_profile``): the agent, or None


def _wire_route(model: Any, base_url: Any, scope: Any = None) -> _WireRoute:
    from providers import profile_for_host
    from agent.provider_route_hooks import ask_profile
    from agent.transports.chat_completions import _model_consumes_thought_signature

    url = str(base_url or "")
    profile = profile_for_host(url)
    keep_fn_sig = profile is not None and bool(ask_profile(
        profile, "replays_function_thought_signature", False, model=model, base_url=url, scope=scope))
    # Same gate as tool-call extra_content, so the two signature carriers can never disagree.
    return _WireRoute(model, url, _model_consumes_thought_signature(model), profile, keep_fn_sig, _message_keys(),
                      scope)


# ``replay_message_carriers`` answers by which carrier keys are present (providers/base.py), so its answer
# for one key set is kept as a key mapping and applied to every message with that key set: one hook call
# per (route, key set) instead of one per carrier-bearing message per request.
_REPLAY_PLANS: dict[tuple, Any] = {}
_REPLAY_PLANS_MAX = 512
_REWRITES = object()  # the hook rewrites values: ask it per message
_FAILED = object()


def _replay_plan(answer: Any, values: dict) -> Any:
    """``((out key, source key), ...)`` reproducing *answer* from *values*; None when ambiguous."""
    if not isinstance(answer, dict):
        return _REWRITES
    plan = []
    for out_key, out_value in answer.items():
        sources = [k for k, v in values.items() if v == out_value]
        if not sources:
            return _REWRITES
        if len(sources) > 1:
            return None
        plan.append((out_key, sources[0]))
    return tuple(plan)


def _replay_carriers(route: _WireRoute, values: dict) -> dict:
    from providers import registry_generation
    from agent.provider_route_hooks import ask_profile, hook_failed

    if hook_failed(route.profile, "replay_message_carriers", route.scope):
        return {}
    key = (registry_generation(), id(route.profile), type(route.profile), route.model, route.base_url,
           tuple(sorted(values)))
    plan = _REPLAY_PLANS.get(key)
    if plan is not None and plan is not _REWRITES:
        return {out: values[src] for out, src in plan}
    answer = ask_profile(route.profile, "replay_message_carriers", _FAILED, values, model=route.model,
                         base_url=route.base_url, scope=route.scope)
    if answer is _FAILED:
        return {}
    if plan is None and (plan := _replay_plan(answer, values)) is not None:
        if len(_REPLAY_PLANS) >= _REPLAY_PLANS_MAX:
            _REPLAY_PLANS.clear()
        _REPLAY_PLANS[key] = plan
    return dict(answer or {})


def _strip_function_signature(tool_calls: list) -> list:
    return [
        {**tc, "function": {k: v for k, v in tc["function"].items() if k != "thought_signature"}}
        if isinstance(tc, dict) and isinstance(tc.get("function"), dict) and "thought_signature" in tc["function"]
        else tc
        for tc in tool_calls
    ]


def _shape(msg: Any, route: _WireRoute) -> Any:
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return msg
    details = msg.get("reasoning_details")
    details = details if isinstance(details, list) else []
    record = next((d for d in details if isinstance(d, dict) and d.get("type") == CARRIER_TYPE), None)
    tool_calls = msg.get("tool_calls")
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    fn_sig = not route.keep_fn_sig and any(
        isinstance(tc, dict) and isinstance(tc.get("function"), dict) and "thought_signature" in tc["function"]
        for tc in tool_calls)
    # A persisted record's own keys are stripped too, so rows from an uninstalled provider plugin never leak.
    keys = route.keys if record is None else (*route.keys, *(k for k in record if k != "type" and k not in route.keys))
    if record is None and not fn_sig and not any(k in msg for k in keys):
        return msg
    out = {k: v for k, v in msg.items() if k not in keys}
    # Live history carries the keys top-level; a resumed row only has the persisted record.
    values = {**(record or {}), **{k: msg[k] for k in keys if k in msg}}
    if record is not None:
        rest = [d for d in details if d is not record]
        if rest:
            out["reasoning_details"] = rest
        else:
            out.pop("reasoning_details", None)
    if route.gemini and values.get(_GEMINI_KEY):
        out[_GEMINI_KEY] = values[_GEMINI_KEY]
    if route.profile is not None:
        out.update(_replay_carriers(route, values))
    if fn_sig:
        out["tool_calls"] = _strip_function_signature(tool_calls)
    return out


def shape_wire_carriers(messages: list, *, model: Any, base_url: Any, scope: Any = None) -> list:
    """Wire copy of ``messages`` with each carrier on the route that reads it and nowhere else.
    Never mutates the input; unchanged messages are returned by identity. *scope* is the agent (failed
    profile hooks stay failed for it; see ``provider_route_hooks.ask_profile``)."""
    from agent.provider_route_hooks import hook_failed

    route = _wire_route(model, base_url, scope)
    failed_before = route.profile is None or hook_failed(route.profile, "replay_message_carriers", scope)
    shaped = [_shape(m, route) for m in messages]
    if not failed_before and hook_failed(route.profile, "replay_message_carriers", scope):
        # Failed mid-request: reshape so every message takes the fallback, as every later request will.
        shaped = [_shape(m, route) for m in messages]
    return messages if all(a is b for a, b in zip(shaped, messages)) else shaped
