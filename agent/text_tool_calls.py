"""Recover a tool call that a provider serialized as TEXT instead of structured ``tool_calls``.

Some OpenRouter backends run a chat template whose tool-call parser intermittently fails to
lift the model's native markup into the OpenAI ``tool_calls`` array, and pass the raw markup
through as content — or, on interleaved-reasoning models, as *reasoning*. The turn then looks
like a clean text answer with zero tool calls, so the loop ends the turn mid-task
(``turn_final_response.py`` promotes the reasoning and reports "complete").

Observed as DeepSeek's DSML on `deepseek-v4-pro` (~7% of calls on one backend, while sibling
backends serving the same model never do it), but the shape is provider-agnostic: a named
invoke wrapper with named parameter children. ``<tool_call>``-style JSON blocks are already
handled by :func:`agent.acp_openai_bridge.extract_tool_calls_from_text`; this module covers
the delimited-invoke family, including the envelope form where a model batches its real calls
inside one synthetic ``invoke name="tool_call"`` wrapper.

Recovery is deliberately narrow — it refuses rather than guesses:

* every extracted name must be a tool actually offered this turn (``valid_names``), so prose
  that merely talks about tools can never become a call;
* every invoke must be closed and every parameter terminated, so a generation cut off
  mid-argument never executes half a command.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

logger = logging.getLogger("agent.conversation_loop")

__all__ = ["extract_delimited_tool_calls", "salvage_text_tool_calls", "strip_tool_call_markup"]

# A delimiter prefix is whatever the vendor wraps its tags in: ``｜DSML｜``, ``antml:``, or
# nothing. Bounded so a stray ``<`` in prose cannot make the scanner run away.
_DELIM = r"[^<>\s]{0,12}?"

_INVOKE_RE = re.compile(
    rf"<{_DELIM}invoke\s+name\s*=\s*\"([^\"]+)\"\s*>",
    re.IGNORECASE,
)
# The batch opener that precedes the first invoke (``<｜DSML｜tool_calls>``); used only to find
# where leaked markup begins so preceding prose can be kept.
_TOOL_CALLS_OPEN_RE = re.compile(rf"<{_DELIM}tool_calls\s*>", re.IGNORECASE)
_PARAM_RE = re.compile(
    rf"<{_DELIM}parameter\s+name\s*=\s*\"([^\"]+)\"[^>]*>",
    re.IGNORECASE,
)
# Any tag that terminates a parameter's value: the next parameter, a closing parameter/invoke
# tag, or a closing tool_calls wrapper. Missing closers are expected, hence the alternation.
_PARAM_END_RE = re.compile(
    rf"<{_DELIM}/?(?:parameter|invoke|tool_calls)\b",
    re.IGNORECASE,
)
# An invoke must be explicitly closed before its arguments are trusted. A generation cut off
# mid-argument would otherwise yield a half-written command, and running half of `rm -rf ...`
# is worse than ending the turn. Real observed leaks are fully closed, so this costs nothing.
_INVOKE_CLOSE_RE = re.compile(rf"<{_DELIM}/{_DELIM}invoke\b", re.IGNORECASE)

# Wrapper names a model uses when it batches its real calls inside one synthetic envelope,
# e.g. ``invoke name="tool_call"`` whose single parameter is ``[{"name": ..., "arguments": ...}]``.
# These are never real tools, so an un-unwrapped envelope would be rejected by the name guard
# and the call lost.
_ENVELOPE_NAMES = frozenset({"tool_call", "tool_calls", "function_calls", "invoke"})


def _unwrap_envelope(name: str, args: dict[str, Any]) -> list[tuple[str, dict[str, Any]]] | None:
    """Inner ``[(name, arguments), ...]`` when ``(name, args)`` is a batched-call envelope.

    Returns ``None`` when this is an ordinary call, so the caller can tell "not an envelope"
    from "an envelope holding nothing usable" (the latter yields ``[]`` and is dropped).
    """
    if name.lower() not in _ENVELOPE_NAMES or len(args) != 1:
        return None
    payload = next(iter(args.values()))
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return None
    if not isinstance(payload, list):
        return None

    inner: list[tuple[str, dict[str, Any]]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        inner_name = entry.get("name") or entry.get("tool") or entry.get("function")
        if not isinstance(inner_name, str) or not inner_name.strip():
            continue
        inner_args = entry.get("arguments", entry.get("args", {}))
        if isinstance(inner_args, str):
            try:
                inner_args = json.loads(inner_args)
            except (ValueError, TypeError):
                inner_args = {}
        inner.append((inner_name.strip(), inner_args if isinstance(inner_args, dict) else {}))
    return inner


def _coerce_param_value(raw: str) -> Any:
    """Keep parameter values as text unless they cleanly parse as a JSON container.

    Tool arguments are re-validated downstream, so the only thing that matters here is not
    mangling a shell command that happens to start with ``[``: a value is decoded only when
    it is a well-formed object/array, and any failure keeps the original string.
    """
    text = raw.strip()
    if text[:1] in ("{", "["):
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return raw
    return raw


def extract_delimited_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    """``[(tool_name, arguments), ...]`` parsed out of delimited-invoke markup in ``text``.

    Every invoke must be closed and every parameter terminated; one incomplete block voids the
    whole result rather than yielding a partial call, because a truncated argument is unsafe to
    execute and a partially-recovered plan is worse than none.
    """
    if not isinstance(text, str) or "invoke" not in text.lower():
        return []

    calls: list[tuple[str, dict[str, Any]]] = []
    invokes = list(_INVOKE_RE.finditer(text))
    for index, invoke in enumerate(invokes):
        name = invoke.group(1).strip()
        if not name:
            continue
        # This invoke's body runs to the next invoke (or end of text), so a missing closer is
        # detectable rather than silently swallowing the rest of the message.
        body_end = invokes[index + 1].start() if index + 1 < len(invokes) else len(text)
        body = text[invoke.end():body_end]

        if not _INVOKE_CLOSE_RE.search(body):
            logger.warning(
                "Text-serialized tool call %r is not closed — refusing to salvage a possibly "
                "truncated argument list", name,
            )
            return []

        args: dict[str, Any] = {}
        params = list(_PARAM_RE.finditer(body))
        for p_index, param in enumerate(params):
            key = param.group(1).strip()
            if not key:
                continue
            value_start = param.end()
            limit = params[p_index + 1].start() if p_index + 1 < len(params) else len(body)
            terminator = _PARAM_END_RE.search(body, value_start, limit)
            if terminator is None:
                logger.warning(
                    "Text-serialized tool call %r has an unterminated parameter %r — refusing "
                    "to salvage", name, key,
                )
                return []
            args[key] = _coerce_param_value(body[value_start:terminator.start()])

        unwrapped = _unwrap_envelope(name, args)
        if unwrapped is not None:
            calls.extend(unwrapped)
        else:
            calls.append((name, args))
    return calls


def salvage_text_tool_calls(
    *, content: str | None, reasoning: str | None, valid_names: Iterable[str] | None,
) -> list[Any]:
    """Structured tool calls rebuilt from text-serialized markup in ``content``/``reasoning``.

    Returns ``[]`` unless every recovered name is a tool offered this turn — the guard that
    keeps prose about tools from becoming a call. ``reasoning`` is searched because
    interleaved-reasoning models file the leaked markup there, which is exactly the case the
    turn loop would otherwise mistake for a finished answer.
    """
    allowed = {str(n) for n in (valid_names or ())}
    if not allowed:
        return []

    from agent.acp_openai_bridge import build_openai_tool_call

    rebuilt: list[Any] = []
    for source in (content, reasoning):
        for name, args in extract_delimited_tool_calls(source or ""):
            if name not in allowed:
                # One unknown name voids the whole salvage: a partially-understood call is
                # worse than none, since the loop would run a truncated version of the plan.
                logger.warning(
                    "Text-serialized tool call names %r which is not an offered tool — "
                    "not salvaging", name,
                )
                return []
            rebuilt.append(build_openai_tool_call(
                call_id=f"salvaged_call_{len(rebuilt) + 1}",
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            ))
        if rebuilt:
            break
    return rebuilt


def strip_tool_call_markup(text: str | None) -> str:
    """``text`` with the leaked tool-call markup removed, keeping any prose that preceded it.

    Once a call is salvaged the markup must not stay in the assistant row's content, where it
    would render as the reply and replay as prose on the next turn.
    """
    if not isinstance(text, str) or not text:
        return ""
    starts = [m.start() for m in (_TOOL_CALLS_OPEN_RE.search(text), _INVOKE_RE.search(text)) if m]
    return text[:min(starts)].strip() if starts else text
