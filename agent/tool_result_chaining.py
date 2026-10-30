"""Bounded tool-result retention and ``{{tool_result:<call_id>.<field>}}`` reference
resolution (#78061): a later tool call can consume a prior tool's structured output
without the model re-emitting the payload.

``remember_tool_result`` keeps raw results keyed by tool call id, bounded three ways:
per-result bytes, entry count, and a total retention budget (oldest evicted first).
``tool_result_reference_hint`` advertises a usable reference path to the model via a
sibling ``_tool_result_hint`` key on the tool message - kept out of ``content`` so the
content stays a parseable JSON string; each wire path folds the hint into the
model-visible text at request-build time (``fold_tool_result_hint``).
``resolve_tool_result_refs`` substitutes references in tool arguments before dispatch.
Advertised path components are restricted to plain identifiers (anything else degrades
to the generic ``field`` placeholder) and the assembled hint is delimiter-neutralized
before it is stored - the hint is folded outside the untrusted-content wrapper, so
result-derived names must never carry delimiter material across that boundary.

References resolve against results committed before dispatch: earlier turns, earlier
batches, and earlier sequential calls of the same batch. Concurrent same-batch
siblings are not yet committed, so a reference to one passes through untouched.
A bare ``{{tool_result:<call_id>}}`` (no dot) resolves to the whole stored payload;
list indices follow Python semantics (negative indices included). Unresolvable
references always pass through untouched - never a crash.

Whole-value references (the argument is exactly one reference) resolve to the stored
value with its native JSON type - a tool schema expecting an array or object receives
one - while embedded references (part of a larger string) substitute as text.
Command-like arguments (``terminal`` command, ``execute_code`` code) only resolve
whole-value references, stringified: their value is executed verbatim, so an embedded
reference could let untrusted result content alter a command the model authored - it
passes through untouched instead. MCP execution tools (a remote ``exec``, a VM command
runner) are NOT covered by that guard - the harness cannot know their argument
semantics, so embedded references still substitute there; keep untrusted result
sources away from them. Every successful resolution logs at debug level (call id,
path, resolved size - never the value).
"""

import json
import logging
import re
from typing import Any

from agent.tool_dispatch_helpers import _DELIMITER_TOKEN_RE, _neutralize_delimiters

logger = logging.getLogger(__name__)

_TOOL_RESULT_REF_RE = re.compile(r"\{\{tool_result:([^{}]+)\}\}")
_MAX_TOOL_RESULT_REF_BYTES = 20 * 1024 * 1024  # per-result cap
_MAX_TOOL_RESULT_REFS = 32  # entry cap
_MAX_TOOL_RESULT_REF_TOTAL_BYTES = 64 * 1024 * 1024  # total budget; oldest evicted first

# Advertised hint path components are restricted to plain identifiers that do not
# themselves carry the delimiter token. The hint is folded into model-visible text
# OUTSIDE the untrusted-result wrapper (transports append it after
# ``_maybe_wrap_untrusted`` runs) and is never re-wrapped, so a content-derived key
# must not be emitted verbatim - a key containing ``</untrusted_tool_result>`` would
# close the trust boundary early. Anything else degrades to the generic ``.field``
# placeholder the model swaps for the real name; the resolver is unchanged (it still
# accepts the real key). Excluding token-carrying identifiers keeps the defang below
# a pure backstop: it never rewrites a path the hint advertised.
_HINT_KEY_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

# Command-like arguments: their value is executed verbatim (shell command / script
# source). Embedded references are never substituted there, so untrusted result
# content cannot alter a larger command string the model authored; whole-value
# references stay allowed (the model explicitly consumed the stored value as the
# command). Plugin/MCP execution surfaces are not listed - the harness cannot know
# their argument semantics.
_COMMAND_LIKE_ARGS: dict[str, frozenset[str]] = {
    "terminal": frozenset({"command"}),
    "execute_code": frozenset({"code"}),
}


def _hint_key_component(key: Any) -> str | None:
    """A hint path component safe to advertise verbatim: a plain identifier that
    does not carry the delimiter token (list indices are integers we synthesize);
    ``None`` when the caller must degrade to the generic ``field`` placeholder."""
    if isinstance(key, int) and not isinstance(key, bool):
        return str(key)
    if (
        isinstance(key, str)
        and _HINT_KEY_RE.fullmatch(key)
        and _DELIMITER_TOKEN_RE.search(key) is None
    ):
        return key
    return None


def remember_tool_result(agent, tool_call_id: str, result: Any) -> None:
    """Keep bounded structured results available for later tool arguments."""
    if not tool_call_id:
        return
    try:
        payload = json.loads(result) if isinstance(result, str) else result
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if size > _MAX_TOOL_RESULT_REF_BYTES:
            return
    except (TypeError, ValueError):
        return
    refs = getattr(agent, "_tool_result_refs", None)
    if refs is None:
        refs = agent._tool_result_refs = {}
    refs.pop(tool_call_id, None)  # re-remembering moves the entry to the end (recency)
    refs[tool_call_id] = (payload, size)
    total = sum(entry[1] for entry in refs.values())
    while refs and (len(refs) > _MAX_TOOL_RESULT_REFS or total > _MAX_TOOL_RESULT_REF_TOTAL_BYTES):
        _, dropped_size = refs.pop(next(iter(refs)))
        total -= dropped_size


def tool_result_reference_hint(tool_call_id: str, result: Any) -> str:
    """Return a compact hint for structured results reusable by later tools."""
    if not tool_call_id:
        return ""
    try:
        payload = json.loads(result) if isinstance(result, str) else result
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, (dict, list)):
        return ""
    # Reveal a real path so the model can form a working reference. For an
    # MCP wrapper {"result": ..., "structuredContent": ...} both values may
    # be JSON strings (double-encoded) - prefer structuredContent, which is
    # the machine-oriented branch and the only one that can hold a real
    # container; fall back to result otherwise. Show a path that reaches a
    # real container: dive through string leaves (double-encoding) so the
    # advertised path is usable, not one that lands on a JSON blob string.
    if isinstance(payload, dict) and payload:
        top_key = ("structuredContent" if "structuredContent" in payload
                   else next(iter(payload)))
        safe_top = _hint_key_component(top_key)
        top_value = payload[top_key]
        if isinstance(top_value, str):
            try:
                top_value = json.loads(top_value)
            except (TypeError, ValueError):
                pass
        if safe_top is None:
            # A content-derived top-level key: never advertise it (see _HINT_KEY_RE).
            ref = f"{{{{tool_result:{tool_call_id}.field}}}}"
        # Dive one more level through a double-encoded container so the hint
        # lands on a real structure, e.g. structuredContent.result.folders.
        elif isinstance(top_value, (dict, list)) and top_value:
            if isinstance(top_value, dict):
                inner_key = next(iter(top_value))
                inner_value = top_value[inner_key]
            else:
                inner_key = 0
                inner_value = top_value[0]
            if isinstance(inner_value, str):
                try:
                    inner_value = json.loads(inner_value)
                except (TypeError, ValueError):
                    pass
            safe_inner = _hint_key_component(inner_key)
            if safe_inner is None:
                ref = f"{{{{tool_result:{tool_call_id}.{safe_top}.field}}}}"
            elif isinstance(inner_value, (dict, list)):
                ref = f"{{{{tool_result:{tool_call_id}.{safe_top}.{safe_inner}.field}}}}"
            else:
                ref = f"{{{{tool_result:{tool_call_id}.{safe_top}.{safe_inner}}}}}"
        elif isinstance(top_value, (dict, list)):
            ref = f"{{{{tool_result:{tool_call_id}.{safe_top}.field}}}}"
        else:
            ref = f"{{{{tool_result:{tool_call_id}.{safe_top}}}}}"
    elif isinstance(payload, list) and payload:
        # A top-level list: the first index is a resolvable path.
        ref = f"{{{{tool_result:{tool_call_id}.0}}}}"
    else:
        # Empty containers: `.field` is a replace-me placeholder the model swaps
        # for a real leaf name (an empty payload has no resolvable path to show).
        ref = f"{{{{tool_result:{tool_call_id}.field}}}}"
    # Defense in depth: defang any live delimiter token before the hint is stored.
    # The gate above never advertises a key this would rewrite, so it only ever
    # fires on the call id (the one advertised component the gate does not cover).
    return _neutralize_delimiters(f"\n\n[Reusable tool result: {ref}]")


def fold_tool_result_hint(content: Any, hint: Any, *, part_type: str = "text") -> Any:
    """Fold a sibling-key tool-result hint into model-visible content: append to a string,
    or add a text part to a block list. No-op for an empty hint or other content shapes."""
    if not hint:
        return content
    if isinstance(content, str):
        return content + str(hint)
    if isinstance(content, list):
        return [*content, {"type": part_type, "text": str(hint).strip()}]
    return content


def _serialize_resolved(value: Any) -> str:
    """Serialize a resolved value for embedding in a larger argument string."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _resolved_size_bytes(value: Any) -> int:
    return len(_serialize_resolved(value).encode("utf-8"))


def _resolve_tool_result_ref(agent, value: Any, *, _command_like: bool = False) -> Any:
    if not isinstance(value, str) or "{{tool_result:" not in value:
        return value
    refs = getattr(agent, "_tool_result_refs", {})

    def resolve(match):
        call_id, _, path = match.group(1).partition(".")
        if call_id not in refs:
            return match.group(0)
        current = refs[call_id][0]
        for part in path.split(".") if path else []:
            # MCP tool results are double-encoded: the wrapper stores
            # {"result": "<json-string>", "structuredContent": ...}, so a
            # walk may land on a string containing JSON. Re-parse it once
            # so chained paths like .result.data resolve against the real
            # payload; unresolvable refs pass through untouched (same
            # contract as the walk below).
            if isinstance(current, str):
                try:
                    current = json.loads(current)
                except (TypeError, ValueError):
                    return match.group(0)
            try:
                current = current[int(part)] if isinstance(current, list) else current[part]
            except (KeyError, IndexError, TypeError, ValueError):
                return match.group(0)
        logger.debug(
            "resolved tool-result reference %s at %s (%d bytes)",
            call_id, path or "<whole>", _resolved_size_bytes(current),
        )
        return current

    match = _TOOL_RESULT_REF_RE.fullmatch(value)
    if match is not None:
        # A whole-value reference (the argument is exactly one reference) keeps
        # native JSON types for regular arguments (a tool schema expecting an
        # array/object receives one); command-like arguments stay strings -
        # they are executed verbatim.
        resolved = resolve(match)
        return _serialize_resolved(resolved) if _command_like else resolved
    if _command_like:
        # Embedded reference inside a command-like argument: leave it literal.
        first = _TOOL_RESULT_REF_RE.search(value)
        logger.debug(
            "kept %d embedded tool-result reference(s) literal in a command-like argument (first: %s)",
            len(_TOOL_RESULT_REF_RE.findall(value)), first.group(1) if first else "<none>",
        )
        return value
    return _TOOL_RESULT_REF_RE.sub(lambda match: _serialize_resolved(resolve(match)), value)


def resolve_tool_result_refs(agent, value: Any, *, tool_name: str | None = None, _command_like: bool = False) -> Any:
    """Resolve ``{{tool_result:call_id.field}}`` recursively in tool args.

    ``tool_name`` scopes the command-like guard: arguments listed in
    ``_COMMAND_LIKE_ARGS`` only resolve whole-value references, so embedded
    references there pass through untouched. ``_command_like`` is internal
    recursion state - callers pass ``tool_name`` only.
    """
    if isinstance(value, dict):
        command_keys = _COMMAND_LIKE_ARGS.get(tool_name or "", frozenset())
        return {
            key: resolve_tool_result_refs(
                agent, item, tool_name=tool_name,
                _command_like=_command_like or key in command_keys,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            resolve_tool_result_refs(agent, item, tool_name=tool_name, _command_like=_command_like)
            for item in value
        ]
    return _resolve_tool_result_ref(agent, value, _command_like=_command_like)
