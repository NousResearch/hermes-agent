"""Recover tool calls a model emitted as TEXT into structured calls.

Some local models — notably ``qwen3-coder`` served via Ollama — sometimes emit
a tool call inside the assistant *content* instead of via the structured
``tool_calls`` field. Two forms are seen in the wild:

  * Qwen-Agent XML::

        <function=write_file>
        <parameter=path>hello.py</parameter>
        <parameter=content>print('hi')</parameter>
        </function>

  * Hermes / Nous JSON-in-tag (documented in the agent's own system prompt)::

        <tool_call>
        {"name": "read_file", "arguments": {"path": "a.txt"}}
        </tool_call>

When the provider's chat template fails to parse these back into structured
``tool_calls``, the call lands in ``content`` and the turn silently ends as a
text response — the intended edit/command is dropped. ``strip_think_blocks``
then *deletes* the ``<tool_call>`` variant outright, so the action vanishes
without a trace. This module parses such text into structured calls so the
normal dispatch path runs.

The recovery is gated narrowly: a call is only recovered when its name is in
the caller-supplied set of known tools, so model prose and examples (e.g.
"I'll use the write_file tool") are never executed.
"""
from __future__ import annotations

import ast
import json
import re
from types import SimpleNamespace
from typing import Iterable, List, Tuple

# <function=NAME> ... </function>  — Qwen-Agent XML. NAME is a bare identifier
# (NOT the ``<function name="...">`` attribute form, which prose can contain).
_FUNC_BLOCK_RE = re.compile(
    r"<function\s*=\s*(?P<name>[A-Za-z_][\w.\-]*)\s*>(?P<body>.*?)</function\s*>",
    re.DOTALL | re.IGNORECASE,
)
# <parameter=KEY> VALUE </parameter>
_PARAM_RE = re.compile(
    r"<parameter\s*=\s*(?P<key>[A-Za-z_][\w.\-]*)\s*>(?P<val>.*?)</parameter\s*>",
    re.DOTALL | re.IGNORECASE,
)
# <tool_call> { ...object... } </tool_call>  — JSON or python-dict literal.
_TOOLCALL_JSON_RE = re.compile(
    r"<tool_call>\s*(?P<obj>\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
# Stray tool-call wrapper tags left dangling after a <function> block is
# consumed (the Qwen-Agent leak wraps the function block in a bare closing
# </tool_call>). Swept from the cleaned text only.
_STRAY_WRAPPER_RE = re.compile(r"</?tool_call>\s*", re.IGNORECASE)


def _parse_obj(raw: str):
    """Parse a JSON object, falling back to a python-dict literal.

    Returns a ``dict`` or ``None``. The single-quoted python-dict form is the
    shape the agent's own system prompt documents, so it is common from local
    models that echo the example.
    """
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = ast.literal_eval(raw)
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _make_call(name: str, arguments, index: int) -> SimpleNamespace:
    """Build a tool_call shaped like an OpenAI ChatCompletion tool_call object.

    Downstream dispatch reads ``.id``/``.call_id``/``.type``/
    ``.function.name``/``.function.arguments`` (and ``getattr``-guards the
    rest), so this mirrors that surface exactly.
    """
    if isinstance(arguments, str):
        args_str = arguments
    else:
        args_str = json.dumps(arguments, ensure_ascii=False)
    call_id = f"recovered_call_{index + 1}"
    return SimpleNamespace(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=SimpleNamespace(name=name, arguments=args_str),
    )


def recover_tool_calls_from_text(
    text, valid_tool_names: Iterable[str]
) -> Tuple[List[SimpleNamespace], str]:
    """Recover text-embedded tool calls into structured call objects.

    Args:
        text: assistant content that may contain a leaked tool call.
        valid_tool_names: the known tool names; a call is recovered only when
            its name is in this set (prevents executing prose/examples).

    Returns:
        ``(recovered_calls, cleaned_text)`` where ``recovered_calls`` is a list
        of OpenAI-tool_call-shaped ``SimpleNamespace`` objects (empty when
        nothing recoverable) and ``cleaned_text`` is ``text`` with the consumed
        tool-call spans removed. When nothing is recovered, ``text`` is
        returned unchanged.
    """
    if not isinstance(text, str) or not text.strip():
        return [], text or ""

    valid = set(valid_tool_names or ())
    recovered: List[SimpleNamespace] = []
    spans: List[Tuple[int, int]] = []

    # 1. Qwen-Agent XML: <function=NAME> <parameter=K>V</parameter> </function>
    for m in _FUNC_BLOCK_RE.finditer(text):
        name = m.group("name").strip()
        if name not in valid:
            continue
        args = {}
        for pm in _PARAM_RE.finditer(m.group("body")):
            args[pm.group("key").strip()] = pm.group("val").strip()
        recovered.append(_make_call(name, args, len(recovered)))
        spans.append((m.start(), m.end()))

    # 2. JSON-in-tag: <tool_call>{ "name": ..., "arguments": ... }</tool_call>.
    #    Only when the XML form produced nothing, so a model that emits BOTH a
    #    <function> block and its <tool_call> JSON echo isn't double-counted.
    if not recovered:
        for m in _TOOLCALL_JSON_RE.finditer(text):
            obj = _parse_obj(m.group("obj"))
            if not obj:
                continue
            name = obj.get("name")
            if not isinstance(name, str) or name.strip() not in valid:
                continue
            recovered.append(_make_call(name.strip(), obj.get("arguments", {}), len(recovered)))
            spans.append((m.start(), m.end()))

    if not spans:
        return [], text

    # Excise the consumed spans, keeping any surrounding prose intact.
    spans.sort()
    pieces: List[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    cleaned = "".join(pieces)
    # Sweep the dangling </tool_call> (or stray <tool_call>) that wrapped a
    # recovered <function> block but wasn't part of its matched span.
    cleaned = _STRAY_WRAPPER_RE.sub("", cleaned).strip()
    return recovered, cleaned
