"""Bounded work evidence for the goal judge.

A goal's judge only accepts DONE when it can see proof, and the proof lives in the turn's tool
messages — a file's new content, a command result — not in the assistant's closing prose. This
module turns a ``run_conversation`` result into a few bounded evidence lines and appends them to
the judge prompt. Split out of ``hermes_cli/goals.py`` so that facade stays under its line cap;
``_truncate`` lives here because evidence shaping is its main caller.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


# Evidence cap: keep the cumulative work artifacts bounded so a long goal's full tool history can't
# blow the judge context (the judge already sees a 4000-char response snippet; evidence is additive).
_JUDGE_EVIDENCE_MAX_ITEMS = 5
_JUDGE_EVIDENCE_MAX_CHARS = 1200
_JUDGE_EVIDENCE_ARGS_MAX_CHARS = 700
_JUDGE_EVIDENCE_RESULT_MAX_CHARS = 400

JUDGE_EVIDENCE_BLOCK_TEMPLATE = (
    "Concrete evidence from the agent's recent work (tool results, file "
    "excerpts, output lines). Use this to VERIFY completion instead of relying "
    "only on the response's assertions:\n{evidence_lines}\n\n"
)


def _append_evidence_to_judge_prompt(prompt: str, evidence: List[str]) -> str:
    """Append a bounded, labelled evidence block to a judge user prompt.

    Only the most recent ``_JUDGE_EVIDENCE_MAX_ITEMS`` items are kept, each truncated to
    ``_JUDGE_EVIDENCE_MAX_CHARS``, so the block stays small and the prompt shape is unchanged
    (same templates + one append) when evidence is absent.
    """
    cleaned = []
    for item in evidence:
        if not isinstance(item, str) or not item.strip():
            continue
        cleaned.append(item.strip())
        if len(cleaned) >= _JUDGE_EVIDENCE_MAX_ITEMS:
            break
    if not cleaned:
        return prompt
    suffix = "… [truncated]"
    def _bounded_line(line: str) -> str:
        if len(line) <= _JUDGE_EVIDENCE_MAX_CHARS:
            return line
        return line[:_JUDGE_EVIDENCE_MAX_CHARS - len(suffix)] + suffix

    lines = "\n".join(f"- {_bounded_line(line)}" for line in cleaned)
    return prompt + JUDGE_EVIDENCE_BLOCK_TEMPLATE.format(evidence_lines=lines)


def _evidence_text(value: Any) -> str:
    """Compact string form of a tool call argument or result for the judge."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _unquote_json_leaves(value: Any) -> Any:
    """Recursively parse string leaves that are themselves JSON so nested content reads clearly.

    write_file's ``content`` argument is JSON *as a string*; dumping the args dict escapes it into
    noise (``{\\"count\\": 25}``) the strict judge can't easily verify. When a string value parses
    as an object/array, embed the parsed value so the concrete content (e.g. ``{"count": 25}``)
    reaches the judge readably. Non-JSON strings are left untouched.
    """
    if isinstance(value, str):
        s = value.strip()
        if s and s[0] in "[{":
            try:
                parsed = json.loads(s)
            except (ValueError, TypeError):
                return value
            if isinstance(parsed, (dict, list)):
                return _unquote_json_leaves(parsed)
            return value
        return value
    if isinstance(value, dict):
        return {k: _unquote_json_leaves(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unquote_json_leaves(v) for v in value]
    return value


def _coerce_args(raw: Any) -> str:
    """Compact, readable display string for a tool call's ``arguments``."""
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        return _evidence_text(_unquote_json_leaves(raw))
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ""
        try:
            return _evidence_text(_unquote_json_leaves(json.loads(s)))
        except (ValueError, TypeError):
            return s
    return _evidence_text(raw)


def extract_turn_evidence(result: Any, *, max_items: int = _JUDGE_EVIDENCE_MAX_ITEMS) -> List[str]:
    """Pull the concrete work artifacts from a ``run_conversation`` result for the judge.

    The judge needs the *evidence* a goal was completed — a file's new content, an output line, a
    command result — which lives in the turn's tool messages, not the final assistant prose. For
    write_file the concrete proof is the tool CALL argument (path + content); the tool RESULT is
    only a receipt (``bytes_written`` / ``verified``). This walks the assistant's ``messages``
    newest-first and emits evidence from PAIRS — each tool call's name + arguments against its
    result message — so the content, not the claim receipt, reaches the judge. Always bounded
    (``max_items``, each truncated by ``_append_evidence_to_judge_prompt``), newest-first,
    non-empty only when proof exists; user pings are never evidence.
    """
    if not isinstance(result, dict):
        return []
    messages = result.get("messages")
    if not isinstance(messages, list):
        return []

    # Map tool_call_id -> (name, arguments) from assistant tool_calls so each tool result can be
    # paired with the concrete call that produced it. Transports rarely emit id-less calls.
    call_by_id: Dict[str, Tuple[str, str]] = {}
    call_by_name: Dict[str, Tuple[str, str]] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(fn.get("name") or tc.get("name") or "")
            args = _coerce_args(fn.get("arguments") or tc.get("arguments"))
            cid = str(tc.get("id") or "")
            if cid:
                call_by_id[cid] = (name, args)
            elif name:
                call_by_name.setdefault(name, (name, args))

    evidence: List[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "tool":
            tag = str(msg.get("tool_name") or msg.get("name") or "")
            cid = str(msg.get("tool_call_id") or "")
            name, args = call_by_id.get(cid) or call_by_name.get(tag) or (tag, "")
            # Preserve both the attempted work AND its outcome.  A large write payload must
            # never crowd out a denial/error receipt that says it did not actually happen.
            args = _truncate(args, _JUDGE_EVIDENCE_ARGS_MAX_CHARS)
            result_txt = _truncate(_evidence_text(content), _JUDGE_EVIDENCE_RESULT_MAX_CHARS)
            if args:
                block = f"{name or tag}: {args}"
                # Append the result only when it adds beyond the call args (e.g. a receipt).
                if result_txt and result_txt != "{}":
                    block = f"{block}  ⇒  {result_txt}"
            else:
                block = f"{name or tag}: {result_txt}" if name or tag else result_txt
            if block.strip():
                evidence.append(block.strip())
        elif role == "assistant" and not msg.get("tool_calls") and isinstance(content, str) and content.strip():
            # Cap assistant additions so the final prose doesn't crowd out the tool proof.
            if len(evidence) == 0:
                continue
            evidence.append(f"response: {content.strip()}")
        if len(evidence) >= max_items:
            break
    return evidence


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "… [truncated]"
