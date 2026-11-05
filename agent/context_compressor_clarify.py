"""Clarify results in compaction: the answer projection and the passes that must keep it (#124169).

A clarify result holds the user's decisions, not disposable tool output. ``_sum_clarify`` projects
it to its answers (legacy runtime notices filtered, raw prefixed text never trusted); the prune,
lean-tail and salvage passes keep that projection instead of a stub, and the lean summary quotes
the answers next to the user's typed messages. Split out of :mod:`agent.context_compressor`
(FILE_LINES ratchet); facade symbols are late-imported inside functions, never at module level.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.compression_marker import elide


# Runtime notices that clarify producers persisted as user_response before per-response
# status existed. Only status-less (legacy) entries are checked; explicit status wins.
_CLARIFY_NON_RESPONSE_PREFIXES = (
    "The user did not provide a response",  # tools/clarify_tool.TIMEOUT_RESPONSE
    "[user did not respond",  # gateway clarify delivery timeout
    "[clarify prompt could not be delivered",  # gateway UNDELIVERED*
    "[oneshot mode:",  # hermes_cli/oneshot.py
)
# Matched whole: as a prefix these would also swallow real answers that start the same way.
_CLARIFY_NON_RESPONSE_TEXTS = (
    "The user cancelled. Use your best judgement to proceed.",  # classic CLI Ctrl+C
)


# Historical callbacks used repr(question) and repr(list[str]). Quoted fields
# must close before a suffix counts: user text can contain the suffix itself.
_CLARIFY_QUOTED_TEXT = r"(?:'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\")"
_CLARIFY_HEADLESS_NOTICE = re.compile(
    r"\[(?:oneshot mode: no user available\. |"
    rf"single-query mode: no user available to answer {_CLARIFY_QUOTED_TEXT}\. )"
    r"(?:Make the most reasonable assumption you can and continue\.|"
    rf"Pick the best (?:option|subset) from \[{_CLARIFY_QUOTED_TEXT}"
    rf"(?:,\s*{_CLARIFY_QUOTED_TEXT})*\] using your own judgment and continue\.)\]",
    re.DOTALL,
)


def _is_clarify_non_response(item) -> bool:
    if not isinstance(item, str):
        return False
    text = item.strip()
    return (text in _CLARIFY_NON_RESPONSE_TEXTS
            or text.startswith(_CLARIFY_NON_RESPONSE_PREFIXES)
            or _CLARIFY_HEADLESS_NOTICE.fullmatch(text) is not None)


def _filter_legacy_clarify_answers(values):
    """Remove complete notices, including contiguous comma-normalized fragments."""
    answers = []
    index = 0
    while index < len(values):
        item = values[index]
        if isinstance(item, str) and item.strip().startswith((
            "[oneshot mode: no user available.",
            "[single-query mode: no user available to answer ",
        )):
            text = ""
            notice_end = None
            for end in range(index, len(values)):
                if not isinstance(values[end], str):
                    break
                text += ("," if end > index else "") + values[end].strip()
                if _CLARIFY_HEADLESS_NOTICE.fullmatch(text):
                    notice_end = end
                    break
            if notice_end is not None:
                index = notice_end + 1
                continue
        if not _is_clarify_non_response(item):
            answers.append(item)
        index += 1
    return answers


CLARIFY_RESPONSE_PREFIX = "[clarify] user responded: "
# A clarify answer is user input, so each one gets the bound of a quoted user message
# (``_LEAN_USER_MESSAGE_MAX_CHARS``), not the 199-char stub cap that cut batch answers down to a
# fragment of the first (#124169).
CLARIFY_ANSWER_MAX_CHARS = 4_000


def clarify_projection_answers(content: Any) -> list[str] | None:
    """The answers in a ``_sum_clarify`` projection, or None when ``content`` is not one.

    Answers are elided inside the JSON, so a projection always parses; prune passes use this
    to leave it alone instead of re-summarizing it to "asked user a question". Callers decide
    by tool identity (the clarify call id), never by this prefix alone."""
    if not isinstance(content, str) or not content.startswith(CLARIFY_RESPONSE_PREFIX):
        return None
    try:
        value = json.loads(content[len(CLARIFY_RESPONSE_PREFIX):])
    except ValueError:
        return None
    values = [value] if isinstance(value, str) else value
    if isinstance(values, list) and values and all(isinstance(v, str) and v for v in values):
        return values
    return None


def clarify_compact_form(content: Any) -> str | None:
    """What a clarify result keeps when tool output is cleared: ``content`` when it is already a
    projection, else its answer projection; None when it holds no answer worth keeping."""
    if clarify_projection_answers(content) is not None:
        return content
    if not isinstance(content, str):
        return None
    projection = _sum_clarify("clarify", {}, content, len(content), 0)
    return projection if clarify_projection_answers(projection) is not None else None


def result_tool_name(msg: dict[str, Any], call_id_to_tool: dict[str, tuple]) -> str:
    """Name of the tool that produced a tool-result row, from its owning call."""
    call = call_id_to_tool.get(str(msg.get("tool_call_id") or ""))
    return str(call[0]) if call else str(msg.get("tool_name") or "")


def _sum_clarify(name, args, content, content_len, line_count):
    from agent.context_compressor import _json_dict

    payload = _json_dict(content)
    responses = payload.get("responses")
    if not isinstance(responses, list) and "user_response" in payload:
        # Clarify results written before per-response status was introduced used
        # the response fields directly on the top-level object.
        responses = [payload]
    answers: list = []
    for entry in responses if isinstance(responses, list) else ():
        if isinstance(entry, dict) and (
            entry.get("status") == "answered"
            or ("status" not in entry and "user_response" in entry)
        ):
            value = entry.get("user_response")
            values = value if isinstance(value, list) else [value]
            # Pre-status sessions also stored timeout/delivery notices as user_response.
            # Only those legacy records need the old sentinel check; explicit status wins.
            # Drop just the notice items so other selections in the same entry survive.
            if "status" not in entry:
                # Scan complete envelopes, not the whole list: genuine selections
                # can surround a notice split by the old multi-select normalizer.
                values = _filter_legacy_clarify_answers(values)
            answers.extend(values)
    # Every answer survives; each is bounded like a quoted user message, elided inside the JSON
    # so the projection still parses (see clarify_projection_answers).
    answers = [elide(answer, CLARIFY_ANSWER_MAX_CHARS) for answer in answers if isinstance(answer, str) and answer]
    if not answers:
        return "[clarify] asked user a question"
    # Escape lone UTF-16 surrogates so the message stays UTF-8/SQLite safe.
    serialized = json.dumps(answers[0] if len(answers) == 1 else answers,
                            ensure_ascii=False).encode("utf-8", errors="backslashreplace")
    return CLARIFY_RESPONSE_PREFIX + serialized.decode("utf-8")

def clarify_answer_quote(msg: dict[str, Any], call_id_to_tool: dict[str, tuple]) -> str | None:
    """A clarify result's answers as the lean summary's verbatim user section quotes them; None for
    any other row. Answers given through clarify are the user's words too, but as tool output they
    were left to the summarizer's paraphrase (#124169)."""
    if msg.get("role") != "tool" or result_tool_name(msg, call_id_to_tool) != "clarify":
        return None
    kept = clarify_compact_form(msg.get("content"))
    answers = clarify_projection_answers(kept) if kept is not None else None
    if not answers:
        return None
    return "[answered via clarify] " + (answers[0] if len(answers) == 1 else "\n".join(f"- {a}" for a in answers))
