"""Refusal detection for compaction summaries: a provider refusal must never replace the compacted turns."""

from __future__ import annotations

import re
from typing import Any

from agent.auxiliary_client import _coerce_llm_message, _message_field

# A provider can return a natural-language refusal with finish_reason="stop". It is
# non-empty, so the usual response validation accepts it, but it contains none of
# the checkpoint needed to safely replace the compacted turns. Keep this narrow:
# a real summary may mention a refusal in a recorded turn, while a refusal as the
# whole response begins with one of these phrases and refers to the requested
# summary/checkpoint.
_SUMMARY_REFUSAL_PREFIX_RE = re.compile(
    r"^\s*(?:(?:sorry|i(?:['’]m| am)\s+sorry|i\s+apologi[sz]e|as\s+an\s+ai)"
    r"\s*[,;:]?\s*(?:but\s+)?)?(?:i|we)\s+"
    r"(?:can(?:\s*not|['’]t)|could\s*not|couldn['’]t|won['’]t|will\s+not|must\s+decline|"
    r"refuse\s+to|am\s+unable\s+to|am\s+not\s+able\s+to)\b"
    r"|^\s*(?:i['’]?m|i\s+am)\s+(?:unable|not\s+able)\b",
    re.IGNORECASE,
)


def _is_summary_refusal(content: str) -> bool:
    """Return whether a complete response is a refusal instead of a summary."""
    normalized = " ".join(content.split())
    if not _SUMMARY_REFUSAL_PREFIX_RE.match(normalized):
        return False
    # A refusal-only body never carries the template's "## " section headings; a real summary
    # that merely opens with a hedging preamble ("I cannot see earlier turns, but here is...") does.
    if re.search(r"(?m)^##\s", content):
        return False
    # Limit the search to the opener so a structured checkpoint that records a
    # historical refusal elsewhere is not rejected. Stems catch summary/summarize/summarise.
    return any(term in normalized[:400].casefold() for term in ("summar", "checkpoint"))


def _response_refusal_text(response: Any) -> str:
    """Explicit provider ``choices[0].message.refusal`` (str, or dict with message/reason/text); ``""`` when absent.

    OpenAI-style structured-output refusals put the refusal here and leave ``content`` as filler or
    empty, so the prose detector never sees it.
    """
    refusal = _message_field(_coerce_llm_message(response), "refusal")
    if isinstance(refusal, dict):
        refusal = refusal.get("message") or refusal.get("reason") or refusal.get("text")
    return refusal.strip() if isinstance(refusal, str) else ""


def _is_refusal_response(response: Any, content: str) -> bool:
    """Single refusal predicate for both summarizer paths.

    An explicit provider ``message.refusal`` wins even when ``content`` looks like a
    summary; otherwise fall back to the prose detector on the extracted content.
    """
    return bool(_response_refusal_text(response)) or _is_summary_refusal(content)
