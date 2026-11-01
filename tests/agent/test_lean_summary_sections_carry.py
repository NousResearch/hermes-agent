"""#136540: lean compaction's deterministic summary sections must carry across compactions.

``ContextCompressor._augment_summary_lean`` appends the Anchor Index / verbatim user quotes /
recovery footer only when the heading is missing from the summary it is handed. The summary it is
handed is either the previous augmented summary echoed back by an obedient summariser (the prompt
says PRESERVE all existing information, and PREVIOUS SUMMARY already carries the sections) or a
rewrite that dropped them. The first case blocks the new region's refresh entirely — its real user
words survive only as the model's paraphrase; the second drops the earlier region's quoted words.
Both are pinned here through the public ``compress`` entry point, on the axis aux summarisers
actually differ by: echoing the previous summary, or rewriting it.
"""
from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _LEAN_ANCHOR_HEADING,
    _LEAN_USER_MESSAGES_HEADING,
)

USER_A = "ALPHA-NEEDLE keep the alpha access code 1111 in src/alpha/access_code.py"
USER_B = "BRAVO-NEEDLE and the bravo access code is 2222 in src/bravo/limits.py"
NARRATIVE = "## Completed Actions\n1. prior work\n\n## Active State\nidle\n"


def _make_compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=8000):
        return ContextCompressor(
            model="test-model", threshold_percent=0.75, protect_first_n=3, protect_last_n=4,
            quiet_mode=True, config_context_length=8000,
        )


def _rounds(first: int, last: int, size: int = 600) -> list[dict]:
    """Tool rounds sized so the recent-tail window cannot swallow the whole transcript."""
    out: list[dict] = []
    for i in range(first, last):
        out.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"row {i} " + "y" * size})
    return out


def _first_transcript(user_a: str) -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "kick off"},
        {"role": "assistant", "content": "ok"},
        *_rounds(0, 4),
        {"role": "user", "content": user_a},
        {"role": "assistant", "content": "noted"},
        *_rounds(4, 18),
        {"role": "user", "content": "tail ask"},
        {"role": "assistant", "content": "tail answer"},
        *_rounds(900, 903),
    ]


def _second_transcript(compacted: list[dict]) -> list[dict]:
    return [
        *[dict(m) for m in compacted],
        *_rounds(20, 26),
        {"role": "user", "content": USER_B},
        {"role": "assistant", "content": "noted"},
        *_rounds(26, 40),
        {"role": "user", "content": "second tail ask"},
        {"role": "assistant", "content": "tail2"},
        *_rounds(990, 993),
    ]


def _verbatim_section(summary: str) -> str:
    """The body of the verbatim-user section, or "" when the summary carries no such section."""
    if _LEAN_USER_MESSAGES_HEADING not in summary:
        return ""
    return summary.split(_LEAN_USER_MESSAGES_HEADING, 1)[1].split("(Every real user message", 1)[0]


def _anchor_section(summary: str) -> str:
    """The body of the anchor-index section, or "" when the summary carries no such section."""
    if _LEAN_ANCHOR_HEADING not in summary:
        return ""
    return summary.split(_LEAN_ANCHOR_HEADING, 1)[1].split("(Exact identifiers from the compacted region", 1)[0]


def _compact_twice(echoes_previous: bool) -> tuple[str, str]:
    """Two consecutive compactions on one compressor; returns (summary1, summary2).

    ``echoes_previous`` models the obedient summariser the prompt asks for (it hands back the
    previous summary, sections included); False models one that rewrites and drops them.
    """
    comp = _make_compressor()

    def fake_summary_call(prompt: str, started_at: float) -> str:
        if echoes_previous and comp._previous_summary:
            return comp._previous_summary
        return NARRATIVE

    with patch.object(comp, "_call_summary_llm", fake_summary_call):
        first = comp.compress(_first_transcript(USER_A), current_tokens=7000, force=True)
        summary1 = comp._previous_summary or ""
        assert USER_A in _verbatim_section(summary1), "fixture: compaction 1 must quote the first region"
        assert "src/alpha/access_code.py" in _anchor_section(summary1), "fixture: compaction 1 must index it"
        comp.compress(_second_transcript(first), current_tokens=7000, force=True)
        return summary1, comp._previous_summary or ""


def test_obedient_summariser_still_quotes_the_new_region():
    """A summariser that preserves the previous summary must not block this region's quotes."""
    _, summary2 = _compact_twice(echoes_previous=True)

    assert NARRATIVE.strip() in summary2, "the model's narrative survives"
    assert summary2.count(_LEAN_USER_MESSAGES_HEADING) == 1, "one verbatim section, never two"
    section = _verbatim_section(summary2)
    assert USER_B in section, "the newest compacted region's user words are quoted verbatim"
    assert USER_A in section, "and the earlier region's quotes are still carried"
    assert summary2.count(_LEAN_ANCHOR_HEADING) == 1, "one anchor index, never two"
    anchors = _anchor_section(summary2)
    assert "src/bravo/limits.py" in anchors, "the newest region's identifiers are harvested"
    assert "src/alpha/access_code.py" in anchors, "and the earlier region's identifiers are carried"


def test_rewriting_summariser_keeps_earlier_quotes():
    """A summariser that drops the deterministic sections must not drop the earlier quotes."""
    _, summary2 = _compact_twice(echoes_previous=False)

    assert summary2.count(_LEAN_USER_MESSAGES_HEADING) == 1, "one verbatim section, never two"
    section = _verbatim_section(summary2)
    assert USER_B in section, "the newest compacted region's user words are quoted verbatim"
    assert USER_A in section, "the earlier region's quotes are re-added, not lost with the rewrite"
    assert summary2.count(_LEAN_ANCHOR_HEADING) == 1, "one anchor index, never two"
    anchors = _anchor_section(summary2)
    assert "src/bravo/limits.py" in anchors, "the newest region's identifiers are harvested"
    assert "src/alpha/access_code.py" in anchors, "and the earlier region's identifiers are re-added"


def test_single_compaction_quotes_a_user_message():
    """Control: the fixture's section is real, not an artifact of the second compaction."""
    comp = _make_compressor()
    with patch.object(comp, "_call_summary_llm", lambda prompt, started_at: NARRATIVE):
        comp.compress(_first_transcript(USER_A), current_tokens=7000, force=True)

    section = _verbatim_section(comp._previous_summary or "")
    assert USER_A in section
