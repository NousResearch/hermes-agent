"""After compaction the first non-system message must be a user turn (#141569).

Once ``protect_first_n`` has decayed (any compaction after the first), the protected head is the
system prompt alone, so the summary is forced to ``role="user"`` and must open the transcript
(#52160). When the token-budget tail opened with an assistant(tool_calls)+tool group followed by a
user turn, the summary was merged into that user row (the template-visible collision repair) and
the tool group stayed in front of it: ``[system, assistant(tool_calls), tool, user(summary), ...]``.
Providers that require the first non-system message to be ``user`` then reject every later request,
and the stored transcript replays the same shape. The leading tool flow is folded into the
summarized window instead, so the summary carrier leads and every tool result still follows its call.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    ContextCompressor,
    _template_visible_role,
)


def _tool_turn(i: int) -> list[dict]:
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "content": "x" * 300, "tool_call_id": f"c{i}"},
    ]


def _transcript(tail_tool_turns: int) -> list[dict]:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first ask " + "q" * 200},
        {"role": "assistant", "content": "ok " + "a" * 200},
    ]
    for i in range(15):
        messages += _tool_turn(i)
    messages += [{"role": "user", "content": "later ask " + "u" * 100}, {"role": "assistant", "content": "r" * 100}]
    for i in range(15, 15 + tail_tool_turns):
        messages += _tool_turn(i)
    return messages + [{"role": "user", "content": "latest ask"}, {"role": "assistant", "content": "done"}]


def _compress(messages: list[dict], *, budget: int, compression_count: int, protect_first_n: int = 3) -> list[dict]:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(model="test/model", threshold_percent=0.5, protect_first_n=protect_first_n,
                              protect_last_n=20, quiet_mode=True)
    c.tail_token_budget = budget
    c.compression_count = compression_count
    with patch.object(c, "_generate_summary", return_value=f"{SUMMARY_PREFIX}\nrolled-up summary"):
        return c.compress(messages, current_tokens=90_000)


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] + ("(tool_calls)" if m.get("tool_calls") else "") for m in messages]


def _assert_valid_shape(out: list[dict]) -> None:
    body = [m for m in out if m.get("role") != "system"]
    assert body and body[0]["role"] == "user", f"first non-system row is not a user turn: {_roles(out)}"
    summary_rows = [i for i, m in enumerate(body) if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert summary_rows and summary_rows[0] == 0, f"summary does not lead the transcript: {_roles(out)}"
    seen_calls: set[str] = set()
    for m in body:
        seen_calls.update(tc["id"] for tc in m.get("tool_calls") or [])
        if m["role"] == "tool":
            assert m["tool_call_id"] in seen_calls, f"tool result ahead of its call: {_roles(out)}"
    # Mistral-family alternation over template-visible roles must still hold.
    expect = "user"
    for role in filter(None, map(_template_visible_role, body)):
        assert role == expect, f"template-visible alternation broken: {_roles(out)}"
        expect = "assistant" if expect == "user" else "user"


@pytest.mark.parametrize("tail_tool_turns", [1, 2])
@pytest.mark.parametrize("budget", [40, 200])
def test_decayed_head_compaction_opens_with_user_turn(tail_tool_turns, budget):
    _assert_valid_shape(_compress(_transcript(tail_tool_turns), budget=budget, compression_count=1))


def test_system_only_head_on_first_compaction_opens_with_user_turn():
    _assert_valid_shape(_compress(_transcript(1), budget=40, compression_count=0, protect_first_n=0))


def test_folded_tool_group_reaches_the_summarizer():
    """The leading tool group is summarized, not silently dropped."""
    messages = _transcript(1)
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        c = ContextCompressor(model="test/model", threshold_percent=0.5, protect_first_n=3,
                              protect_last_n=20, quiet_mode=True)
    c.tail_token_budget = 40
    c.compression_count = 1
    seen: list[list[dict]] = []

    def fake_summary(turns, *args, **kwargs):
        seen.append(list(turns))
        return f"{SUMMARY_PREFIX}\nrolled-up summary"

    with patch.object(c, "_generate_summary", side_effect=fake_summary):
        out = c.compress(messages, current_tokens=90_000)
    kept_calls = {tc["id"] for m in out for tc in m.get("tool_calls") or []}
    summarized_calls = {tc["id"] for t in seen[0] for tc in t.get("tool_calls") or []}
    every_call = {f"c{i}" for i in range(16)}
    assert every_call <= kept_calls | summarized_calls


def test_protected_head_with_user_turn_is_unchanged():
    """A head that still holds a user turn keeps the existing placement (no forced-leading case)."""
    out = _compress(_transcript(1), budget=40, compression_count=0)
    assert [m["role"] for m in out[:3]] == ["system", "user", "assistant"]
