"""Lean stale-tail demotion: a demoted ``skill_view`` body keeps its reload marker (#32106)."""
import json

import pytest

import agent.context_compressor as cc
from agent.context_compressor_tail import _LEAN_TAIL_KEEP_TOOL_ROUNDS

SKILL = "synthetic-tail-skill"
BODY = "Skill instructions " + "s" * 2000


def _call(call_id, name, args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _round(call_id, name, args, content, tool_name=True):
    tool = {"role": "tool", "tool_call_id": call_id, "content": content}
    if tool_name:
        tool["tool_name"] = name
    return [{"role": "assistant", "tool_calls": [_call(call_id, name, args)]}, tool]


def _fresh_rounds(n=_LEAN_TAIL_KEEP_TOOL_ROUNDS):
    rows = []
    for i in range(n):
        rows += _round(f"fresh-{i}", "terminal", {"command": f"echo {i}"}, f"Step {i} done.")
    return rows


def _stub_for(rows, call_id):
    return next(r["content"] for r in rows if r.get("role") == "tool" and r.get("tool_call_id") == call_id)


@pytest.fixture
def compressor(monkeypatch):
    monkeypatch.setattr(cc, "get_model_context_length", lambda *a, **k: 100_000)
    monkeypatch.setattr(cc, "call_llm", lambda **kw: {"choices": [{"message": {
        "content": "## Goal\nSynthetic work is recorded."}, "finish_reason": "stop"}]})
    comp = cc.ContextCompressor(model="test/model", config_context_length=100_000, protect_first_n=0,
                                protect_last_n=3, quiet_mode=True, tail_mode="lean")
    comp.tail_token_budget = 2000
    return comp


def test_reload_marker_survives_the_next_compaction(compressor):
    # With a session id the marked stub passes the Phase-1 prune floor (200 chars); it must not be
    # re-summarised into a plain "[skill_view] ..." line that drops the marker.
    compressor.bind_session_state(session_id="0123abcd-4567-89ef-0123-456789abcdef")
    rows = [
        {"role": "system", "content": "Synthetic test conversation."},
        {"role": "user", "content": "Summarize the completed earlier work."},
        {"role": "assistant", "content": "Old notes " + "x" * 40_000},
        *_round("older-terminal", "terminal", {"command": "old-job"}, "Earlier output " + "o" * 8000),
        {"role": "user", "content": "Continue the current work."},
        *_round("skill-1", "skill_view", {"name": SKILL}, BODY),
        *_fresh_rounds(),
        {"role": "assistant", "content": "All rounds completed."},
    ]
    first = compressor.compress(rows + [{"role": "user", "content": "Report the current work."}], force=True)
    stub = _stub_for(first, "skill-1")
    assert len(stub) > cc._PRUNE_MIN_CHARS and cc._extract_pruned_skill_names(stub) == [SKILL]

    later = [{"role": "assistant", "content": "Reported."}, {"role": "user", "content": "Next task."}]
    for i in range(8):
        # Enough newer output that the stub falls outside Phase-1 tail protection (tail_token_budget).
        later += _round(f"later-{i}", "terminal", {"command": f"echo later {i}"}, f"Later step {i} " + "l" * 1400)
    second = compressor.compress(first + later + [{"role": "assistant", "content": "Done."}], force=True)

    assert compressor.compression_count == 2 and not compressor._last_summary_fallback_used
    text = "\n".join(str(r.get("content") or "") for r in second)
    assert SKILL in cc._extract_pruned_skill_names(text), "reload marker lost at the next compaction"
