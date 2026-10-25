"""Invariants of the fast-jev-compaction eval arm (evals/compaction/jev_arm.py).

Offline: the fake asker stands in for Jev. The contract under test is the
plugin's own: nothing is rewritten, only tool calls/results go, and no tool
result is ever left without its call (or vice versa).
"""
from __future__ import annotations

from evals.compaction.fixtures import synthetic_transcript, total_tokens
from evals.compaction.jev_arm import JevCompactor, JevOptions, fake_asker, message_text


def _pairs(messages):
    call_ids = {tc["id"] for m in messages for tc in m.get("tool_calls") or []}
    result_ids = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    return call_ids, result_ids


def test_drop_decisions_never_orphan_and_never_rewrite_text():
    msgs = synthetic_transcript(40)
    texts_before = [message_text(m) for m in msgs if m.get("role") != "tool"]
    comp = JevCompactor(asker=fake_asker(keep_call=0.2, keep_result=0.1))
    out = comp.compress(msgs)

    call_ids, result_ids = _pairs(out)
    assert call_ids == result_ids, "a dropped call must take its result with it, and only its result"
    assert total_tokens(out) < total_tokens(msgs)
    # user/assistant text is preserved verbatim and in order (only tool_calls / tool rows change)
    texts_after = [message_text(m) for m in out if m.get("role") != "tool"]
    assert texts_after == [t for t in texts_before if t.strip()]
    # pinned calls (first row / newest rows) are never candidates
    assert comp.stats["pinned"] >= 1 and comp.stats["calls_dropped"] == comp.stats["calls"] - comp.stats["pinned"]


def test_drop_result_keeps_call_and_bounded_head():
    msgs = synthetic_transcript(30)
    comp = JevCompactor(asker=fake_asker(keep_call=0.9, keep_result=0.1), options=JevOptions(truncate_head_chars=50))
    out = comp.compress(msgs)

    call_ids, result_ids = _pairs(out)
    assert call_ids == result_ids and len(out) == len(msgs)
    truncated = [m for m in out if m.get("role") == "tool" and "fast-jev-compaction truncated" in m["content"]]
    assert truncated and all(m["content"].startswith("step output") for m in truncated)
    assert all(len(m["content"]) < 300 for m in truncated)
    assert comp.stats["results_dropped"] == len(truncated)
