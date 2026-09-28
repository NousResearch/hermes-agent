"""kanban_show compression keep: a dispatcher-owned worker's own card survives (#126702)."""
import json

from agent.context_compressor import (
    ContextCompressor,
    _is_kept_own_card,
    _is_summary_stub,
    _summarize_tool_result,
)


def _show(task_id, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "T1")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    card = {"task": {"id": task_id, "title": "own card title", "body": "OWNCARD " + "x" * 6000}}
    return _summarize_tool_result("kanban_show", "{}", json.dumps(card))


def test_own_card_kept(monkeypatch):
    out = _show("T1", monkeypatch)
    assert "own card title" in out and "OWNCARD" in out and len(out) <= 4000
    assert _is_kept_own_card(out)  # the soft demotion readers spare the card via this predicate
    assert not _is_summary_stub(out)  # a kept card is not a 1-line stub: distinct shapes


def test_other_card_and_junk_stay_stubs(monkeypatch):
    assert "OWNCARD" not in _show("T2", monkeypatch)
    assert "chars" in _summarize_tool_result("kanban_show", "{}", "not json " + "y" * 5000)


def test_own_card_soft_keep_and_pressure_escape(monkeypatch):
    """The keep holds through the ordinary demotion pass but is not absolute: the
    pressure pass (keep_own_card=False) must still be able to re-stub it, or a
    huge kept card would wedge the hard budget backstop (#32106 rule)."""
    call_map = {"c1": ("kanban_show", "{}")}
    out = _show("T1", monkeypatch)

    result = [{"role": "tool", "tool_call_id": "c1", "content": out}]
    assert ContextCompressor._demote_tool_result_at(result, 0, call_map, 200) is False
    assert _is_kept_own_card(result[0]["content"])

    result2 = [{"role": "tool", "tool_call_id": "c1", "content": out}]
    assert ContextCompressor._demote_tool_result_at(
        result2, 0, call_map, 200, keep_own_card=False,
    ) is True
    assert not _is_kept_own_card(result2[0]["content"])
