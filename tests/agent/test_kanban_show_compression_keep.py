"""kanban_show compression keep: a dispatcher-owned worker's own card survives (#126702)."""
import json

from agent.context_compressor import _is_summary_stub, _summarize_tool_result


def _show(task_id, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "T1")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    card = {"task": {"id": task_id, "title": "own card title", "body": "OWNCARD " + "x" * 6000}}
    return _summarize_tool_result("kanban_show", "{}", json.dumps(card))


def test_own_card_kept(monkeypatch):
    out = _show("T1", monkeypatch)
    assert "own card title" in out and "OWNCARD" in out and len(out) <= 4000
    assert _is_summary_stub(out)  # demotion passes must not re-stub the kept card


def test_other_card_and_junk_stay_stubs(monkeypatch):
    assert "OWNCARD" not in _show("T2", monkeypatch)
    assert "chars" in _summarize_tool_result("kanban_show", "{}", "not json " + "y" * 5000)
