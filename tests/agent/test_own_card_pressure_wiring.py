"""Pressure-pass wiring for the own-card keep: the documented escape hatch is
actually reachable from the production pass-4 caller, through both of its
demotion arms. Note: kanban_show's generic stub ends in ``chars result)``, so
``_is_summary_stub`` (which keys on ``chars)``) is not the right verifier here —
the stub property we assert is: one short ``[kanban_show]`` line, no card text."""
import json

from agent.context_compressor import (
    ContextCompressor,
    _summarize_tool_result,
)


def _show(task_id, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "T1")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    card = {"task": {"id": task_id, "title": "own card title", "body": "OWNCARD " + "x" * 6000}}
    return _summarize_tool_result("kanban_show", "{}", json.dumps(card))


def _setup(monkeypatch, own_card_pos):
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor.quiet_mode = True
    rows, call_map = [], {}
    for i in range(6):
        cid = "c%d" % i
        if i == own_card_pos:
            rows.append({"role": "tool", "tool_call_id": cid, "content": _show("T1", monkeypatch)})
            call_map[cid] = ("kanban_show", "{}")
        else:
            rows.append({"role": "tool", "tool_call_id": cid, "content": "P " + "y" * 6000})
            call_map[cid] = ("generic_read", "{}")
    return compressor, rows, call_map


def _is_stubbed(content):
    return content.startswith("[kanban_show]") and "OWNCARD" not in content and len(content) < 200


def test_pressure_pass_re_stubs_old_own_card(monkeypatch):
    """Own card inside the demote window: the pass-4 loop arm (keep_own_card=False
    wiring at the _shrink_at call site) must be able to re-stub it."""
    compressor, rows, call_map = _setup(monkeypatch, own_card_pos=0)
    compressor._pressure_demote_tail(
        rows, prune_boundary=0, protect_tail_tokens=200,
        call_id_to_tool=call_map, min_prune_chars=200, spared=range(0),
    )
    assert _is_stubbed(rows[0]["content"]), "old own card must be re-stubbable via the real pressure pass"


def test_pressure_pass_re_stubs_newest_own_card(monkeypatch):
    """Own card as the newest tool row: only the last-resort arm can reach it
    (keep_own_card=False wiring at the last_tool_idx call site) — a huge kept
    card must never wedge the hard budget backstop."""
    compressor, rows, call_map = _setup(monkeypatch, own_card_pos=5)
    compressor._pressure_demote_tail(
        rows, prune_boundary=0, protect_tail_tokens=200,
        call_id_to_tool=call_map, min_prune_chars=200, spared=range(0),
    )
    assert _is_stubbed(rows[5]["content"]), "newest own card must be reachable by the last-resort arm"


def test_pressure_pass_spares_own_card_without_override(monkeypatch):
    """Control: the guard itself is intact — a kept card fed through the same
    demotion primitive with the default keep_own_card=True stays kept."""
    compressor, rows, call_map = _setup(monkeypatch, own_card_pos=0)
    assert compressor._demote_tool_result_at(rows, 0, call_map, 200) is False
    assert rows[0]["content"].startswith("[kanban_show own-card:")
