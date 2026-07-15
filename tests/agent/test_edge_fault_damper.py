"""Tests for edge fault-loop damper and scratchpad fault lines."""

from unittest.mock import patch

from agent.edge_fault_damper import (
    edge_precheck_tool_repeat,
    edge_tool_signature,
    parse_sig_markers_from_text,
    resync_edge_failed_signatures,
)
from agent.edge_working_memory import append_auto_fault_blocker, default_scratchpad


def test_edge_tool_signature_stable():
    s1 = edge_tool_signature("terminal", {"command": "ls"})
    s2 = edge_tool_signature("terminal", {"command": "ls"})
    assert s1 == s2
    assert "|" in s1


def test_parse_sig_markers():
    text = "x [sig:terminal|abcd] y [sig:read_file|efgh]"
    got = parse_sig_markers_from_text(text)
    assert "terminal|abcd" in got
    assert "read_file|efgh" in got


def test_precheck_blocks_repeat():
    class A:
        edge_mode = True
        _edge_scratchpad = ""
        _edge_failed_signatures = set()

    a = A()
    sig = edge_tool_signature("terminal", {"command": "nope"})
    a._edge_failed_signatures.add(sig.lower())
    msg = edge_precheck_tool_repeat(a, "terminal", {"command": "nope"})
    assert msg is not None
    assert "fault damper" in msg.lower()


def test_append_auto_fault_under_blockers():
    base = default_scratchpad("g")
    out = append_auto_fault_blocker(
        base, "terminal", "terminal|deadbeef", "Command not found: xyz",
    )
    assert "[sig:terminal|deadbeef]" in out
    assert "Faults" in out


def test_resync_merges_scratchpad_sigs():
    class A:
        edge_mode = True
        _edge_scratchpad = "- [auto] x [sig:terminal|aaa]\n"
        _edge_failed_signatures = {"read_file|bbb"}

    a = A()
    resync_edge_failed_signatures(a)
    assert "terminal|aaa" in a._edge_failed_signatures
    assert "read_file|bbb" in a._edge_failed_signatures


def test_record_marks_scratchpad_on_failure():
    from agent.edge_fault_damper import edge_record_tool_result_for_damper

    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad("goal")
        _edge_failed_signatures = set()

    a = A()
    bad = '{"success": false, "error": "permission denied"}'
    with patch("agent.display._detect_tool_failure", return_value=(True, "x")):
        edge_record_tool_result_for_damper(a, "write_file", {"path": "/tmp/x"}, bad)
    assert a._edge_failed_signatures
    assert "[sig:" in a._edge_scratchpad


def test_consecutive_failures_set_interrupt_when_capped():
    from agent.edge_fault_damper import edge_record_tool_result_for_damper

    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad("goal")
        _edge_failed_signatures = set()
        _edge_max_consecutive_tool_failures = 2
        _interrupt_requested = False

    a = A()
    bad = '{"success": false, "error": "e"}'
    with patch("agent.display._detect_tool_failure", return_value=(True, "x")):
        edge_record_tool_result_for_damper(a, "terminal", {"command": "x"}, bad)
        assert not a._interrupt_requested
        edge_record_tool_result_for_damper(a, "terminal", {"command": "y"}, bad)
        assert a._interrupt_requested


def test_success_resets_consecutive_counter():
    from agent.edge_fault_damper import edge_record_tool_result_for_damper

    class A:
        edge_mode = True
        _edge_scratchpad = default_scratchpad("goal")
        _edge_failed_signatures = set()
        _edge_consecutive_tool_failures = 3

    a = A()
    with patch("agent.display._detect_tool_failure", return_value=(False, "")):
        edge_record_tool_result_for_damper(a, "terminal", {"command": "x"}, '{"ok": true}')
    assert a._edge_consecutive_tool_failures == 0
