"""Tests for the prose grounding gate (agent.verification_stop.build_grounding_nudge)."""
from agent.verification_stop import build_grounding_nudge


def _tc(name):
    return {"tool_calls": [{"function": {"name": name, "arguments": "{}"}}]}


def test_fabrication_analytical_prose_no_grounding_nudges():
    r = build_grounding_nudge(
        changed_paths=["/w/analysis.md"],
        session_messages=[_tc("write_file"), _tc("kanban_complete")],
    )
    assert r and "analytical deliverable" in r


def test_grounded_read_file_is_silent():
    r = build_grounding_nudge(
        changed_paths=["/w/analysis.md"],
        session_messages=[_tc("read_file"), _tc("write_file")],
    )
    assert r is None


def test_grounded_terminal_is_silent():
    r = build_grounding_nudge(
        changed_paths=["/w/report.md"],
        session_messages=[_tc("terminal"), _tc("write_file")],
    )
    assert r is None


def test_grounded_web_search_is_silent():
    r = build_grounding_nudge(
        changed_paths=["/w/research-brief.md"],
        session_messages=[_tc("web_search")],
    )
    assert r is None


def test_non_analytical_prose_is_silent():
    r = build_grounding_nudge(
        changed_paths=["/w/README.md"], session_messages=[_tc("write_file")]
    )
    assert r is None


def test_code_file_is_silent():
    r = build_grounding_nudge(
        changed_paths=["/w/main.py"], session_messages=[_tc("write_file")]
    )
    assert r is None


def test_bounded_by_attempts():
    r = build_grounding_nudge(
        changed_paths=["/w/analysis.md"],
        session_messages=[_tc("write_file")],
        attempts=2,
    )
    assert r is None


def test_grounded_execute_code_is_silent():
    """execute_code is terminal-class and can read sources; it must count."""
    r = build_grounding_nudge(
        changed_paths=["/w/analysis.md"],
        session_messages=[_tc("execute_code"), _tc("write_file")],
    )
    assert r is None


def test_grounded_vision_analyze_is_silent():
    """Inspecting an image IS inspecting source material."""
    r = build_grounding_nudge(
        changed_paths=["/w/findings.md"],
        session_messages=[_tc("vision_analyze"), _tc("write_file")],
    )
    assert r is None


def test_kanban_list_is_not_grounding():
    r = build_grounding_nudge(
        changed_paths=["/w/findings.md"],
        session_messages=[_tc("kanban_list"), _tc("write_file")],
    )
    assert r and "analytical deliverable" in r


def test_empty_session_nudges():
    r = build_grounding_nudge(changed_paths=["/w/audit-report.md"], session_messages=[])
    assert r is not None
