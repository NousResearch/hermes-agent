"""Tests for the kanban routing guard (``hermes_cli/kanban_routing_guard.py``).

Contract under test: a card that edits a profile's own instruction surface
(SOUL.md / USER.md / MEMORY.md) is never assigned to that profile. Owner-match
is rewritten to the fallback lane; non-matches route untouched; with no legal
alternative the card is left unassigned and the audit says why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_routing_guard as guard


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB (same pattern as the
    decompose/specify DB tests)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    return home


# --- detection -------------------------------------------------------------


@pytest.mark.parametrize(
    "path,owner",
    [
        ("profiles/wordsmith/SOUL.md", "wordsmith"),
        ("profiles/wordsmith/USER.md", "wordsmith"),
        ("profiles/wordsmith/MEMORY.md", "wordsmith"),
        ("profiles/wordsmith/memories/MEMORY.md", "wordsmith"),
        ("profiles/wordsmith/memories/USER.md", "wordsmith"),
        ("~/.hermes/profiles/wordsmith/SOUL.md", "wordsmith"),
        ("/Users/sgt.snips/.hermes/profiles/wordsmith/SOUL.md", "wordsmith"),
        ("~/.hermes/SOUL.md", "default"),
        ("~/.hermes/memories/MEMORY.md", "default"),
    ],
)
def test_surface_path_forms_are_detected(path, owner):
    refs = guard.find_instruction_surface_refs(f"see {path} for details")
    assert [ref.profile for ref in refs] == [owner]


def test_surface_detection_requires_a_path_not_a_mention():
    assert guard.find_instruction_surface_refs(
        "A card about writers that names wordsmith as a subject."
    ) == []
    assert guard.find_instruction_surface_refs("profiles/wordsmith/config.yaml") == []


# --- guard_assignee --------------------------------------------------------


def test_owner_match_is_rewritten_to_fallback_lane():
    decision = guard.guard_assignee(
        "wordsmith",
        "Edit profiles/wordsmith/SOUL.md section 1",
        fallback_assignee="devops",
        default_assignee="",
    )
    assert decision.rewritten
    assert decision.assignee == "devops"
    assert decision.original_assignee == "wordsmith"
    assert decision.lane_source == "fallback"
    assert decision.matched_profiles == ["wordsmith"]
    assert decision.matched_paths == ["profiles/wordsmith/SOUL.md"]


def test_absolute_path_is_detected_and_rewritten():
    """The real 2026-10-05 incident card (t_e98a4f1a) used the absolute form."""
    decision = guard.guard_assignee(
        "wordsmith",
        "Edit `/Users/sgt.snips/.hermes/profiles/wordsmith/SOUL.md` to add a passage.",
        fallback_assignee="devops",
    )
    assert decision.rewritten
    assert decision.assignee == "devops"
    assert decision.matched_paths == ["profiles/wordsmith/SOUL.md"]


def test_mixed_case_assignee_matches_the_owner():
    decision = guard.guard_assignee(
        "Wordsmith", "edit profiles/wordsmith/SOUL.md", fallback_assignee="devops"
    )
    assert decision.rewritten
    assert decision.assignee == "devops"


def test_default_home_surface_rewrites_the_default_profile():
    decision = guard.guard_assignee(
        "default", "edit ~/.hermes/SOUL.md", fallback_assignee="devops"
    )
    assert decision.rewritten
    assert decision.assignee == "devops"
    assert decision.matched_profiles == ["default"]


def test_mention_only_card_is_untouched():
    decision = guard.guard_assignee(
        "wordsmith",
        "A card about writers that names wordsmith as a subject; write the guide.",
        fallback_assignee="devops",
        default_assignee="",
    )
    assert not decision.rewritten
    assert decision.assignee == "wordsmith"
    assert decision.matched_paths == []


def test_other_profiles_surface_does_not_rewrite():
    decision = guard.guard_assignee(
        "wordsmith", "edit profiles/lightwright/SOUL.md", fallback_assignee="devops"
    )
    assert not decision.rewritten
    assert decision.assignee == "wordsmith"


def test_legitimate_lane_routes_unchanged():
    """A card whose chosen lane is legitimately correct still routes unchanged."""
    decision = guard.guard_assignee(
        "devops", "edit profiles/wordsmith/SOUL.md", fallback_assignee="devops"
    )
    assert not decision.rewritten
    assert decision.assignee == "devops"
    assert decision.matched_paths == ["profiles/wordsmith/SOUL.md"]  # detected, just not the owner


def test_no_alternative_leaves_unassigned_with_reason():
    decision = guard.guard_assignee(
        "wordsmith",
        "edit profiles/wordsmith/SOUL.md",
        fallback_assignee="",
        default_assignee="",
    )
    assert decision.rewritten
    assert decision.assignee is None
    assert decision.lane_source == "none"
    assert "unassigned" in decision.reason
    assert "no eligible alternative lane" in decision.reason


def test_lanes_equal_to_the_owner_are_not_alternatives():
    decision = guard.guard_assignee(
        "wordsmith",
        "edit profiles/wordsmith/SOUL.md",
        fallback_assignee="wordsmith",
        default_assignee="wordsmith",
    )
    assert decision.assignee is None
    assert decision.lane_source == "none"


def test_empty_fallback_falls_through_to_default_assignee():
    decision = guard.guard_assignee(
        "wordsmith",
        "edit profiles/wordsmith/SOUL.md",
        fallback_assignee="",
        default_assignee="orchestrator",
    )
    assert decision.rewritten
    assert decision.assignee == "orchestrator"
    assert decision.lane_source == "default_assignee"


def test_eligible_names_restrict_lane_choice():
    decision = guard.guard_assignee(
        "wordsmith",
        "edit profiles/wordsmith/SOUL.md",
        fallback_assignee="devops",
        default_assignee="junior-dev",
        eligible_names={"wordsmith", "junior-dev"},
    )
    assert decision.assignee == "junior-dev"
    assert decision.lane_source == "default_assignee"

    decision = guard.guard_assignee(
        "wordsmith",
        "edit profiles/wordsmith/SOUL.md",
        fallback_assignee="devops",
        default_assignee="junior-dev",
        eligible_names={"wordsmith"},
    )
    assert decision.assignee is None
    assert decision.lane_source == "none"


def test_no_assignee_is_untouched():
    decision = guard.guard_assignee(None, "edit profiles/wordsmith/SOUL.md")
    assert not decision.rewritten
    assert decision.assignee is None


# --- config home -----------------------------------------------------------


def test_config_defaults_define_the_fallback_lane():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["routing_guard_fallback_assignee"] == "devops"


def test_configured_fallback_defaults_to_devops(kanban_home):
    assert guard.configured_fallback_assignee() == "devops"


def test_configured_fallback_reads_config(kanban_home):
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  routing_guard_fallback_assignee: qa\n", encoding="utf-8"
    )
    assert guard.configured_fallback_assignee() == "qa"
    decision = guard.guard_assignee("wordsmith", "edit profiles/wordsmith/SOUL.md")
    assert decision.assignee == "qa"
    assert decision.lane_source == "fallback"


def test_configured_empty_string_disables_the_fallback_lane(kanban_home):
    (kanban_home / "config.yaml").write_text(
        'kanban:\n  routing_guard_fallback_assignee: ""\n', encoding="utf-8"
    )
    assert guard.configured_fallback_assignee() == ""
    # Config-driven call: fallback disabled, default_assignee unset -> no legal lane.
    decision = guard.guard_assignee("wordsmith", "edit profiles/wordsmith/SOUL.md")
    assert decision.assignee is None
    assert decision.lane_source == "none"


# --- audit record ----------------------------------------------------------


def _make_task(conn, assignee="wordsmith"):
    from hermes_cli import kanban_db as kb

    return kb.create_task(conn, title="edit SOUL", assignee=assignee)


def _routing_guard_events(conn, task_id):
    from hermes_cli import kanban_db as kb

    return [event for event in kb.list_events(conn, task_id) if event.kind == "routing_guard"]


def test_record_guard_decision_writes_event_and_comment(kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        task_id = _make_task(conn)
    decision = guard.guard_assignee(
        "wordsmith", "edit profiles/wordsmith/SOUL.md", fallback_assignee="devops"
    )
    with kbc.connect() as conn:
        guard.record_guard_decision(conn, task_id, decision)
    with kbc.connect() as conn:
        events = _routing_guard_events(conn, task_id)
        comments = kb.list_comments(conn, task_id)

    assert len(events) == 1
    payload = events[0].payload
    assert payload["original_assignee"] == "wordsmith"
    assert payload["new_assignee"] == "devops"
    assert payload["lane_source"] == "fallback"
    assert payload["matched_profiles"] == ["wordsmith"]
    assert payload["matched_paths"] == ["profiles/wordsmith/SOUL.md"]
    assert any("Routing guard fired" in comment.body for comment in comments)


def test_record_guard_decision_is_safe_inside_open_write_txn(kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    decision = guard.guard_assignee(
        "wordsmith", "edit profiles/wordsmith/SOUL.md", fallback_assignee="devops"
    )
    with kbc.connect() as conn:
        task_id = _make_task(conn)
        with kb.write_txn(conn):
            guard.record_guard_decision(conn, task_id, decision)
        assert len(_routing_guard_events(conn, task_id)) == 1
    with kbc.connect() as conn:
        assert len(_routing_guard_events(conn, task_id)) == 1


def test_record_guard_decision_audits_the_unassigned_case(kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        task_id = _make_task(conn)
    decision = guard.guard_assignee(
        "wordsmith",
        "edit profiles/wordsmith/SOUL.md",
        fallback_assignee="",
        default_assignee="",
    )
    with kbc.connect() as conn:
        guard.record_guard_decision(conn, task_id, decision)
    with kbc.connect() as conn:
        events = _routing_guard_events(conn, task_id)
        comments = kb.list_comments(conn, task_id)

    assert len(events) == 1
    assert events[0].payload["new_assignee"] is None
    assert any("left unassigned" in comment.body for comment in comments)


def test_record_guard_decision_noop_when_not_rewritten(kanban_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect() as conn:
        task_id = _make_task(conn, assignee="junior-dev")
    decision = guard.guard_assignee(
        "junior-dev", "edit profiles/wordsmith/SOUL.md", fallback_assignee="devops"
    )
    assert not decision.rewritten
    with kbc.connect() as conn:
        guard.record_guard_decision(conn, task_id, decision)
        assert _routing_guard_events(conn, task_id) == []
        assert not any("Routing guard" in c.body for c in kb.list_comments(conn, task_id))
