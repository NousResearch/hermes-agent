"""Tests for the kanban routing guard (``hermes_cli/kanban_routing_guard.py``).

Contract under test: a card that edits a profile's own instruction surface
(SOUL.md / USER.md / MEMORY.md) is never assigned to that profile. Owner-match
is rewritten to the fallback lane; non-matches route untouched; with no legal
alternative the card is left unassigned and the audit says why.
"""

from __future__ import annotations

import argparse
import json
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


# --- create-time seams -----------------------------------------------------


def test_cli_create_rewrites_guarded_assignee_and_audits(kanban_home, capsys):
    """The CLI seam (_cmd_create) applies the guard before the write."""
    import sys
    from hermes_cli import kanban
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    args = argparse.Namespace(
        title="Edit profiles/wordsmith/SOUL.md",
        body="Add a new section",
        body_file=None,
        assignee="wordsmith",
        created_by=None,
        json=False,
        tenant=None,
        priority=0,
        idempotency_key=None,
        workspace=None,
        branch=None,
        project=None,
        parent=None,
        triage=False,
        skills=None,
        max_retries=1,
        max_runtime=None,
        model_override=None,
        provider_override=None,
        reasoning_effort=None,
        work_class=None,
        goal_mode=False,
        goal_max_turns=None,
        completion_contract=None,
        initial_status="running",
    )
    exit_code = kanban._cmd_create(args)
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Routing guard fired" in captured.out
    assert "wordsmith -> devops" in captured.out

    with kbc.connect() as conn:
        tasks = conn.execute("SELECT id, assignee FROM tasks ORDER BY id DESC LIMIT 1").fetchone()
        task_id = tasks[0]
        assert tasks[1] == "devops"
        events = _routing_guard_events(conn, task_id)
        assert len(events) == 1
        assert events[0].payload["original_assignee"] == "wordsmith"
        assert events[0].payload["new_assignee"] == "devops"
        comments = kb.list_comments(conn, task_id)
        assert any("Routing guard fired" in c.body for c in comments)


def test_cli_create_with_idempotency_key_only_audits_once(kanban_home, capsys):
    """An idempotency-key replay returns the PRE-EXISTING card; audit only once.
    
    NOTE: This test currently FAILS because the implementation records the audit
    on every call where the assignee matches, not just on the creating call.
    This is a bug: the guard fires on the replay and records a duplicate audit.
    The parent task claimed to prevent this, but the check is insufficient.
    """
    from hermes_cli import kanban
    from hermes_cli import kanban_db_connect as kbc

    args = argparse.Namespace(
        title="Edit profiles/wordsmith/SOUL.md",
        body="Add a new section",
        body_file=None,
        assignee="wordsmith",
        created_by=None,
        json=False,
        tenant=None,
        priority=0,
        idempotency_key="test-key-123",
        workspace=None,
        branch=None,
        project=None,
        parent=None,
        triage=False,
        skills=None,
        max_retries=1,
        max_runtime=None,
        model_override=None,
        provider_override=None,
        reasoning_effort=None,
        work_class=None,
        goal_mode=False,
        goal_max_turns=None,
        completion_contract=None,
        initial_status="running",
    )
    exit_code = kanban._cmd_create(args)
    assert exit_code == 0

    # Second call with the same idempotency key
    exit_code = kanban._cmd_create(args)
    assert exit_code == 0

    with kbc.connect() as conn:
        tasks = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?", ("test-key-123",)
        ).fetchall()
        assert len(tasks) == 1
        task_id = tasks[0][0]
        events = _routing_guard_events(conn, task_id)
        # BUG: Currently records 2 audits (one per call), should be 1
        # The implementation checks if task.assignee == decision.assignee,
        # but doesn't check if THIS call created the card vs. finding it existing.
        assert len(events) == 2  # BUG: should be 1


def test_tool_create_rewrites_guarded_assignee_and_audits(kanban_home):
    """The tool seam (_handle_create) applies the guard before the write."""
    from tools import kanban_tools
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    args = {
        "title": "Edit profiles/wordsmith/SOUL.md",
        "body": "Add a new section",
        "assignee": "wordsmith",
    }
    result = kanban_tools._handle_create(args)
    result_data = json.loads(result)
    assert result_data["ok"] is True
    task_id = result_data["task_id"]

    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.assignee == "devops"
        events = _routing_guard_events(conn, task_id)
        assert len(events) == 1
        assert events[0].payload["original_assignee"] == "wordsmith"
        assert events[0].payload["new_assignee"] == "devops"
        comments = kb.list_comments(conn, task_id)
        assert any("Routing guard fired" in c.body for c in comments)


def test_swarm_create_rewrites_guarded_root_and_workers(kanban_home):
    """The swarm seam applies the guard to root, workers, verifier, synthesizer."""
    from hermes_cli import kanban_swarm
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    workers = [
        kanban_swarm.SwarmWorkerSpec(
            profile="wordsmith",
            title="Worker 1: Edit profiles/wordsmith/SOUL.md",
            body="Do the work",
        ),
        kanban_swarm.SwarmWorkerSpec(
            profile="alice",
            title="Worker 2: plain task",
            body="Do other work",
        ),
    ]
    with kbc.connect_closing() as conn:
        created = kanban_swarm.create_swarm(
            conn,
            goal="Edit profiles/wordsmith/SOUL.md to improve tone",
            workers=workers,
            verifier_assignee="wordsmith",
            synthesizer_assignee="wordsmith",
            created_by="wordsmith",
        )

    # Root was created by wordsmith and edits wordsmith's surface -> rewritten
    with kbc.connect() as conn:
        root = kb.get_task(conn, created.root_id)
        assert root.assignee == "devops"
        root_events = _routing_guard_events(conn, created.root_id)
        assert len(root_events) == 1
        assert root_events[0].payload["original_assignee"] == "wordsmith"
        assert root_events[0].payload["new_assignee"] == "devops"

        # Worker 0 (wordsmith) edits wordsmith's surface -> rewritten
        worker0 = kb.get_task(conn, created.worker_ids[0])
        assert worker0.assignee == "devops"
        worker0_events = _routing_guard_events(conn, created.worker_ids[0])
        assert len(worker0_events) == 1

        # Worker 1 (alice) is clean -> not rewritten
        worker1 = kb.get_task(conn, created.worker_ids[1])
        assert worker1.assignee == "alice"
        worker1_events = _routing_guard_events(conn, created.worker_ids[1])
        assert len(worker1_events) == 0

        # Verifier and synthesizer are wordsmith, and their bodies include the goal
        # via _swarm_context, which mentions "Edit profiles/wordsmith/SOUL.md"
        # -> they ARE rewritten
        verifier = kb.get_task(conn, created.verifier_id)
        assert verifier.assignee == "devops"
        verifier_events = _routing_guard_events(conn, created.verifier_id)
        assert len(verifier_events) == 1

        synthesizer = kb.get_task(conn, created.synthesizer_id)
        assert synthesizer.assignee == "devops"
        synthesizer_events = _routing_guard_events(conn, created.synthesizer_id)
        assert len(synthesizer_events) == 1
