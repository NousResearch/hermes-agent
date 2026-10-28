"""Tests for the dispatch-time routing net (``hermes_cli/kanban_db_dispatch_routing.py``).

Contract under test: no card whose own assignee owns an instruction surface it
edits can reach a worker, whatever write path put it in ``ready``. The net runs
inside ``_dispatch_lane_task`` - before the claim - and either reroutes the row,
skips the spawn (no legal lane), or leaves it alone.

The cross-seam suite (create-time seams + the module's own 30 guard tests) lives
in ``tests/hermes_cli/test_kanban_routing_guard.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db_dispatch_routing as routing


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with a clean kanban DB (same pattern as the guard tests)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb.init_db()
    return home


def _fake_spawn(*args, **kwargs):
    return 12345


def _events(conn, task_id, kind=None):
    sql = "SELECT kind, payload FROM task_events WHERE task_id = ?"
    params: tuple = (task_id,)
    if kind:
        sql += " AND kind = ?"
        params += (kind,)
    return [json.loads(row["payload"]) if row["payload"] else None
            for row in conn.execute(sql + " ORDER BY id", params)]


def _comments(conn, task_id):
    return [row["body"] for row in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id", (task_id,))]


# --- lane resolution ---------------------------------------------------------


def test_lanes_are_filtered_to_spawnable_profiles():
    """A candidate lane this home cannot place a worker on is not a legal alternative."""
    lanes = routing.spawnable_guard_lanes(
        lambda name: name == "bob", fallback_assignee="devops", default_assignee="bob",
    )
    assert lanes is not None
    assert lanes.names == ["bob"]
    assert lanes.default_assignee == "bob"


def test_lanes_are_unknown_without_a_profile_predicate():
    """Unimportable profiles module keeps the guard on its config-only semantics."""
    assert routing.spawnable_guard_lanes(None, fallback_assignee="devops") is None


def test_lanes_dedupe_and_keep_an_empty_default_lane():
    lanes = routing.spawnable_guard_lanes(
        lambda name: True, fallback_assignee="devops", default_assignee=" devops ",
    )
    assert lanes is not None
    assert lanes.names == ["devops"]
    assert routing.spawnable_guard_lanes(
        lambda name: True, fallback_assignee="", default_assignee="",
    ).names == []


# --- decision (read-only) ----------------------------------------------------


def test_owner_surface_card_is_rerouted_off_the_owner(all_assignees_spawnable):
    decision = routing.decide_dispatch_routing(
        "wordsmith", "Fix the tone section", "Edit profiles/wordsmith/SOUL.md please.",
    )
    assert decision.rewritten is True
    assert decision.assignee == "devops"


def test_mention_only_card_is_not_rerouted(all_assignees_spawnable):
    decision = routing.decide_dispatch_routing(
        "wordsmith", "Review wordsmith output", "A card about writers, not about a path.",
    )
    assert decision.rewritten is False
    assert decision.assignee == "wordsmith"


def test_lane_eligibility_comes_from_the_tick_not_from_config(all_assignees_spawnable):
    """The dispatcher's lane set wins: an explicit empty default stays disabled."""
    lanes = routing.GuardLaneSet(default_assignee="", names=["devops"])
    decision = routing.decide_dispatch_routing(
        "wordsmith", "t", "profiles/wordsmith/SOUL.md", lanes=lanes,
    )
    assert decision.rewritten is True
    assert decision.assignee == "devops"


# --- applying the decision ---------------------------------------------------


def test_rewrite_persists_assignee_events_and_audit(kanban_home, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md",
            assignee="wordsmith",
        )
        outcome = routing.apply_dispatch_routing_guard(
            conn, task_id, "wordsmith", "tone", "Edit profiles/wordsmith/SOUL.md",
        )
        assert outcome.skipped is False
        assert outcome.assignee == "devops"
        assert outcome.audit_written is True

        assert kb.get_task(conn, task_id).assignee == "devops"
        assigned = _events(conn, task_id, "assigned")
        assert assigned == [{"assignee": "devops", "from": "wordsmith", "source": "routing_guard"}]
        audits = _events(conn, task_id, "routing_guard")
        assert len(audits) == 1
        assert audits[0]["new_assignee"] == "devops"
        assert audits[0]["matched_paths"] == ["profiles/wordsmith/SOUL.md"]
        assert "profiles/wordsmith/SOUL.md" in _comments(conn, task_id)[-1]


def test_no_legal_lane_skips_the_spawn_and_audits_once(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import profiles

    # No candidate lane exists on this home.
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "wordsmith")
    lanes = routing.spawnable_guard_lanes(
        profiles.profile_exists, fallback_assignee="devops", default_assignee="",
    )
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md",
            assignee="wordsmith",
        )
        first = routing.apply_dispatch_routing_guard(
            conn, task_id, "wordsmith", "tone", "Edit profiles/wordsmith/SOUL.md", lanes=lanes,
        )
        assert first.skipped is True
        assert first.assignee is None
        assert "no eligible alternative lane" in first.reason
        assert first.audit_written is True
        # The row keeps the operator's routing intent; only the spawn is denied.
        assert kb.get_task(conn, task_id).assignee == "wordsmith"

        # A second pass over the same unchanged row must not re-audit.
        second = routing.apply_dispatch_routing_guard(
            conn, task_id, "wordsmith", "tone", "Edit profiles/wordsmith/SOUL.md", lanes=lanes,
        )
        assert second.skipped is True
        assert second.audit_written is False
        assert len(_events(conn, task_id, "routing_guard")) == 1
        assert len(_comments(conn, task_id)) == 1


def test_dry_run_reports_without_writing(kanban_home, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md",
            assignee="wordsmith",
        )
        outcome = routing.apply_dispatch_routing_guard(
            conn, task_id, "wordsmith", "tone", "Edit profiles/wordsmith/SOUL.md", dry_run=True,
        )
        assert outcome.assignee == "devops"
        assert outcome.skipped is False
        assert outcome.audit_written is False
        assert kb.get_task(conn, task_id).assignee == "wordsmith"
        assert _events(conn, task_id, "routing_guard") == []


def test_rewrite_aborts_when_the_row_moved_underneath(kanban_home, all_assignees_spawnable):
    """The UPDATE is a CAS on the assignee we saw; a concurrent writer wins."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md",
            assignee="wordsmith",
        )
        # Somebody else reassigned between the lane snapshot and the net.
        kb.assign_task(conn, task_id, "bob")
        outcome = routing.apply_dispatch_routing_guard(
            conn, task_id, "wordsmith", "tone", "Edit profiles/wordsmith/SOUL.md",
        )
        assert outcome.skipped is True
        assert outcome.assignee is None
        assert "changed" in outcome.reason
        assert kb.get_task(conn, task_id).assignee == "bob"
        assert _events(conn, task_id, "routing_guard") == []


def test_a_broken_guard_never_stops_the_board(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    def _boom(*args, **kwargs):
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(routing, "decide_dispatch_routing", _boom)
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md",
            assignee="wordsmith",
        )
        outcome = routing.apply_dispatch_routing_guard(
            conn, task_id, "wordsmith", "tone", "Edit profiles/wordsmith/SOUL.md",
        )
        assert outcome.assignee == "wordsmith"
        assert outcome.skipped is False


# --- dispatcher integration --------------------------------------------------


def test_dispatch_reroutes_an_owner_card_before_the_spawn(kanban_home, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md",
            assignee="wordsmith",
        )
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)

    assert res.routed_by_guard == [(task_id, "wordsmith", "devops")]
    assert [s[0] for s in res.spawned] == [task_id]
    assert res.spawned[0][1] == "devops"
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).assignee == "devops"
        assert len(_events(conn, task_id, "routing_guard")) == 1


def test_dispatch_skips_a_card_with_no_legal_lane(kanban_home, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"wordsmith", "alice"})
    with kbc.connect_closing() as conn:
        owner_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md", assignee="wordsmith",
        )
        clean_id = kb.create_task(conn, title="other", body="nothing to see", assignee="alice")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)

    assert [tid for tid, _reason in res.skipped_routing_guard] == [owner_id]
    assert "no eligible alternative lane" in res.skipped_routing_guard[0][1]
    assert [s[0] for s in res.spawned] == [clean_id]
    with kbc.connect_closing() as conn:
        # Nothing was spawned for the owning profile and the row is untouched.
        assert kb.get_task(conn, owner_id).assignee == "wordsmith"
        assert kb.get_task(conn, owner_id).status == "ready"


def test_dispatch_leaves_a_mention_only_card_alone(kanban_home, all_assignees_spawnable):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="Review wordsmith output", body="A card about writers.",
            assignee="wordsmith",
        )
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn)

    assert res.routed_by_guard == []
    assert res.skipped_routing_guard == []
    assert res.spawned[0][:2] == (task_id, "wordsmith")


def test_default_assignee_is_corrected_by_the_net_in_the_same_tick(kanban_home, monkeypatch):
    """The documented interaction: auto-assign writes the OPERATOR's intent,
    then the net reroutes it before the claim."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"wordsmith", "devops"})
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="tone", body="Edit profiles/wordsmith/SOUL.md", assignee=None,
        )
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, default_assignee="wordsmith")

    assert res.auto_assigned_default == [task_id]
    assert res.routed_by_guard == [(task_id, "wordsmith", "devops")]
    assert res.spawned[0][:2] == (task_id, "devops")
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).assignee == "devops"
        sources = [p["source"] for p in _events(conn, task_id, "assigned") if p]
        assert sources == ["kanban.default_assignee", "routing_guard"]


def _review_card(conn, kb, *, body, assignee="alice"):
    """A card claimed and handed to the review column (the canonical handoff)."""
    task_id = kb.create_task(conn, title="review work", body=body, assignee=assignee)
    assert kb.claim_task(conn, task_id) is not None
    kb.request_review(
        conn, task_id, summary="done",
        expected_run_id=kb.get_task(conn, task_id).current_run_id,
    )
    return task_id


def test_dispatch_guards_the_review_lane_too(kanban_home, monkeypatch):
    """The net is not ready-lane-only: a review row is rerouted/skipped as well."""
    import hermes_cli.config as cfgmod
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "alice")
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    with kbc.connect_closing() as conn:
        review_id = _review_card(conn, kb, body="Edit profiles/alice/SOUL.md")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_spawn=2)

    assert [tid for tid, _r in res.skipped_routing_guard] == [review_id]
    assert res.spawned == []


def test_guard_skipped_review_row_does_not_hold_ready_capacity(kanban_home, monkeypatch):
    """A review row the net will skip must not starve the ready lane's budget."""
    import hermes_cli.config as cfgmod
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "alice")
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    with kbc.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="ready work", body="plain", assignee="alice")
        _review_card(conn, kb, body="Edit profiles/alice/SOUL.md")
    with kbc.connect_closing() as conn:
        # max_spawn=1: a review row that reserved the slot would spawn nothing.
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_spawn=1)

    assert [s[0] for s in res.spawned] == [ready_id]


def test_spawnable_review_row_still_reserves_ready_capacity(kanban_home, monkeypatch):
    """Negative control: the reservation still engages for a row the net allows."""
    import hermes_cli.config as cfgmod
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "alice")
    monkeypatch.setattr(
        cfgmod, "load_config", lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    with kbc.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="ready work", body="plain", assignee="alice")
        review_id = _review_card(conn, kb, body="plain review work")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_spawn=1)

    # The review row holds the shared slot; the ready card waits its turn.
    assert [s[0] for s in res.spawned] == [review_id]
    assert ready_id not in [s[0] for s in res.spawned]
