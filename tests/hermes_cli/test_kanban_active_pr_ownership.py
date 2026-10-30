"""Respawn guard: ``active_pr`` must be scoped to the PR the card OWNS.

The guard exists to stop a card being respawned into a *duplicate* of work
its own PR is already doing. Before this change it matched ANY GitHub PR URL
in ANY of the card's recent comments, so a card that merely *referenced*
another card's PR (a merge gate, a review, an audit) was held for the whole
24 h window even though it had no PR of its own.

Ownership is resolved locally, from the card's own durable records:
``task_runs.metadata.published_pr`` and ``tasks.completion_contract``. A PR
URL the card never recorded as its own does not suppress it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


OWN_PR = "https://github.com/example/repo/pull/42"
OTHER_PR = "https://github.com/example/repo/pull/99"


def _publish_pr(conn, task_id: str, url: str) -> None:
    """Record on the card's own run that this card published ``url``."""
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at, metadata) "
            "VALUES (?, 'worker', 'done', 'completed', 1, 2, ?)",
            (task_id, json.dumps({"published_pr": url})),
        )


def test_card_recording_its_own_pr_is_still_guarded(kanban_home: Path) -> None:
    """The guard's purpose is preserved: a card that opened a PR is held."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="opened a PR", assignee="worker")
        _publish_pr(conn, tid, OWN_PR)
        kb.add_comment(conn, tid, "worker", f"Opened {OWN_PR} for review.")

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_card_citing_only_another_cards_pr_is_not_guarded(kanban_home: Path) -> None:
    """A merge gate / review that references someone else's PR must not be held.

    The cited PR belongs to another card, so re-spawning this one cannot
    duplicate it — the guard must not fire.
    """
    with kbc.connect() as conn:
        owner = kb.create_task(conn, title="owner", assignee="worker")
        _publish_pr(conn, owner, OWN_PR)

        gate = kb.create_task(conn, title="merge gate for PR 42", assignee="worker")
        kb.add_comment(conn, gate, "worker", f"Gate round for {OWN_PR} — see the PR.")

        assert kbd.check_respawn_guard(conn, gate) is None


def test_card_with_no_pr_record_is_not_guarded(kanban_home: Path) -> None:
    """A bare PR URL in a comment is not evidence this card opened a PR."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mentions a PR", assignee="worker")
        kb.add_comment(conn, tid, "worker", f"Related upstream: {OTHER_PR}")

        assert kbd.check_respawn_guard(conn, tid) is None


def test_completion_contract_pr_also_counts_as_ownership(kanban_home: Path) -> None:
    """A card whose completion contract names the PR owns it."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="contract PR", assignee="worker")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                (OWN_PR, tid),
            )
        kb.add_comment(conn, tid, "worker", f"Opened {OWN_PR} for review.")

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_own_pr_guard_survives_a_cross_card_citation(kanban_home: Path) -> None:
    """Ownership is per-card: another card's PR does not transfer ownership."""
    with kbc.connect() as conn:
        owner = kb.create_task(conn, title="owner", assignee="worker")
        _publish_pr(conn, owner, OTHER_PR)
        kb.add_comment(conn, owner, "worker", f"Opened {OTHER_PR} for review.")

        tid = kb.create_task(conn, title="owns 42 only", assignee="worker")
        _publish_pr(conn, tid, OWN_PR)
        kb.add_comment(conn, tid, "worker", f"Opened {OWN_PR}; also saw {OTHER_PR}.")

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"
        assert kbd.check_respawn_guard(conn, owner) == "active_pr"


def test_dispatch_spawns_a_card_that_only_cites_another_cards_pr(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the dispatcher spawns the citing card instead of holding it."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    with kbc.connect() as conn:
        owner = kb.create_task(conn, title="owner", assignee="worker")
        _publish_pr(conn, owner, OWN_PR)

        gate = kb.create_task(conn, title="merge gate", assignee="worker")
        kb.add_comment(conn, gate, "worker", f"Gate round for {OWN_PR}.")

        res = kbd.dispatch_once(conn, dry_run=True)
        assert gate in [s[0] for s in res.spawned]
        assert gate not in dict(res.respawn_guarded)


def test_review_lane_is_unaffected(kanban_home: Path) -> None:
    """The review lane already skipped the PR rule; it still does."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        _publish_pr(conn, tid, OWN_PR)
        kb.add_comment(conn, tid, "worker", f"Opened {OWN_PR} for review.")

        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
