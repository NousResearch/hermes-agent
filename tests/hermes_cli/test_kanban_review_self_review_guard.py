"""Review-lane self-review guard (t_035c6c0d): an agent must not review its own card.

Two cooperating pieces:

* ``request_review``'s first-review path best-effort routes to the
  operator-configured ``kanban.default_reviewer`` when no explicit
  ``reviewer=`` is given, so the row doesn't land in the review lane
  self-assigned in the first place. This is additive and never refuses —
  ``request_review`` is documented as NOT a blocker (see
  test_kanban_review_lifecycle.py) and every existing caller that omits
  ``reviewer=`` on a first review must keep working unchanged when
  ``default_reviewer`` is unset (the shipped default).
* ``check_respawn_guard`` is the actual enforcement point: a review-lane row
  whose ``assignee`` still equals the implementer recorded on the latest
  ``review_requested`` event is withheld from dispatch with
  reason="self_review" — whether it got that way because default_reviewer
  is unset/unresolvable, or because of a hand-edit (CLI reassign, direct DB
  write, an older Hermes version) that points assignee back at the author.
"""
from __future__ import annotations

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


def _fake_spawn(*_args, **_kwargs):
    return 4242


# ---------------------------------------------------------------------------
# check_respawn_guard: the enforcement point
# ---------------------------------------------------------------------------


def test_self_assigned_review_row_is_guarded(kanban_home: Path) -> None:
    """No reviewer=, no default_reviewer configured: assignee stays == the
    implementer (existing, unchanged first-review behavior) — the dispatcher
    must not hand the card back to its own author."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "worker"  # unchanged: no fallback configured

        assert kbd.check_respawn_guard(conn, tid, lane="review") == "self_review"


def test_distinct_reviewer_is_not_guarded(kanban_home: Path) -> None:
    """An explicit, distinct reviewer= is never treated as self-review."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="paired", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).assignee == "reviewer"
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_hand_reassigned_back_to_implementer_is_still_guarded(kanban_home: Path) -> None:
    """The guard reads current ``assignee``, not just request_review's
    immediate choice — a later CLI reassign back onto the implementer is
    caught too (the row didn't predate the guard; it was mutated after)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reassigned back", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        assert kb.assign_task(conn, tid, "worker")
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "self_review"


def test_guard_is_review_lane_only(kanban_home: Path) -> None:
    """The ready lane has its own checks (recent_success / active_pr); a
    ready row with assignee == its own most recent implementer must not
    trip self_review — that concept doesn't apply outside the review lane."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ready row", assignee="worker")
        assert kbd.check_respawn_guard(conn, tid, lane="ready") is None


def test_missing_review_requested_event_is_not_guarded(kanban_home: Path) -> None:
    """A review-status row with no ``review_requested`` event on record (e.g.
    hand-crafted via direct DB access) has no implementer provenance to
    compare against — fail open rather than guess."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no provenance", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


# ---------------------------------------------------------------------------
# dispatch_once: the guard actually withholds the spawn
# ---------------------------------------------------------------------------


def test_dispatch_withholds_self_assigned_review_row(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    assert tid not in [s[0] for s in res.spawned]
    assert (tid, "self_review") in res.respawn_guarded


def test_dispatch_still_spawns_distinct_reviewer(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="paired", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)

    assert tid in [s[0] for s in res.spawned]
    assert res.respawn_guarded == []


# ---------------------------------------------------------------------------
# kanban.default_reviewer: opt-in fallback so the row never gets there
# ---------------------------------------------------------------------------


def test_default_reviewer_routes_first_review_when_unnamed(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  default_reviewer: reviewer\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.assignee == "reviewer"
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None

        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert tid in [s[0] for s in res.spawned]


def test_explicit_reviewer_overrides_default_reviewer(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  default_reviewer: fallback-reviewer\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="explicit wins", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done", reviewer="named-reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).assignee == "named-reviewer"


def test_default_reviewer_naming_nonexistent_profile_is_ignored(
    kanban_home: Path,
) -> None:
    """default_reviewer naming a profile that doesn't exist on this host is
    treated the same as unset — never written to the row — so the guard
    (not a broken assignment) is what catches the self-review case."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  default_reviewer: ghost-profile\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ghost fallback", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.assignee == "worker"
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "self_review"


def test_default_reviewer_unset_matches_prior_behavior(kanban_home: Path) -> None:
    """No config key at all: identical to the pre-guard contract — reviewer
    stays None, assignee is left as the implementer. Only the (separate)
    dispatch-time guard changes behavior, never this handoff."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="baseline", assignee="worker")
        kb.claim_task(conn, tid)
        ok, reviewer_ish = kb.request_review(
            conn, tid, summary="done",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
            with_reason=True,
        )
        assert ok is True
        assert reviewer_ish is None
        assert kb.get_task(conn, tid).assignee == "worker"


# ---------------------------------------------------------------------------
# re-review path: _prior_reviewer takes precedence over default_reviewer
# ---------------------------------------------------------------------------


def test_default_reviewer_does_not_override_re_review_provenance(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """A re-review (after changes_requested) must keep routing to the
    reviewer recorded on that event, never the configured default — the
    default only fills the gap on a genuine first review."""
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  default_reviewer: fallback-reviewer\n", encoding="utf-8",
    )
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="cycle", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.request_review(
            conn, tid, summary="v1", reviewer="original-reviewer",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        review = kb.claim_review_task(conn, tid)
        assert review is not None
        assert kb.request_changes(
            conn, tid, reason="needs work", expected_run_id=review.current_run_id,
        ) == (True, "worker")

        retry = kb.claim_task(conn, tid, claimer="worker:retry")
        assert retry is not None
        assert kb.request_review(
            conn, tid, summary="v2", expected_run_id=retry.current_run_id,
        )
        assert kb.get_task(conn, tid).assignee == "original-reviewer"
