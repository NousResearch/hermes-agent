"""Focused tests for structured blocker ownership.

Proves the acceptance criteria from the "structured blocker ownership" work:

* reviewer-owned cards (review-required) never read as human asks;
* true explicit human decisions DO read as human asks and carry the owner;
* automation / parked / acceptance work is distinguishable and never nagged;
* a non-human owner kind cannot smuggle a name in via ``blocker_owner``;
* legacy rows (NULL) stay unknown — migration never invents a human owner;
* completion and unblock both clear the ownership fields.

These assert kernel behaviour (contracts about how ownership is normalised
and persisted), not a snapshot of any particular board's data.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked(conn, *, reason=None, owner_kind=None, owner=None, kind=None):
    """Create → claim → block a task and return the reloaded Task."""
    t = kb.create_task(conn, title="x", assignee="a")
    kb.claim_task(conn, t)
    assert kb.block_task(
        conn, t,
        reason=reason,
        kind=kind,
        blocker_owner_kind=owner_kind,
        blocker_owner=owner,
    )
    return kb.get_task(conn, t)


# ---------------------------------------------------------------------------
# derive_blocker_owner — the pure normaliser
# ---------------------------------------------------------------------------

def test_review_required_defaults_to_reviewer_not_human():
    ok, owner = kb.derive_blocker_owner(
        "review-required: PR #123 needs eyes", None, None
    )
    assert ok == "reviewer"
    assert owner is None


def test_kernel_never_defaults_to_human():
    # A reason that merely mentions a person must NOT become a human owner
    # unless the caller explicitly classifies it as such.
    ok, owner = kb.derive_blocker_owner("waiting on Daniel to merge", None, None)
    assert ok is None
    assert owner is None


def test_explicit_human_keeps_owner():
    ok, owner = kb.derive_blocker_owner("need a call", "human", "daniel")
    assert ok == "human"
    assert owner == "daniel"


def test_external_keeps_owner():
    ok, owner = kb.derive_blocker_owner("upstream ci", "external", "github-ci")
    assert ok == "external"
    assert owner == "github-ci"


@pytest.mark.parametrize("kind", ["reviewer", "automation", "acceptance", "parked"])
def test_non_human_kinds_drop_owner(kind):
    # A name cannot ride in behind a non-human/external classification.
    ok, owner = kb.derive_blocker_owner("x", kind, "daniel")
    assert ok == kind
    assert owner is None


def test_explicit_kind_beats_review_required_inference():
    ok, owner = kb.derive_blocker_owner(
        "review-required: but actually parked", "parked", None
    )
    assert ok == "parked"


def test_unknown_normalises_to_none():
    ok, owner = kb.derive_blocker_owner("x", "unknown", None)
    assert ok is None
    assert owner is None


def test_invalid_owner_kind_rejected():
    with pytest.raises(ValueError):
        kb.derive_blocker_owner("x", "boss", None)


# ---------------------------------------------------------------------------
# block_task — persistence + event payload
# ---------------------------------------------------------------------------

def test_review_required_block_is_reviewer_owned(kanban_home):
    with kb.connect() as conn:
        task = _blocked(conn, reason="review-required: needs eyes on PR #9")
        assert task.status == "blocked"
        assert task.blocker_owner_kind == "reviewer"
        assert task.blocker_owner is None


def test_human_block_persists_owner(kanban_home):
    with kb.connect() as conn:
        task = _blocked(
            conn, reason="which pricing model?",
            owner_kind="human", owner="daniel",
        )
        assert task.blocker_owner_kind == "human"
        assert task.blocker_owner == "daniel"


def test_automation_block_never_human(kanban_home):
    with kb.connect() as conn:
        task = _blocked(
            conn, reason="waiting on the merge bot",
            owner_kind="automation", owner="daniel",
        )
        assert task.blocker_owner_kind == "automation"
        # owner dropped: automation is not a person
        assert task.blocker_owner is None


def test_parked_block_distinguishable(kanban_home):
    with kb.connect() as conn:
        task = _blocked(conn, reason="shelved for now", owner_kind="parked")
        assert task.blocker_owner_kind == "parked"


def test_block_event_carries_structured_owner(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        kb.block_task(
            conn, t, reason="need a decision",
            blocker_owner_kind="human", blocker_owner="daniel",
        )
        events = kb.list_events(conn, t)
        blocked_ev = [e for e in events if e.kind == "blocked"][-1]
        assert blocked_ev.payload["blocker_owner_kind"] == "human"
        assert blocked_ev.payload["blocker_owner"] == "daniel"


def test_invalid_owner_kind_rejected_in_block(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        with pytest.raises(ValueError):
            kb.block_task(conn, t, reason="x", blocker_owner_kind="boss")


# ---------------------------------------------------------------------------
# clearing — unblock and complete wipe ownership
# ---------------------------------------------------------------------------

def test_unblock_clears_owner(kanban_home):
    with kb.connect() as conn:
        task = _blocked(conn, reason="need input", owner_kind="human", owner="d")
        assert task.blocker_owner_kind == "human"
        assert kb.unblock_task(conn, task.id)
        reloaded = kb.get_task(conn, task.id)
        assert reloaded.blocker_owner_kind is None
        assert reloaded.blocker_owner is None


def test_complete_clears_owner(kanban_home):
    with kb.connect() as conn:
        task = _blocked(conn, reason="need input", owner_kind="human", owner="d")
        assert kb.complete_task(conn, task.id, result="done")
        reloaded = kb.get_task(conn, task.id)
        assert reloaded.status == "done"
        assert reloaded.blocker_owner_kind is None
        assert reloaded.blocker_owner is None


# ---------------------------------------------------------------------------
# migration / legacy compatibility
# ---------------------------------------------------------------------------

def test_legacy_rows_read_as_unknown(kanban_home):
    """A row with NULL ownership (legacy / un-classified) reads as unknown,
    never as a human ask."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="legacy", assignee="a")
        kb.claim_task(conn, t)
        kb.block_task(conn, t, reason="waiting on Daniel")  # no owner kind
        task = kb.get_task(conn, t)
        # No explicit classification and reason isn't review-required →
        # stays unknown (NULL), so reports don't auto-file it under a person.
        assert task.blocker_owner_kind is None
        assert task.blocker_owner is None


def test_migration_adds_columns_to_legacy_db(kanban_home):
    """Opening a DB missing the ownership columns adds them without error."""
    with kb.connect() as conn:
        conn.execute("ALTER TABLE tasks DROP COLUMN blocker_owner_kind")
        conn.execute("ALTER TABLE tasks DROP COLUMN blocker_owner")
        conn.commit()
    # Re-init should re-add the columns idempotently.
    kb.init_db()
    with kb.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "blocker_owner_kind" in cols
        assert "blocker_owner" in cols


# ---------------------------------------------------------------------------
# CLI surface — `hermes kanban block --owner-kind ... --owner ...`
# ---------------------------------------------------------------------------

def test_cli_block_flags_persist_ownership(kanban_home):
    import argparse

    from hermes_cli import kanban as kc

    with kb.connect() as conn:
        t = kb.create_task(conn, title="cli", assignee="a")
        kb.claim_task(conn, t)

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    args = parser.parse_args(
        ["kanban", "block", t, "which tier?", "--owner-kind", "human", "--owner", "daniel"]
    )
    assert kc.kanban_command(args) == 0

    with kb.connect() as conn:
        task = kb.get_task(conn, t)
    assert task.status == "blocked"
    assert task.blocker_owner_kind == "human"
    assert task.blocker_owner == "daniel"


def test_cli_block_review_required_defaults_reviewer(kanban_home):
    import argparse

    from hermes_cli import kanban as kc

    with kb.connect() as conn:
        t = kb.create_task(conn, title="cli", assignee="a")
        kb.claim_task(conn, t)

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    # No owner flags; a review-required reason must classify as reviewer.
    args = parser.parse_args(
        ["kanban", "block", t, "review-required: PR #9 needs eyes"]
    )
    assert kc.kanban_command(args) == 0

    with kb.connect() as conn:
        task = kb.get_task(conn, t)
    assert task.blocker_owner_kind == "reviewer"
    assert task.blocker_owner is None
