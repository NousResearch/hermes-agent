"""Tests for the explicit, audited "close as merged" Kanban workflow.

Covers ``kanban_db.close_task_as_merged`` (DB layer) and the
``hermes kanban close-merged`` CLI verb. The scenarios mirror the
requirements: a no-merge card later explicitly authorized and closed
merged, missing/invalid evidence failing safely, ordinary completion
staying unchanged, and archived/terminal consistency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


# The kinds the gateway kanban-notifier subscribes to (mirrors
# ``gateway.kanban_watchers.TERMINAL_KINDS`` — a function-local constant we
# can't import). Used to prove the merged-close is auto-delivered to Discord
# et al. through the shared ``completed`` event path.
NOTIFIER_TERMINAL_KINDS = (
    "completed", "blocked", "gave_up", "crashed", "timed_out",
    "status", "archived", "unblocked",
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _events(conn, task_id):
    return kb.list_events(conn, task_id)


def _event_kinds(conn, task_id):
    return [e.kind for e in _events(conn, task_id)]


def _first_event(conn, task_id, kind):
    for e in _events(conn, task_id):
        if e.kind == kind:
            return e
    return None


def _make_blocked_card(conn, *, title="ship feature", body="the original body"):
    """A card that was worked but never merged — then blocked for review."""
    tid = kb.create_task(conn, title=title, body=body, initial_status="running")
    assert kb.block_task(conn, tid, reason="needs review", kind="needs_input")
    task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    return tid


# ---------------------------------------------------------------------------
# Happy path — no-merge card explicitly authorized and closed merged
# ---------------------------------------------------------------------------

def test_close_merged_blocked_card_success(kanban_home):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        kinds_before = _event_kinds(conn, tid)

        ok = kb.close_task_as_merged(
            conn, tid,
            reason="PR landed on main after review",
            pr="https://github.com/NousResearch/hermes-agent/pull/123",
            commit="abc1234",
            authorized=True,
            authorized_by="bishop",
        )
        assert ok is True

        task = kb.get_task(conn, tid)
        # Terminal done state with a clear merged result.
        assert task.status == "done"
        assert task.completed_at is not None
        assert "Closed as merged" in task.result
        assert "#123" not in task.result  # full URL is normalized, not a bare #
        assert "pull/123" in task.result
        # Immutable body preserved.
        assert task.body == "the original body"
        # Block bookkeeping cleared on the terminal transition.
        assert task.block_kind is None
        assert task.block_recurrences == 0

        # Durable, dedicated audit event with full metadata.
        ev = _first_event(conn, tid, "closed_merged")
        assert ev is not None
        assert ev.payload["outcome"] == "merged"
        assert ev.payload["pull_request"] == (
            "https://github.com/NousResearch/hermes-agent/pull/123"
        )
        assert ev.payload["commit"] == "abc1234"
        assert ev.payload["reason"] == "PR landed on main after review"
        assert ev.payload["authorized"] is True
        assert ev.payload["authorized_by"] == "bishop"
        assert ev.payload["prior_status"] == "blocked"

        # Prior audit history is preserved (not rewritten).
        for k in kinds_before:
            assert k in _event_kinds(conn, tid)

        # A run carries the merged outcome for attempt history / stats.
        runs = kb.list_runs(conn, tid)
        assert any(r.outcome == "merged" for r in runs)


def test_close_merged_emits_completed_event_for_discord_delivery(kanban_home):
    """The merged-close must auto-deliver through the notifier's shared
    ``completed`` path, and the non-terminal ``closed_merged`` audit row must
    not wedge the cursor."""
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        kb.add_notify_sub(
            conn, task_id=tid, platform="discord", chat_id="chan-1",
        )
        assert kb.close_task_as_merged(
            conn, tid,
            reason="merged",
            pr="42",
            commit="deadbeef",
            authorized=True,
        )

        # The completed event exists and is flagged as a merged close.
        completed = _first_event(conn, tid, "completed")
        assert completed is not None
        assert completed.payload["closed_as_merged"] is True
        assert completed.payload["outcome"] == "merged"
        assert completed.payload["summary"].startswith("Closed as merged")

        # What the gateway notifier would actually claim & deliver.
        _old, _new, delivered = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="discord", chat_id="chan-1",
            kinds=NOTIFIER_TERMINAL_KINDS,
        )
        delivered_kinds = [e.kind for e in delivered]
        # The completed event IS delivered (Discord auto-delivery confirmed);
        # the closed_merged audit row is skipped, not wedged behind the cursor.
        assert "completed" in delivered_kinds
        assert "closed_merged" not in delivered_kinds


@pytest.mark.parametrize(
    "pr_in,expected",
    [
        (123, "#123"),
        ("123", "#123"),
        ("#123", "#123"),
        (
            "https://github.com/o/r/pull/7/files",
            "https://github.com/o/r/pull/7",
        ),
        (
            "https://github.com/o/r/pull/7#discussion_r99",
            "https://github.com/o/r/pull/7",
        ),
    ],
)
def test_pr_reference_normalization(pr_in, expected):
    assert kb._normalize_pr_reference(pr_in) == expected


def test_commit_sha_is_lowercased(kanban_home):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        assert kb.close_task_as_merged(
            conn, tid, reason="x", pr="1", commit="ABCDEF1234", authorized=True,
        )
        ev = _first_event(conn, tid, "closed_merged")
        assert ev.payload["commit"] == "abcdef1234"


# ---------------------------------------------------------------------------
# Missing / invalid evidence fails safely (no mutation)
# ---------------------------------------------------------------------------

def test_close_merged_requires_explicit_authorization(kanban_home):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        with pytest.raises(kb.MergeAuthorizationError):
            kb.close_task_as_merged(
                conn, tid, reason="x", pr="1", commit="abc1234",
                authorized=False,
            )
        # Untouched.
        assert kb.get_task(conn, tid).status == "blocked"
        assert "closed_merged" not in _event_kinds(conn, tid)


@pytest.mark.parametrize("reason", ["", "   ", "\n\t "])
def test_close_merged_requires_non_empty_reason(kanban_home, reason):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        with pytest.raises(kb.MergeEvidenceError):
            kb.close_task_as_merged(
                conn, tid, reason=reason, pr="1", commit="abc1234",
                authorized=True,
            )
        assert kb.get_task(conn, tid).status == "blocked"
        assert "closed_merged" not in _event_kinds(conn, tid)


@pytest.mark.parametrize(
    "bad_pr",
    [
        None,
        "",
        "   ",
        0,
        -3,
        True,
        "not-a-pr",
        "https://github.com/o/r/compare/main...feat",
        "https://github.com/o/r/commit/abc1234",
        "https://gitlab.com/o/r/pull/1",
        "feature/my-branch",
    ],
)
def test_close_merged_rejects_invalid_pr(kanban_home, bad_pr):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        with pytest.raises(kb.MergeEvidenceError):
            kb.close_task_as_merged(
                conn, tid, reason="x", pr=bad_pr, commit="abc1234",
                authorized=True,
            )
        assert kb.get_task(conn, tid).status == "blocked"
        assert "closed_merged" not in _event_kinds(conn, tid)


@pytest.mark.parametrize(
    "bad_sha",
    [
        None,
        "",
        "   ",
        True,
        "HEAD",
        "main",
        "master",
        "feature/x",
        "abc",          # too short (< 7)
        "z" * 8,        # non-hex
        "g1234567",     # non-hex leading char
        "a" * 41,       # too long (> 40)
    ],
)
def test_close_merged_rejects_invalid_commit(kanban_home, bad_sha):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        with pytest.raises(kb.MergeEvidenceError):
            kb.close_task_as_merged(
                conn, tid, reason="x", pr="1", commit=bad_sha,
                authorized=True,
            )
        assert kb.get_task(conn, tid).status == "blocked"
        assert "closed_merged" not in _event_kinds(conn, tid)


# ---------------------------------------------------------------------------
# Ordinary completion is unchanged
# ---------------------------------------------------------------------------

def test_ordinary_completion_unchanged(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="normal", initial_status="running")
        assert kb.complete_task(conn, tid, result="did the thing")
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.result == "did the thing"
        # No merged provenance leaked into the ordinary path.
        assert "closed_merged" not in _event_kinds(conn, tid)
        assert "completed" in _event_kinds(conn, tid)
        runs = kb.list_runs(conn, tid)
        assert any(r.outcome == "completed" for r in runs)
        assert not any(r.outcome == "merged" for r in runs)


# ---------------------------------------------------------------------------
# Archived / terminal consistency
# ---------------------------------------------------------------------------

def test_close_merged_rejects_already_done(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="done card", initial_status="running")
        assert kb.complete_task(conn, tid, result="ordinary result")
        # Cannot merged-close a card that already reached done.
        ok = kb.close_task_as_merged(
            conn, tid, reason="x", pr="1", commit="abc1234", authorized=True,
        )
        assert ok is False
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.result == "ordinary result"  # untouched
        assert "closed_merged" not in _event_kinds(conn, tid)


def test_close_merged_rejects_archived(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="arch card", initial_status="running")
        assert kb.complete_task(conn, tid, result="r")
        assert kb.archive_task(conn, tid)
        assert kb.get_task(conn, tid).status == "archived"
        ok = kb.close_task_as_merged(
            conn, tid, reason="x", pr="1", commit="abc1234", authorized=True,
        )
        assert ok is False
        assert kb.get_task(conn, tid).status == "archived"


def test_close_merged_double_close_is_safe(kanban_home):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
        assert kb.close_task_as_merged(
            conn, tid, reason="first", pr="1", commit="abc1234", authorized=True,
        )
        # Second attempt is a no-op — card is already terminal.
        assert kb.close_task_as_merged(
            conn, tid, reason="second", pr="2", commit="def5678", authorized=True,
        ) is False
        merged_events = [e for e in _events(conn, tid) if e.kind == "closed_merged"]
        assert len(merged_events) == 1
        assert merged_events[0].payload["reason"] == "first"


def test_close_merged_unknown_task_returns_false(kanban_home):
    with kb.connect() as conn:
        assert kb.close_task_as_merged(
            conn, "t_doesnotexist", reason="x", pr="1", commit="abc1234",
            authorized=True,
        ) is False


# ---------------------------------------------------------------------------
# CLI verb: hermes kanban close-merged
# ---------------------------------------------------------------------------

def _parse(tokens):
    """Parse a ``kanban …`` token list through the real argparse tree."""
    root = argparse.ArgumentParser(prog="hermes")
    subs = root.add_subparsers(dest="command")
    kanban_parser = kc.build_parser(subs)
    kanban_parser.set_defaults(_kanban_parser=kanban_parser)
    return root.parse_args(["kanban", *tokens])


def test_cli_close_merged_success(kanban_home, capsys):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
    args = _parse([
        "close-merged", tid,
        "--yes-merged",
        "--reason", "landed in prod",
        "--pr", "123",
        "--commit", "abc1234",
    ])
    assert kc.kanban_command(args) == 0
    out = capsys.readouterr().out
    assert "as merged" in out
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "done"


def test_cli_close_merged_json(kanban_home, capsys):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
    args = _parse([
        "close-merged", tid,
        "--yes-merged", "--reason", "x", "--pr", "9", "--commit", "abcdef0",
        "--json",
    ])
    assert kc.kanban_command(args) == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["ok"] is True
    assert payload["outcome"] == "merged"
    assert payload["status"] == "done"


def test_cli_close_merged_without_authorization_flag_fails(kanban_home, capsys):
    """Omitting --yes-merged is refused (the DB raises; the handler reports)."""
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
    args = _parse([
        "close-merged", tid,
        "--reason", "x", "--pr", "1", "--commit", "abc1234",
    ])
    rc = kc.kanban_command(args)
    assert rc != 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_cli_close_merged_invalid_evidence_fails(kanban_home):
    with kb.connect() as conn:
        tid = _make_blocked_card(conn)
    args = _parse([
        "close-merged", tid,
        "--yes-merged", "--reason", "x", "--pr", "not-a-pr", "--commit", "abc1234",
    ])
    assert kc.kanban_command(args) != 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize(
    "missing",
    [
        ["close-merged", "t_x", "--yes-merged", "--pr", "1", "--commit", "abc1234"],
        ["close-merged", "t_x", "--yes-merged", "--reason", "r", "--commit", "abc1234"],
        ["close-merged", "t_x", "--yes-merged", "--reason", "r", "--pr", "1"],
    ],
)
def test_cli_close_merged_requires_evidence_flags(kanban_home, missing):
    # argparse enforces required=True on --reason/--pr/--commit.
    with pytest.raises(SystemExit):
        _parse(missing)
