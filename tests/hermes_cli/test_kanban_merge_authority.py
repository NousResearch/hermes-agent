"""Acceptance contract for SMA-2026-07-21 single merge authority.

These tests intentionally exercise the real SQLite path. In particular, the
PR #257 replay seeds two already-in-flight cards, the state shape that bypassed
creation-time protection before this change.
"""
from __future__ import annotations

import json
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli import kanban_db as kb


PR_REF = "nousresearch/hermes-agent#257"


def _body(*, hard: bool, pr_ref: str = "NousResearch/hermes-agent#257") -> str:
    gate = (
        "HARD GATE: do not merge until the required independent security "
        "re-review is complete."
        if hard
        else "Merge gate: CI must be green."
    )
    return f"ROLE: VERIFIER\nMERGE-TARGET: {pr_ref}\n{gate}"


@pytest.fixture
def board(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._INITIALIZED_PATHS.clear()
    path = kb.init_db()
    conn = kb.connect(path)
    try:
        yield conn, path
    finally:
        conn.close()
        kb._INITIALIZED_PATHS.clear()


def _seed_duplicate_cards(conn, *, running: bool = False) -> tuple[str, str]:
    """Create neutral cards first, then turn them into legacy duplicates."""
    standard = kb.create_task(conn, title="neutral standard", assignee="reviewer")
    hard = kb.create_task(conn, title="neutral hard", assignee="security")
    now = 1_700_000_000
    status = "running" if running else "ready"
    started = now if running else None
    conn.execute(
        "UPDATE tasks SET title = ?, body = ?, status = ?, started_at = ?, "
        "claim_lock = ? WHERE id = ?",
        (
            "VERIFY+MERGE CI verifier",
            _body(hard=False),
            status,
            started,
            "in-flight-standard" if running else None,
            standard,
        ),
    )
    conn.execute(
        "UPDATE tasks SET title = ?, body = ?, status = ?, started_at = ?, "
        "claim_lock = ? WHERE id = ?",
        (
            "Orphaned PR verifier",
            _body(hard=True),
            status,
            started,
            "in-flight-hard" if running else None,
            hard,
        ),
    )
    return standard, hard


# AC-1: marker + legacy parsing and fail-closed behavior.
@pytest.mark.parametrize(
    ("text", "default_repo", "expected", "source"),
    [
        ("MERGE-TARGET: NousResearch/hermes-agent#257", None, PR_REF, "marker"),
        (
            "Review https://github.com/NousResearch/hermes-agent/pull/257/files",
            None,
            PR_REF,
            "github_url",
        ),
        (
            "(https://github.com/NousResearch/hermes-agent/pull/257)",
            None,
            PR_REF,
            "github_url",
        ),
        ("Verifier for #257", "NousResearch/hermes-agent", PR_REF, "bare"),
        (
            "MERGE-TARGET: NousResearch/hermes-agent#257\n"
            "MERGE-TARGET: nousresearch/hermes-agent#257",
            None,
            PR_REF,
            "marker",
        ),
    ],
)
def test_ac1_parse_merge_target(text, default_repo, expected, source):
    parsed = kb.parse_merge_target(text, default_repo=default_repo)
    assert parsed.ok is True
    assert parsed.pr_ref == expected
    assert parsed.source == source


@pytest.mark.parametrize(
    ("text", "default_repo", "error_fragment"),
    [
        ("MERGE-TARGET: not-a-pr", None, "invalid"),
        (
            "MERGE-TARGET: NousResearch/hermes-agent#257\nMERGE-TARGET:",
            None,
            "invalid",
        ),
        (
            "MERGE-TARGET: broken\n"
            "https://github.com/NousResearch/hermes-agent/pull/257",
            None,
            "invalid",
        ),
        (
            "MERGE-TARGET: a/repo#1\nMERGE-TARGET: b/repo#2",
            None,
            "ambiguous",
        ),
        (
            "https://github.com/a/repo/pull/1 and https://github.com/b/repo/pull/2",
            None,
            "ambiguous",
        ),
        ("Verifier for #257", None, "default_repo"),
        ("Verifier for #257 and #258", "NousResearch/hermes-agent", "ambiguous"),
        (
            "https://github.com/NousResearch/hermes-agent/pull/257x",
            None,
            "no parseable",
        ),
        ("No PR here", "NousResearch/hermes-agent", "no parseable"),
    ],
)
def test_ac1_parse_merge_target_fails_closed(text, default_repo, error_fragment):
    parsed = kb.parse_merge_target(text, default_repo=default_repo)
    assert parsed.ok is False
    assert parsed.pr_ref is None
    assert error_fragment in (parsed.error or "")


def test_ac1_bare_target_uses_board_default_repo_end_to_end(board):
    conn, _ = board
    kb.write_board_metadata("default", default_repo="NousResearch/hermes-agent")
    task_id = kb.create_task(
        conn,
        title="VERIFY+MERGE legacy verifier #257",
        body="Merge gate: CI must be green.",
        assignee="reviewer",
    )
    candidates = kb.list_merge_authority_candidates(conn)
    assert [(candidate.task_id, candidate.pr_ref) for candidate in candidates] == [
        (task_id, PR_REF)
    ]
    assert kb.check_merge_authority(conn, task_id).allowed is True


# AC-2: gate strength, seniority, then lexicographic task id is a total order.
def test_ac2_arbitration_total_order():
    candidates = [
        kb.MergeAuthorityCandidate("t_old_standard", PR_REF, "standard", 1, "ready"),
        kb.MergeAuthorityCandidate("t_new_hard", PR_REF, "hard", 9, "ready"),
        kb.MergeAuthorityCandidate("t_z_hard", PR_REF, "hard", 2, "ready"),
        kb.MergeAuthorityCandidate("t_a_hard", PR_REF, "hard", 2, "ready"),
    ]
    winner, reason = kb.arbitrate_merge_authority(candidates)
    assert winner.task_id == "t_a_hard"
    assert "hard-gated" in reason
    assert kb.merge_gate_strength("Orphaned PR verifier", _body(hard=True)) == "hard"
    assert kb.merge_gate_strength("VERIFY+MERGE", _body(hard=False)) == "standard"


# AC-3: create_task arbitrates before its write transaction commits.
def test_ac3_creation_time_dedupe_and_ledger(board):
    conn, _ = board
    standard = kb.create_task(
        conn,
        title="VERIFY+MERGE CI verifier",
        body=_body(hard=False),
        assignee="reviewer",
    )
    hard = kb.create_task(
        conn,
        title="Orphaned PR verifier",
        body=_body(hard=True),
        assignee="security",
    )

    assert kb.get_task(conn, standard).status == "archived"
    assert kb.get_task(conn, hard).status == "ready"
    ledger = conn.execute(
        "SELECT * FROM merge_authority WHERE pr_ref = ?", (PR_REF,)
    ).fetchone()
    assert ledger["task_id"] == hard
    assert ledger["gate_strength"] == "hard"
    assert ledger["last_disposition"] == "deduped"
    assert kb.check_merge_authority(conn, hard).allowed is True
    for task_id in (standard, hard):
        comments = [
            comment for comment in kb.list_comments(conn, task_id)
            if comment.author == kb.MERGE_AUTHORITY_COMMENT_AUTHOR
        ]
        events = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == kb.MERGE_AUTHORITY_EVENT_KIND
        ]
        assert len(comments) == 1
        assert len(events) == 1
        payload = events[0].payload
        assert payload["pr_ref"] == PR_REF
        assert payload["kept_task_id"] == hard
        assert set(payload["demoted_task_ids"] + payload["archived_task_ids"]) == {
            standard
        }
        assert payload["reason"]


# AC-4: periodic/manual sweep is idempotent and runs before dispatch work.
def test_ac4_sweep_dispatch_step_zero_and_manual_cli_are_idempotent(
    board, monkeypatch, capsys
):
    conn, _ = board
    standard, hard = _seed_duplicate_cards(conn)

    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _name: True)
    result = kb.dispatch_once(conn, dry_run=True, max_spawn=0)
    assert len(result.merge_authority_actions) == 1
    assert result.merge_authority_actions[0].kept_task_id == hard
    assert kb.get_task(conn, standard).status == "archived"

    before_comments = conn.execute(
        "SELECT COUNT(*) FROM task_comments WHERE author = ?",
        (kb.MERGE_AUTHORITY_COMMENT_AUTHOR,),
    ).fetchone()[0]
    before_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = ?",
        (kb.MERGE_AUTHORITY_EVENT_KIND,),
    ).fetchone()[0]

    from hermes_cli import kanban

    rc = kanban._cmd_authority(Namespace(authority_action="sweep", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == {"dedupe_count": 0, "actions": []}
    assert kb.sweep_merge_authority(conn) == []
    assert conn.execute(
        "SELECT COUNT(*) FROM task_comments WHERE author = ?",
        (kb.MERGE_AUTHORITY_COMMENT_AUTHOR,),
    ).fetchone()[0] == before_comments
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE kind = ?",
        (kb.MERGE_AUTHORITY_EVENT_KIND,),
    ).fetchone()[0] == before_events


# AC-5: replay the two-in-flight-verifier incident for PR #257.
def test_ac5_pr257_hard_security_rereview_keeps_authority(board, monkeypatch):
    conn, _ = board
    standard, hard = _seed_duplicate_cards(conn, running=True)
    # The hard-gated card has started but is waiting on the required security
    # re-review; the CI-only competitor is runnable/in-flight. Gate strength,
    # not current runnability, determines authority.
    conn.execute(
        "UPDATE tasks SET status = 'blocked', claim_lock = NULL WHERE id = ?",
        (hard,),
    )

    # Keep the test focused on dispatch step zero; the seeded synthetic worker
    # locks are intentionally not real PIDs and must not be reclaimed here.
    monkeypatch.setattr(kb, "release_stale_claims", lambda _conn: 0)
    monkeypatch.setattr(
        kb, "detect_stale_running", lambda _conn, stale_timeout_seconds=0: []
    )
    monkeypatch.setattr(kb, "detect_crashed_workers", lambda _conn: [])
    monkeypatch.setattr(kb, "enforce_max_runtime", lambda _conn: [])
    result = kb.dispatch_once(conn, dry_run=True, max_spawn=0)

    assert result.merge_authority_actions[0].kept_task_id == hard
    assert kb.get_task(conn, hard).status == "blocked"
    loser = kb.get_task(conn, standard)
    assert loser.status == "running"
    assert loser.title.startswith("[ADVISORY]")
    assert "STAND-DOWN" in (loser.body or "")
    assert "MERGE-AUTHORITY: NONE" in (loser.body or "")
    assert kb.check_merge_authority(conn, hard).allowed is True
    denied = kb.check_merge_authority(conn, standard)
    assert denied.allowed is False
    assert "advisory/stand-down" in denied.reason
    event = next(
        event for event in kb.list_events(conn, standard)
        if event.kind == kb.MERGE_AUTHORITY_EVENT_KIND
    )
    assert event.payload["pr_ref"] == PR_REF
    assert event.payload["kept_task_id"] == hard
    assert event.payload["demoted_task_ids"] == [standard]
    assert event.payload["reason"]

    from tools import kanban_tools

    monkeypatch.setenv("HERMES_KANBAN_TASK", hard)
    allowed_tool_result = json.loads(kanban_tools._handle_authority({}))
    assert allowed_tool_result["allowed"] is True

    monkeypatch.setenv("HERMES_KANBAN_TASK", standard)
    tool_result = json.loads(kanban_tools._handle_authority({}))
    assert tool_result["allowed"] is False


# AC-6: BEGIN IMMEDIATE makes concurrent creates converge on one winner.
def test_ac6_concurrent_create_safety(board):
    conn, path = board
    barrier = threading.Barrier(2)

    def create(hard: bool) -> str:
        local = kb.connect(path)
        try:
            barrier.wait(timeout=5)
            return kb.create_task(
                local,
                title="Orphaned PR verifier" if hard else "VERIFY+MERGE CI verifier",
                body=_body(hard=hard),
                assignee="security" if hard else "reviewer",
            )
        finally:
            local.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(create, (False, True)))

    candidates = [
        candidate
        for candidate in kb.list_merge_authority_candidates(conn)
        if candidate.pr_ref == PR_REF
    ]
    assert len(candidates) == 1
    assert candidates[0].gate_strength == "hard"
    assert {kb.get_task(conn, task_id).status for task_id in ids} == {
        "ready", "archived"
    }
    ledger = conn.execute(
        "SELECT task_id FROM merge_authority WHERE pr_ref = ?", (PR_REF,)
    ).fetchone()
    assert ledger["task_id"] == candidates[0].task_id


# AC-7: unrelated/non-verifier cards are strict no-ops.
def test_ac7_unrelated_and_non_verifier_cards_are_unchanged(board):
    conn, _ = board
    ordinary = kb.create_task(
        conn,
        title="Implement feature",
        body="MERGE-TARGET: NousResearch/hermes-agent#257",
        assignee="builder",
    )
    other_pr = kb.create_task(
        conn,
        title="VERIFY+MERGE other PR",
        body=_body(hard=False, pr_ref="NousResearch/hermes-agent#258"),
        assignee="reviewer",
    )
    before = {
        task_id: (kb.get_task(conn, task_id).title, kb.get_task(conn, task_id).status)
        for task_id in (ordinary, other_pr)
    }
    assert kb.sweep_merge_authority(conn) == []
    after = {
        task_id: (kb.get_task(conn, task_id).title, kb.get_task(conn, task_id).status)
        for task_id in (ordinary, other_pr)
    }
    assert after == before
    assert kb.check_merge_authority(conn, ordinary).allowed is False


# AC-8: unsafe losers are advisory and cannot pass the merge-time check.
@pytest.mark.parametrize("child_status", ["todo", "done", "archived"])
def test_ac8_never_started_loser_with_any_child_is_demoted(board, child_status):
    conn, _ = board
    standard, hard = _seed_duplicate_cards(conn)
    child = kb.create_task(
        conn, title="dependent implementation", assignee="builder", parents=[standard]
    )
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (child_status, child))

    actions = kb.sweep_merge_authority(conn)
    assert actions[0].kept_task_id == hard
    assert actions[0].demoted_task_ids == (standard,)
    assert actions[0].archived_task_ids == ()
    assert kb.get_task(conn, standard).status == "ready"
    assert kb.get_task(conn, child).status == child_status
    assert kb.check_merge_authority(conn, standard).allowed is False


# AC-9: unparseable, ambiguous, missing, and stale ledger states all deny.
def test_ac9_fail_closed_ambiguous_unparseable_and_stale_ledger(board):
    conn, _ = board
    winner = kb.create_task(
        conn,
        title="VERIFY+MERGE verifier",
        body=_body(hard=True),
        assignee="security",
    )
    assert kb.check_merge_authority(conn, winner).allowed is True

    conn.execute("DELETE FROM merge_authority WHERE pr_ref = ?", (PR_REF,))
    assert "stale ledger" in kb.check_merge_authority(conn, winner).reason

    kb.sweep_merge_authority(conn)
    conn.execute(
        "UPDATE merge_authority SET task_id = 't_stale' WHERE pr_ref = ?", (PR_REF,)
    )
    assert "stale ledger" in kb.check_merge_authority(conn, winner).reason

    ambiguous = kb.create_task(conn, title="neutral verifier seed", assignee="reviewer")
    conn.execute(
        "UPDATE tasks SET title = ?, body = ? WHERE id = ?",
        ("VERIFY+MERGE duplicate", _body(hard=False), ambiguous),
    )
    assert "ambiguous/no authority" in kb.check_merge_authority(conn, winner).reason

    conn.execute(
        "UPDATE tasks SET body = 'MERGE-TARGET: malformed' WHERE id = ?", (winner,)
    )
    unparseable = kb.check_merge_authority(conn, winner)
    assert unparseable.allowed is False
    assert "invalid" in unparseable.reason


def test_schema_migration_is_idempotent_and_preserves_ledger(board):
    conn, path = board
    columns = {
        row["name"]: row for row in conn.execute("PRAGMA table_info(merge_authority)")
    }
    assert set(columns) == {
        "pr_ref", "task_id", "gate_strength", "claimed_at", "last_disposition"
    }
    assert columns["pr_ref"]["pk"] == 1
    conn.execute(
        "INSERT INTO merge_authority VALUES (?, ?, ?, ?, ?)",
        ("example/repo#1", "t_existing", "standard", 123, "test"),
    )
    kb.init_db(path)
    reopened = conn.execute(
        "SELECT * FROM merge_authority WHERE pr_ref = 'example/repo#1'"
    ).fetchone()
    assert reopened["task_id"] == "t_existing"


def test_authority_cli_check_returns_nonzero_when_denied(board, capsys):
    conn, _ = board
    task_id = kb.create_task(conn, title="ordinary", assignee="builder")
    from hermes_cli import kanban

    rc = kanban._cmd_authority(
        Namespace(authority_action="check", task_id=task_id, json=True)
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["allowed"] is False
