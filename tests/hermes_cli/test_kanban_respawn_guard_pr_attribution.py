"""Regression tests for the two-tier ``active_pr`` respawn guard window.

Bug (registered case, card t_60055c4e on a fleet board): a ready-lane card
whose prior worker posted a *delivery report* citing an EXTERNAL GitHub PR as
an artifact link (the PR belongs to another repo's workflow; its required
checks keep failing and cycle-rerunning) was deferred by ``check_respawn_guard``
with reason ``active_pr`` for the full 24 h — and every follow-up report
renewed the window, so the card could never be re-dispatched even though real
work remained and no duplicate PR risk existed.

Fix: a PR URL only counts for the FULL window when it is ATTRIBUTED to this
card's own work — i.e. the same normalized URL also appears in a structured
run handoff (``task_runs.summary``/``metadata``) or ``tasks.result``. A URL
that lives only in comment prose (QA/audit/operator reports) is held for the
short unattributed window, after which the ready card re-dispatches.

The review lane is untouched (it skips the guard entirely — see
``test_kanban_review_lifecycle.py``).
"""

from __future__ import annotations

import json
import time
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


def _backdate_comment(conn, task_id: str, age_seconds: int, body: str) -> None:
    """Add a comment then push its ``created_at`` into the past."""
    comment_id = kb.add_comment(conn, task_id, author="worker", body=body)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = ? WHERE id = ?",
            (int(time.time()) - age_seconds, comment_id),
        )


def _ready_card(conn, title: str, assignee: str = "worker") -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,),
        )
    return tid


PR_URL = "https://github.com/example-org/external-repo/pull/42"


# ---------------------------------------------------------------------------
# The registered false-positive: comment-only (unattributed) PR URL
# ---------------------------------------------------------------------------


def test_unattributed_pr_comment_releases_ready_card_after_short_window(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delivery/QA report citing an external PR (URL nowhere in this card's
    structured handoffs) must NOT hold the ready card for the full 24 h.

    Inside the short unattributed window the guard still defers (duplicate-
    storm protection immediately after the report); once that window elapses
    — while the comment is still inside the 24 h window — the card
    re-dispatches. This is the exact t_60055c4e shape: report at T, checks
    on the external PR still failing at T+11 h, card must run.
    """
    monkeypatch.setattr(
        kbd, "_RESPAWN_GUARD_PR_UNATTRIBUTED_WINDOW", 3600,
    )
    with kbc.connect() as conn:
        tid = _ready_card(conn, "ci unblock follow-up")
        report = (
            f"Delivery report: adapter landed, see {PR_URL} — external "
            "required checks are failing and cycle-rerunning; work remains."
        )
        # 11 h old: inside the 24 h window, outside the 1 h short window.
        _backdate_comment(conn, tid, 11 * 3600, report)

        assert kbd.check_respawn_guard(conn, tid) is None

        # A FRESH unattributed report still defers (short window active).
        fresh = _ready_card(conn, "fresh report card")
        kb.add_comment(conn, fresh, author="qa", body=f"see {PR_URL}")
        assert kbd.check_respawn_guard(conn, fresh) == "active_pr"


def test_unattributed_pr_comment_defers_dispatch_within_short_window(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatcher integration: within the short window the ready row is
    respawn-guarded; the same row spawns once the comment ages past it."""
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(profmod, "profile_exists", lambda name: True)
    with kbc.connect() as conn:
        tid = _ready_card(conn, "integration card")
        kb.add_comment(conn, tid, author="worker", body=f"opened {PR_URL}.")

        res = kbd.dispatch_once(conn, dry_run=True)
        assert tid not in [s[0] for s in res.spawned]
        assert dict(res.respawn_guarded).get(tid) == "active_pr"

        # Age the comment past the short window (still inside 24 h).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                (int(time.time()) - 7200, tid),
            )
        res2 = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [s[0] for s in res2.spawned]
        assert tid not in dict(res2.respawn_guarded)


# ---------------------------------------------------------------------------
# The valid case stays protected: attributed PR URL keeps the full window
# ---------------------------------------------------------------------------


def test_attributed_pr_url_keeps_full_window(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the SAME URL is declared in a structured run handoff (the worker
    that delivered on this card opened the PR), a comment citing it holds the
    full 24 h window — the duplicate-PR risk the guard exists for."""
    with kbc.connect() as conn:
        tid = _ready_card(conn, "worker opened the pr")
        # Structured handoff from a run on this very card.
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at, summary, metadata) "
                "VALUES (?, 'worker', 'done', 'blocked', ?, ?, ?, ?)",
                (
                    tid, int(time.time()) - 86000, int(time.time()) - 40000,
                    f"Implementation handoff: PR {PR_URL} opened.",
                    json.dumps({"published_pr": PR_URL}),
                ),
            )
        # Comment is 20 h old: past the 1 h short window, inside 24 h.
        _backdate_comment(
            conn, tid, 20 * 3600, f"See {PR_URL} for the change.",
        )

        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_attribution_matches_despite_punctuation_and_case(
    kanban_home: Path,
) -> None:
    """Normalization: trailing punctuation and case differences must not
    break attribution (handoff writes ``.../pull/42,`` vs comment
    ``.../PULL/42.``)."""
    with kbc.connect() as conn:
        tid = _ready_card(conn, "normalization card")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at, summary) "
                "VALUES (?, 'worker', 'done', 'blocked', ?, ?, ?)",
                (
                    tid, int(time.time()) - 86000, int(time.time()) - 40000,
                    f"handoff: {PR_URL.upper()}, checks pending",
                ),
            )
        _backdate_comment(
            conn, tid, 20 * 3600, f"details in {PR_URL}.",
        )
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_tasks_result_attribution_also_counts(
    kanban_home: Path,
) -> None:
    """``tasks.result`` (legacy/manual completion text) counts as a
    structured handoff for attribution."""
    with kbc.connect() as conn:
        tid = _ready_card(conn, "result attribution")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET result = ? WHERE id = ?",
                (f"shipped via {PR_URL}", tid),
            )
        _backdate_comment(conn, tid, 20 * 3600, f"ref {PR_URL}")
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


# ---------------------------------------------------------------------------
# Guard boundaries: unrelated URLs, review lane, other reasons untouched
# ---------------------------------------------------------------------------


def test_other_pr_urls_do_not_extend_hold(
    kanban_home: Path,
) -> None:
    """An attributed URL for PR #42 must not keep holding the card when the
    only RECENT comment cites a different PR (#43) past the short window."""
    with kbc.connect() as conn:
        tid = _ready_card(conn, "two prs")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at, summary) "
                "VALUES (?, 'worker', 'done', 'blocked', ?, ?, ?)",
                (
                    tid, int(time.time()) - 200000, int(time.time()) - 100000,
                    f"old handoff: {PR_URL}",
                ),
            )
        other = "https://github.com/example-org/external-repo/pull/43"
        _backdate_comment(conn, tid, 5 * 3600, f"audit cites {other} only")
        assert kbd.check_respawn_guard(conn, tid) is None


def test_review_lane_still_skips_active_pr(
    kanban_home: Path,
) -> None:
    """Review lane is untouched by the two-tier change: a fresh PR-URL
    comment on a review-parked card never guards (the URL is the review
    input)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="reviewer")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        kb.add_comment(conn, tid, author="worker", body=f"opened {PR_URL}")
        assert kb.request_review(
            conn, tid, summary="PR ready",
            expected_run_id=claimed.current_run_id,
        )
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_rate_limit_and_success_guards_untouched(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The change is confined to step 4: rate-limit cooldown and
    recent_success keep their semantics."""
    now = int(time.time())
    with kbc.connect() as conn:
        # recent_success: completed run within 1 h, no requeue after.
        tid = _ready_card(conn, "recent success")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'worker', 'done', "
                "'completed', ?, ?)",
                (tid, now - 300, now - 60),
            )
        assert kbd.check_respawn_guard(conn, tid) == "recent_success"

        # rate_limit_cooldown still precedes everything on the latest run.
        tid2 = _ready_card(conn, "rate limited")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, "
                "started_at, ended_at) VALUES (?, 'worker', 'rate_limited', "
                "'rate_limited', ?, ?)",
                (tid2, now - 30, now - 5),
            )
        assert kbd.check_respawn_guard(conn, tid2) == "rate_limit_cooldown"
