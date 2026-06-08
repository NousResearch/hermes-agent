"""Tests for ``hermes kanban resolve-fanin`` — the Kanban root-fix slice
that classifies a final review task's verdict / ack / remediation and,
in apply mode, creates one deduped fix card + one fix-review card.

Bounded scope: see /tmp/t_330e19c2_claude_prompt.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Helpers — build a small graph: origin -> impl -> final_review (done).
# ---------------------------------------------------------------------------

def _make_graph(*, verdict: str, ack_status: str | None = None,
                unsafe_phrase: str | None = None):
    """Create origin → impl → final_review and mark final_review done with
    a ``Verdict: <verdict>`` summary. Returns ``(origin, impl, final)``.

    When ``unsafe_phrase`` is supplied, it is appended to the final
    review's result so the safety guard fires.
    """
    conn = kb.connect()
    try:
        origin = kb.create_task(conn, title="origin work")
        kb.complete_task(conn, origin, result="origin done")
        impl = kb.create_task(conn, title="impl",
                              parents=[origin])
        kb.complete_task(conn, impl, result="impl done")
        final = kb.create_task(
            conn,
            title="final review",
            parents=[impl],
            assignee="ccreviewer",
        )
        # ready -> done with the verdict baked into summary.
        body = f"Verdict: {verdict}\nreview-required: parent worker asked for human review."
        if unsafe_phrase:
            body += f"\n{unsafe_phrase}"
        ok = kb.complete_task(conn, final, result=body, summary=body)
        assert ok, "fixture: final review task did not transition to done"

        if ack_status:
            # Record ack outcome as a comment with the documented prefix.
            kb.add_comment(conn, final, author="gateway-watchdog",
                           body=f"ack-status: {ack_status}")
    finally:
        conn.close()
    return origin, impl, final


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classify_go_with_ack_failed_is_not_need_more(kanban_home):
    """GO verdict + ACK failure must classify as task_verdict=GO,
    ack_status=FAILED, remediation_status=NONE. The ack failure is an
    operational/delivery issue — it must NOT be misreported as a code
    NEED_MORE that triggers a fix card."""
    _, _, final = _make_graph(verdict="GO", ack_status="FAILED")

    out = kc.run_slash(f"resolve-fanin {final} --dry-run --json")
    payload = _extract_json(out)

    assert payload["task_verdict"] == "GO"
    assert payload["ack_status"] == "FAILED"
    assert payload["remediation_status"] == "NONE"
    assert payload["remediation_task_ids"] in (None, {}, {"fix": None, "fix_review": None})


def test_classify_reads_verdict_from_run_summary_when_result_empty(kanban_home):
    """kanban_complete commonly stores the final ACK text in run
    summary/event payload while tasks.result is empty; resolver must
    still see the Verdict marker."""
    conn = kb.connect()
    try:
        origin = kb.create_task(conn, title="origin work")
        kb.complete_task(conn, origin, result="origin done")
        final = kb.create_task(conn, title="final review", parents=[origin])
        summary = "Verdict: BLOCK\nreview-required: summary-only blocker"
        assert kb.complete_task(conn, final, result=None, summary=summary)
    finally:
        conn.close()

    out = kc.run_slash(f"resolve-fanin {final} --dry-run --json")
    payload = _extract_json(out)
    assert payload["task_verdict"] == "BLOCK"
    assert payload["remediation_status"] == "REQUIRED"


def test_dry_run_writes_nothing_on_block(kanban_home):
    """A BLOCK verdict in dry-run reports REQUIRED but creates no cards."""
    _, _, final = _make_graph(verdict="BLOCK")

    before_n = _board_task_count()
    out = kc.run_slash(f"resolve-fanin {final} --dry-run --json")
    after_n = _board_task_count()

    payload = _extract_json(out)
    assert payload["task_verdict"] == "BLOCK"
    assert payload["remediation_status"] == "REQUIRED"
    assert payload["dry_run"] is True
    assert payload["remediation_task_ids"] in (None, {"fix": None, "fix_review": None})
    assert after_n == before_n, "dry-run must not insert any cards"


def test_apply_block_creates_dedup_fix_and_fix_review(kanban_home):
    """BLOCK verdict + --apply creates exactly one fix card and one
    fix-review card, both linked to the final task. A second run must
    not duplicate either card."""
    _, _, final = _make_graph(verdict="BLOCK")

    out1 = kc.run_slash(
        f"resolve-fanin {final} --apply --json "
        f"--fix-assignee ccsupervisor --review-assignee ccreviewer"
    )
    payload1 = _extract_json(out1)
    assert payload1["task_verdict"] == "BLOCK"
    assert payload1["remediation_status"] == "CREATED"
    fix_id = payload1["remediation_task_ids"]["fix"]
    rev_id = payload1["remediation_task_ids"]["fix_review"]
    assert fix_id and rev_id and fix_id != rev_id

    conn = kb.connect()
    try:
        fix = kb.get_task(conn, fix_id)
        rev = kb.get_task(conn, rev_id)
        assert fix is not None and rev is not None
        # Fix-review depends on fix.
        assert fix_id in kb.parent_ids(conn, rev_id)
        # Sanitization: no raw "review-required:" sentinel leak in bodies.
        assert "review-required:" not in (fix.body or "")
        assert "review-required:" not in (rev.body or "")
        # Assignees flowed through.
        assert fix.assignee == "ccsupervisor"
        assert rev.assignee == "ccreviewer"
    finally:
        conn.close()

    # Second run is a no-op for card creation.
    out2 = kc.run_slash(
        f"resolve-fanin {final} --apply --json "
        f"--fix-assignee ccsupervisor --review-assignee ccreviewer"
    )
    payload2 = _extract_json(out2)
    assert payload2["remediation_task_ids"]["fix"] == fix_id
    assert payload2["remediation_task_ids"]["fix_review"] == rev_id
    # No third card appeared.
    conn = kb.connect()
    try:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE id IN (?, ?)", (fix_id, rev_id)
        ).fetchall()
        assert len(rows) == 2
        # Total task count grew by exactly 2 over the original graph.
        # origin + impl + final + fix + fix_review = 5.
        total = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
        assert total == 5
    finally:
        conn.close()


def test_apply_unsafe_blocker_refuses_card_creation(kanban_home):
    """A BLOCK verdict whose review body mentions an unsafe sentinel
    category (secret leak, destructive data loss, auth/credential, user
    input required, live money/trading) must classify as
    remediation_status=BLOCKED and create no cards even in --apply."""
    _, _, final = _make_graph(
        verdict="BLOCK",
        unsafe_phrase="reason: leaked SECRET_KEY in commit",
    )

    before_n = _board_task_count()
    out = kc.run_slash(f"resolve-fanin {final} --apply --json")
    after_n = _board_task_count()
    payload = _extract_json(out)

    assert payload["task_verdict"] == "BLOCK"
    assert payload["remediation_status"] == "BLOCKED"
    assert payload["remediation_task_ids"] in (None, {"fix": None, "fix_review": None})
    assert after_n == before_n


def test_apply_need_more_creates_fix_with_need_more_marker(kanban_home):
    """NEED_MORE verdict also drives remediation; created cards must
    reflect that this came from NEED_MORE (not generic BLOCK)."""
    _, _, final = _make_graph(verdict="NEED_MORE")

    out = kc.run_slash(f"resolve-fanin {final} --apply --json")
    payload = _extract_json(out)

    assert payload["task_verdict"] == "NEED_MORE"
    assert payload["remediation_status"] == "CREATED"
    assert payload["remediation_task_ids"]["fix"]
    assert payload["remediation_task_ids"]["fix_review"]


def test_apply_can_gate_final_reporter_behind_fix_review(kanban_home):
    """When requested, apply mode creates one deduped final fan-in
    reporter gated behind the fix-review card and reports it in the
    machine-readable ledger."""
    _, _, final = _make_graph(verdict="BLOCK")

    out1 = kc.run_slash(
        f"resolve-fanin {final} --apply --json "
        f"--fix-assignee ccsupervisor --review-assignee ccreviewer "
        f"--reporter-assignee ccsupervisor"
    )
    payload1 = _extract_json(out1)
    review_id = payload1["remediation_task_ids"]["fix_review"]
    reporter_id = payload1["reporter_task_id"]
    assert reporter_id

    conn = kb.connect()
    try:
        reporter = kb.get_task(conn, reporter_id)
        assert reporter is not None
        assert reporter.assignee == "ccsupervisor"
        assert review_id in kb.parent_ids(conn, reporter_id)
    finally:
        conn.close()

    out2 = kc.run_slash(
        f"resolve-fanin {final} --apply --json "
        f"--fix-assignee ccsupervisor --review-assignee ccreviewer "
        f"--reporter-assignee ccsupervisor"
    )
    payload2 = _extract_json(out2)
    assert payload2["reporter_task_id"] == reporter_id
    assert _board_task_count() == 6


def test_apply_respects_max_fan_in_threshold(kanban_home):
    """The apply path is bounded by transitive graph size so a single
    command cannot silently spawn remediation for an unexpectedly large
    fan-in. This uses the real temp kanban DB fixture and verifies no
    cards are inserted when the bound is exceeded."""
    _, _, final = _make_graph(verdict="BLOCK")

    before_n = _board_task_count()
    out = kc.run_slash(
        f"resolve-fanin {final} --apply --json --max-fan-in-threshold 2"
    )
    after_n = _board_task_count()
    payload = _extract_json(out)

    assert payload["graph_task_count"] == 3
    assert payload["max_fan_in_threshold"] == 2
    assert payload["remediation_status"] == "BLOCKED"
    assert "exceeds threshold 2" in payload["blocked_reason"]
    assert payload["remediation_task_ids"] in (
        None,
        {"fix": None, "fix_review": None},
    )
    assert after_n == before_n


def test_apply_allows_explicitly_raised_fan_in_threshold(kanban_home):
    """Operators can raise the threshold after inspecting a larger graph;
    the real DB apply path then creates the deduped remediation pair."""
    _, _, final = _make_graph(verdict="BLOCK")

    out = kc.run_slash(
        f"resolve-fanin {final} --apply --json --max-fan-in-threshold 3"
    )
    payload = _extract_json(out)

    assert payload["graph_task_count"] == 3
    assert payload["max_fan_in_threshold"] == 3
    assert payload["remediation_status"] == "CREATED"
    assert payload["remediation_task_ids"]["fix"]
    assert payload["remediation_task_ids"]["fix_review"]


def test_invalid_max_fan_in_threshold_fails_before_writes(kanban_home):
    _, _, final = _make_graph(verdict="BLOCK")

    before_n = _board_task_count()
    out = kc.run_slash(
        f"resolve-fanin {final} --apply --json --max-fan-in-threshold 0"
    )
    after_n = _board_task_count()
    payload = _extract_json(out)

    assert "max_fan_in_threshold must be >= 1" in payload["error"]
    assert after_n == before_n


# ---------------------------------------------------------------------------
# Internals — ledger shape
# ---------------------------------------------------------------------------

def test_ledger_carries_graph_and_origin_return(kanban_home):
    origin, impl, final = _make_graph(verdict="GO")
    out = kc.run_slash(f"resolve-fanin {final} --dry-run --json")
    payload = _extract_json(out)
    assert payload["final_task"] == final
    assert payload["board"]  # current board slug
    assert set(payload["graph_task_ids"]) >= {origin, impl, final}
    assert payload["origin_return"] == origin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(out: str) -> dict:
    # ``run_slash`` returns captured stdout — the CLI emits one JSON
    # document when ``--json`` is set.
    out = out.strip()
    # Tolerate a trailing newline / extra noise: find the first ``{`` and
    # last ``}``.
    start = out.find("{")
    end = out.rfind("}")
    assert start != -1 and end != -1, f"no JSON in output: {out!r}"
    return json.loads(out[start:end + 1])


def _board_task_count() -> int:
    conn = kb.connect()
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    finally:
        conn.close()
