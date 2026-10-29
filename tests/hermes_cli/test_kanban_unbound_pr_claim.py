"""Fail-closed gate: a completion whose evidence cites a GitHub PR the card's
``completion_contract`` does not bind is refused before any state change.

The contract is the card's publication claim — ``local-only`` makes no PR
claim, ``owner/repo`` binds every PR of that repo, an exact PR URL binds that
one PR. A completion citing a PR outside the bound scope would let a worker
close a publication card on a sibling PR (or a PR of another repo) whose CI
the card never declared. The gate refuses after an auditable
``completion_blocked_unbound_pr`` event; nothing is mutated.
"""
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    yield


def _events(conn, task_id, kind):
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=?", (task_id, kind))]


def test_unbound_pr_in_result_is_refused(board):
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        with pytest.raises(kb.UnboundPrClaimError) as exc:
            kb.complete_task(conn, tid, result="shipped in https://github.com/other/repo/pull/9")
        assert exc.value.task_id == tid
        assert exc.value.pr_url == "https://github.com/other/repo/pull/9"
        assert exc.value.contract == "acme/repo"
        assert kb.get_task(conn, tid).status != "done"
        events = _events(conn, tid, "completion_blocked_unbound_pr")
        assert len(events) == 1
        assert events[0]["contract"] == "acme/repo"
        assert events[0]["unbound_prs"] == ["https://github.com/other/repo/pull/9"]


def test_unbound_pr_in_summary_is_refused(board):
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        with pytest.raises(kb.UnboundPrClaimError):
            kb.complete_task(conn, tid, summary="see https://github.com/other/repo/pull/9")
        assert kb.get_task(conn, tid).status != "done"


def test_unbound_pr_in_metadata_is_refused(board):
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        with pytest.raises(kb.UnboundPrClaimError):
            kb.complete_task(conn, tid, result="done",
                             metadata={"published_pr": "https://github.com/other/repo/pull/9"})
        assert kb.get_task(conn, tid).status != "done"


def test_sibling_pr_in_same_repo_is_refused(board):
    """An exact-URL contract binds that one PR; a sibling PR of the same repo
    is outside the bound scope."""
    with connect() as conn:
        tid = kb.create_task(conn, title="publish",
                             completion_contract="https://github.com/acme/repo/pull/7")
        with pytest.raises(kb.UnboundPrClaimError) as exc:
            kb.complete_task(conn, tid, result="done",
                             metadata={"published_pr": "https://github.com/acme/repo/pull/8"})
        assert exc.value.pr_url == "https://github.com/acme/repo/pull/8"
        assert kb.get_task(conn, tid).status != "done"


def test_bound_pr_passes_gate(board):
    """A PR inside the bound scope passes the gate (the green acceptance check
    is a separate concern — here the gate must not block it)."""
    with connect() as conn:
        # owner/repo contract: any PR of that repo is bound.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        # The gate itself must not raise for a bound PR; acceptance may still
        # refuse (no gh here), but the refusal must NOT be the unbound-PR gate.
        try:
            kb.complete_task(conn, tid, result="done",
                             metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        except kb.UnboundPrClaimError:
            pytest.fail("bound PR must not trip the unbound-PR gate")
        # exact-URL contract: the bound PR itself is allowed.
        tid2 = kb.create_task(conn, title="publish2",
                              completion_contract="https://github.com/acme/repo/pull/7")
        try:
            kb.complete_task(conn, tid2, result="done",
                             metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        except kb.UnboundPrClaimError:
            pytest.fail("bound PR must not trip the unbound-PR gate")


def test_local_only_contract_is_exempt(board):
    """A local-only card makes no PR claim; citing a PR is background context,
    not a publication claim."""
    with connect() as conn:
        tid = kb.create_task(conn, title="local", completion_contract="local-only")
        # No gh on PATH here, so acceptance would fail — but the gate must not
        # be the reason. The local-only contract is exempt from the gate.
        try:
            kb.complete_task(conn, tid,
                             summary="https://github.com/acme/repo/pull/7 is background context")
        except kb.UnboundPrClaimError:
            pytest.fail("local-only contract must be exempt from the unbound-PR gate")


def test_no_pr_cited_passes_gate(board):
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        try:
            kb.complete_task(conn, tid, result="done locally")
        except kb.UnboundPrClaimError:
            pytest.fail("no PR cited must not trip the unbound-PR gate")


def test_review_status_is_exempt(board):
    """A human vouches for a review card; the approval is the record."""
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert kb.request_review(conn, tid, summary="ready for review")
        assert kb.get_task(conn, tid).status == "review"
        try:
            kb.complete_task(conn, tid, result="done",
                             metadata={"published_pr": "https://github.com/other/repo/pull/9"})
        except kb.UnboundPrClaimError:
            pytest.fail("review status must be exempt from the unbound-PR gate")


def test_gate_leaves_no_state_change(board):
    """The refusal is fail-closed: no status change, no pr_acceptance receipt,
    exactly one audit event."""
    with connect() as conn:
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        with pytest.raises(kb.UnboundPrClaimError):
            kb.complete_task(conn, tid, result="shipped in https://github.com/other/repo/pull/9")
        task = kb.get_task(conn, tid)
        assert task.status != "done"
        assert conn.execute(
            "SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'",
            (tid,)).fetchone()[0] == 0
        assert len(_events(conn, tid, "completion_blocked_unbound_pr")) == 1
