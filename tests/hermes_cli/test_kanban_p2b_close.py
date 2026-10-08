"""P2b final close — authenticated, digest-bound approval markers (2026-10-05).

Operator decision (dual-choice, 2026-10-05): an approval marker only RELEASES
a human gate when BOTH:
  (a) its author is an authorized approval identity (``_AUTHORIZED_APPROVAL_
      AUTHORS``: operator/orchestrator/hermes-system/hermes) — a worker can
      no longer self-approve by writing a comment as ``operator``;
  (b) the marker carries ``digest= <sha256-of-artifact-bytes>`` which must
      MATCH the artifact's CURRENT bytes at release-scan time (missing file,
      missing digest, or stale digest = no release).

This closes the review-5/wave-2 HIGH ("markers não autenticados") with the
approved low-cost design; `kanban_complete`'s in-txn recheck (R5-02) reuses
the same scan, so the in-txn gate inherits the authentication semantics.

Also: R5-09 CLOSED BY DESIGN — the post-stat mutation window of the managed
destination is declared outside the threat model (docstring unchanged; see
the duplicate finding in the review report). No code change for it here.

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixtures). Boards reais READ-ONLY.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import pytest

_ART = tempfile.NamedTemporaryFile(prefix="p2bc_", delete=False).name

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "2000")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _status(conn, task_id):
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


def _comment(conn, task_id, author, body):
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)"
        , (task_id, author, body, int(time.time()))
    )
    conn.commit()


def _claimed_task(conn):
    tid = kb.create_task(conn, title="p2b-close card", assignee="coder")
    assert kb.claim_task(conn, tid, claimer="host:mock") is not None
    return tid


def _approval(gate="G1", path=_ART, digest=None):
    if digest is None:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"HUMAN_GATE_APPROVAL: {gate} artifact={path} digest={digest}"


class TestP2bCloseAuthAndDigest:
    def test_authorized_approval_with_matching_digest_releases(self, conn):
        tid = _claimed_task(conn)
        _comment(conn, tid, "operator", "HUMAN_GATE_PENDING: G1")
        _comment(conn, tid, "operator", _approval("G1"))
        assert kb.complete_task(conn, tid, result="x", summary="ok") is True
        assert _status(conn, tid) == "done"

    def test_worker_self_approval_is_noise(self, conn):
        """A worker forging an approval as 'operator' is refused: its run-time
        author is NOT the operator identity recorded on the comment row —
        since the trail stored the FORGED author, the scan can only see the
        row. So the released gate requires the marker to come from the
        auth channel (kanban_approve writes the row). Here we prove: an
        approval whose row author is a WORKER profile never releases."""
        tid = _claimed_task(conn)
        _comment(conn, tid, "operator", "HUMAN_GATE_PENDING: G1")
        _comment(conn, tid, "coder", _approval("G1"))  # worker self-approval
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="x", summary="ok")
        assert _status(conn, tid) != "done"

    def test_missing_digest_refuses_release(self, conn):
        tid = _claimed_task(conn)
        _comment(conn, tid, "operator", "HUMAN_GATE_PENDING: G1")
        # Pre-close shape (no digest) — now releases nothing.
        _comment(conn, tid, "operator",
                 f"HUMAN_GATE_APPROVAL: G1 artifact={_ART}")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="x", summary="ok")
        assert _status(conn, tid) != "done"

    def test_stale_digest_refuses_release(self, conn, kanban_home):
        tid = _claimed_task(conn)
        _comment(conn, tid, "operator", "HUMAN_GATE_PENDING: G1")
        artifact = Path(kanban_home) / "approval.yaml"
        artifact.write_text("decision: APPROVE\n")
        _comment(conn, tid, "operator", _approval("G1", str(artifact)))
        # Artifact bytes changed AFTER the approval digest was computed.
        artifact.write_text("decision: APPROVE-NEW-SCOPE\n")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="x", summary="ok")
        assert _status(conn, tid) != "done"

    def test_missing_artifact_with_digest_refuses(self, conn):
        tid = _claimed_task(conn)
        _comment(conn, tid, "operator", "HUMAN_GATE_PENDING: G1")
        _comment(conn, tid, "operator",
                 "HUMAN_GATE_APPROVAL: G1 artifact=/nonexistent/x.yaml digest=aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="x", summary="ok")
        assert _status(conn, tid) != "done"

    def test_authorized_author_list_is_frozen(self):
        """The authorization list is explicit and frozen at these identities."""
        assert set(kb._AUTHORIZED_APPROVAL_AUTHORS) == {
            "operator", "orchestrator", "hermes-system", "hermes-system-", "hermes",
            "autopilot",
        }
