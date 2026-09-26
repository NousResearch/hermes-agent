"""R4 — a landing card cannot read ``done`` on unlanded evidence.

The failure this fences: a card closes ``done`` on "designed, implemented and
tested" while the change — and the card that was going to land it — is still
queued. Approval is not a landing, and neither is a review handoff: the only
accepted evidence is the merged commit, the deploy stamp naming it, or a live
hash equal to the merged blob.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

MERGED_SHA = "a1b2c3d4" * 5  # 40 hex
BLOB_SHA = "b" * 64


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _running(conn, **kw) -> str:
    tid = kb.create_task(conn, title="land it", **kw)
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def _event_payloads(conn, tid, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows if r["payload"]]


def _kinds(conn, tid) -> list[str]:
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)).fetchall()]


def _status(conn, tid) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"]


# --- the claim path: a claimed landing must substantiate it -------------------

def test_landing_claim_without_evidence_is_refused(conn):
    tid = _running(conn)
    with pytest.raises(kb.UnlandedCardError) as exc:
        kb.complete_task(
            conn, tid, summary="designed, implemented and tested",
            metadata={"landed": "PR #12 opened"},
        )
    # The refusal names the exact missing evidence.
    message = str(exc.value)
    assert "merged commit" in message
    assert "deploy stamp" in message
    assert "live hash" in message
    assert _status(conn, tid) == "running"
    blocked = _event_payloads(conn, tid, "completion_blocked_landing_evidence")
    assert blocked, _kinds(conn, tid)
    assert blocked[-1]["landing_evidence"] is False
    assert blocked[-1]["missing_evidence"]
    assert "completed" not in _kinds(conn, tid)


def test_merged_commit_lands_the_card(conn):
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="landed", metadata={"landed": {"merged_commit": MERGED_SHA}},
    ) is True
    assert _status(conn, tid) == "done"
    assert not _event_payloads(conn, tid, "completion_blocked_landing_evidence")


def test_deploy_stamp_lands_the_card(conn):
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="landed",
        metadata={"landed_evidence": {"deploy_stamp": f"{MERGED_SHA} @ 2026-09-25T23:20"}},
    ) is True
    assert _status(conn, tid) == "done"


def test_live_hash_equal_to_the_merged_blob_lands_the_card(conn):
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="landed",
        metadata={"merged_commit": MERGED_SHA, "live_sha256": BLOB_SHA,
                  "merged_blob_sha256": BLOB_SHA},
    ) is True
    assert _status(conn, tid) == "done"


def test_live_hash_disagreeing_with_the_merged_blob_is_refused(conn):
    tid = _running(conn)
    with pytest.raises(kb.UnlandedCardError):
        kb.complete_task(
            conn, tid, summary="looks deployed",
            metadata={"merged_commit": MERGED_SHA, "live_sha256": "c" * 64,
                      "merged_blob_sha256": BLOB_SHA},
        )
    assert _status(conn, tid) == "running"


# --- a checked non-landing card completes unchanged --------------------------

def test_non_landing_card_completes_unchanged(conn):
    tid = _running(conn)
    assert kb.complete_task(conn, tid, summary="shipped the analysis") is True
    assert _status(conn, tid) == "done"
    assert not _event_payloads(conn, tid, "completion_blocked_landing_evidence")


def test_published_pr_alone_is_left_to_pr_acceptance(conn):
    """A declared PR contract is gated by the acceptance path, not by this gate:
    no landing *claim* was made, so nothing here refuses it."""
    tid = _running(conn)
    gap = kb._landing.landing_gap({"published_pr": "https://github.com/acme/repo/pull/7"})
    assert gap is None


def test_a_claim_key_that_is_not_evidence_never_refuses_a_plain_card(conn):
    """Report fields unrelated tooling sets must not read as a landing claim."""
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="reported", metadata={"landing_evidence": False, "artifacts": ["x"]},
    ) is True
    assert _status(conn, tid) == "done"


# --- the chain-head path: a parent must not close over its own landing leg ----

def test_chain_head_cannot_close_over_a_pending_landing_child(conn):
    tid = _running(conn)
    child = kb.create_task(conn, title="land the PR", assignee="coder",
                           completion_contract="acme/repo", parents=[tid])
    with pytest.raises(kb.UnlandedCardError) as exc:
        kb.complete_task(conn, tid, summary="designed, implemented and tested")
    assert child in str(exc.value)
    assert _status(conn, tid) == "running"
    blocked = _event_payloads(conn, tid, "completion_blocked_landing_evidence")
    assert blocked[-1]["pending_landing_children"] == [child]
    # The landing leg lands, the head closes.
    kb.archive_task(conn, child)
    assert kb.complete_task(conn, tid, summary="landed by the leg") is True
    assert _status(conn, tid) == "done"


def test_a_local_only_child_is_the_normal_review_flow(conn):
    """A review child declares nothing, so the parent completes unchanged —
    completion is what releases the child."""
    tid = _running(conn)
    kb.create_task(conn, title="review it", assignee="reviewer", parents=[tid])
    assert kb.complete_task(conn, tid, summary="implemented, handing off") is True
    assert _status(conn, tid) == "done"


# --- no bypass through review, one through the operator ----------------------

def test_review_approval_of_a_landing_card_is_not_exempt(conn):
    tid = _running(conn)
    assert kb.request_review(conn, tid, summary="please land it") is True
    assert _status(conn, tid) == "review"
    with pytest.raises(kb.UnlandedCardError):
        kb.complete_task(conn, tid, summary="approved",
                         metadata={"landed": "approved, CI green"})
    assert _status(conn, tid) == "review"


def test_force_is_the_operator_override(conn):
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="landed out of band", force=True,
        metadata={"landed": "done by hand on the box"},
    ) is True
    assert _status(conn, tid) == "done"


# --- the evidence is read from the target it names, not the claim -------------

DEPLOY_STAMP = (
    "repo: /srv/app\n"
    "branch: main\n"
    f"commit: {MERGED_SHA}\n"
    "tree: 86e97c751337c0a7f8eed47b63be10d55072ac8c\n"
    "deployed: 2026-09-25T23:24:22-0400\n"
    "verified: copied tree matches git archive HEAD\n"
)


def _deploy_stamp(tmp_path, body: str) -> Path:
    """A stamp shaped like the one a deploy writes into the deployed tree."""
    stamp = tmp_path / "deployed" / ".deployed-from"
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(body)
    return stamp


def test_deploy_stamp_file_is_read_and_lands_the_card(conn, tmp_path):
    """The evidence is the commit the stamp NAMES, and the card records it."""
    stamp = _deploy_stamp(tmp_path, DEPLOY_STAMP)
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="shipped", metadata={"deploy_stamp": str(stamp)},
    ) is True
    assert _status(conn, tid) == "done"
    completed = _event_payloads(conn, tid, "completed")[-1]
    assert completed["landing_evidence"]["deploy_stamp_commit"] == MERGED_SHA


def test_deploy_stamp_naming_no_commit_is_not_evidence(conn, tmp_path):
    """A stamp that says only WHEN something was deployed lands nothing."""
    stamp = _deploy_stamp(tmp_path, "deployed: 2026-09-25T23:24:22-0400\nunit: board-sweep\n")
    tid = _running(conn)
    with pytest.raises(kb.UnlandedCardError) as exc:
        kb.complete_task(conn, tid, summary="shipped", metadata={"deploy_stamp": str(stamp)})
    assert "deploy stamp" in str(exc.value)
    assert _status(conn, tid) == "running"


def test_deploy_stamp_naming_another_commit_is_refused(conn, tmp_path):
    """A target contradicting the claim it is offered for is self-refuting."""
    stamp = _deploy_stamp(tmp_path, "commit: 0f1e2d3c4b5a69788796a5b4c3d2e1f0a1b2c3d4\n")
    tid = _running(conn)
    with pytest.raises(kb.UnlandedCardError) as exc:
        kb.complete_task(
            conn, tid, summary="shipped",
            metadata={"merged_commit": MERGED_SHA, "deploy_stamp": str(stamp)},
        )
    assert "different commit" in str(exc.value)


def test_live_hash_is_checked_against_the_artifact(conn, tmp_path):
    """The live hash must be the hash the named artifact actually has."""
    artifact = tmp_path / "greeter.py"
    artifact.write_text("print('hello')\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="live and hashed",
        metadata={"live_sha256": digest, "merged_blob_sha256": digest,
                  "artifact_path": str(artifact)},
    ) is True
    stale = tmp_path / "stale.py"
    stale.write_text("print('stale')\n")
    other = _running(conn)
    with pytest.raises(kb.UnlandedCardError) as exc:
        kb.complete_task(
            conn, other, summary="live but not hashed",
            metadata={"live_sha256": digest, "merged_blob_sha256": digest,
                      "artifact_path": str(stale)},
        )
    assert "disagree" in str(exc.value)
    assert _status(conn, other) == "running"


def test_an_artifact_path_alone_is_not_a_landing_claim(conn, tmp_path):
    """A report field that happens to name a path must not refuse the card."""
    tid = _running(conn)
    assert kb.complete_task(
        conn, tid, summary="no landing claimed",
        metadata={"artifact_path": str(tmp_path / "report.md")},
    ) is True

