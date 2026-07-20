"""Kernel-level tests for the external-worker transaction seam.

These tests assert the *frozen* public contract of
``hermes_cli.kanban_external_worker``: raw-byte spec identity, strict JSON
parser negatives, atomic submit rollback, attachment swap/mutation detection,
copied claim identity, expected-spec/lease CAS, bind mismatch + required
pgid, heartbeat ownership/expiry, typed uncertain hold, affirmative
no-start / process-absence recovery, no requeue at two holds, uniform
COMPLETE/REQUEUE/BLOCK exact result bytes, same-hash same-tuple idempotency,
divergent hash or tuple rejection, public list/get/read APIs, and
artifact same-name collision behavior.

All tests run against a temp ``HERMES_HOME`` board. They never assert broad
status alternatives — every assertion pins an exact outcome or exception
class.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_external_worker as xw


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kb.connect()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_json(*, objective: str = "implement the requested change") -> bytes:
    return json.dumps(
        {
            "schema_version": xw.SPEC_SCHEMA_VERSION,
            "board": "default",
            "repo_key": "fixture",
            "objective": objective,
            "acceptance_criteria": ["focused tests pass"],
            "scope": {"include": ["src/**"], "exclude": [".env"]},
            "base_sha": "a" * 40,
            "risk": "small",
            "workflow": "implement-verify",
            "execution": {
                "timeout_seconds": 60,
                "max_attempts": 2,
                "max_tokens": 1000,
                "max_cost_usd": None,
            },
            "verification": {
                "check_ids": ["focused"],
                "fresh_reviewer": True,
                "security_review": False,
            },
            "delivery": {"mode": "review-only", "push": False, "deploy": False},
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def _result_json(
    *,
    run_id: int,
    task_id: str,
    spec: xw.SpecIdentity,
    disposition: str,
    block_kind: str | None = None,
    attempt: int = 1,
    process: xw.BoundProcess | None = None,
    outcome: str | None = None,
    absence_proven: bool = True,
    summary: str = "fixture result",
    deliverable: dict | None = None,
    artifacts: list[dict] | None = None,
) -> bytes:
    if outcome is None:
        outcome = "completed" if process is not None else "aborted_before_start"
    identity = None
    if process is not None:
        identity = {
            "host": process.host,
            "pid": process.pid,
            "pgid": process.pgid,
            "start_token": process.start_token,
        }
    return json.dumps(
        {
            "mas": {
                "schema_version": xw.RESULT_SCHEMA_VERSION,
                "spec_sha256": spec.spec_hash,
                "submitted_attachment_id": spec.attachment_id,
                "run_id": run_id,
                "attempt": attempt,
                "outcome": outcome,
                "disposition": disposition.lower(),
                "block_kind": block_kind,
                "writer": {"backend": None, "model": None},
                "reviewer": {"backend": None, "model": None, "verdict": None},
                "process": {
                    "identity": identity,
                    "termination_status": "exited" if process else "not_started",
                    "absence_proven": absence_proven,
                },
                "git": {
                    "base_sha": "a" * 40,
                    "head_sha": None,
                    "branch": None,
                    "diff_sha256": None,
                    "changed_files": [],
                    "primary_unchanged": True,
                },
                "deliverable": deliverable
                or {"kind": None, "attachment": None, "sha256": None},
                "checks": [],
                "usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "reasoning_tokens": None,
                    "cost_usd": None,
                    "cost_status": "unknown",
                    "source": "unknown",
                },
                "failure_signature": None,
                "summary": summary,
                "artifacts": artifacts or [],
            }
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")


def _make_task_with_spec(
    conn, *, title: str = "t", body: str = "b"
) -> tuple[str, int, bytes]:
    """Create a task, upload a spec attachment, return (task_id, att_id, raw)."""
    tid = kb.create_task(conn, title=title, body=body, triage=True)
    raw = _spec_json(objective=title)
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=raw
    )
    return tid, aid, raw


def _submit_and_claim(
    conn, *, lease_token: str = "lease-1", ttl: int = 600
) -> tuple[str, int, bytes, xw.SpecIdentity, xw.Lease]:
    tid, aid, raw = _make_task_with_spec(conn)
    spec = xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    lease = xw.claim_external(
        conn,
        task_id=tid,
        expected_spec=spec,
        lease_token=lease_token,
        lease_expires_at=int(time.time()) + ttl,
    )
    return tid, aid, raw, spec, lease


def _expire_lease(conn, lease: xw.Lease) -> xw.Lease:
    expired_at = int(time.time()) - 1
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
            (expired_at, lease.run_id),
        )
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (expired_at, lease.task_id),
        )
    return replace(lease, lease_expires_at=expired_at)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_module_exposes_api_version():
    assert xw.EXTERNAL_WORKER_API_VERSION == 1
    assert xw.SPEC_SCHEMA_VERSION == "mas-task-spec.v1"
    assert xw.RESULT_SCHEMA_VERSION == "mas-execution-result.v1"
    assert xw.SPEC_ATTACHMENT_NAME == "mas-task-spec.v1.json"


def test_public_seam_opens_named_board_without_kanban_db_import(kanban_home):
    with xw.connect(board="external-fixture") as public_conn:
        task_id = kb.create_task(public_conn, title="public connection")
        assert kb.get_task(public_conn, task_id) is not None


# ---------------------------------------------------------------------------
# submit — raw-byte hash, exact attachment id, atomic lock
# ---------------------------------------------------------------------------


def test_candidate_read_is_non_mutating(kanban_home, conn):
    tid, aid, raw = _make_task_with_spec(conn)

    observed = xw.read_candidate_attachment(
        conn,
        task_id=tid,
        attachment_id=aid,
    )

    assert observed == raw
    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert task.external_spec_hash is None
    assert task.external_spec_attachment_id is None


def test_submit_expected_hash_rejects_candidate_mutation_atomically(kanban_home, conn):
    tid, aid, raw = _make_task_with_spec(conn)
    observed = xw.read_candidate_attachment(
        conn,
        task_id=tid,
        attachment_id=aid,
    )
    assert observed == raw
    replacement = _spec_json(objective="x")
    assert len(replacement) == len(raw)
    attachment = kb.get_attachment(conn, aid)
    Path(attachment.stored_path).write_bytes(replacement)

    with pytest.raises(xw.SpecMutationError, match="candidate hash changed"):
        xw.submit(
            conn,
            task_id=tid,
            spec_attachment_id=aid,
            expected_spec_hash=hashlib.sha256(raw).hexdigest(),
        )

    task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert task.external_spec_hash is None
    assert task.external_spec_attachment_id is None


def test_submit_records_raw_byte_hash_and_attachment_id(kanban_home, conn):
    tid, aid, raw = _make_task_with_spec(conn)
    spec = xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    # Identity is the exact raw-byte SHA-256 — never normalized.
    assert spec.spec_hash == hashlib.sha256(raw).hexdigest()
    assert spec.attachment_id == aid
    assert spec.schema_version == xw.SPEC_SCHEMA_VERSION

    t = kb.get_task(conn, tid)
    assert t is not None
    assert t.status == "ready"
    assert t.external_spec_hash == spec.spec_hash
    assert t.external_spec_attachment_id == aid
    assert t.external_spec_schema == xw.SPEC_SCHEMA_VERSION
    assert t.external_spec_locked_at is not None


def test_submit_does_not_touch_title_body(kanban_home, conn):
    """The spec attachment is the sole authoritative input after submit;
    submit must not rewrite title/body in a separate transaction."""
    tid = kb.create_task(conn, title="orig-title", body="orig-body", triage=True)
    raw = _spec_json(objective="DIFFERENT")
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=raw
    )
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    t = kb.get_task(conn, tid)
    assert t is not None
    # Task row's title/body are exactly what create_task set.
    assert t.title == "orig-title"
    assert t.body == "orig-body"


def test_submit_rejects_already_locked_task(kanban_home, conn):
    tid, aid, _ = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    with pytest.raises(xw.SpecMutationError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)


def test_submit_rejects_attachment_on_other_task(kanban_home, conn):
    tid1, aid1, _ = _make_task_with_spec(conn)
    tid2 = kb.create_task(conn, title="other", triage=True)
    with pytest.raises(xw.AttachmentNotOwnedError):
        xw.submit(conn, task_id=tid2, spec_attachment_id=aid1)


def test_submit_rejects_wrong_filename(kanban_home, conn):
    tid = kb.create_task(conn, title="t", body="b", triage=True)
    aid = kb.store_attachment_bytes(conn, tid, filename="not-the-spec.json", data=b"{}")
    with pytest.raises(xw.AttachmentMismatchError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)


def test_submit_rejects_more_than_one_canonical_attachment(kanban_home, conn):
    tid, aid, _raw = _make_task_with_spec(conn)
    att = kb.get_attachment(conn, aid)
    kb.add_attachment(
        conn,
        tid,
        filename=xw.SPEC_ATTACHMENT_NAME,
        stored_path=att.stored_path,
        size=att.size,
    )
    with pytest.raises(xw.AttachmentMismatchError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)


def test_submit_routes_unfinished_dependency_to_todo(kanban_home, conn):
    parent = kb.create_task(conn, title="parent", body="b")
    tid = kb.create_task(conn, title="child", body="b", triage=True, parents=[parent])
    raw = _spec_json()
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=raw
    )
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    assert kb.get_task(conn, tid).status == "todo"


def test_submitted_canonical_attachment_cannot_be_added_or_deleted(kanban_home, conn):
    tid, aid, _raw = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    att = kb.get_attachment(conn, aid)
    with pytest.raises(kb.ExternalTaskConflict) as added:
        kb.add_attachment(
            conn,
            tid,
            filename=xw.SPEC_ATTACHMENT_NAME,
            stored_path=att.stored_path,
            size=att.size,
        )
    assert added.value.code == "external_spec_locked"
    with pytest.raises(kb.ExternalTaskConflict) as deleted:
        kb.delete_attachment(conn, aid)
    assert deleted.value.code == "external_spec_locked"


def test_submit_atomic_rollback_on_bad_json(kanban_home, conn):
    """A failed submit must leave the task row untouched — no partial lock."""
    tid = kb.create_task(conn, title="t", body="b", triage=True)
    bad = b"{not valid json"
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=bad
    )
    with pytest.raises(xw.SpecParseError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    t = kb.get_task(conn, tid)
    assert t is not None
    # Atomic rollback: no spec lock, task status unchanged.
    assert t.external_spec_hash is None
    assert t.external_spec_attachment_id is None
    assert t.external_spec_locked_at is None


def test_submit_is_atomic_under_concurrent_lock(kanban_home, conn):
    """If the task row is locked by a concurrent writer mid-submit, the
    spec validation must still roll back (the WHERE clause external_spec_hash
    IS NULL matches 0 rows → no partial state)."""
    tid, aid, _ = _make_task_with_spec(conn)
    # Simulate a concurrent writer locking the spec manually.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET external_spec_hash = 'existing', "
            "external_spec_schema = 'mas-task-spec.v1', "
            "external_spec_attachment_id = 999, "
            "external_spec_locked_at = 1 WHERE id = ?",
            (tid,),
        )
    with pytest.raises(xw.SpecMutationError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    t = kb.get_task(conn, tid)
    assert t is not None
    # The concurrent lock is intact; submit did not overwrite.
    assert t.external_spec_hash == "existing"
    assert t.external_spec_attachment_id == 999


# ---------------------------------------------------------------------------
# read_submitted_attachment
# ---------------------------------------------------------------------------


def test_read_submitted_attachment_returns_exact_bytes(kanban_home, conn):
    tid, aid, raw = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    out = xw.read_submitted_attachment(conn, task_id=tid, attachment_id=aid)
    assert out == raw


def test_read_submitted_attachment_rejects_other_task(kanban_home, conn):
    tid, aid, _ = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    tid2 = kb.create_task(conn, title="other")
    with pytest.raises(xw.AttachmentNotOwnedError):
        xw.read_submitted_attachment(conn, task_id=tid2, attachment_id=aid)


def test_read_submitted_attachment_detects_byte_mutation(kanban_home, conn):
    tid, aid, _ = _make_task_with_spec(conn)
    spec = xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    # Mutate the on-disk bytes.
    att = kb.get_attachment(conn, aid)
    Path(att.stored_path).write_bytes(b"TAMPERED")
    with pytest.raises(xw.AttachmentMismatchError):
        xw.read_submitted_attachment(conn, task_id=tid, attachment_id=aid)


def test_read_submitted_attachment_detects_attachment_swap(kanban_home, conn):
    """If someone swaps the spec attachment row to point at different bytes
    (same task, same name), the hash check fires."""
    tid, aid, raw = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    # Upload a second attachment with the same name (collision-resolved)
    # and repoint the original row's stored_path at it.
    raw2 = _spec_json(objective="DIFFERENT")
    aid2 = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=raw2
    )
    att2 = kb.get_attachment(conn, aid2)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_attachments SET stored_path = ?, size = ? WHERE id = ?",
            (att2.stored_path, len(raw2), aid),
        )
    with pytest.raises(xw.AttachmentMismatchError):
        xw.read_submitted_attachment(conn, task_id=tid, attachment_id=aid)


# ---------------------------------------------------------------------------
# Strict JSON parser negatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_bytes",
    [
        b"\xff\xfe" + b'"x"',
        b"\xef\xbb\xbf" + _spec_json(),
        _spec_json() + b"\x00extra",
        b'{"a": "\xff"}',
        b'{"schema_version":"mas-task-spec.v1","schema_version":"duplicate"}',
        b"NaN",
    ],
)
def test_strict_json_parser_negatives(kanban_home, conn, bad_bytes):
    tid = kb.create_task(conn, title="t", body="b", triage=True)
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=bad_bytes
    )
    with pytest.raises(xw.SpecParseError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong_nested", "infinity"])
def test_strict_taskspec_shape_rejected(kanban_home, conn, mutation):
    tid = kb.create_task(conn, title="t", body="b", triage=True)
    payload = json.loads(_spec_json())
    if mutation == "extra":
        payload["extra"] = True
    elif mutation == "missing":
        del payload["workflow"]
    elif mutation == "wrong_nested":
        payload["scope"] = []
    else:
        payload["execution"]["max_cost_usd"] = float("inf")
    bad = json.dumps(payload, allow_nan=True).encode()
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=bad
    )
    with pytest.raises(xw.SpecParseError):
        xw.submit(conn, task_id=tid, spec_attachment_id=aid)


def test_strict_json_allows_trailing_whitespace(kanban_home, conn):
    tid = kb.create_task(conn, title="t", body="b", triage=True)
    raw = _spec_json() + b"\n  "
    aid = kb.store_attachment_bytes(
        conn, tid, filename=xw.SPEC_ATTACHMENT_NAME, data=raw
    )
    spec = xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    assert spec.spec_hash == hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# claim_external — copied spec identity, expected-spec CAS
# ---------------------------------------------------------------------------


def test_claim_copies_spec_identity_into_task_runs(kanban_home, conn):
    tid, aid, _, spec, lease = _submit_and_claim(conn)
    assert lease.lease_state == xw.LEASE_ACTIVE
    assert lease.attempt == 1
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["worker_kind"] == xw.WORKER_KIND
    assert int(row["external_spec_attachment_id"]) == aid
    assert row["external_spec_hash"] == spec.spec_hash
    assert row["external_spec_schema"] == spec.schema_version
    assert int(row["external_attempt"]) == 1
    assert row["external_substate"] == xw.SUBSTATE_CLAIMED
    assert row["external_lease_state"] == xw.LEASE_ACTIVE
    assert int(row["external_recovery_count"]) == 0
    assert row["claim_lock"] == lease.lease_token
    events = kb.list_events(conn, tid)
    claimed = next(event for event in events if event.kind == "external_claimed")
    assert lease.lease_token not in json.dumps(claimed.payload)


def test_claim_refuses_second_open_run_after_task_state_drift(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
    with pytest.raises(xw.ClaimRejected):
        xw.claim_external(
            conn,
            task_id=tid,
            expected_spec=spec,
            lease_token="second",
            lease_expires_at=int(time.time()) + 60,
        )
    active = xw.list_active(conn)
    assert [item.run_id for item in active] == [lease.run_id]


def test_claim_rejects_wrong_expected_spec_hash(kanban_home, conn):
    tid, aid, _, spec, _ = _submit_and_claim(conn)
    # Requeue back to ready by direct SQL so we can re-claim.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
    bad_spec = xw.SpecIdentity(
        attachment_id=aid,
        spec_hash="0" * 64,
        schema_version=spec.schema_version,
    )
    with pytest.raises(xw.ClaimRejected):
        xw.claim_external(
            conn,
            task_id=tid,
            expected_spec=bad_spec,
            lease_token="other",
            lease_expires_at=int(time.time()) + 60,
        )


def test_claim_rejects_already_claimed(kanban_home, conn):
    tid, aid, _, spec, _ = _submit_and_claim(conn)
    with pytest.raises(xw.ClaimRejected):
        xw.claim_external(
            conn,
            task_id=tid,
            expected_spec=spec,
            lease_token="other",
            lease_expires_at=int(time.time()) + 60,
        )


def test_claim_rejects_when_spec_attachment_swapped_after_submit(kanban_home, conn):
    """Re-reads and rehashes the accepted attachment in the same txn. If
    the on-disk bytes have drifted between submit and claim, the recomputed
    hash disagrees and CAS refuses."""
    tid, aid, raw, spec, _ = _submit_and_claim(conn)
    # Requeue back to ready for re-claim.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
    # Tamper with the spec attachment bytes — submit-time hash no longer matches.
    att = kb.get_attachment(conn, aid)
    Path(att.stored_path).write_bytes(b"TAMPERED")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_attachments SET size = ? WHERE id = ?",
            (len(b"TAMPERED"), aid),
        )
    with pytest.raises(xw.ClaimRejected):
        xw.claim_external(
            conn,
            task_id=tid,
            expected_spec=spec,
            lease_token="other",
            lease_expires_at=int(time.time()) + 60,
        )


def test_claim_rejects_past_expiry(kanban_home, conn):
    tid, aid, _, spec, _ = _submit_and_claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
    with pytest.raises(xw.ExternalWorkerError):
        xw.claim_external(
            conn,
            task_id=tid,
            expected_spec=spec,
            lease_token="x",
            lease_expires_at=int(time.time()) - 1,
        )


# ---------------------------------------------------------------------------
# bind_process — required pgid, mismatch, idempotency
# ---------------------------------------------------------------------------


def test_bind_process_requires_pgid(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(ValueError):
        xw.BoundProcess(host="h", pid=1, pgid=0, start_token="t")


def test_bind_process_requires_positive_pid(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(ValueError):
        xw.BoundProcess(host="h", pid=0, pgid=1, start_token="t")


def test_bind_process_records_durable_bound_substate(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    proc = xw.BoundProcess(host="h", pid=123, pgid=123, start_token="tok")
    bound = xw.bind_process(conn, lease=lease, process=proc)
    assert bound.lease_state == xw.LEASE_BOUND
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["external_substate"] == xw.SUBSTATE_BOUND
    assert row["external_lease_state"] == xw.LEASE_BOUND
    assert row["external_host"] == "h"
    assert int(row["external_pid"]) == 123
    assert int(row["external_pgid"]) == 123
    assert row["external_start_token"] == "tok"


def test_bind_process_idempotent_for_same_identity(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    proc = xw.BoundProcess(host="h", pid=123, pgid=123, start_token="tok")
    bound1 = xw.bind_process(conn, lease=lease, process=proc)
    # Same identity quad → idempotent.
    bound2 = xw.bind_process(conn, lease=bound1, process=proc)
    assert bound2.lease_state == xw.LEASE_BOUND


def test_bind_process_lost_response_replays_with_original_lease(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    proc = xw.BoundProcess(host="h", pid=123, pgid=123, start_token="tok")
    first = xw.bind_process(conn, lease=lease, process=proc)
    replay = xw.bind_process(conn, lease=lease, process=proc)
    assert replay == first


def test_bind_process_rejects_divergent_identity(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    proc1 = xw.BoundProcess(host="h", pid=123, pgid=123, start_token="tok1")
    bound = xw.bind_process(conn, lease=lease, process=proc1)
    proc2 = xw.BoundProcess(host="h", pid=456, pgid=456, start_token="tok2")
    # Second call must carry the BOUND-state lease returned by the first
    # bind — the lease_state field is the caller's assertion about the
    # run's current state.
    with pytest.raises(xw.BindMismatch):
        xw.bind_process(conn, lease=bound, process=proc2)


def test_bind_process_rejects_wrong_lease_state(kanban_home, conn):
    """Bind requires LEASE_ACTIVE; calling it with a BOUND-state lease
    (i.e. the lease returned by a prior bind) on a divergent identity
    must raise BindMismatch, not silently accept."""
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    proc1 = xw.BoundProcess(host="h", pid=1, pgid=1, start_token="a")
    bound = xw.bind_process(conn, lease=lease, process=proc1)
    # Pass lease with lease_state=BOUND and a divergent process — should
    # hit the BindMismatch path (identity quad differs).
    proc2 = xw.BoundProcess(host="h", pid=2, pgid=2, start_token="b")
    with pytest.raises(xw.BindMismatch):
        xw.bind_process(conn, lease=bound, process=proc2)


# ---------------------------------------------------------------------------
# heartbeat / still_owns — exact ownership and expiry
# ---------------------------------------------------------------------------


def test_heartbeat_returns_updated_lease(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    new_expiry = int(time.time()) + 900
    hb = xw.heartbeat(conn, lease=lease, lease_expires_at=new_expiry)
    assert hb.lease_expires_at == new_expiry
    assert hb.lease_state == lease.lease_state
    row = conn.execute(
        "SELECT claim_expires FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert int(row["claim_expires"]) == new_expiry


def test_heartbeat_rejects_stale_expected_expiry(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    new_expiry = int(time.time()) + 900
    xw.heartbeat(conn, lease=lease, lease_expires_at=new_expiry)
    with pytest.raises(xw.NotOwner):
        xw.heartbeat(conn, lease=lease, lease_expires_at=new_expiry + 60)


def test_heartbeat_cannot_resurrect_expired_lease(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    lease = _expire_lease(conn, lease)
    with pytest.raises(xw.NotOwner):
        xw.heartbeat(conn, lease=lease, lease_expires_at=int(time.time()) + 60)


def test_heartbeat_rejects_wrong_lease_token(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    bad_lease = xw.Lease(
        run_id=lease.run_id,
        task_id=lease.task_id,
        spec=lease.spec,
        lease_token="wrong-token",
        lease_state=xw.LEASE_ACTIVE,
        lease_expires_at=lease.lease_expires_at,
        attempt=lease.attempt,
    )
    with pytest.raises(xw.NotOwner):
        xw.heartbeat(conn, lease=bad_lease, lease_expires_at=int(time.time()) + 60)


def test_heartbeat_rejects_wrong_expected_state(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    # Caller asserts state=BOUND but the DB is ACTIVE.
    bad = xw.Lease(
        run_id=lease.run_id,
        task_id=lease.task_id,
        spec=lease.spec,
        lease_token=lease.lease_token,
        lease_state=xw.LEASE_BOUND,
        lease_expires_at=lease.lease_expires_at,
        attempt=lease.attempt,
    )
    with pytest.raises(xw.LeaseStateError):
        xw.heartbeat(conn, lease=bad, lease_expires_at=int(time.time()) + 60)


def test_still_owns_checks_exact_ownership_and_expiry(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    assert not xw.still_owns(conn, lease=replace(lease, lease_token="nope"))
    expired = _expire_lease(conn, lease)
    assert not xw.still_owns(conn, lease=expired)
    # Refresh and verify True.
    refreshed_at = int(time.time()) + 60
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
            (refreshed_at, lease.run_id),
        )
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (refreshed_at, lease.task_id),
        )
    assert xw.still_owns(conn, lease=replace(lease, lease_expires_at=refreshed_at))


# ---------------------------------------------------------------------------
# hold_for_recovery — typed bounded proof, durable counter, uncertainty
# ---------------------------------------------------------------------------


def test_hold_for_recovery_increments_counter_and_extends_lease(kanban_home, conn):
    # Short initial TTL so the recovery extension is observably longer.
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn, ttl=60)
    proof = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=lease.task_id, bound=None, evidence="lost"
    )
    held = xw.hold_for_recovery(conn, lease=lease, proof=proof, extension_seconds=300)
    assert held.lease_state == xw.LEASE_HOLDING
    assert held.lease_expires_at > lease.lease_expires_at
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert int(row["external_recovery_count"]) == 1
    assert row["external_substate"] == xw.SUBSTATE_HOLDING
    assert row["external_lease_state"] == xw.LEASE_HOLDING


def test_hold_for_recovery_can_quarantine_an_expired_lease(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    lease = _expire_lease(conn, lease)
    proof = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=tid, bound=None, evidence="uncertain"
    )
    held = xw.hold_for_recovery(conn, lease=lease, proof=proof, extension_seconds=60)
    assert held.lease_state == xw.LEASE_HOLDING
    assert held.lease_expires_at > int(time.time())


def test_second_hold_requires_manual_recovery_and_third_is_rejected(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    proof = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=tid, bound=None, evidence="uncertain"
    )
    held = xw.hold_for_recovery(conn, lease=lease, proof=proof, extension_seconds=60)
    held = xw.hold_for_recovery(conn, lease=held, proof=proof, extension_seconds=60)
    assert held.recovery_count == 2
    events = kb.list_events(conn, tid)
    assert [e.kind for e in events].count("external_manual_recovery_required") == 1
    with pytest.raises(xw.RecoveryRejected):
        xw.hold_for_recovery(conn, lease=held, proof=proof, extension_seconds=60)


def test_hold_for_recovery_rejects_proof_with_wrong_identity(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    bad = xw.RecoveryHoldProof(
        run_id=lease.run_id + 999,
        task_id=lease.task_id,
        bound=None,
        evidence="x",
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.hold_for_recovery(conn, lease=lease, proof=bad)


def test_hold_for_recovery_rejects_bound_proof_for_unbound_run(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    # Run is unbound (no bind_process called), but proof claims a bound proc.
    proof = xw.RecoveryHoldProof(
        run_id=lease.run_id,
        task_id=lease.task_id,
        bound=xw.BoundProcess(host="h", pid=1, pgid=1, start_token="x"),
        evidence="x",
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.hold_for_recovery(conn, lease=lease, proof=proof)


def test_hold_for_recovery_uncertainty_does_not_release(kanban_home, conn):
    """A failed hold (wrong lease state) leaves the run fully active."""
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    bad = xw.Lease(
        run_id=lease.run_id,
        task_id=lease.task_id,
        spec=lease.spec,
        lease_token=lease.lease_token,
        lease_state=xw.LEASE_BOUND,  # DB is ACTIVE — mismatch
        lease_expires_at=lease.lease_expires_at,
        attempt=lease.attempt,
    )
    proof = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=lease.task_id, bound=None, evidence="x"
    )
    with pytest.raises(xw.LeaseStateError):
        xw.hold_for_recovery(conn, lease=bad, proof=proof)
    # Run is still active and owned.
    row = conn.execute(
        "SELECT ended_at, external_lease_state, claim_lock FROM task_runs WHERE id = ?",
        (lease.run_id,),
    ).fetchone()
    assert row["ended_at"] is None
    assert row["external_lease_state"] == xw.LEASE_ACTIVE
    assert row["claim_lock"] == lease.lease_token


# ---------------------------------------------------------------------------
# recover_expired — affirmative proof, no os.kill inference, no requeue @ 2 holds
# ---------------------------------------------------------------------------


def test_recover_expired_with_no_start_proof_for_unbound_run(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    lease = _expire_lease(conn, lease)
    proof = xw.NoStartAckProof(
        run_id=lease.run_id, task_id=tid, evidence="never-started"
    )
    result_bytes = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    result = xw.ExecutionResult(
        disposition="COMPLETE",
        block_kind=None,
        result_bytes=result_bytes,
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    out = xw.recover_expired(
        conn,
        lease=lease,
        proof=proof,
        result=result,
    )
    assert out.disposition == "COMPLETE"
    assert not out.requeued
    assert not out.blocked
    t = kb.get_task(conn, tid)
    assert t is not None and t.status == "done"


def test_recover_expired_with_absence_proof_for_bound_run(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    proc = xw.BoundProcess(host="h", pid=12345, pgid=12345, start_token="tok")
    lease = xw.bind_process(conn, lease=lease, process=proc)
    lease = _expire_lease(conn, lease)
    proof = xw.ProcessAbsenceProof(
        host="h", pid=12345, pgid=12345, start_token="tok", evidence="waitpid:9"
    )
    result = xw.ExecutionResult(
        disposition="COMPLETE",
        block_kind=None,
        result_bytes=_result_json(
            run_id=lease.run_id,
            task_id=tid,
            spec=spec,
            disposition="COMPLETE",
            process=proc,
            outcome="recovered_after_crash",
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    out = xw.recover_expired(
        conn,
        lease=lease,
        proof=proof,
        result=result,
    )
    assert out.disposition == "COMPLETE"


def test_recover_expired_rejects_unexpired_lease(kanban_home, conn):
    """Uncertainty (lease not expired) leaves the run active."""
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    proof = xw.NoStartAckProof(run_id=lease.run_id, task_id=tid, evidence="x")
    result = xw.ExecutionResult(
        disposition="COMPLETE",
        block_kind=None,
        result_bytes=_result_json(
            run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.recover_expired(
            conn,
            lease=lease,
            proof=proof,
            result=result,
        )
    row = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["ended_at"] is None


def test_recover_expired_rejects_wrong_expected_lease_state(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    lease = _expire_lease(conn, lease)
    wrong = replace(lease, lease_state=xw.LEASE_BOUND)
    proof = xw.NoStartAckProof(
        run_id=lease.run_id, task_id=tid, evidence="never-started"
    )
    result = xw.ExecutionResult(
        disposition="COMPLETE",
        block_kind=None,
        result_bytes=_result_json(
            run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.recover_expired(conn, lease=wrong, proof=proof, result=result)


def test_recover_expired_rejects_absence_proof_for_unbound_run(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    lease = _expire_lease(conn, lease)
    proof = xw.ProcessAbsenceProof(
        host="h", pid=1, pgid=1, start_token="x", evidence="x"
    )
    result = xw.ExecutionResult(
        disposition="COMPLETE",
        block_kind=None,
        result_bytes=_result_json(
            run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.recover_expired(
            conn,
            lease=lease,
            proof=proof,
            result=result,
        )


def test_recover_expired_rejects_no_start_proof_for_bound_run(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    proc = xw.BoundProcess(host="h", pid=12345, pgid=12345, start_token="t")
    lease = xw.bind_process(
        conn,
        lease=lease,
        process=proc,
    )
    lease = _expire_lease(conn, lease)
    proof = xw.NoStartAckProof(run_id=lease.run_id, task_id=tid, evidence="x")
    result = xw.ExecutionResult(
        disposition="COMPLETE",
        block_kind=None,
        result_bytes=_result_json(
            run_id=lease.run_id,
            task_id=tid,
            spec=spec,
            disposition="COMPLETE",
            process=proc,
            outcome="recovered_after_crash",
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.recover_expired(
            conn,
            lease=lease,
            proof=proof,
            result=result,
        )


def test_recover_expired_rejects_requeue_at_two_holds(kanban_home, conn):
    """After two inconclusive holds, REQUEUE is refused. The run stays
    active; the caller must supply a non-REQUEUE result orBLOCK disposition."""
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    proof_hold = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=tid, bound=None, evidence="h1"
    )
    held_lease = xw.hold_for_recovery(
        conn, lease=lease, proof=proof_hold, extension_seconds=60
    )
    proof_hold2 = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=tid, bound=None, evidence="h2"
    )
    held_lease = xw.hold_for_recovery(
        conn, lease=held_lease, proof=proof_hold2, extension_seconds=60
    )
    held_lease = _expire_lease(conn, held_lease)
    result = xw.ExecutionResult(
        disposition="REQUEUE",
        block_kind=None,
        result_bytes=_result_json(
            run_id=lease.run_id, task_id=tid, spec=spec, disposition="REQUEUE"
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    proof = xw.NoStartAckProof(run_id=lease.run_id, task_id=tid, evidence="x")
    with pytest.raises(xw.RecoveryRejected):
        xw.recover_expired(
            conn,
            lease=held_lease,
            proof=proof,
            result=result,
        )
    # Run still active (uncertainty leaves it active).
    row = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["ended_at"] is None


def test_recover_expired_at_two_holds_accepts_block(kanban_home, conn):
    """At the requeue limit, a BLOCK recovery is accepted."""
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    proof_hold = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=tid, bound=None, evidence="h1"
    )
    held = xw.hold_for_recovery(
        conn, lease=lease, proof=proof_hold, extension_seconds=60
    )
    proof_hold2 = xw.RecoveryHoldProof(
        run_id=lease.run_id, task_id=tid, bound=None, evidence="h2"
    )
    held = xw.hold_for_recovery(
        conn, lease=held, proof=proof_hold2, extension_seconds=60
    )
    held = _expire_lease(conn, held)
    result = xw.ExecutionResult(
        disposition="BLOCK",
        block_kind="environment-error",
        result_bytes=_result_json(
            run_id=lease.run_id,
            task_id=tid,
            spec=spec,
            disposition="BLOCK",
            block_kind="environment-error",
        ),
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    proof = xw.NoStartAckProof(run_id=lease.run_id, task_id=tid, evidence="x")
    out = xw.recover_expired(
        conn,
        lease=held,
        proof=proof,
        result=result,
    )
    assert out.disposition == "BLOCK"
    assert out.blocked
    assert out.block_kind == "environment-error"


# ---------------------------------------------------------------------------
# finalize — uniform COMPLETE/REQUEUE/BLOCK, same-hash same-tuple, divergent
# ---------------------------------------------------------------------------


def _bind(_lease):
    """Helper: no-op placeholder for tests that don't bind."""


def test_finalize_complete_persists_exact_result_bytes(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    out = xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    assert out.status == xw.FINALIZE_COMMITTED
    assert out.result_hash == hashlib.sha256(rb).hexdigest()
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["external_result_hash"] == out.result_hash
    assert row["external_result_json"] == rb.decode("utf-8")
    assert row["metadata"] == rb.decode("utf-8")
    assert row["summary"] == "fixture result"
    assert row["external_terminal_disposition"] == "COMPLETE"
    assert row["external_block_kind"] is None
    assert row["external_substate"] == xw.SUBSTATE_COMMITTED
    assert row["external_lease_state"] == xw.LEASE_COMMITTED
    t = kb.get_task(conn, tid)
    assert t is not None and t.status == "done"
    assert t.result == "fixture result"


def test_finalize_rejects_result_without_absence_proof(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="COMPLETE",
        absence_proven=False,
    )
    with pytest.raises(xw.ResultRejected):
        xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    assert xw.get_run(conn, run_id=lease.run_id).lease_state == xw.LEASE_ACTIVE


def test_finalize_rejects_mismatched_bound_process_identity(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    bound = xw.BoundProcess(host="h", pid=10, pgid=10, start_token="bound")
    lease = xw.bind_process(conn, lease=lease, process=bound)
    other = xw.BoundProcess(host="h", pid=11, pgid=11, start_token="other")
    rb = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="COMPLETE",
        process=other,
    )
    with pytest.raises(xw.ResultRejected):
        xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)


def test_finalize_requeue_persists_exact_result_bytes(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="REQUEUE"
    )
    out = xw.finalize(conn, lease=lease, disposition="REQUEUE", result_bytes=rb)
    assert out.status == xw.FINALIZE_COMMITTED
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["external_result_json"] == rb.decode("utf-8")
    assert row["external_terminal_disposition"] == "REQUEUE"
    t = kb.get_task(conn, tid)
    assert t is not None and t.status == "ready"


def test_finalize_block_persists_exact_result_bytes(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="BLOCK",
        block_kind="needs_input",
    )
    out = xw.finalize(
        conn,
        lease=lease,
        disposition="BLOCK",
        result_bytes=rb,
        block_kind="needs_input",
    )
    assert out.status == xw.FINALIZE_COMMITTED
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert row["external_terminal_disposition"] == "BLOCK"
    assert row["external_block_kind"] == "needs_input"
    t = kb.get_task(conn, tid)
    assert t is not None and t.status == "blocked"
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, tid).status == "blocked"


def test_finalize_same_hash_same_tuple_is_idempotent(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    out = xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    assert out.status == xw.FINALIZE_ALREADY_COMMITTED_SAME_HASH


def test_finalize_same_hash_different_disposition_is_rejected(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    # Same bytes but caller passes REQUEUE — the bytes encode COMPLETE so
    # this fails at result JSON validation, but if we encode a REQUEUE
    # version with otherwise-identical contents the hash differs. The
    # divergent-tuple check fires when we craft a REQUEUE result that
    # still produces a *different* hash.
    rb2 = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="REQUEUE"
    )
    out = xw.finalize(conn, lease=lease, disposition="REQUEUE", result_bytes=rb2)
    assert out.status == xw.FINALIZE_REJECTED
    assert out.prior_hash is not None


def test_finalize_divergent_hash_is_rejected(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb1 = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="COMPLETE",
    )
    xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb1)
    # Build a different valid result that hashes differently.
    rb2 = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="BLOCK",
        block_kind="capability",
    )
    out = xw.finalize(
        conn,
        lease=lease,
        disposition="BLOCK",
        result_bytes=rb2,
        block_kind="capability",
    )
    assert out.status == xw.FINALIZE_REJECTED
    assert out.prior_hash == hashlib.sha256(rb1).hexdigest()


def test_finalize_rejects_wrong_lease_token(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    bad = xw.Lease(
        run_id=lease.run_id,
        task_id=lease.task_id,
        spec=spec,
        lease_token="wrong",
        lease_state=xw.LEASE_ACTIVE,
        lease_expires_at=lease.lease_expires_at,
        attempt=1,
    )
    with pytest.raises(xw.NotOwner):
        xw.finalize(conn, lease=bad, disposition="COMPLETE", result_bytes=rb)


def test_finalize_rejects_wrong_run_id(kanban_home, conn):
    """Authorization is never by run id alone — wrong run id + right token
    must still reject (the run row's identity doesn't match)."""
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    bad = xw.Lease(
        run_id=lease.run_id + 999,
        task_id=lease.task_id,
        spec=spec,
        lease_token=lease.lease_token,
        lease_state=xw.LEASE_ACTIVE,
        lease_expires_at=lease.lease_expires_at,
        attempt=1,
    )
    with pytest.raises(xw.NotOwner):
        xw.finalize(conn, lease=bad, disposition="COMPLETE", result_bytes=rb)


# ---------------------------------------------------------------------------
# put_result — persist before finalize
# ---------------------------------------------------------------------------


def test_put_result_persists_without_finalizing(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    h = xw.put_result(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    assert h == hashlib.sha256(rb).hexdigest()
    row = conn.execute(
        "SELECT external_result_hash, external_terminal_disposition, "
        "external_substate, external_lease_state, ended_at "
        "FROM task_runs WHERE id = ?",
        (lease.run_id,),
    ).fetchone()
    assert row["external_result_hash"] == h
    assert row["external_terminal_disposition"] == "COMPLETE"
    # Run still active.
    assert row["external_substate"] == xw.SUBSTATE_CLAIMED
    assert row["external_lease_state"] == xw.LEASE_ACTIVE
    assert row["ended_at"] is None


def test_put_result_then_finalize_commits_same_bytes(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    xw.put_result(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    out = xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    assert out.status == xw.FINALIZE_COMMITTED


def test_put_result_rejects_divergent_staged_result(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    first = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    second = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="BLOCK",
        block_kind="capability",
    )
    xw.put_result(conn, lease=lease, disposition="COMPLETE", result_bytes=first)
    with pytest.raises(xw.ResultRejected):
        xw.put_result(
            conn,
            lease=lease,
            disposition="BLOCK",
            block_kind="capability",
            result_bytes=second,
        )


def test_recovery_cannot_overwrite_divergent_staged_result(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    first = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    xw.put_result(conn, lease=lease, disposition="COMPLETE", result_bytes=first)
    lease = _expire_lease(conn, lease)
    second = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="BLOCK",
        block_kind="environment-error",
    )
    result = xw.ExecutionResult(
        disposition="BLOCK",
        block_kind="environment-error",
        result_bytes=second,
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
    )
    proof = xw.NoStartAckProof(
        run_id=lease.run_id, task_id=tid, evidence="never-started"
    )
    with pytest.raises(xw.RecoveryRejected):
        xw.recover_expired(conn, lease=lease, proof=proof, result=result)
    row = conn.execute(
        "SELECT external_result_json, ended_at FROM task_runs WHERE id=?",
        (lease.run_id,),
    ).fetchone()
    assert row["external_result_json"] == first.decode()
    assert row["ended_at"] is None


# ---------------------------------------------------------------------------
# Public list/get/read APIs
# ---------------------------------------------------------------------------


def test_list_ready_returns_only_locked_external_tasks(kanban_home, conn):
    # A task with no spec lock must not appear even if status is ready.
    tid_no_spec = kb.create_task(conn, title="no-spec", body="b")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready' WHERE id = ?",
            (tid_no_spec,),
        )
    # A properly locked task with no open run SHOULD appear.
    tid, aid, _raw = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    ready = xw.list_ready(conn)
    ids = {t.id for t in ready}
    assert tid in ids
    assert tid_no_spec not in ids


def test_list_ready_excludes_task_with_open_run_despite_task_state_drift(
    kanban_home, conn
):
    tid, _aid, _raw, _spec, _lease = _submit_and_claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, current_run_id=NULL WHERE id=?",
            (tid,),
        )
    assert tid not in {task.id for task in xw.list_ready(conn)}


def test_list_active_returns_active_external_runs(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    active = xw.list_active(conn)
    assert len(active) == 1
    assert active[0].run_id == lease.run_id


def test_list_active_ignores_corrupted_task_status_and_run_pointer(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL WHERE id = ?",
            (tid,),
        )
    active = xw.list_active(conn)
    assert [item.run_id for item in active] == [lease.run_id]


def test_get_run_returns_typed_lease(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    got = xw.get_run(conn, run_id=lease.run_id)
    assert got.run_id == lease.run_id
    assert got.task_id == tid
    assert got.spec == spec


def test_get_run_rejects_unknown_run(kanban_home, conn):
    with pytest.raises(xw.ExternalWorkerError):
        xw.get_run(conn, run_id=999999)


# ---------------------------------------------------------------------------
# Native claim_task excludes external tasks
# ---------------------------------------------------------------------------


def test_native_claim_excludes_ready_external_task(kanban_home, conn):
    tid, _aid, _raw, _spec, _lease = _submit_and_claim(conn)
    # Requeue to ready.
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, current_run_id=NULL WHERE id = ?",
            (tid,),
        )
    # Native claim_task must surface the still-open run even when mutable task
    # state no longer points at it.
    with pytest.raises(kb.ExternalTaskConflict) as caught:
        kb.claim_task(conn, tid)
    assert caught.value.code == "external_run_active"
    assert caught.value.run_id == _lease.run_id


def test_native_claim_rejects_submitted_external_task_without_run(kanban_home, conn):
    tid, aid, _raw = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    with pytest.raises(kb.ExternalTaskConflict) as caught:
        kb.claim_task(conn, tid)
    assert caught.value.code == "external_spec_locked"
    assert caught.value.run_id is None


def test_end_run_defense_refuses_open_external_run(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(kb.ExternalTaskConflict) as caught:
        kb._end_run(conn, tid, outcome="reclaimed")
    assert caught.value.operation == "_end_run"
    assert caught.value.run_id == lease.run_id
    assert xw.get_run(conn, run_id=lease.run_id).lease_state == xw.LEASE_ACTIVE


@pytest.mark.parametrize(
    "operation",
    [
        "assign_task",
        "reassign_task",
        "complete_task",
        "block_task",
        "schedule_task",
        "archive_task",
        "delete_task",
        "reclaim_task",
    ],
)
def test_native_lifecycle_refuses_open_external_run_despite_task_drift(
    kanban_home, conn, operation
):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL WHERE id = ?",
            (tid,),
        )

    calls = {
        "assign_task": lambda: kb.assign_task(conn, tid, "other"),
        "reassign_task": lambda: kb.reassign_task(conn, tid, "other"),
        "complete_task": lambda: kb.complete_task(conn, tid, result="native"),
        "block_task": lambda: kb.block_task(conn, tid, reason="native"),
        "schedule_task": lambda: kb.schedule_task(conn, tid, reason="native"),
        "archive_task": lambda: kb.archive_task(conn, tid),
        "delete_task": lambda: kb.delete_task(conn, tid),
        "reclaim_task": lambda: kb.reclaim_task(conn, tid, reason="native"),
    }
    with pytest.raises(kb.ExternalTaskConflict) as caught:
        calls[operation]()
    assert caught.value.code == "external_run_active"
    assert caught.value.operation == operation
    assert caught.value.run_id == lease.run_id
    assert kb.get_task(conn, tid).status == "ready"
    run = conn.execute(
        "SELECT ended_at FROM task_runs WHERE id = ?", (lease.run_id,)
    ).fetchone()
    assert run["ended_at"] is None


def test_native_recovery_scans_ignore_open_external_run(kanban_home, conn, monkeypatch):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    lease = _expire_lease(conn, lease)
    host_lock = f"{kb._claimer_id().split(':', 1)[0]}:fixture"
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at=1, last_heartbeat_at=NULL, "
            "worker_pid=999999, max_runtime_seconds=1, claim_lock=?, "
            "current_run_id=NULL WHERE id=?",
            (host_lock, tid),
        )
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    signals = []

    assert (
        kb.release_stale_claims(conn, signal_fn=lambda *args: signals.append(args)) == 0
    )
    assert (
        kb.detect_stale_running(
            conn, stale_timeout_seconds=1, signal_fn=lambda *args: signals.append(args)
        )
        == []
    )
    assert kb.detect_crashed_workers(conn) == []
    assert (
        kb.enforce_max_runtime(conn, signal_fn=lambda *args: signals.append(args)) == []
    )
    assert signals == []
    assert xw.get_run(conn, run_id=lease.run_id).lease_state == xw.LEASE_ACTIVE


def test_native_heartbeat_refuses_open_external_run(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(kb.ExternalTaskConflict) as caught:
        kb.heartbeat_worker(conn, tid, expected_run_id=lease.run_id)
    assert caught.value.code == "external_run_active"
    assert caught.value.operation == "heartbeat_worker"


def test_dispatcher_does_not_spawn_external_ready_task(kanban_home, conn, monkeypatch):
    tid, aid, _raw = _make_task_with_spec(conn)
    xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET assignee='worker' WHERE id=?", (tid,))
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    spawned = []
    kb.dispatch_once(conn, spawn_fn=lambda *args: spawned.append(args))
    assert spawned == []
    assert kb.get_task(conn, tid).status == "ready"
    assert kb.has_spawnable_ready(conn) is False


def test_cli_surfaces_external_lifecycle_conflict(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    from hermes_cli.kanban import run_slash

    output = run_slash(f"complete {tid}")
    assert "external_run_active" in output
    assert f"run_id={lease.run_id}" in output
    assert kb.get_task(conn, tid).status == "running"


# ---------------------------------------------------------------------------
# put_artifact / read_artifact — idempotency + collision
# ---------------------------------------------------------------------------


def test_put_artifact_idempotent_for_same_name_same_bytes(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    ref1 = xw.put_artifact(conn, lease=lease, name="out.txt", data=b"hello")
    ref2 = xw.put_artifact(conn, lease=lease, name="out.txt", data=b"hello")
    assert ref1 == ref2
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM task_external_artifacts WHERE run_id = ?",
        (lease.run_id,),
    ).fetchone()
    assert int(rows["n"]) == 1


def test_put_artifact_rejects_divergent_bytes_same_name(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    xw.put_artifact(conn, lease=lease, name="out.txt", data=b"hello")
    with pytest.raises(xw.ArtifactCollision):
        xw.put_artifact(conn, lease=lease, name="out.txt", data=b"DIFFERENT")


def test_put_artifact_rejects_bad_name(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(ValueError):
        # Shell-meta characters are not in the allowed alphabet.
        xw.put_artifact(conn, lease=lease, name="out;rm.txt", data=b"x")


def test_read_artifact_returns_exact_bytes(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    xw.put_artifact(conn, lease=lease, name="out.txt", data=b"payload")
    out = xw.read_artifact(
        conn,
        run_id=lease.run_id,
        name="out.txt",
    )
    assert out == b"payload"


def test_read_artifact_remains_available_after_finalize(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    xw.put_artifact(conn, lease=lease, name="out.txt", data=b"payload")
    rb = _result_json(
        run_id=lease.run_id, task_id=tid, spec=spec, disposition="COMPLETE"
    )
    xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=rb)
    assert xw.read_artifact(conn, run_id=lease.run_id, name="out.txt") == b"payload"


def test_read_artifact_rejects_unknown_name(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(xw.ArtifactNotFound):
        xw.read_artifact(
            conn,
            run_id=lease.run_id,
            name="missing.txt",
        )


def test_put_artifact_rejects_wrong_lease(kanban_home, conn):
    _tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    bad = xw.Lease(
        run_id=lease.run_id,
        task_id=lease.task_id,
        spec=spec,
        lease_token="wrong",
        lease_state=xw.LEASE_ACTIVE,
        lease_expires_at=lease.lease_expires_at,
        attempt=1,
    )
    with pytest.raises(xw.NotOwner):
        xw.put_artifact(conn, lease=bad, name="x", data=b"y")


# ---------------------------------------------------------------------------
# Module never trusts caller-provided hash
# ---------------------------------------------------------------------------


def test_execution_result_dataclass_rejects_bad_block_kind(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="BLOCK",
        block_kind="not-a-real-kind",
    )
    with pytest.raises(ValueError):
        xw.ExecutionResult(
            disposition="BLOCK",
            block_kind="not-a-real-kind",
            result_bytes=rb,
            run_id=lease.run_id,
            task_id=tid,
            spec=spec,
        )


def test_external_block_kinds_include_review_and_environment():
    """The two block kinds added by the external result contract are valid."""
    assert "review-required" in kb.VALID_BLOCK_KINDS
    assert "environment-error" in kb.VALID_BLOCK_KINDS


def test_finalize_block_with_new_block_kinds(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    rb = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="BLOCK",
        block_kind="environment-error",
    )
    out = xw.finalize(
        conn,
        lease=lease,
        disposition="BLOCK",
        result_bytes=rb,
        block_kind="environment-error",
    )
    assert out.status == xw.FINALIZE_COMMITTED
    t = kb.get_task(conn, tid)
    assert t is not None and t.status == "blocked"


# Cross-surface lifecycle and restart regressions
# ---------------------------------------------------------------------------


def test_claim_rejects_caller_supplied_attempt_drift(kanban_home, conn):
    _tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(xw.NotOwner, match="attempt mismatch"):
        xw.heartbeat(
            conn,
            lease=replace(lease, attempt=lease.attempt + 1),
            lease_expires_at=int(time.time()) + 700,
        )


def test_attempt_ignores_native_task_run_history(kanban_home, conn):
    tid, aid, _raw = _make_task_with_spec(conn)
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs "
            "(task_id, status, worker_kind, started_at, ended_at) "
            "VALUES (?, 'completed', 'native', ?, ?)",
            (tid, now - 1, now),
        )
    spec = xw.submit(conn, task_id=tid, spec_attachment_id=aid)
    lease = xw.claim_external(
        conn,
        task_id=tid,
        expected_spec=spec,
        lease_token="external-first",
        lease_expires_at=now + 600,
    )
    assert lease.attempt == 1


def test_claim_rechecks_unfinished_parents_after_force_promote(kanban_home, conn):
    parent_id = kb.create_task(conn, title="parent", body="p", triage=True)
    child_id, child_aid, _raw = _make_task_with_spec(conn, title="child")
    kb.link_tasks(conn, parent_id, child_id)
    child_spec = xw.submit(conn, task_id=child_id, spec_attachment_id=child_aid)
    assert kb.promote_task(
        conn, child_id, actor="test", reason="force", force=True
    ) == (True, None)
    with pytest.raises(xw.ClaimRejected, match="unfinished parent"):
        xw.claim_external(
            conn,
            task_id=child_id,
            expected_spec=child_spec,
            lease_token="must-not-win",
            lease_expires_at=int(time.time()) + 600,
        )
    assert kb.get_task(conn, child_id).status == "ready"


def test_complete_atomically_promotes_external_dependents(kanban_home, conn):
    parent_id, parent_aid, _ = _make_task_with_spec(conn, title="parent")
    child_id, child_aid, _ = _make_task_with_spec(conn, title="child")
    kb.link_tasks(conn, parent_id, child_id)
    parent_spec = xw.submit(conn, task_id=parent_id, spec_attachment_id=parent_aid)
    xw.submit(conn, task_id=child_id, spec_attachment_id=child_aid)
    assert kb.get_task(conn, child_id).status == "todo"
    lease = xw.claim_external(
        conn,
        task_id=parent_id,
        expected_spec=parent_spec,
        lease_token="parent",
        lease_expires_at=int(time.time()) + 600,
    )
    result = _result_json(
        run_id=lease.run_id,
        task_id=parent_id,
        spec=parent_spec,
        disposition="COMPLETE",
    )
    xw.finalize(conn, lease=lease, disposition="COMPLETE", result_bytes=result)
    assert kb.get_task(conn, child_id).status == "ready"
    assert [task.id for task in xw.list_ready(conn)] == [child_id]


def test_named_board_artifacts_follow_connected_database(kanban_home):
    kb.create_board("named")
    named = xw.connect(board="named")
    try:
        _tid, _aid, _raw, _spec, lease = _submit_and_claim(named)
        xw.put_artifact(named, lease=lease, name="out.txt", data=b"named")
        row = named.execute(
            "SELECT stored_path FROM task_external_artifacts WHERE run_id = ?",
            (lease.run_id,),
        ).fetchone()
        stored = Path(row["stored_path"]).resolve()
        assert stored.is_relative_to(kb.attachments_root(board="named").resolve())
        assert not stored.is_relative_to(kb.attachments_root(board="default").resolve())
        kb.set_current_board("default")
        assert xw.read_artifact(named, run_id=lease.run_id, name="out.txt") == b"named"
    finally:
        named.close()


def test_board_removal_rejects_active_external_run(kanban_home):
    kb.create_board("named")
    named = xw.connect(board="named")
    try:
        _tid, _aid, _raw, _spec, lease = _submit_and_claim(named)
        with pytest.raises(kb.ExternalTaskConflict) as exc_info:
            kb.remove_board("named")
        assert exc_info.value.code == "external_run_active"
        assert exc_info.value.run_id == lease.run_id
        assert kb.board_exists("named")
        assert xw.get_run(named, run_id=lease.run_id).lease_state == xw.LEASE_ACTIVE
    finally:
        named.close()


def test_stale_connection_cannot_claim_after_board_archive(kanban_home, monkeypatch):
    external_attachments = kanban_home / "external-attachments"
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(external_attachments))
    kb.create_board("named")
    named = xw.connect(board="named")
    try:
        tid, aid, _raw = _make_task_with_spec(named)
        spec = xw.submit(named, task_id=tid, spec_attachment_id=aid)
        kb.remove_board("named")
        assert not kb.kanban_db_path(board="named").exists()
        with pytest.raises(xw.StaleBoardConnection, match="removed board"):
            xw.claim_external(
                named,
                task_id=tid,
                expected_spec=spec,
                lease_token="stale",
                lease_expires_at=int(time.time()) + 600,
            )
        assert xw.list_active(named) == []
    finally:
        named.close()


def test_native_heartbeat_cannot_extend_external_lease(kanban_home, conn):
    tid, _aid, _raw, _spec, lease = _submit_and_claim(conn)
    with pytest.raises(kb.ExternalTaskConflict):
        kb.heartbeat_claim(conn, tid, claimer=lease.lease_token)
    assert (
        xw.get_run(conn, run_id=lease.run_id).lease_expires_at == lease.lease_expires_at
    )


def test_status_independent_native_helpers_reject_external_run(kanban_home, conn):
    tid, _aid, _raw, _spec, _lease = _submit_and_claim(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
    with pytest.raises(kb.ExternalTaskConflict):
        kb.specify_triage_task(conn, tid, title="changed")
    with pytest.raises(kb.ExternalTaskConflict):
        kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="owner",
            children=[{"title": "child"}],
        )
    parent = kb.create_task(conn, title="parent", body="p", triage=True)
    with pytest.raises(kb.ExternalTaskConflict):
        kb.link_tasks(conn, parent, tid)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent, tid),
        )
    with pytest.raises(kb.ExternalTaskConflict):
        kb.unlink_tasks(conn, parent, tid)


def test_persisted_result_can_be_read_and_recovered_after_reopen(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    result_bytes = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="COMPLETE",
    )
    expected_hash = xw.put_result(
        conn,
        lease=lease,
        disposition="COMPLETE",
        result_bytes=result_bytes,
    )

    reopened = xw.connect(board="default")
    try:
        recovered_lease = xw.list_active(reopened)[0]
        persisted = xw.read_result(reopened, lease=recovered_lease)
        assert persisted is not None
        assert persisted.result_bytes == result_bytes
        assert persisted.result_hash == expected_hash
        recovered_lease = _expire_lease(reopened, recovered_lease)
        outcome = xw.recover_expired(
            reopened,
            lease=recovered_lease,
            proof=xw.NoStartAckProof(
                run_id=lease.run_id,
                task_id=tid,
                evidence="restart found no start ACK",
            ),
            result=xw.ExecutionResult(
                disposition=persisted.disposition,
                block_kind=persisted.block_kind,
                result_bytes=persisted.result_bytes,
                run_id=lease.run_id,
                task_id=tid,
                spec=spec,
            ),
        )
        assert outcome.disposition == "COMPLETE"
        assert kb.get_task(reopened, tid).status == "done"
    finally:
        reopened.close()


def test_recovery_hold_rejects_already_persisted_result(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    result_bytes = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="REQUEUE",
    )
    xw.put_result(
        conn,
        lease=lease,
        disposition="REQUEUE",
        result_bytes=result_bytes,
    )
    with pytest.raises(xw.RecoveryRejected, match="persisted result"):
        xw.hold_for_recovery(
            conn,
            lease=lease,
            proof=xw.RecoveryHoldProof(
                run_id=lease.run_id,
                task_id=tid,
                bound=None,
                evidence="must replay staged result",
            ),
        )
    assert xw.get_run(conn, run_id=lease.run_id).recovery_count == 0


def test_recovery_replays_staged_requeue_after_two_prior_holds(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    for index in range(2):
        lease = xw.hold_for_recovery(
            conn,
            lease=lease,
            proof=xw.RecoveryHoldProof(
                run_id=lease.run_id,
                task_id=tid,
                bound=None,
                evidence=f"inconclusive inspection {index + 1}",
            ),
            extension_seconds=60,
        )
    result_bytes = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="REQUEUE",
    )
    xw.put_result(
        conn,
        lease=lease,
        disposition="REQUEUE",
        result_bytes=result_bytes,
    )
    lease = _expire_lease(conn, lease)
    persisted = xw.read_result(conn, lease=lease)
    assert persisted is not None
    outcome = xw.recover_expired(
        conn,
        lease=lease,
        proof=xw.NoStartAckProof(
            run_id=lease.run_id,
            task_id=tid,
            evidence="restart proved no child start",
        ),
        result=xw.ExecutionResult(
            disposition=persisted.disposition,
            block_kind=persisted.block_kind,
            result_bytes=persisted.result_bytes,
            run_id=lease.run_id,
            task_id=tid,
            spec=spec,
        ),
    )
    assert outcome.requeued is True
    assert kb.get_task(conn, tid).status == "ready"


def test_finalize_rejects_declared_artifact_that_is_not_durable(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    result_bytes = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="COMPLETE",
        deliverable={
            "kind": "patch",
            "attachment": "missing.patch",
            "sha256": "1" * 64,
        },
        artifacts=[{"name": "also-missing.json", "sha256": "2" * 64}],
    )
    with pytest.raises(xw.ResultRejected, match="not persisted"):
        xw.finalize(
            conn,
            lease=lease,
            disposition="COMPLETE",
            result_bytes=result_bytes,
        )
    assert xw.get_run(conn, run_id=lease.run_id).lease_state == xw.LEASE_ACTIVE


def test_finalize_accepts_declared_durable_artifact(kanban_home, conn):
    tid, _aid, _raw, spec, lease = _submit_and_claim(conn)
    payload = b"patch bytes"
    ref = xw.put_artifact(conn, lease=lease, name="change.patch", data=payload)
    result_bytes = _result_json(
        run_id=lease.run_id,
        task_id=tid,
        spec=spec,
        disposition="COMPLETE",
        deliverable={
            "kind": "patch",
            "attachment": ref.name,
            "sha256": ref.sha256,
        },
        artifacts=[{"name": ref.name, "sha256": ref.sha256}],
    )
    outcome = xw.finalize(
        conn,
        lease=lease,
        disposition="COMPLETE",
        result_bytes=result_bytes,
    )
    assert outcome.status == xw.FINALIZE_COMMITTED
