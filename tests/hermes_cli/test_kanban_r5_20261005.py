"""R5-01 / R5-02 regression tests — review 5 (GPT-6.1 Sol, 2026-10-05).

R5-01: the attachment UPLOAD path used a precheck-then-``write_bytes``
sequence; a concurrent P3b preservation could ``os.link``-publish and bind a
proof on the same (task, basename) between the two steps, and the upload then
TRUNCATED a committed ``done`` card's proof while both calls returned ok.
The upload path now publishes through a staging file with O_EXCL ``os.link``
name reservation — an existing file is never opened for writing.

R5-02: the P2b human gate was only evaluated BEFORE the completion write
lock; a ``HUMAN_GATE_PENDING`` marker armed between the pre-lock check and
the txn completed unimpeded (the status UPDATE fence covers statuses only,
not marker state). The gate is now re-evaluated INSIDE the txn; the trip
rolls the completion back and emits the same auditable refusal.

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixture conventions).
Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

_ART = tempfile.NamedTemporaryFile(prefix="r5_art_", delete=False).name

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_external_artifacts as kea


import hashlib as _hashlib_mod


def _dig(_p) -> str:
    """sha256 of the file at _p (bytes) — P2b-close digest binding."""
    from pathlib import Path as _P
    return _hashlib_mod.sha256(_P(_p).read_bytes()).hexdigest()

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME/kanban home with an empty kanban DB."""
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
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]


def _event_kinds(conn, task_id):
    return [
        r[0]
        for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def _event_payloads(conn, task_id, kind):
    return [
        json.loads(r[0]) if r[0] else {}
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (task_id, kind),
        ).fetchall()
    ]


def _comment_now(conn, task_id, author, body):
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, author, body, int(time.time())),
    )
    conn.commit()


def _claimed_task(conn, title="r5 card"):
    tid = kb.create_task(conn, title=title, assignee="coder")
    claimed = kb.claim_task(conn, tid, claimer="host:mock")
    assert claimed is not None, "claim must succeed for the completion path"
    return tid


def _external_contract_task(conn, kanban_home, name="proof.txt", content=b"evidence"):
    """A claimed card whose contract declares one EXTERNAL artifact (forces
    the P3b capture/publish pre-lock I/O into ``complete_task``)."""
    tid = _claimed_task(conn, title=f"r5 {name}")
    ext = Path(kanban_home) / f"ext-{name}"
    ext.write_bytes(content)
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        (json.dumps({"required_artifacts": [str(ext)]}), tid),
    )
    conn.commit()
    return tid, ext


class TestR501UploadNeverTruncatesAPublishedProof:
    def test_store_shifts_aside_when_name_already_taken(self, conn):
        """The review's cross-writer interleaving, collapsed to its
        deterministic core: the name is won by the preservation copy BEFORE
        the upload's write step. The upload must land on a shifted name and
        leave the proof byte-identical."""
        tid = kb.create_task(conn, title="r5-01")
        dest_dir = kb.task_attachments_dir(tid, board=None)
        dest_dir.mkdir(parents=True, exist_ok=True)
        proof = dest_dir / "proof.txt"
        proof.write_bytes(b"PRESERVED-PROOF-BYTES")

        kb.store_attachment_bytes(conn, tid, "proof.txt", b"unrelated-upload")

        assert proof.read_bytes() == b"PRESERVED-PROOF-BYTES", (
            "the pre-existing file must never be truncated by an upload"
        )
        rows = kb.list_attachments(conn, tid)
        assert len(rows) == 1
        stored = Path(rows[0].stored_path)
        assert stored.read_bytes() == b"unrelated-upload"
        assert stored != proof, "the upload must take a DIFFERENT name"

    def test_fresh_name_is_still_used_verbatim(self, conn):
        """Backwards compatibility: with no pre-existing file the blob lands
        on the exact requested basename."""
        tid = kb.create_task(conn, title="r5-01 fresh")
        kb.store_attachment_bytes(conn, tid, "proof.txt", b"payload")
        rows = kb.list_attachments(conn, tid)
        assert len(rows) == 1
        assert Path(rows[0].stored_path).name == "proof.txt"
        assert Path(rows[0].stored_path).read_bytes() == b"payload"

    def test_row_failure_cleans_only_its_own_blob(self, conn, monkeypatch):
        """The no-orphan cleanup must keep the identity guard: a name whose
        file was replaced meanwhile (another writer's proof) is not ours to
        unlink."""
        tid = kb.create_task(conn, title="r5-01 cleanup")
        dest_dir = kb.task_attachments_dir(tid, board=None)
        dest_dir.mkdir(parents=True, exist_ok=True)
        proof = dest_dir / "proof.txt"
        proof.write_bytes(b"PRESERVED-PROOF-BYTES")

        def _boom(*a, **kw):
            raise RuntimeError("row insert fails")

        monkeypatch.setattr(kb, "add_attachment", _boom)
        with pytest.raises(RuntimeError):
            kb.store_attachment_bytes(conn, tid, "proof.txt", b"unrelated-upload")
        # The OTHER writer's proof survived, and no stray upload blob leaked.
        assert proof.read_bytes() == b"PRESERVED-PROOF-BYTES"
        leftovers = sorted(p.name for p in dest_dir.iterdir())
        assert leftovers == ["proof.txt"]

    def test_two_uploads_same_basename_both_survive(self, conn):
        """Sequential double upload keeps both blobs (shifted names), like
        the old ``foo (1)`` chooser promised."""
        tid = kb.create_task(conn, title="r5-01 twice")
        kb.store_attachment_bytes(conn, tid, "proof.txt", b"first")
        kb.store_attachment_bytes(conn, tid, "proof.txt", b"second")
        rows = kb.list_attachments(conn, tid)
        assert len(rows) == 2
        blobs = {Path(r.stored_path).read_bytes() for r in rows}
        assert blobs == {b"first", b"second"}


class TestR502HumanGateArmedBetweenPrecheckAndTxn:
    def test_marker_armed_after_precheck_blocks_completion(self, conn, kanban_home, monkeypatch):
        """R5-02 exact shape: the gate is armed AFTER ``complete_task``'s
        pre-lock human-gate check (the P3b publish seam sits inside that
        window). The completion must roll back and emit the auditable
        refusal — never close."""
        tid, _ext = _external_contract_task(conn, kanban_home)
        armed = threading.Event()

        def _arm_during_publish(_origin_path):
            if armed.is_set():
                return
            armed.set()
            with kbc.connect() as c2:
                c2.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (tid, "operator", "HUMAN_GATE_PENDING: R52", int(time.time())),
                )
                c2.commit()

        monkeypatch.setattr(kea, "_PUBLISH_PRE_LINK_HOOK", _arm_during_publish)
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="r5-02", summary="s")
        assert _status(conn, tid) != "done"
        kinds = _event_kinds(conn, tid)
        assert "completed" not in kinds
        assert "completion_blocked_human_gate" in kinds
        assert _event_payloads(conn, tid, "completion_blocked_human_gate") == [
            {"gate_ids": ["R52"]}
        ]

    def test_approval_after_late_arm_still_completes(self, conn, kanban_home, monkeypatch):
        """The legitimate exit survives the fix: an approval marker (with a
        real artifact on disk) recorded after the late arm releases the gate
        in-txn and the completion closes normally."""
        tid, _ext = _external_contract_task(conn, kanban_home, name="ok.txt")

        def _arm_then_approve(_origin_path):
            with kbc.connect() as c2:
                c2.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (tid, "operator", "HUMAN_GATE_PENDING: R52", int(time.time())),
                )
                c2.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        tid, "operator",
                        f"HUMAN_GATE_APPROVAL: R52 artifact={_ART} digest={_dig(_ART)}",
                        int(time.time()),
                    ),
                )
                c2.commit()

        monkeypatch.setattr(kea, "_PUBLISH_PRE_LINK_HOOK", _arm_then_approve)
        assert kb.complete_task(conn, tid, result="r5-02", summary="s") is True
        assert _status(conn, tid) == "done"
        assert "completion_blocked_human_gate" not in _event_kinds(conn, tid)

    def test_prelock_armed_marker_is_still_the_specific_refusal(self, conn):
        """Regression guard for the pre-lock path: an ALREADY-armed gate is
        refused before the evidence gates, with the same event + payload."""
        tid = _claimed_task(conn)
        _comment_now(conn, tid, "operator", "HUMAN_GATE_PENDING: PRE")
        with pytest.raises(kb.HumanGatePendingError):
            kb.complete_task(conn, tid, result="x", summary="s")
        assert _status(conn, tid) != "done"
        assert _event_payloads(conn, tid, "completion_blocked_human_gate") == [
            {"gate_ids": ["PRE"]}
        ]
