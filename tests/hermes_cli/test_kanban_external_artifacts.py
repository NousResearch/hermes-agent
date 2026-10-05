"""P3b external-artifact durable preservation (audit 2026-10-04, item P3b).

Closes the ``delete-after-recheck`` race: before P3b an EXTERNAL artifact
declared in ``completion_contract`` (``required_artifacts``) was validated by
``is_file()`` at the gate and re-checked in-txn, but NEVER copied — a
delete/replace between (or after) the in-txn recheck left a ``done`` card
pointing at a pathname that no longer held the validated content (and a
pathname is not proof of content). P3b adds a Hermes-managed durable copy:
capture → atomic publish → in-txn binding + auditable
``external_artifact_preserved`` event.

Backwards compatibility pinned here: a contract WITHOUT external artifacts
(on-board / scratch-only) keeps the exact pre-P3b behaviour — no new event,
no extra attachment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_external_artifacts as kea


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (never a real board)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_contract(conn, task_id: str, contract) -> None:
    value = contract if isinstance(contract, str) else json.dumps(contract)
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        (value, task_id),
    )
    conn.commit()


def _scratch_task(conn, kanban_home, title: str):
    """A task with a real managed-scratch workspace directory."""
    t = kb.create_task(conn, title=title)
    task = kb.get_task(conn, t)
    ws = kbw.resolve_workspace(task)
    kbw.set_workspace_path(conn, t, ws)
    return t, ws


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()[0]


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
        ).fetchall()
    ]


def _event_payloads(conn, task_id: str, kind: str) -> list[dict]:
    return [
        json.loads(r["payload"]) if isinstance(r["payload"], str) else (r["payload"] or {})
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
            (task_id, kind),
        ).fetchall()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Preserve + survive deletion/replacement of the external origin
# ---------------------------------------------------------------------------


def test_external_artifact_preserved_and_survives_origin_deletion(kanban_home):
    """A declared external artifact is copied to managed storage with a recorded
    sha256; deleting/replacing the origin afterwards leaves the managed copy
    intact and matching the recorded digest (proof by command)."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "preserve external")
        origin = Path(kanban_home) / "evidence.md"
        payload = b"validated evidence bytes\n"
        origin.write_bytes(payload)
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        assert kb.complete_task(conn, tid, result="done with external") is True

        # A managed attachment copy exists and holds the validated bytes.
        atts = kb.list_attachments(conn, tid)
        assert len(atts) == 1, "the external artifact must be persisted as an attachment"
        stored = Path(atts[0].stored_path)
        assert stored.is_file()
        assert stored.read_bytes() == payload
        digest = _sha256(stored)

        # The auditable event names origin + managed dest + digest + size.
        payloads = _event_payloads(conn, tid, "external_artifact_preserved")
        assert payloads, "preservation must be auditable"
        rec = payloads[0]
        assert rec["origin"] == str(origin)
        assert rec["stored_path"] == str(stored.resolve())
        assert rec["sha256"] == digest
        assert rec["size"] == len(payload)

        # The binding is on the closing run's metadata too.
        run = kb.latest_run(conn, tid)
        bound = (run.metadata or {}).get("external_artifacts_preserved")
        assert bound and bound[0]["sha256"] == digest
        assert bound[0]["stored_path"] == str(stored.resolve())

        # The completed event references the MANAGED copy (not only the origin).
        completed = _event_payloads(conn, tid, "completed")
        assert any(str(stored.resolve()) in json.dumps(p) for p in completed)

        # Delete + replace the origin AFTER completion: the managed copy is the
        # durable proof and still matches the recorded digest.
        origin.unlink()
        origin.write_bytes(b"tampered replacement content")
        assert stored.is_file()
        assert _sha256(stored) == rec["sha256"] == digest


# ---------------------------------------------------------------------------
# 2. Race: origin removed between capture and the closing transaction
# ---------------------------------------------------------------------------


def test_origin_deleted_between_capture_and_txn_still_references_copy(
    kanban_home, monkeypatch
):
    """The exact race the audit flagged: the origin is deleted AFTER the in-txn
    recheck has run but before the commit. The completed card must reference the
    intact managed copy captured before the lock — never a broken reference."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "delete-after-recheck")
        origin = Path(kanban_home) / "raced.bin"
        payload = b"the validated bytes"
        origin.write_bytes(payload)
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_recheck = kb._recheck_contract_in_txn

        def recheck_then_delete(c, task_id):
            real_recheck(c, task_id)
            origin.unlink()  # the race: gone after recheck, before commit

        monkeypatch.setattr(kb, "_recheck_contract_in_txn", recheck_then_delete)

        assert kb.complete_task(conn, tid, result="raced") is True
        assert not origin.exists(), "fixture must actually delete the origin"

        atts = kb.list_attachments(conn, tid)
        assert atts, "a done card must reference the durable managed copy"
        stored = Path(atts[0].stored_path)
        assert stored.is_file()
        assert stored.read_bytes() == payload


def test_origin_deleted_between_capture_and_intxn_recheck_never_breaks(
    kanban_home, monkeypatch
):
    """Variant: the origin vanishes between capture and the in-txn recheck. The
    existing recheck mechanism may refuse (the requirement's origin is gone) —
    which is a valid, safe outcome — but a completed card must NEVER reference a
    missing/changed file: either it closes onto the digest-verified managed copy,
    or it is refused with the card left un-done."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "delete-before-recheck")
        origin = Path(kanban_home) / "vanish.bin"
        origin.write_bytes(b"bytes that must survive")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_capture = kea.capture_external_artifacts

        def capture_then_delete(c, task_id):
            caps = real_capture(c, task_id)
            origin.unlink()
            return caps

        monkeypatch.setattr(kea, "capture_external_artifacts", capture_then_delete)

        try:
            completed = kb.complete_task(conn, tid, result="vanish race")
        except kb.ContractSpecError:
            completed = False  # origin gone before the in-txn recheck: refused
        if completed:
            atts = kb.list_attachments(conn, tid)
            assert atts, "a completed card must not point at a vanished origin"
            stored = Path(atts[0].stored_path)
            assert stored.is_file()
            assert stored.read_bytes() == b"bytes that must survive"
        else:
            assert _status(conn, tid) != "done"
            # No dangling reference from a refusal either.
            for att in kb.list_attachments(conn, tid):
                assert Path(att.stored_path).is_file()
            # The published copy was discarded on rollback: no orphan blob left
            # behind (an orphan would make a retry stage a duplicate).
            att_dir = kb.task_attachments_dir(tid)
            leftovers = list(att_dir.iterdir()) if att_dir.is_dir() else []
            assert not leftovers, f"rollback left orphan copies: {leftovers}"


# ---------------------------------------------------------------------------
# 3. Too large / unstable capture — no undue completion
# ---------------------------------------------------------------------------


def test_external_artifact_over_cap_refused(kanban_home, monkeypatch):
    """A declared external artifact above the attachment cap is a typed refusal:
    no completion, no managed copy, card stays in its prior status."""
    monkeypatch.setattr(kb, "KANBAN_ATTACHMENT_MAX_BYTES", 1024)
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "over cap")
        origin = Path(kanban_home) / "huge.bin"
        origin.write_bytes(b"x" * 4096)
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(conn, tid, result="too big")
        assert _status(conn, tid) == "ready"
        assert "completed" not in _event_kinds(conn, tid)
        assert not kb.list_attachments(conn, tid)


def test_unstable_capture_refused(kanban_home):
    """A file that changes WHILE being captured fails the completion: a card is
    never closed against unstable bytes."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "unstable")
        origin = Path(kanban_home) / "unstable.bin"
        origin.write_bytes(b"stable original content")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_hook = kea._CAPTURE_MID_READ_HOOK

        def mutate_during_read(path):
            # Different LENGTH guarantees detection regardless of mtime grain.
            Path(path).write_bytes(b"mutated during the read!!")

        kea._CAPTURE_MID_READ_HOOK = mutate_during_read
        try:
            with pytest.raises(kea.ExternalArtifactPreservationError):
                kb.complete_task(conn, tid, result="unstable capture")
        finally:
            kea._CAPTURE_MID_READ_HOOK = real_hook
        assert _status(conn, tid) == "ready"
        assert "completed" not in _event_kinds(conn, tid)


# ---------------------------------------------------------------------------
# 4. No external contract — behaviour byte-for-byte unchanged
# ---------------------------------------------------------------------------


def test_scratch_only_contract_behaviour_unchanged(kanban_home):
    """A scratch-only requirements contract keeps the exact pre-P3b path: the
    scratch artifact is staged as before, no external-preservation noise."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "scratch only")
        artifact = ws / "report.md"
        artifact.write_text("# report", encoding="utf-8")
        _set_contract(conn, tid, {"required_artifacts": [str(artifact)]})

        assert kb.complete_task(conn, tid, result="scratch only") is True
    kinds = _event_kinds(conn, tid)
    assert "external_artifact_preserved" not in kinds
    assert "external_artifact_recorded" not in kinds


def test_inert_contract_completion_unchanged(kanban_home):
    """'local-only' never triggers external capture (no extra events/attachments)."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "inert")
        _set_contract(conn, tid, "local-only")
        assert kb.complete_task(conn, tid, summary="substantive summary") is True
    kinds = _event_kinds(conn, tid)
    assert "external_artifact_preserved" not in kinds
    assert "external_artifact_recorded" not in kinds
