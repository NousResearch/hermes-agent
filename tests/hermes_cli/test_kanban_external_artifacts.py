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

import contextlib
import hashlib
import json
import os
import sqlite3
import stat
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

        def capture_then_delete(c, task_id, **kwargs):
            caps = real_capture(c, task_id, **kwargs)
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


# ===========================================================================
# ROUND 2 (adversarial review d329e2a5e6, GPT-6.1 Sol) — regression guards
# for the 3 HIGH + 4 MEDIUM + 1 LOW defects closed by this round.
# ===========================================================================


def _attachment_rows(conn, task_id):
    return conn.execute(
        "SELECT filename, stored_path FROM task_attachments WHERE task_id = ?",
        (task_id,),
    ).fetchall()


def _attached_events(conn, task_id):
    return _event_payloads(conn, task_id, "attached")


def _att_dir(task_id):
    return kb.task_attachments_dir(task_id)


def _leftovers(task_id):
    d = _att_dir(task_id)
    return list(d.iterdir()) if d.is_dir() else []


# ---------------------------------------------------------------------------
# HIGH #1 — concurrent publication: per-attempt unique destination, no overwrite
# ---------------------------------------------------------------------------


def test_publish_is_unique_per_attempt_and_never_overwrites(kanban_home, monkeypatch):
    """Two attempts publishing the SAME artifact name must land on DISTINCT
    finals and NEVER overwrite an existing file. The pre-link seam makes the
    collision deterministic: a concurrent attempt wins one intermediate name;
    the loser advances past it without touching any existing bytes."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "concurrent publish")
        origin = Path(kanban_home) / "proof.md"
        origin.write_bytes(b"validated bytes")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        published_a = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        a_path = Path(published_a[0].published_path)
        assert a_path.name == "proof.md"
        a_bytes = a_path.read_bytes()
        assert a_bytes == b"validated bytes"

        # A concurrent attempt wins the FIRST still-free name (a real attempt
        # links atomically; here we occupy it directly). Exactly once.
        stolen = {"done": False}

        def other_attempt_wins(candidate):
            p = Path(candidate)
            if stolen["done"] or p.exists():
                return
            stolen["done"] = True
            p.write_bytes(b"OTHER-ATTEMPT")  # foreign copy occupies the name

        monkeypatch.setattr(kea, "_PUBLISH_PRE_LINK_HOOK", other_attempt_wins)
        published_b = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        b_path = Path(published_b[0].published_path)

        assert a_path != b_path, "concurrent attempts must not share a final name"
        assert b_path.name == "proof_2.md", "loser must advance past the won name"
        assert a_path.read_bytes() == a_bytes, "attempt B must not overwrite A's proof"
        assert (a_path.parent / "proof_1.md").read_bytes() == b"OTHER-ATTEMPT", (
            "the foreign copy at the contested name must not be overwritten"
        )
        assert b_path.read_bytes() == b"validated bytes"


def test_discard_of_one_attempt_never_removes_the_other(kanban_home):
    """The cleanup of a refused attempt must remove ONLY its own files."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "discard isolation")
        origin = Path(kanban_home) / "iso.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        published_a = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        published_b = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        a_path = Path(published_a[0].published_path)
        b_path = Path(published_b[0].published_path)
        assert a_path != b_path and a_path.is_file() and b_path.is_file()

        kea.discard_published_artifacts(published_b)
        assert a_path.is_file(), "discarding one attempt must not delete another"
        assert not b_path.exists()


# ---------------------------------------------------------------------------
# HIGH #2 — incomplete capture: absent-at-capture refuses; A→B→A not preserved
# ---------------------------------------------------------------------------


def test_declared_external_absent_at_capture_is_refused(kanban_home, monkeypatch):
    """A declared external requirement that is PRESENT at the gate but GONE at
    capture must refuse (typed), not silently close with no copy — even if it
    reappears before the in-txn recheck."""
    monkeypatch.setattr(kb, "_gate_contract_completion", lambda c, t: None)
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "absent at capture")
        origin = Path(kanban_home) / "ghost.md"
        origin.write_bytes(b"evidence")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_capture = kea.capture_external_artifacts

        def delete_then_capture(c, task_id, **kwargs):
            origin.unlink()
            try:
                return real_capture(c, task_id, **kwargs)
            finally:
                origin.write_bytes(b"reappeared")  # reappears before recheck

        monkeypatch.setattr(kea, "capture_external_artifacts", delete_then_capture)

        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(
                conn, tid, result="must not close without a preserved copy"
            )
        assert _status(conn, tid) != "done"
        assert not _attachment_rows(conn, tid)
        assert "completed" not in _event_kinds(conn, tid)
        assert not _leftovers(tid), "a refusal must not leave orphan copies"
        assert origin.is_file(), "fixture must have let it reappear"


def test_contract_aba_swap_refuses_and_does_not_preserve_wrong_version(
    kanban_home, monkeypatch
):
    """Gate sees contract A; a concurrent writer swaps to B (a DIFFERENT external
    requirement) and back. Capture rides A's snapshot; the txn validates what is
    in force. No version the txn did not validate may be preserved, and the card
    must not close riding an unvalidated set."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "A->B->A")
        a = Path(kanban_home) / "a.md"
        b = Path(kanban_home) / "b.md"
        a.write_bytes(b"contract A evidence")
        b.write_bytes(b"contract B evidence")
        contract_a = json.dumps({"required_artifacts": [str(a)]}, separators=(",", ":"))
        contract_b = json.dumps({"required_artifacts": [str(b)]}, separators=(",", ":"))
        _set_contract(conn, tid, contract_a)

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_aba(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (contract_b, task_id),
                    )
                    c2.commit()
                    c2.execute(
                        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                        (contract_a, task_id),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_aba)

        result = None
        try:
            result = kb.complete_task(conn, tid, result="A->B->A probe")
        except kb.ContractSpecError:
            result = None  # acceptable: audible refusal
        # Whatever happens, the transaction's own verdict must be the authority.
        if result:
            bound = (kb.latest_run(conn, tid).metadata or {}).get(
                "external_artifacts_preserved"
            )
            assert bound, "a completed external card must carry a binding"
            covered = {Path(x["origin"]).name for x in bound}
            assert covered <= {"a.md"}, "must not preserve a version not validated"
            for x in bound:
                assert Path(x["stored_path"]).is_file()
        else:
            assert _status(conn, tid) != "done"
            assert not _attachment_rows(conn, tid)
            assert not _leftovers(tid), "refusal left orphan copies"


def test_preserved_set_covers_all_required_externals(kanban_home, monkeypatch):
    """Coverage: a multi-requirement external contract whose pre-lock capture is
    PARTIAL is refused in-txn (the captured set must cover every requirement)."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "coverage positive")
        one = Path(kanban_home) / "one.md"
        two = Path(kanban_home) / "two.md"
        one.write_bytes(b"one")
        two.write_bytes(b"two")
        _set_contract(conn, tid, {"required_artifacts": [str(one), str(two)]})

        real_capture = kea.capture_external_artifacts
        monkeypatch.setattr(
            kea, "capture_external_artifacts",
            lambda c, task_id, **kw: real_capture(c, task_id, **kw)[:1],
        )
        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(conn, tid, result="partial capture")
        assert _status(conn, tid) != "done"
        assert not _leftovers(tid)


# ---------------------------------------------------------------------------
# HIGH #3 / MEDIUM #4 — cleanup discipline: no orphans on return False; no
# deletion after commit
# ---------------------------------------------------------------------------


def test_return_false_after_publish_leaves_no_orphans(kanban_home, monkeypatch):
    """A ``return False`` AFTER publication (parent reopened inside the txn) must
    not leave orphan copies behind: the uniform ``finally`` cleanup must fire on
    the non-exception negative exit too."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "return False orphan")
        origin = Path(kanban_home) / "rf.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        # Reopen the parent AFTER the pre-lock snapshot: the in-txn hard
        # invariant returns False with the copies already published.
        real_ps = kb._parents_satisfied
        calls = {"n": 0}

        def ps(c, t):
            calls["n"] += 1
            return True if calls["n"] == 1 else False

        monkeypatch.setattr(kb, "_parents_satisfied", ps)
        del real_ps

        assert kb.complete_task(conn, tid, result="parent reopened") is False
        assert not _attachment_rows(conn, tid)
        assert not _leftovers(tid), f"return False left orphans: {_leftovers(tid)}"


def test_failure_after_commit_keeps_bound_copies(kanban_home, monkeypatch):
    """A failure AFTER the completion COMMIT (post-commit invariant) must NOT
    delete the bound copies — the binding is already durable."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "post-commit failure")
        origin = Path(kanban_home) / "keep.bin"
        origin.write_bytes(b"durable proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        def boom(*a, **k):
            raise sqlite3.DatabaseError("simulated post-commit failure")

        monkeypatch.setattr(kb, "recompute_ready", boom)

        with pytest.raises(sqlite3.DatabaseError):
            kb.complete_task(conn, tid, result="commits then fails")
        atts = _attachment_rows(conn, tid)
        assert atts, "the binding committed before the failure"
        for row in atts:
            assert Path(row["stored_path"]).is_file(), (
                "a post-commit failure must NOT delete a bound copy"
            )


def test_write_txn_post_commit_invariant_failure_keeps_copies(kanban_home, monkeypatch):
    """The exact HIGH #3 path: ``write_txn`` COMMITs and THEN runs
    ``_check_file_length_invariant``, which can raise. The status flip and the
    attachment rows are already committed, so the copies must be kept."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "torn-extend after commit")
        origin = Path(kanban_home) / "torn.bin"
        origin.write_bytes(b"durable proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_invariant = kbc._check_file_length_invariant

        def boom(_conn):
            raise sqlite3.DatabaseError("simulated post-COMMIT torn-extend")

        monkeypatch.setattr(kbc, "_check_file_length_invariant", boom)
        with pytest.raises(sqlite3.DatabaseError):
            kb.complete_task(conn, tid, result="committed then invariant fails")
        monkeypatch.setattr(kbc, "_check_file_length_invariant", real_invariant)

        assert _status(conn, tid) == "done", "the status flip committed"
        atts = _attachment_rows(conn, tid)
        assert atts, "the binding committed with the status flip"
        for row in atts:
            assert Path(row["stored_path"]).is_file(), (
                "a post-COMMIT invariant failure must NOT delete a bound copy"
            )


def test_no_orphans_on_contract_swap_refusal(kanban_home, monkeypatch):
    """The in-txn contract-swap refusal cleans its published copies."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "swap refusal cleanup")
        origin = Path(kanban_home) / "swap.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = 'local-only' "
                        "WHERE id = ?",
                        (task_id,),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)
        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(conn, tid, result="contract swapped to inert")
        assert not _attachment_rows(conn, tid)
        assert not _leftovers(tid), f"swap refusal left orphans: {_leftovers(tid)}"


# ---------------------------------------------------------------------------
# MEDIUM #6 — mixed contract (scratch + external) must not duplicate
# ---------------------------------------------------------------------------


def test_mixed_contract_does_not_duplicate_external_attachment(kanban_home):
    """A contract mixing a scratch artifact and an external one binds the
    external copy exactly ONCE (one row, one ``attached`` event)."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "mixed contract")
        scratch = ws / "report.md"
        scratch.write_text("# report", encoding="utf-8")
        external = Path(kanban_home) / "external.md"
        external.write_bytes(b"external evidence")
        _set_contract(
            conn, tid, {"required_artifacts": [str(scratch), str(external)]}
        )

        assert kb.complete_task(conn, tid, result="mixed") is True

        rows = _attachment_rows(conn, tid)
        extern_names = [r["filename"] for r in rows if "external" in r["filename"]]
        assert len(extern_names) == 1, f"external attachment duplicated: {rows}"

        ext_attached = [
            p for p in _attached_events(conn, tid)
            if "external" in str(p.get("filename", ""))
        ]
        assert len(ext_attached) == 1, f"external attached event duplicated: {ext_attached}"

        # And no spurious external_artifact_recorded for OUR managed copy.
        for p in _event_payloads(conn, tid, "external_artifact_recorded"):
            assert "/attachments/" not in str(p.get("artifact", "")), (
                "must not re-record our own managed copy as external"
            )

        # The scratch artifact is still preserved exactly once.
        scratch_rows = [r for r in rows if "report" in r["filename"]]
        assert len(scratch_rows) == 1, f"scratch attachment duplicated: {rows}"
        for r in rows:
            assert Path(r["stored_path"]).is_file()


# ---------------------------------------------------------------------------
# MEDIUM #7 — strengthened stability fingerprint (same length + mtime restored)
# ---------------------------------------------------------------------------


def test_same_length_rewrite_with_restored_mtime_is_rejected(kanban_home):
    """A same-length, in-place rewrite that restores the mtime is detected via
    the ctime in the fingerprint — a card must never close onto bytes that were
    mixed during the read."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "same-length rewrite")
        origin = Path(kanban_home) / "stable.bin"
        origin.write_bytes(b"AAAA")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_hook = kea._CAPTURE_MID_READ_HOOK

        def mutate_same_length(path):
            st = origin.stat()
            origin.write_bytes(b"BBBB")  # same length
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))  # restore mtime

        kea._CAPTURE_MID_READ_HOOK = mutate_same_length
        try:
            with pytest.raises(kea.ExternalArtifactPreservationError):
                kb.complete_task(conn, tid, result="same-length mutation")
        finally:
            kea._CAPTURE_MID_READ_HOOK = real_hook
        assert _status(conn, tid) != "done"
        assert "completed" not in _event_kinds(conn, tid)


# ---------------------------------------------------------------------------
# LOW — raw OSError wrapped in the typed, recoverable refusal
# ---------------------------------------------------------------------------


def test_raw_oserror_from_open_is_typed(kanban_home, monkeypatch):
    """A raw ``OSError`` from ``path.open`` (e.g. deleted between the check and
    the open, EACCES, EIO) is wrapped in the recoverable typed error."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "raw oserror")
        origin = Path(kanban_home) / "oserr.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_open = Path.open

        def flaky_open(self, *a, **k):
            if self == origin:
                raise OSError("EIO: simulated raw open failure")
            return real_open(self, *a, **k)

        monkeypatch.setattr(Path, "open", flaky_open)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(conn, tid, result="raw os error")


# ---------------------------------------------------------------------------
# MEDIUM #5 — directory durability is not silently ignored
# ---------------------------------------------------------------------------


def test_dir_fsync_failure_is_typed(kanban_home, monkeypatch):
    """MEDIUM #5: a failed directory fsync is a TYPED refusal, never a silent
    durability promise."""
    d = kanban_home / "adurdir"
    d.mkdir()

    def flaky_fsync(fd):
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(os, "fsync", flaky_fsync)
    with pytest.raises(kea.ExternalArtifactPreservationError):
        kea._fsync_dir(d)


def test_publish_dir_fsync_failure_is_typed_and_no_orphans(kanban_home, monkeypatch):
    """The publish path surfaces a directory-fsync refusal as the typed error and
    leaves no orphan copy behind."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "dir fsync fails")
        origin = Path(kanban_home) / "dur.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_fsync_dir = kea._fsync_dir

        def boom(_directory):
            raise kea.ExternalArtifactPreservationError("simulated dir fsync failure")

        monkeypatch.setattr(kea, "_fsync_dir", boom)
        caps = kea.capture_external_artifacts(conn, tid)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kea.publish_external_artifacts(caps, tid)
        monkeypatch.setattr(kea, "_fsync_dir", real_fsync_dir)
        assert not _leftovers(tid), f"dir fsync failure left orphans: {_leftovers(tid)}"


# ===========================================================================
# ROUND 3 (adversarial re-review 376bd9cffc, GPT-6.1 Sol) — regression guards
# for the 3 HIGH + 1 MEDIUM + 1 LOW defects closed by this round (2 of which
# were regressions the round-2 fix introduced).
# ===========================================================================


# ---------------------------------------------------------------------------
# HIGH #1 — ONE discard per attempt, ownership by device+inode, bind verifies
# ---------------------------------------------------------------------------


def test_discard_runs_once_on_refusal_path(kanban_home, monkeypatch):
    """Round-3 HIGH #1: a refusal that rolls the completion back must discard the
    published copies EXACTLY once — the round-2 ``except`` + ``finally`` pair ran
    the cleanup twice, letting the second pass delete a copy a concurrent
    attempt had since republished onto the freed name."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "single discard")
        origin = Path(kanban_home) / "one.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_gate = kb._gate_contract_completion
        swapped = {"done": False}

        def gate_then_swap(c, task_id):
            real_gate(c, task_id)
            if not swapped["done"]:
                swapped["done"] = True
                with kbc.connect() as c2:
                    c2.execute(
                        "UPDATE tasks SET completion_contract = 'local-only' "
                        "WHERE id = ?",
                        (task_id,),
                    )
                    c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap)

        calls = {"n": 0}
        real_discard = kea.discard_published_artifacts

        def counting_discard(*a, **k):
            calls["n"] += 1
            return real_discard(*a, **k)

        monkeypatch.setattr(kea, "discard_published_artifacts", counting_discard)
        with pytest.raises(kb.ContractSpecError):
            kb.complete_task(conn, tid, result="swap refusal")
        assert calls["n"] == 1, (
            f"published copies discarded {calls['n']} times on one refusal"
        )
        assert not _leftovers(tid), f"refusal left orphans: {_leftovers(tid)}"


def test_stale_discard_never_deletes_another_attempts_republished_copy(
    kanban_home,
):
    """Round-3 HIGH #1 (identity): attempt A publishes ``proof.md``, its rollback
    deletes it; attempt B republishes onto the now-free name. A stale/second
    discard of A's record must NOT delete B's file (its device+inode differs)."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "identity discard")
        origin = Path(kanban_home) / "proof.md"
        origin.write_bytes(b"attempt bytes")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        published_a = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        a_path = Path(published_a[0].published_path)
        assert a_path.name == "proof.md"
        a_identity = (
            published_a[0].published_ino,
            published_a[0].published_dev,
            published_a[0].published_ctime_ns,
        )

        # Attempt A's own rollback removes its copy (single discard).
        kea.discard_published_artifacts(published_a)
        assert not a_path.exists()

        # Attempt B publishes onto the freed name — a DIFFERENT file identity.
        published_b = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        b_path = Path(published_b[0].published_path)
        assert b_path == a_path, "fixture expects B to reuse the freed name"
        b_identity = (
            published_b[0].published_ino,
            published_b[0].published_dev,
            published_b[0].published_ctime_ns,
        )
        assert b_identity != a_identity, (
            "B's republished copy must carry a distinct ownership identity"
        )

        # A stale discard of A's record: identity guard must spare B's copy.
        kea.discard_published_artifacts(published_a)
        assert b_path.is_file(), "stale discard deleted another attempt's proof"
        assert b_path.read_bytes() == b"attempt bytes"


def test_bind_refuses_when_published_destination_disappeared(
    kanban_home, monkeypatch
):
    """Round-3 HIGH #1 (bind): if a concurrent attempt's stale cleanup removed the
    published copy before binding, ``bind_external_artifacts`` must REFUSE with
    the typed error — never bind a broken reference onto the closing run."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "bind gone")
        origin = Path(kanban_home) / "gone.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        published = kea.publish_external_artifacts(
            kea.capture_external_artifacts(conn, tid), tid
        )
        dest = Path(published[0].published_path)
        assert dest.is_file()
        dest.unlink()  # the destination vanished before the txn binds it

        with pytest.raises(kea.ExternalArtifactPreservationError):
            with kbc.write_txn(conn):
                kea.bind_external_artifacts(conn, tid, published, {}, 0)
        assert not _attachment_rows(conn, tid), "must not bind a broken reference"


def test_bind_missing_destination_refuses_completion_typed(kanban_home, monkeypatch):
    """End-to-end: a published copy that disappears before the closing txn makes
    ``complete_task`` refuse (typed) with the card NOT done and no dangling row."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "bind gone e2e")
        origin = Path(kanban_home) / "vanish2.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_bind = kea.bind_external_artifacts

        def bind_after_delete(c, task_id, published, metadata, now):
            for cap in published:
                if cap.published_path is not None:
                    with contextlib.suppress(OSError):
                        Path(cap.published_path).unlink()
            return real_bind(c, task_id, published, metadata, now)

        monkeypatch.setattr(kea, "bind_external_artifacts", bind_after_delete)

        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(conn, tid, result="bind after delete")
        assert _status(conn, tid) != "done"
        assert not _attachment_rows(conn, tid)
        assert "completed" not in _event_kinds(conn, tid)


# ---------------------------------------------------------------------------
# HIGH #2 — caller metadata cannot forge external_artifacts_preserved
# ---------------------------------------------------------------------------


def test_forged_external_artifacts_preserved_does_not_skip_scratch(kanban_home):
    """Round-3 HIGH #2: a caller-forged ``external_artifacts_preserved`` naming
    the resolved scratch requirement must NOT make staging skip it. The scratch
    artifact is still copied to managed storage, and the forged key is stripped
    from the caller metadata (never reaches the closing run)."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "forged metadata")
        scratch = ws / "req.md"
        scratch.write_text("scratch requirement", encoding="utf-8")
        _set_contract(conn, tid, {"required_artifacts": [str(scratch)]})

        forged = {
            "external_artifacts_preserved": [
                {"stored_path": str(scratch.resolve())}
            ]
        }
        assert kb.complete_task(conn, tid, result="forged", metadata=forged) is True

        rows = _attachment_rows(conn, tid)
        assert rows, "scratch requirement was NOT preserved (forged key skipped it)"
        copies = [Path(r["stored_path"]) for r in rows]
        assert any(
            c.is_file() and c.read_bytes() == b"scratch requirement" for c in copies
        ), f"scratch requirement not copied to managed storage: {copies}"

        run = kb.latest_run(conn, tid)
        bound = (run.metadata or {}).get("external_artifacts_preserved")
        assert not bound, f"forged key leaked into the closing run metadata: {bound}"


# ---------------------------------------------------------------------------
# HIGH #3 — A→B→A on a MIXED contract must not lose a scratch requirement
# ---------------------------------------------------------------------------


def test_mixed_contract_aba_swap_preserves_gate_scratch_requirement(
    kanban_home, monkeypatch
):
    """Round-3 HIGH #3: gate enforces A (external ``a`` + scratch ``s``); a
    concurrent writer swaps to B (scratch ``t``) AFTER the gate and back to A
    only during capture (before the txn). The scratch requirement of the GATE
    contract (``s``) must still be staged — the merge must ride the gate
    snapshot, not an autonomous re-read — so the card never closes losing ``s``
    to the post-commit cleanup."""
    with kbc.connect() as conn:
        tid, ws = _scratch_task(conn, kanban_home, "mixed A->B->A")
        a_ext = Path(kanban_home) / "a.md"
        a_ext.write_bytes(b"external A evidence")
        s = ws / "s.md"
        s.write_text("scratch S", encoding="utf-8")
        t = ws / "t.md"
        t.write_text("scratch T", encoding="utf-8")
        contract_a = json.dumps(
            {"required_artifacts": [str(a_ext), str(s)]}, separators=(",", ":")
        )
        contract_b = json.dumps(
            {"required_artifacts": [str(t)]}, separators=(",", ":")
        )
        # Contract A is in force when the gate runs.
        _set_contract(conn, tid, contract_a)

        real_gate = kb._gate_contract_completion

        def gate_then_swap_to_b(c, task_id):
            real_gate(c, task_id)
            # B is now in force for the merge window (A is only the gate snapshot).
            with kbc.connect() as c2:
                c2.execute(
                    "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                    (contract_b, task_id),
                )
                c2.commit()

        monkeypatch.setattr(kb, "_gate_contract_completion", gate_then_swap_to_b)

        real_capture = kea.capture_external_artifacts

        def capture_then_restore_a(c, task_id, **kw):
            caps = real_capture(c, task_id, **kw)
            # Restore A before the txn: the fence validates A, so A's scratch
            # requirement must be the one preserved.
            with kbc.connect() as c2:
                c2.execute(
                    "UPDATE tasks SET completion_contract = ? WHERE id = ?",
                    (contract_a, task_id),
                )
                c2.commit()
            return caps

        monkeypatch.setattr(kea, "capture_external_artifacts", capture_then_restore_a)

        result = None
        try:
            result = kb.complete_task(conn, tid, result="mixed A->B->A")
        except kb.ContractSpecError:
            result = None  # an audible refusal is also acceptable

        # In-force contract is A (external a + scratch s); the gate accepted it,
        # so the completion MUST close — never silently done with a lost `s`.
        assert result is True, "the in-force contract A was satisfied; must close"
        # Post-commit cleanup deletes the scratch workspace: `s` is gone on disk.
        assert not s.exists(), "fixture expects workspace cleanup to remove s"
        atts = [Path(r["stored_path"]) for r in _attachment_rows(conn, tid)]
        assert any(
            p.is_file() and p.read_bytes() == b"scratch S" for p in atts
        ), f"gate scratch requirement s was lost to cleanup: {atts}"
        for p in atts:
            assert p.is_file()
        bound = (kb.latest_run(conn, tid).metadata or {}).get(
            "external_artifacts_preserved"
        )
        assert bound, "the external requirement of the in-force contract"
        for x in bound:
            assert Path(x["stored_path"]).is_file()


# ---------------------------------------------------------------------------
# MEDIUM #4 — _ensure_dir_durable re-syncs the parent on a retry
# ---------------------------------------------------------------------------


def test_ensure_dir_durable_resyncs_parent_on_retry(kanban_home, monkeypatch):
    """Round-3 MEDIUM #4: the first attempt creates the tree but a parent fsync
    then fails (typed refusal leaves the tree in place). On the retry
    ``to_create`` is empty; the directory<->parent link that never got persisted
    must STILL be re-synced before a COMMIT may claim durability."""
    base = kanban_home / "att-root"
    base.mkdir()
    target = base / "sub" / "task"

    synced: list[Path] = []
    fail_once = {"done": False}

    def recording_fsync_dir(directory):
        synced.append(Path(directory))
        if not fail_once["done"] and Path(directory) == target.parent:
            fail_once["done"] = True
            raise kea.ExternalArtifactPreservationError(
                "simulated parent fsync failure"
            )

    monkeypatch.setattr(kea, "_fsync_dir", recording_fsync_dir)

    with pytest.raises(kea.ExternalArtifactPreservationError):
        kea._ensure_dir_durable(target)
    assert target.is_dir(), "mkdir must have created the tree before the fsync failed"

    synced.clear()
    kea._ensure_dir_durable(target)  # retry: to_create is now empty
    assert Path(target.parent) in synced, (
        "retry must re-sync the directory's link to its parent, not only itself"
    )
    assert Path(target) in synced


# ---------------------------------------------------------------------------
# LOW #5 — publication OSErrors are typed
# ---------------------------------------------------------------------------


def test_publish_staging_open_oserror_is_typed(kanban_home, monkeypatch):
    """Round-3 LOW #5: a raw OSError from the staging ``os.open`` is wrapped in
    the typed, recoverable refusal (no bare OSError escapes publication)."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "staging open oserror")
        origin = Path(kanban_home) / "so.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        att_dir = _att_dir(tid)
        real_open = os.open

        def flaky_open(path, flags, *a, **k):
            p = str(path)
            if p.startswith(str(att_dir)) and ".p3b-staging-" in p:
                raise OSError("EIO: simulated staging open failure")
            return real_open(path, flags, *a, **k)

        monkeypatch.setattr(os, "open", flaky_open)
        caps = kea.capture_external_artifacts(conn, tid)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kea.publish_external_artifacts(caps, tid)
        monkeypatch.setattr(os, "open", real_open)
        assert not _leftovers(tid), f"staging failure left orphans: {_leftovers(tid)}"


def test_publish_staging_write_oserror_is_typed(kanban_home, monkeypatch):
    """Round-3 LOW #5: a raw OSError from the staging write/flush/fsync path is
    wrapped in the typed refusal."""
    with kbc.connect() as conn:
        tid, _ws = _scratch_task(conn, kanban_home, "staging write oserror")
        origin = Path(kanban_home) / "sw.bin"
        origin.write_bytes(b"proof")
        _set_contract(conn, tid, {"required_artifacts": [str(origin)]})

        real_fsync = os.fsync

        def flaky_fsync(fd):
            # Fail ONLY on a regular-file fd (the staging file's fsync) — a
            # directory fsync already routes through the typed ``_fsync_dir``.
            st = os.fstat(fd)
            if stat.S_ISREG(st.st_mode):
                raise OSError("EIO: simulated staging fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(os, "fsync", flaky_fsync)
        caps = kea.capture_external_artifacts(conn, tid)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kea.publish_external_artifacts(caps, tid)
        monkeypatch.setattr(os, "fsync", real_fsync)
        assert not _leftovers(tid), f"staging failure left orphans: {_leftovers(tid)}"


def test_mkdir_oserror_is_typed(kanban_home, monkeypatch):
    """Round-3 LOW #5: a raw OSError from ``mkdir`` in ``_ensure_dir_durable`` is
    the typed refusal, not a bare OSError."""
    target = kanban_home / "cannot" / "create"
    real_mkdir = Path.mkdir

    def flaky_mkdir(self, *a, **k):
        if str(self) == str(target):
            raise OSError("EACCES: simulated mkdir failure")
        return real_mkdir(self, *a, **k)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    with pytest.raises(kea.ExternalArtifactPreservationError):
        kea._ensure_dir_durable(target)
