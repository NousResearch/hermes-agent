"""R5-05 / R5-06 / R5-08 regression tests — review 5, round-5c follow-up.

R5-05 (MEDIUM): a contract requirement pointing INSIDE ANOTHER task's managed
scratch fell into a preservation gap: it is not ``external`` (a managed-scratch
path), and not the CURRENT task's scratch (the staging only covers its own
workspace root). The consumer completed with zero preserved copies; the
producer's later completion removed the origin — the consumer's ``done`` card
lost its declared evidence. Fix: partition ONCE by ownership — a requirement
inside ANY managed scratch that is NOT this task's own workspace is preserved
via the EXTERNAL capture lane (a cross-task scratch requirement is exactly a
foreign durable copy: nobody else guarantees it survives).

R5-06 (MEDIUM): a requirement declared via the RESOLVED (physical) spelling of
a symlinked managed workspace was simultaneously classified external
(lexical check fails for the physical form) and scratch (resolved containment
succeeds) — a VALID contract was refused (externally captured, then the
scratch coverage demanded an extra copy that the merge never staged). Fix:
one authoritative classifier used by capture, merge, staging and coverage —
a path that resolves inside this task's workspace root is scratch, period;
capture uses the same verdict, so the two sides can never double-claim.

R5-08 (LOW): on the scratch-coverage refusal path, the already-staged copies
were leaked (no rollback cleanup, no rows) — retries then created
``name_1.ext`` orphans. Fix: staged scratch copies are discarded on every
non-commit exit via the completion's uniform ``finally`` (identity-guarded
like the external copies).

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixture conventions).
Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


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


def _scratch_task(conn, title: str, contract=None):
    t = kb.create_task(conn, title=title, workspace_kind="scratch",
                       completion_contract=contract)
    ws = kbw.resolve_workspace(kb.get_task(conn, t))
    kbw.set_workspace_path(conn, t, ws)
    return t, Path(ws)


def _status(conn, task_id):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]


def _attachments(conn, task_id):
    return kb.list_attachments(conn, task_id)


class TestR505OtherTaskScratchRequirementPreserved:
    def test_consumer_preserves_producers_scratch_requirement(self, conn):
        """The review-5 probe's happy path, inverted: a requirement inside
        ANOTHER task's managed scratch is preserved for the consumer via the
        external lane; the producer's later completion then deletes its
        workspace and the CONSUMER's proof survives."""
        producer, pws = _scratch_task(conn, "producer")
        required = pws / "result.txt"
        required.write_text("required evidence")
        contract = json.dumps({"required_artifacts": [str(required)]})
        consumer, _cws = _scratch_task(conn, "consumer", contract=contract)

        assert kb.complete_task(conn, consumer, summary="evidence") is True, (
            "a cross-task scratch requirement must be PRESERVED, not refused"
        )
        atts = _attachments(conn, consumer)
        assert len(atts) == 1, "the requirement must have exactly one durable copy"
        copy = Path(atts[0].stored_path)
        assert copy.read_text() == "required evidence"

        # Producer completes later; its workspace is cleaned up.
        assert kb.complete_task(conn, producer, summary="done") is True
        # The consumer's proof must survive the producer's cleanup.
        assert copy.exists(), "preserved copy deleted by the producer's completion"
        kinds = [e.kind for e in kb.list_events(conn, consumer)]
        assert "external_artifact_preserved" in kinds

    def test_own_scratch_requirement_is_not_double_captured(self, conn):
        """A requirement inside the task's OWN workspace stays scratch-owned:
        exactly one preserved copy (the staged attachment), no external
        duplicate row."""
        tid, ws = _scratch_task(conn, "own")
        mine = ws / "own.txt"
        mine.write_text("own artifact")
        contract = json.dumps({"required_artifacts": [str(mine)]})
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?",
            (contract, tid),
        )
        conn.commit()
        assert kb.complete_task(conn, tid, summary="evidence") is True
        atts = _attachments(conn, tid)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).read_text() == "own artifact"


class TestR506SymlinkedWorkspaceSingleClassification:
    def test_resolved_spelling_completes_without_refusal(self, conn):
        """The review-5 probe inverted: requirement declared via the PHYSICAL
        spelling of a symlinked managed workspace root. One classifier rules:
        it is scratch (resolved containment) — capture does not also claim it,
        the staging covers it via the corresponding declarable spelling, and
        the completion SUCCEEDS instead of refusing."""
        root = kb.workspaces_root()
        real = Path(kanban_home_home(conn)) / "relocated"
        real.mkdir(parents=True, exist_ok=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            root.unlink()
        root.symlink_to(real, target_is_directory=True)
        ws = root / "named"
        ws.mkdir()
        origin_physical = ws.resolve() / "required.txt"
        origin_physical.write_text("required content")
        contract = json.dumps({"required_artifacts": [str(origin_physical)]})
        tid = kb.create_task(
            conn, title="physical spelling", workspace_kind="scratch",
            workspace_path=str(ws), completion_contract=contract,
        )
        assert kb.complete_task(conn, tid, summary="evidence") is True, (
            "a valid contract spelled via the symlink target must complete"
        )
        atts = _attachments(conn, tid)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).read_text() == "required content"

    def test_lexical_spelling_through_symlinked_root_still_works(self, conn):
        """Backwards compat (round-4 MEDIUM): the LEXICAL spelling through the
        symlinked root must keep completing too."""
        root = kb.workspaces_root()
        real = Path(kanban_home_home(conn)) / "relocated2"
        real.mkdir(parents=True, exist_ok=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            root.unlink()
        root.symlink_to(real, target_is_directory=True)
        ws = root / "named2"
        ws.mkdir()
        origin_lexical = ws / "required.txt"
        origin_lexical.write_text("lexical content")
        contract = json.dumps({"required_artifacts": [str(origin_lexical)]})
        tid = kb.create_task(
            conn, title="lexical spelling", workspace_kind="scratch",
            workspace_path=str(ws), completion_contract=contract,
        )
        assert kb.complete_task(conn, tid, summary="evidence") is True
        atts = _attachments(conn, tid)
        assert len(atts) == 1


class TestR508ScratchRefusalDiscardsStagedCopies:
    def test_coverage_refusal_leaves_no_orphan_blob(self, conn, monkeypatch):
        """A refusal AFTER scratch staging must discard the staged copies (the
        review-5 probe left ``b.txt`` AND ``b_1.txt`` orphans; retry then used
        new names)."""
        tid, ws = _scratch_task(conn, "leak")
        first = ws / "a.txt"
        first.write_text("A")
        second = ws / "b.txt"
        second.write_text("B")
        contract = json.dumps({"required_artifacts": [str(first), str(second)]})
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?", (contract, tid),
        )
        conn.commit()
        # Deterministic refusal seam: fail the LAST staging row insert (after
        # both copies are on disk), so the rollback runs with staged copies
        # present and the uniform finally must clean them.
        real_insert = kb._insert_completion_attachment
        def failing_insert(conn2, task_id, *, filename, stored_path, size, created_at, uploaded_by):
            if Path(filename).name == "b.txt":
                raise RuntimeError("injected staging row failure")
            return real_insert(conn2, task_id, filename=filename, stored_path=stored_path,
                               size=size, created_at=created_at, uploaded_by=uploaded_by)
        monkeypatch.setattr(kb, "_insert_completion_attachment", failing_insert)
        with pytest.raises(RuntimeError):
            kb.complete_task(conn, tid, summary="evidence")
        att_dir = kb.task_attachments_dir(tid, board=None)
        leftovers = sorted(p.name for p in att_dir.iterdir()) if att_dir.is_dir() else []
        assert leftovers == [], f"staged copies leaked on refusal: {leftovers}"
        assert _status(conn, tid) == "ready"  # rolled back, retryable


def kanban_home_home(conn):
    """The kanban home this connection is bound to (for fixture-local roots)."""
    import os
    return Path(os.environ["HERMES_KANBAN_HOME"])
