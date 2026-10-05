"""F5-FINAL-01 / F5-FINAL-02 regression tests — final review (GPT-6.1 Sol).

F5-FINAL-01 (HIGH, introduced by 95ddde0cc6): the R5-08 finally discarded
staged scratch copies on ANY non-commit exit — but a post-COMMIT invariant
failure (``_check_file_length_invariant`` runs AFTER the COMMIT inside the
context manager) also reaches it with ``committed`` still False, while the
transaction HAS committed and a card is already ``done`` pointing at the
blob. The finally unlinked a durable proof. Fix: the scratch discard probe is
the same committed-row read the external lane uses (fail-closed), so a bound
scratch proof survives and an orphan is still cleaned.

F5-FINAL-02 (MEDIUM): the R5-05/06 ownership classifier declared a
requirement "own scratch" whenever it resolved under the task's
``workspace_path`` — WITHOUT checking that the workspace itself is MANAGED
scratch. Cards with ``workspace_kind='dir'`` (or any unmanaged path) hit the
staging's managed-guard empty set AND the classifier's own-scratch verdict:
the requirement fell into BOTH-empty limbo (task completes, zero preserved
copies) where BASE preserved it externally. Fix: own only when the workspace
passes the same managed guard the staging uses; otherwise external.

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixture conventions).
Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_external_artifacts as kea


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
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]


def _attachments(conn, task_id):
    return kb.list_attachments(conn, task_id)


class TestF5FINAL01PostCommitInvariantKeepsBoundScratchBlob:
    def test_post_commit_invariant_failure_keeps_committed_scratch_blob(
        self, conn, kanban_home, monkeypatch,
    ):
        """The final review's HIGH probe: a scratch requirement staged and its
        attachment row COMMITTED, then the post-commit invariant check raises.
        The done card must keep its proof — the finally may clean ONLY
        unbound copies."""
        tid, ws = None, None
        t = kb.create_task(conn, title="f5f1", workspace_kind="scratch")
        tid = t
        ws = Path(kbw.resolve_workspace(kb.get_task(conn, tid)))
        kbw.set_workspace_path(conn, tid, ws)
        mine = ws / "own.txt"
        mine.write_text("own artifact")
        contract = json.dumps({"required_artifacts": [str(mine)]})
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?", (contract, tid),
        )
        conn.commit()
        # Fail the post-COMMIT invariant: the txn commits, then the context
        # manager's boundary check raises — the exact torn-extend seam.
        real_check = kbc._check_file_length_invariant
        def boom(conn2):
            real_check(conn2)  # keep the DB honest
            raise RuntimeError("injected post-commit invariant failure")
        monkeypatch.setattr(kbc, "_check_file_length_invariant", boom)
        with pytest.raises(RuntimeError):
            kb.complete_task(conn, tid, summary="evidence")
        assert _status(conn, tid) == "done"
        atts = _attachments(conn, tid)
        assert len(atts) == 1
        stored = Path(atts[0].stored_path)
        assert stored.exists(), (
            "the committed scratch proof was erased by the R5-08 finally"
        )
        assert stored.read_text() == "own artifact"


class TestF5FINAL02UnmanagedWorkspaceRequirementsGoExternal:
    def test_dir_kind_workspace_requirement_is_preserved(self, conn, kanban_home):
        """A ``workspace_kind='dir'`` card (unmanaged by design — completion
        preserves the dir) whose contract requires one of its files: the
        requirement must ride the EXTERNAL lane and produce a durable copy,
        not vanish in the staging's managed-guard empty set."""
        tdir = Path(kanban_home) / "proj-dir"
        tdir.mkdir()
        mine = tdir / "deliverable.txt"
        mine.write_text("dir artifact")
        contract = json.dumps({"required_artifacts": [str(mine)]})
        tid = kb.create_task(
            conn, title="dir card", workspace_kind="dir",
            workspace_path=str(tdir), completion_contract=contract,
        )
        assert kb.complete_task(conn, tid, summary="evidence") is True
        atts = _attachments(conn, tid)
        assert len(atts) == 1, "unmanaged-workspace requirement must be preserved"
        assert Path(atts[0].stored_path).read_text() == "dir artifact"
        # The dir itself is preserved (not removed) — completion never rmtree'd
        # unmanaged storage.
        assert mine.exists()

    def test_scratch_kind_with_physical_ws_spelling_preserves(self, conn, kanban_home):
        """F5-FINAL-02's exact regression: kind=scratch but the stored
        workspace_path is the PHYSICAL (resolved) spelling of a symlinked
        managed root. The staging guard fails lexically (empty set) while the
        old classifier claimed own-scratch by resolved containment — both
        lanes empty, zero attachments, probe completed. With the fix, the
        workspace's own manageability is judged like the staging judges it:
        unmanaged lexically → external lane preserves."""
        root = kb.workspaces_root()
        real = Path(kanban_home) / "relocated-f5f2"
        real.mkdir(parents=True, exist_ok=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        if root.is_symlink():
            root.unlink()
        root.symlink_to(real, target_is_directory=True)
        ws = root / "named-phys"
        ws.mkdir()
        mine = ws.resolve() / "deliverable.txt"
        mine.write_text("physical artifact")
        contract = json.dumps({"required_artifacts": [str(mine)]})
        tid = kb.create_task(
            conn, title="physical ws spelling", workspace_kind="scratch",
            workspace_path=str(ws.resolve()),  # the PHYSICAL spelling stored
            completion_contract=contract,
        )
        assert kb.complete_task(conn, tid, summary="evidence") is True
        atts = _attachments(conn, tid)
        assert len(atts) == 1, (
            "physical-spelling workspace requirement must be preserved exactly once"
        )
        assert Path(atts[0].stored_path).read_text() == "physical artifact"

    def test_plain_kind_card_with_external_requirement_unaffected(self, conn, kanban_home):
        """No regression: a card with NO workspace keeps the base behaviour
        (external requirement, one durable copy)."""
        origin = Path(kanban_home) / "plain.txt"
        origin.write_bytes(b"plain evidence")
        contract = json.dumps({"required_artifacts": [str(origin)]})
        tid = kb.create_task(conn, title="plain card", completion_contract=contract)
        assert kb.complete_task(conn, tid, summary="evidence") is True
        atts = _attachments(conn, tid)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).read_bytes() == b"plain evidence"

    def test_managed_scratch_requirement_still_own_scratch(self, conn, kanban_home):
        """Guard the partition: a genuine scratch card keeps the in-txn
        staging lane (exactly one copy, via staging)."""
        t = kb.create_task(conn, title="scratch card", workspace_kind="scratch")
        ws = Path(kbw.resolve_workspace(kb.get_task(conn, t)))
        kbw.set_workspace_path(conn, t, ws)
        mine = ws / "own.txt"
        mine.write_text("scratch artifact")
        contract = json.dumps({"required_artifacts": [str(mine)]})
        conn.execute(
            "UPDATE tasks SET completion_contract = ? WHERE id = ?", (contract, t),
        )
        conn.commit()
        assert kb.complete_task(conn, t, summary="evidence") is True
        atts = _attachments(conn, t)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).read_text() == "scratch artifact"
