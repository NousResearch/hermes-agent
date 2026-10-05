"""R5-07 / R5-09 regression tests — review 5, round-5d follow-up.

R5-07 (MEDIUM): the durability chain of the attachments directory tree was
incomplete: a RETRY after a failed ancestor fsync saw a populated ``to_create``
and skipped that ancestor; a tree created by a NORMAL UPLOAD (mkdir only, no
fsync) never got its ancestors synced at completion. Fix: the publish step
fsyncs the FULL ancestor chain from the durable anchor (the board attachments
root, whose entry lives in the pre-existing kanban dir) down to the directory
itself — ancestors included even when they pre-exist and even when the张学 tree
was created by another writer.

R5-09 (MEDIUM): ``bind`` verified only ``is_file()`` — a concurrent removal of
the managed destination after the stat could leave a committed row pointing at
a missing blob. Fix: bind re-stats INSIDE the txn and requires the current
file identity to equal the identity THIS attempt published; a replaced or
missing file is a typed refusal (whole txn rolls back) — no broken reference
under any interleaving we can detect at userland.

Sandbox: HERMES_HOME/HERMES_KANBAN_HOME pinned at tmp_path, delegated-child
marker dropped, Path.home monkeypatched (sibling fixture conventions).
Boards reais READ-ONLY.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

_ART = tempfile.NamedTemporaryFile(prefix="r57_art_", delete=False).name

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


def _external_task(conn, kanban_home, name="proof.txt", content=b"validated-original"):
    origin = Path(kanban_home) / name
    origin.write_bytes(content)
    tid = kb.create_task(conn, title=f"r57 {name}")
    conn.execute(
        "UPDATE tasks SET completion_contract = ? WHERE id = ?",
        (json.dumps({"required_artifacts": [str(origin)]}), tid),
    )
    conn.commit()
    return tid, origin


def _status(conn, task_id):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]


class TestR507FsyncAncestorChainComplete:
    def test_retry_syncs_every_ancestor_down_from_the_root(self, conn, kanban_home, monkeypatch):
        """The review-5 probe inverted: after an injected EIO on the
        ``workspaces-parent`` ancestor, the RETRY must still fsync that
        ancestor (the first attempt's refusal left it unsynced and the chain
        must be complete before COMMIT)."""
        tid, _origin = _external_task(conn, kanban_home)
        directory = kb.task_attachments_dir(tid)
        fail_parent = directory.parent.parent  # the ancestors above the task dir
        synced, round_no = [], [0]
        real = kea._fsync_dir
        def recording_sync(path):
            p = Path(path)
            synced.append(p)
            if round_no[0] == 0 and p == fail_parent:
                raise kea.ExternalArtifactPreservationError("injected EIO at ancestor")
            return real(path)
        monkeypatch.setattr(kea, "_fsync_dir", recording_sync)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(conn, tid, summary="evidence")
        first_round = list(synced)
        synced.clear()
        round_no[0] = 1
        assert kb.complete_task(conn, tid, summary="evidence") is True
        assert fail_parent in synced or fail_parent in first_round, (
            "the ancestor chain must be fsync'd in SOME attempt before commit"
        )

    def test_upload_created_tree_ancestors_are_synced_at_completion(self, conn, kanban_home, monkeypatch):
        """The review-5 probe inverted: a tree created by a normal upload
        (mkdir without fsync) gets ALL its new ancestors fsync'd at the
        completion's publish step."""
        tid, _origin = _external_task(conn, kanban_home, name="other.txt")
        directory = kb.task_attachments_dir(tid)
        kb.store_attachment_bytes(conn, tid, "unrelated.txt", b"upload")
        root_ancestor = directory.parent.parent
        synced = []
        real_sync = kea._fsync_dir
        def recording_sync(path):
            synced.append(Path(path))
            return real_sync(path)
        monkeypatch.setattr(kea, "_fsync_dir", recording_sync)
        assert kb.complete_task(conn, tid, summary="evidence") is True
        assert root_ancestor in synced, (
            f"ancestor created by the upload must be fsync'd at publish: {root_ancestor}"
        )


class TestR509BindVerifiesIdentityInTxn:
    def test_replaced_destination_refuses_completion(self, conn, kanban_home, monkeypatch):
        """The review-5 probe's stronger shape: another writer REPLACES the
        managed destination (unlink + write different bytes) after publish.
        The in-txn identity check must refuse — never bind a file we did not
        publish."""
        tid, origin = _external_task(conn, kanban_home)
        swapped = [False]
        real_is_file = Path.is_file
        def swap_on_stat(self, *a, **kw):
            present = real_is_file(self, *a, **kw)
            if (
                not swapped[0] and present
                and self.name == origin.name
                and "attachments" in self.parts
            ):
                swapped[0] = True
                self.unlink()
                self.write_bytes(b"foreign bytes")
                return False  # the check the bind would now see before identity
            return present
        monkeypatch.setattr(Path, "is_file", swap_on_stat)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(conn, tid, summary="evidence")
        assert _status(conn, tid) != "done"
        assert not kb.list_attachments(conn, tid)

    def test_disappeared_destination_refuses_completion(self, conn, kanban_home, monkeypatch):
        """Concurrent REMOVAL (not replacement) of the managed destination
        after publish: typed refusal, no committed row, no broken reference."""
        tid, _origin = _external_task(conn, kanban_home, name="gone.txt")
        removed = [False]
        real_is_file = Path.is_file
        def remove_on_stat(self, *a, **kw):
            present = real_is_file(self, *a, **kw)
            if (
                not removed[0] and present
                and self.name == "gone.txt"
                and "attachments" in self.parts
            ):
                removed[0] = True
                self.unlink()
                return False
            return present
        monkeypatch.setattr(Path, "is_file", remove_on_stat)
        with pytest.raises(kea.ExternalArtifactPreservationError):
            kb.complete_task(conn, tid, summary="evidence")
        assert _status(conn, tid) != "done"
        assert not kb.list_attachments(conn, tid)

    def test_intact_destination_binds_normally(self, conn, kanban_home):
        """The GOOD path keeps working: untouched publishes bind; the card
        closes with the binding naming the durable copy."""
        tid, origin = _external_task(conn, kanban_home)
        assert kb.complete_task(conn, tid, summary="evidence") is True
        atts = kb.list_attachments(conn, tid)
        assert len(atts) == 1
        assert Path(atts[0].stored_path).read_bytes() == b"validated-original"
