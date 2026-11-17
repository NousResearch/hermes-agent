"""A Docker worker's ``/workspace/...`` artifact is found, checked and preserved on the host (REA-1329).

The worker context names the path its sandbox mounts the scratch workspace at, so that is the path
it declares in ``kanban_complete(artifacts=[...])``. It is not a host path: before this mapping the
artifact was kept as a bare string, the scratch workspace was cleaned up, and the file was lost.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _complete_with(conn, declared, files):
    t = kb.create_task(conn, title="write artifact")
    task = kb.get_task(conn, t)
    ws = kbw.resolve_workspace(task)
    kbw.set_workspace_path(conn, t, ws)
    for rel, data in files.items():
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_bytes(data)
    assert kb.complete_task(conn, t, result="ok", metadata={"artifacts": declared})
    return t, ws


@pytest.mark.parametrize("declared, rel", [
    ("/workspace/verification.txt", "verification.txt"),
    ("/workspace/out/report.txt", "out/report.txt"),
    ("/host-cwd/verification.txt", "verification.txt"),
])
def test_sandbox_artifact_is_preserved_before_scratch_cleanup(kanban_home, declared, rel):
    with kbc.connect() as conn:
        t, ws = _complete_with(conn, [declared], {rel: b"SLACK_E2E_PASS"})
        attachments = kb.list_attachments(conn, t)
    assert not ws.exists(), "scratch workspace is still cleaned up"
    assert [a.filename for a in attachments] == [Path(rel).name]
    assert Path(attachments[0].stored_path).read_bytes() == b"SLACK_E2E_PASS"


def test_missing_sandbox_artifact_keeps_the_task_in_flight_like_a_host_path(kanban_home):
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="write artifact")
        task = kb.get_task(conn, t)
        kbw.set_workspace_path(conn, t, kbw.resolve_workspace(task))
        with pytest.raises(kb.ArtifactPreservationError):
            kb.complete_task(conn, t, result="ok", metadata={"artifacts": ["/workspace/never-written.txt"]})


def test_sandbox_path_cannot_reach_outside_the_task_workspace(kanban_home):
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="write artifact")
        task = kb.get_task(conn, t)
        ws = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, t, ws)
        outside = ws.parent / "outside.txt"
        outside.write_bytes(b"secret")
        assert kb.complete_task(conn, t, result="ok", metadata={"artifacts": ["/workspace/../outside.txt"]})
        assert kb.list_attachments(conn, t) == []
    assert outside.read_bytes() == b"secret"


def test_host_artifact_path_leaves_other_paths_alone(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    assert kbw._host_artifact_path(str(root / "a.txt"), root) == root / "a.txt"
    assert kbw._host_artifact_path("/workspacefoo/a.txt", root) == Path("/workspacefoo/a.txt")
    assert kbw._host_artifact_path("/workspace/a.txt", root) == root / "a.txt"
    assert kbw._host_artifact_path("/workspace", root) == root
