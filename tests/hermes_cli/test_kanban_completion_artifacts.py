"""Completion artifact confinement contracts ported from PR #86610."""
import os
import sqlite3
from pathlib import Path
import pytest
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_db_artifacts as kba

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    kb.init_db()
    return home

@pytest.mark.parametrize("memory", [False, True])
def test_unknown_connection_board_never_uses_ambient_board(
    kanban_home, tmp_path, monkeypatch, memory,
):
    monkeypatch.setattr(kb, "get_current_board", lambda: "unrelated-board")
    conn = sqlite3.connect(":memory:" if memory else tmp_path / "unknown.db")
    conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(kb.ArtifactPreservationError, match="board ownership"):
            kba._connection_board(conn)
    finally:
        conn.close()

def test_oversized_scratch_prose_preserves_workspace_on_refusal(kanban_home):
    """A failed preservation must not report done and clean away the output."""
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="oversized report")
        workspace = kbw.resolve_workspace(kb.get_task(conn, task_id))
        kbw.set_workspace_path(conn, task_id, workspace)
        artifact = workspace / "large-report.bin"
        with artifact.open("wb") as stream:
            stream.truncate(kb.KANBAN_ATTACHMENT_MAX_BYTES + 1)
        with pytest.raises(kb.ArtifactPreservationError, match="exceeds"):
            kb.complete_task(conn, task_id, summary=f"Produced {artifact}", fire_lifecycle_hook=False)
        assert kb.get_task(conn, task_id).status == "ready"
        assert not kb.list_attachments(conn, task_id)
        assert not any(event.kind == "completed" for event in kb.list_events(conn, task_id))
        assert artifact.stat().st_size == kb.KANBAN_ATTACHMENT_MAX_BYTES + 1
        assert not list(kb.task_attachments_dir(task_id).glob("*"))


def test_complete_task_stages_dir_workspace_declared_artifact(
    kanban_home,
    tmp_path,
):
    """Persistent workspaces also stage a durable path outside worker control."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    artifact.write_bytes(b"expected report")

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="persistent report",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary=f"produced {artifact}",
            metadata={"artifacts": [str(artifact)]},
        )
        completed = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "completed"
        ][-1]
        persisted = Path(completed.payload["artifacts"][0])
        run = kb.latest_run(conn, task_id)
        attachments = kb.list_attachments(conn, task_id)

    assert artifact.exists(), "persistent workspace content must remain in place"
    assert persisted.parent == kb.task_attachments_dir(task_id)
    assert persisted != artifact
    assert persisted.read_bytes() == b"expected report"
    assert run is not None
    assert run.metadata["artifacts"] == [str(persisted)]
    assert [(item.filename, item.stored_path) for item in attachments] == [
        ("report.pdf", str(persisted.resolve()))
    ]

def test_complete_task_stages_dir_artifact_on_connection_board(
    kanban_home,
    tmp_path,
):
    """An explicit board connection owns staging even when current is default."""
    kb.create_board("other")
    workspace = tmp_path / "other-project"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    artifact.write_bytes(b"other board")

    with kbc.connect(board="other") as conn:
        task_id = kb.create_task(
            conn,
            title="other-board report",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        assert kb.complete_task(
            conn,
            task_id,
            summary="done",
            metadata={"artifacts": [str(artifact)]},
        )
        completed = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "completed"
        ][-1]
        persisted = Path(completed.payload["artifacts"][0])

    assert persisted.parent == kb.task_attachments_dir(task_id, board="other")
    assert not kb.task_attachments_dir(task_id, board="default").exists()

def test_complete_task_does_not_clobber_existing_durable_artifact(
    kanban_home,
    tmp_path,
):
    """Descriptor-relative collision handling must remain append-only."""
    workspace = tmp_path / "collision-project"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    artifact.write_bytes(b"new report")

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="collision-safe staging",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        destination = kb.task_attachments_dir(task_id)
        destination.mkdir(parents=True)
        existing = destination / artifact.name
        existing.write_bytes(b"existing report")

        assert kb.complete_task(
            conn,
            task_id,
            result="done",
            metadata={"artifacts": [str(artifact)]},
        )
        completed = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "completed"
        ][-1]
        persisted = Path(completed.payload["artifacts"][0])

    assert existing.read_bytes() == b"existing report"
    assert persisted.name == "report_1.pdf"
    assert persisted.read_bytes() == b"new report"

@pytest.mark.parametrize("board", ["default", "other"])
@pytest.mark.parametrize("force_path_fallback", [False, True])
@pytest.mark.parametrize("symlink_level", ["attachments-root", "task-dir"])
def test_complete_task_rejects_symlinked_destination_directory(
    kanban_home,
    tmp_path,
    monkeypatch,
    board,
    force_path_fallback,
    symlink_level,
):
    """A pre-planted task attachment symlink must not redirect durable output."""
    if force_path_fallback:
        monkeypatch.setattr(kba, "_attachment_dir_fd_supported", lambda: False)
    if board != "default":
        kb.create_board(board)

    workspace = tmp_path / f"{board}-project"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    artifact.write_bytes(b"expected report")

    with kbc.connect(board=board) as conn:
        task_id = kb.create_task(
            conn,
            title=f"{board} destination confinement",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        destination = kb.task_attachments_dir(task_id, board=board)
        outside = tmp_path / f"{board}-outside"
        outside.mkdir()
        if symlink_level == "attachments-root":
            symlink = destination.parent
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            symlink = destination
        symlink.parent.mkdir(parents=True, exist_ok=True)
        try:
            symlink.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlinks unavailable: {exc}")

        with pytest.raises(kb.ArtifactPreservationError):
            kb.complete_task(
                conn,
                task_id,
                result="done",
                metadata={"artifacts": [str(artifact)]},
            )

        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.list_attachments(conn, task_id) == []
        assert all(event.kind != "completed" for event in kb.list_events(conn, task_id))

    assert list(outside.iterdir()) == []
    assert artifact.read_bytes() == b"expected report"

@pytest.mark.parametrize("via_workspace_symlink", [False, True])
def test_complete_task_rejects_artifacts_outside_scratch_workspace(
    kanban_home,
    tmp_path,
    via_workspace_symlink,
):
    """A completion artifact cannot turn the notifier into a host-file reader."""
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("host secret\n", encoding="utf-8")

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="bounded artifact")
        task = kb.get_task(conn, task_id)
        workspace = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, task_id, workspace)
        declared = outside
        if via_workspace_symlink:
            declared = workspace / "artifact.txt"
            declared.symlink_to(outside)

        with pytest.raises(
            kb.ArtifactPreservationError,
            match="outside its task workspace",
        ):
            kb.complete_task(
                conn,
                task_id,
                result="done",
                metadata={"artifacts": [str(declared)]},
            )

        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.list_attachments(conn, task_id) == []
        assert all(event.kind != "completed" for event in kb.list_events(conn, task_id))

    assert workspace.exists(), "rejected completion must preserve the workspace"
    assert outside.read_text(encoding="utf-8") == "host secret\n"

@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"),
    reason="platform has no O_NOFOLLOW support",
)
def test_complete_task_rejects_artifact_swapped_to_symlink_before_open(
    kanban_home,
    tmp_path,
    monkeypatch,
):
    """Opening the source must not follow a symlink installed after validation."""
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("host secret\n", encoding="utf-8")

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="race-safe artifact")
        task = kb.get_task(conn, task_id)
        workspace = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, task_id, workspace)
        artifact = workspace / "artifact.txt"
        artifact.write_text("expected output\n", encoding="utf-8")

        original_unique_path = kba._unique_attachment_path

        def swap_before_open(directory, filename, used):
            artifact.unlink()
            artifact.symlink_to(outside)
            return original_unique_path(directory, filename, used)

        monkeypatch.setattr(kba, "_unique_attachment_path", swap_before_open)

        with pytest.raises(kb.ArtifactPreservationError):
            kb.complete_task(
                conn,
                task_id,
                result="done",
                metadata={"artifacts": [str(artifact)]},
            )

        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.list_attachments(conn, task_id) == []
        assert all(event.kind != "completed" for event in kb.list_events(conn, task_id))

    assert workspace.exists(), "rejected completion must preserve the workspace"
    assert outside.read_text(encoding="utf-8") == "host secret\n"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("dir_fd", [True, False])
def test_source_parent_swapped_after_confinement_is_not_copied(tmp_path, monkeypatch, dir_fd):
    """O_NOFOLLOW guards only the leaf; a swapped parent must not redirect the read,
    with pinned descriptors or with the path-based fallback's check after open."""
    if not dir_fd:
        monkeypatch.setattr(kba, "_attachment_dir_fd_supported", lambda: False)
    root = tmp_path.resolve()
    workspace = root / "workspace"
    parent = workspace / "inputs"
    parent.mkdir(parents=True)
    source = parent / "report.txt"
    source.write_text("authorized report")
    outside = root / "outside"
    outside.mkdir()
    (outside / "report.txt").write_text("outside fixture")
    destination = root / "attachments"
    destination.mkdir()
    real_stat = os.stat
    swapped = []

    def swap_parent_on_first_check(path, *args, **kwargs):
        if str(path) == str(source) and kwargs.get("follow_symlinks") is False and not swapped:
            swapped.append(True)
            parent.rename(workspace / "old-inputs")
            parent.symlink_to(outside, target_is_directory=True)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", swap_parent_on_first_check)
    with pytest.raises(kb.ArtifactPreservationError):
        kba._stage_one_artifact(str(source), workspace, destination, None, set(), lambda: None)
    assert swapped
    assert list(destination.iterdir()) == []
