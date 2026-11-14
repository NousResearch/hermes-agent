"""Configured root aliases must not weaken descendant no-follow checks."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_artifacts as kba


@pytest.mark.parametrize("fallback", [False, True])
def test_completion_accepts_alias_above_configured_kanban_root(tmp_path, monkeypatch, fallback):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    home = alias / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if fallback:
        monkeypatch.setattr(kba, "_attachment_dir_fd_supported", lambda: False)
    kb.init_db()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "report.txt"
    artifact.write_bytes(b"root alias report")
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="root alias", workspace_kind="dir", workspace_path=str(workspace))
        assert kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(artifact)]}, fire_lifecycle_hook=False)
        event = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1]
        stored = Path(event.payload["artifacts"][0])
        assert kb.get_task(conn, task_id).status == "done"
        assert len(kb.list_attachments(conn, task_id)) == 1
    assert stored == real / "home" / "kanban" / "attachments" / task_id / "report.txt"
    assert stored.read_bytes() == b"root alias report"
    assert artifact.read_bytes() == b"root alias report"


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("level", ["kanban", "attachments", "task"])
def test_root_alias_does_not_authorize_descendant_symlinks(tmp_path, monkeypatch, fallback, level):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(alias))
    monkeypatch.setenv("HERMES_HOME", str(alias))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if fallback:
        monkeypatch.setattr(kba, "_attachment_dir_fd_supported", lambda: False)
    kb.init_db()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "report.txt"
    artifact.write_bytes(b"keep workspace intact")
    outside = tmp_path / "outside"
    outside.mkdir()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="reject link", workspace_kind="dir", workspace_path=str(workspace))
        if level == "kanban":
            destination = alias / "kanban"
        elif level == "attachments":
            destination = alias / "kanban" / "attachments"
        else:
            destination = kb.task_attachments_dir(task_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(outside, target_is_directory=True)
        with pytest.raises(kb.ArtifactPreservationError):
            kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(artifact)]}, fire_lifecycle_hook=False)
        assert kb.get_task(conn, task_id).status == "ready"
        assert not kb.list_attachments(conn, task_id)
        assert not any(e.kind == "completed" for e in kb.list_events(conn, task_id))
    assert list(outside.iterdir()) == []
    assert artifact.read_bytes() == b"keep workspace intact"
