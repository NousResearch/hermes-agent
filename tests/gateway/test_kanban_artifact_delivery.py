"""Notifier artifact confinement contracts ported from PR #86610."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
import pytest
from gateway.kanban_watchers import GatewayKanbanWatchersMixin

class _ArtifactAdapter:
    def __init__(self):
        self.documents: list[str] = []

    @staticmethod
    def extract_local_files(content: str):
        return ([part for part in content.split() if part.endswith(".pdf")], content)

    async def send_document(self, *, file_path: str, **_kwargs):
        self.documents.append(file_path)


def test_notifier_rejects_unstaged_completion_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "inside.pdf"
    outside = tmp_path / "outside.pdf"
    inside.write_bytes(b"inside")
    outside.write_bytes(b"outside")
    adapter = _ArtifactAdapter()
    task = SimpleNamespace(
        id="t_bounded",
        workspace_path=str(workspace),
        result=f"legacy {outside}",
    )

    asyncio.run(
        GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat",
            metadata={},
            event_payload={
                "artifacts": [str(outside)],
                "summary": f"deliver {inside} not {outside}",
            },
            task=task,
        )
    )

    assert adapter.documents == []


def test_durable_task_attachment_remains_deliverable(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as kb

    task_id = "t_durable"
    stored_dir = kb.task_attachments_dir(task_id, board="default")
    stored_dir.mkdir(parents=True)
    stored = stored_dir / "report.pdf"
    stored.write_bytes(b"report")
    adapter = _ArtifactAdapter()
    task = SimpleNamespace(
        id=task_id,
        workspace_path=str(tmp_path / "cleaned-workspace"),
        result=None,
    )

    asyncio.run(
        GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat",
            metadata={},
            event_payload={"artifacts": [str(stored)]},
            task=task,
            board="default",
        )
    )

    assert adapter.documents == [str(stored.resolve())]


def test_notifier_rejects_symlinked_task_attachment_root(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as kb

    task_id = "t_symlinked_root"
    stored_dir = kb.task_attachments_dir(task_id, board="default")
    stored_dir.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "report.pdf"
    external.write_bytes(b"host secret")
    try:
        stored_dir.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    adapter = _ArtifactAdapter()
    task = SimpleNamespace(
        id=task_id,
        workspace_path=str(tmp_path / "workspace"),
        result=None,
    )

    asyncio.run(
        GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat",
            metadata={},
            event_payload={"artifacts": [str(stored_dir / external.name)]},
            task=task,
            board="default",
        )
    )

    assert adapter.documents == []


def test_notifier_never_hands_mutable_workspace_path_to_adapter(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    outside = tmp_path / "outside.pdf"
    artifact.write_bytes(b"expected report")
    outside.write_bytes(b"host secret")

    class SwappingAdapter(_ArtifactAdapter):
        def __init__(self):
            super().__init__()
            self.uploaded: list[bytes] = []

        async def send_document(self, *, file_path: str, **_kwargs):
            artifact.unlink()
            artifact.symlink_to(outside)
            self.uploaded.append(Path(file_path).read_bytes())

    adapter = SwappingAdapter()
    task = SimpleNamespace(
        id="t_raced",
        workspace_path=str(workspace),
        result=None,
    )

    asyncio.run(
        GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat",
            metadata={},
            event_payload={"artifacts": [str(artifact)]},
            task=task,
        )
    )

    assert adapter.uploaded == []


@pytest.mark.parametrize("board", ["default", "other"])
def test_completion_delivers_original_bytes_after_workspace_mutation(tmp_path, monkeypatch, board):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()
    if board != "default":
        kb.create_board(board)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    artifact.write_bytes(b"original report")
    with kbc.connect(board=board) as conn:
        task_id = kb.create_task(conn, title="deliver report", workspace_kind="dir", workspace_path=str(workspace))
        assert kb.complete_task(conn, task_id, summary=f"done: {artifact}", metadata={"artifacts": [str(artifact)]})
        task = kb.get_task(conn, task_id)
        payload = [e.payload for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1]
    artifact.write_bytes(b"modified by worker after completion")

    class ReadingAdapter(_ArtifactAdapter):
        def __init__(self):
            super().__init__()
            self.uploaded: list[tuple[str, bytes]] = []

        async def send_document(self, *, file_path, **kwargs):
            self.uploaded.append((file_path, Path(file_path).read_bytes()))

    adapter = ReadingAdapter()
    asyncio.run(GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
        adapter=adapter, chat_id="chat", metadata={}, event_payload=payload, task=task, board=board,
    ))
    assert adapter.uploaded == [(str(kb.task_attachments_dir(task_id, board=board) / "report.pdf"), b"original report")]


def _aliased_kanban_home(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    home = alias / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _deliver(task_id, artifacts):
    adapter = _ArtifactAdapter()
    task = SimpleNamespace(id=task_id, workspace_path=None, result=None)
    asyncio.run(
        GatewayKanbanWatchersMixin()._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat",
            metadata={},
            event_payload={"artifacts": artifacts},
            task=task,
            board="default",
        )
    )
    return adapter.documents


def test_artifact_staged_under_configured_root_alias_is_delivered(tmp_path, monkeypatch):
    """Staging accepts an alias above the configured root; delivery must too."""
    _aliased_kanban_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    kb.init_db()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / "report.pdf"
    artifact.write_bytes(b"report")
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="alias", workspace_kind="dir", workspace_path=str(workspace))
        assert kb.complete_task(conn, task_id, summary="done", metadata={"artifacts": [str(artifact)]}, fire_lifecycle_hook=False)
        event = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1]
    stored = event.payload["artifacts"]

    assert _deliver(task_id, stored) == stored


def test_configured_root_alias_does_not_authorize_symlinked_attachments(tmp_path, monkeypatch):
    home = _aliased_kanban_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb

    outside = tmp_path / "outside"
    (outside / "t_alias").mkdir(parents=True)
    external = outside / "t_alias" / "report.pdf"
    external.write_bytes(b"host secret")
    (home / "kanban").mkdir()
    (home / "kanban" / "attachments").symlink_to(outside, target_is_directory=True)

    assert _deliver("t_alias", [str(kb.task_attachments_dir("t_alias", board="default") / "report.pdf")]) == []
