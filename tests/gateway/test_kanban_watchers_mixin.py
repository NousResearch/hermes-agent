"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


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
