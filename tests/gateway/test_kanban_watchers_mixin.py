"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.platforms.base import BasePlatformAdapter

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


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


class _ArtifactAdapter:
    extract_local_files = staticmethod(BasePlatformAdapter.extract_local_files)

    def __init__(self):
        self.send_document = AsyncMock()
        self.send_multiple_images = AsyncMock()
        self.send_video = AsyncMock()


def test_kanban_artifact_delivery_skips_internal_workflow_control_files(tmp_path):
    workflow = tmp_path / ".hermes" / "workflows" / "run-1"
    artifacts = workflow / "artifacts"
    artifacts.mkdir(parents=True)
    manifest = workflow / "manifest.yaml"
    planner_bundle = artifacts / "planner-handoff-bundle.yaml"
    report = artifacts / "final-report.html"
    manifest.write_text("run_id: run-1\n", encoding="utf-8")
    planner_bundle.write_text("schema: hermes.planner_handoff_bundle.v1\n", encoding="utf-8")
    report.write_text("<html>ok</html>\n", encoding="utf-8")

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(manifest), str(planner_bundle), str(report)]},
            task=None,
        )
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1", file_path=str(report), metadata={}
    )
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_video.assert_not_awaited()


def test_kanban_summary_path_skips_internal_workflow_control_files_only(tmp_path):
    workflow = tmp_path / ".hermes" / "workflows" / "run-2"
    artifacts = workflow / "artifacts"
    artifacts.mkdir(parents=True)
    planner_bundle = artifacts / "planner-handoff-bundle.yaml"
    rendered_report = artifacts / "dashboard.html"
    project_yaml = workflow / "project.yaml"
    unrelated_yaml = tmp_path / "customer-facing-config.yaml"
    for path in (planner_bundle, rendered_report, project_yaml, unrelated_yaml):
        path.write_text("ok\n", encoding="utf-8")

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()
    summary = (
        f"evidence: {planner_bundle}\n"
        f"report: {rendered_report}\n"
        f"state: {project_yaml}\n"
        f"yaml: {unrelated_yaml}"
    )

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={"thread_id": "t1"},
            event_payload={"summary": summary},
            task=SimpleNamespace(result=f"legacy path: {unrelated_yaml}"),
        )
    )

    assert adapter.send_document.await_count == 2
    sent_paths = [call.kwargs["file_path"] for call in adapter.send_document.call_args_list]
    assert sent_paths == [str(rendered_report), str(unrelated_yaml.resolve())]


def test_kanban_summary_keeps_non_internal_text_artifacts(tmp_path):
    report_md = tmp_path / "notes.md"
    report_txt = tmp_path / "summary.txt"
    report_md.write_text("# notes\n", encoding="utf-8")
    report_txt.write_text("summary\n", encoding="utf-8")

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()
    summary = f"notes: {report_md}\nsummary: {report_txt}"

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"summary": summary},
            task=None,
        )
    )

    sent_paths = [call.kwargs["file_path"] for call in adapter.send_document.call_args_list]
    assert sent_paths == [str(report_md.resolve()), str(report_txt.resolve())]


def test_explicit_non_workflow_manifest_can_still_be_delivered(tmp_path):
    manifest = tmp_path / "customer" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"deliverable": true}\n', encoding="utf-8")

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(manifest)]},
            task=None,
        )
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1", file_path=str(manifest.resolve()), metadata={}
    )


def test_non_workflow_planner_bundle_can_still_be_delivered(tmp_path):
    bundle = tmp_path / "customer" / "planner-handoff-bundle.yaml"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("deliverable: true\n", encoding="utf-8")

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(bundle)]},
            task=None,
        )
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1", file_path=str(bundle.resolve()), metadata={}
    )


def test_manifest_path_with_non_adjacent_hermes_workflows_parts_is_not_suppressed(tmp_path):
    manifest = tmp_path / "workflows" / "customer" / ".hermes" / "report" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("kind: deliverable\n", encoding="utf-8")

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(manifest)]},
            task=None,
        )
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1", file_path=str(manifest.resolve()), metadata={}
    )


def test_internal_workflow_symlink_control_file_is_not_delivered(tmp_path):
    workflow = tmp_path / ".hermes" / "workflows" / "run-3"
    workflow.mkdir(parents=True)
    external_target = tmp_path / "target-state.json"
    external_target.write_text('{"private": true}\n', encoding="utf-8")
    symlinked_state = workflow / "state.json"
    try:
        symlinked_state.symlink_to(external_target)
    except OSError:
        return

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(symlinked_state)]},
            task=None,
        )
    )

    adapter.send_document.assert_not_awaited()
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_video.assert_not_awaited()


def test_external_symlink_to_internal_workflow_control_file_is_not_delivered(tmp_path):
    workflow = tmp_path / ".hermes" / "workflows" / "run-4"
    workflow.mkdir(parents=True)
    internal_state = workflow / "state.json"
    internal_state.write_text('{"private": true}\n', encoding="utf-8")
    external_link = tmp_path / "linked-state.json"
    try:
        external_link.symlink_to(internal_state)
    except OSError:
        return

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(external_link)]},
            task=None,
        )
    )

    adapter.send_document.assert_not_awaited()
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_video.assert_not_awaited()


def test_artifact_delivery_uses_normalized_safe_path(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    report = real_dir / "report.pdf"
    report.write_bytes(b"%PDF-1.4")
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError:
        return
    symlinked_report = link_dir / "report.pdf"

    adapter = _ArtifactAdapter()
    runner = GatewayKanbanWatchersMixin()

    asyncio.run(
        runner._deliver_kanban_artifacts(
            adapter=adapter,
            chat_id="chat-1",
            metadata={},
            event_payload={"artifacts": [str(symlinked_report)]},
            task=None,
        )
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1", file_path=str(report.resolve()), metadata={}
    )
