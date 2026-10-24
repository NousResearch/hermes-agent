"""A completed task uploads only the artifacts it declared, never a file its prose mentions.

``event_payload['summary']`` and ``task.result`` are worker free text and both reach the
artifact delivery path. Scanning them for paths made prose an upload authorization: a
worker (or an injection in its output) that wrote "/Users/someone/.e.g. notes.txt" caused an
existing host file to be uploaded to the chat. Only the explicit ``artifacts`` list — written
by the board's staging step — may cause an upload.

Every file here is synthetic and lives in a tmp_path.
"""

from __future__ import annotations

import asyncio
import pathlib

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


class _Recorder:
    """Minimal adapter recording what would have been sent to the chat."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.images: list[str] = []

    @staticmethod
    def extract_local_files(content: str):
        # Delegate to the real scanner: the point is what the notifier does with its output.
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.extract_local_files(content)

    async def send_document(self, *, chat_id, file_path, metadata):
        self.documents.append(file_path)

    async def send_video(self, *, chat_id, video_path, metadata):
        self.documents.append(video_path)

    async def send_multiple_images(self, *, chat_id, images, metadata):
        self.images.extend(url for url, _ in images)


def _deliver(adapter, payload, task=None):
    mixin = GatewayKanbanWatchersMixin()
    asyncio.run(
        mixin._deliver_kanban_artifacts(
            adapter=adapter, chat_id="1", metadata={}, event_payload=payload, task=task
        )
    )


@pytest.fixture()
def secret_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "private-notes.txt"
    p.write_text("synthetic secret\n")
    return p


@pytest.fixture()
def declared_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "report.txt"
    p.write_text("synthetic artifact\n")
    return p


def test_prose_path_in_summary_does_not_authorize_an_upload(secret_file):
    adapter = _Recorder()
    _deliver(adapter, {"summary": f"NEXT_STEP: see {secret_file} for details"})
    assert adapter.documents == []


def test_prose_path_in_task_result_does_not_authorize_an_upload(secret_file):
    adapter = _Recorder()
    task = type("T", (), {"result": f"RESULT: wrote {secret_file}"})()
    _deliver(adapter, {"summary": "done"}, task=task)
    assert adapter.documents == []


def test_declared_artifact_is_still_delivered(declared_file):
    adapter = _Recorder()
    _deliver(adapter, {"artifacts": [str(declared_file)]})
    assert adapter.documents == [str(declared_file)]


def test_declared_artifact_survives_a_prose_mention_of_another_file(
    declared_file, secret_file
):
    adapter = _Recorder()
    _deliver(
        adapter,
        {
            "artifacts": [str(declared_file)],
            "summary": f"also see {secret_file}",
        },
    )
    assert adapter.documents == [str(declared_file)]
    assert str(secret_file) not in adapter.documents


def test_missing_declared_artifact_is_skipped(tmp_path):
    adapter = _Recorder()
    _deliver(adapter, {"artifacts": [str(tmp_path / "absent.txt")]})
    assert adapter.documents == []


def test_no_artifacts_and_no_prose_sends_nothing():
    adapter = _Recorder()
    _deliver(adapter, {}, task=None)
    assert adapter.documents == [] and adapter.images == []
