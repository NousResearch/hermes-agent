"""A completion notification must not upload one document per produced file.

Workers that list every file they produced (per-step logs, action dumps, CSVs)
used to flood the human's chat with hundreds of uploads. Only the smallest
documents ride along; the rest are logged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


# The documented cap. Kept as a literal on purpose: importing the constant would make
# the test fail at collection instead of on the behaviour it is meant to lock down.
DOC_ARTIFACT_LIMIT = 5


class _Watcher(GatewayKanbanWatchersMixin):
    pass


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.extract_local_files.return_value = ([], "")
    adapter.send_document = AsyncMock()
    adapter.send_video = AsyncMock()
    adapter.send_multiple_images = AsyncMock()
    return adapter


def _files(tmp_path, sizes):
    paths = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"artifact_{index}.csv"
        path.write_bytes(b"x" * size)
        paths.append(str(path))
    return paths


@pytest.fixture(autouse=True)
def _no_path_filter(monkeypatch):
    from gateway.platforms.base import BasePlatformAdapter

    monkeypatch.setattr(
        BasePlatformAdapter,
        "filter_local_delivery_paths",
        classmethod(lambda cls, paths: list(paths)),
    )


@pytest.mark.asyncio
async def test_document_artifacts_are_capped_to_the_smallest(tmp_path):
    watcher = _Watcher()
    adapter = _adapter()
    # Sizes are deliberately unique so the kept set is unambiguous.
    paths = _files(tmp_path, [10, 20, 30, 40, 50, 60, 70, 80])

    await watcher._deliver_kanban_artifacts(
        adapter=adapter,
        chat_id="chat-1",
        metadata={},
        event_payload={"artifacts": paths},
        task=None,
    )

    assert adapter.send_document.await_count == DOC_ARTIFACT_LIMIT
    sent = [call.kwargs["file_path"] for call in adapter.send_document.await_args_list]
    assert sent == sorted(paths, key=lambda p: __import__("os").path.getsize(p))[:DOC_ARTIFACT_LIMIT]


@pytest.mark.asyncio
async def test_small_completion_still_uploads_everything(tmp_path):
    """Negative control: the cap must not drop anything below the limit."""
    watcher = _Watcher()
    adapter = _adapter()
    paths = _files(tmp_path, [10, 20, 30])

    await watcher._deliver_kanban_artifacts(
        adapter=adapter,
        chat_id="chat-2",
        metadata={},
        event_payload={"artifacts": paths},
        task=None,
    )

    assert adapter.send_document.await_count == 3
    assert sorted(call.kwargs["file_path"] for call in adapter.send_document.await_args_list) == sorted(paths)
