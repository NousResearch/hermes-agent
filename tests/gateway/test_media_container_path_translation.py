"""Bare container paths named in reply prose are delivered as local files.

``MEDIA:/workspace/chart.png`` is translated to its host file by
``_translate_docker_container_media_path`` inside ``validate_media_delivery_path``.
A reply that names the same path in prose ("I saved it to /workspace/chart.png")
goes through ``BasePlatformAdapter.extract_local_files`` instead, which used to
check the container path with a plain ``os.path.isfile`` on the host and drop it.

The second half pins the two delivery log lines on the prose route: intent
before the send loop and a per-file confirmation after each successful send, so
an upload that hangs (no result, no exception) still leaves a trace.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key

pytestmark = pytest.mark.platforms("linux", "macos")


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Host directory bind-mounted at /workspace in the Docker sandbox."""
    workspace = tmp_path / "host-workspace"
    workspace.mkdir()
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{workspace}:/workspace"]))
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    return workspace


# ---------------------------------------------------------------------------
# Bare paths in reply text
# ---------------------------------------------------------------------------

def test_bare_container_path_in_a_reply_is_extracted_as_the_host_file(sandbox):
    chart = sandbox / "chart.png"
    chart.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    files, cleaned = BasePlatformAdapter.extract_local_files(
        "Done. I saved the chart to /workspace/chart.png for you."
    )
    assert files == [str(chart.resolve())]
    assert "/workspace/chart.png" not in cleaned


def test_bare_container_path_passes_the_delivery_filter(sandbox):
    """The translated path is an ordinary host path for the delivery policy."""
    report = sandbox / "report.md"
    report.write_text("# report\n")

    files, _cleaned = BasePlatformAdapter.extract_local_files("Written to /workspace/report.md")
    assert BasePlatformAdapter.filter_local_delivery_paths(files) == [str(report.resolve())]


def test_bare_container_path_with_no_file_is_still_dropped(sandbox):
    files, cleaned = BasePlatformAdapter.extract_local_files("See /workspace/imaginary.png")
    assert files == []
    assert cleaned == "See /workspace/imaginary.png"


def test_unmounted_container_path_is_still_dropped(sandbox):
    files, _cleaned = BasePlatformAdapter.extract_local_files("See /scratch/job-a/chart.png")
    assert files == []


def test_translation_cannot_escape_the_mount(sandbox, tmp_path):
    """``..`` out of the mount must not reach a host file next to it."""
    (tmp_path / "outside.pdf").write_bytes(b"%PDF-1.4")
    files, _cleaned = BasePlatformAdapter.extract_local_files("See /workspace/../outside.pdf")
    assert files == []


def test_host_path_still_extracted_unchanged(sandbox):
    chart = sandbox / "chart.png"
    chart.write_bytes(b"data" * 32)
    files, _cleaned = BasePlatformAdapter.extract_local_files(f"Saved to {chart}")
    assert files == [str(chart)]


def test_no_docker_volumes_leaves_behaviour_unchanged(monkeypatch):
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    files, _cleaned = BasePlatformAdapter.extract_local_files("See /workspace/chart.png")
    assert files == []


def test_paths_inside_code_blocks_are_still_ignored(sandbox):
    """Translation must not start dragging code samples into delivery."""
    (sandbox / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    files, _cleaned = BasePlatformAdapter.extract_local_files(
        "Run this:\n```\nopen /workspace/chart.png\n```\n"
    )
    assert files == []


def test_unresolved_prose_path_does_not_raise_the_media_warning(monkeypatch, tmp_path, caplog):
    """Prose names many paths that are not files; the Docker MEDIA warning is
    for ``MEDIA:`` tags, and the prose route already logs its own skip line."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{tmp_path}:/workspace"]))
    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        files, _cleaned = BasePlatformAdapter.extract_local_files("See /workspace/imaginary.png")
    assert files == []
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Docker MEDIA path" in m for m in messages)
    assert any("Skipping bare file path in reply" in m for m in messages)


# ---------------------------------------------------------------------------
# Prose-route delivery logging
# ---------------------------------------------------------------------------

class _RecordingAdapter(BasePlatformAdapter):
    """Minimal adapter that records what the delivery loop hands it."""

    def __init__(self, platform: Platform = Platform.DISCORD):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.documents: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                            reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.documents.append(str(file_path))
        return SendResult(success=True, message_id="doc-1")


class _EmptyStore:
    """No prior turns, so nothing is suppressed by the history dedup."""

    def peek_session_id(self, session_key):
        return None

    def load_transcript(self, session_id):
        return []


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _prose_delivery_adapter(reply: str) -> _RecordingAdapter:
    adapter = _RecordingAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_session_store(_EmptyStore())

    async def handler(_event):
        return reply

    adapter.set_message_handler(handler)
    return adapter


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="build me the workbook",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="111", chat_type="dm"),
        message_id="m1",
    )


async def _deliver(adapter, caplog) -> list[str]:
    event = _make_event()
    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))
    return [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_container_path_in_prose_is_sent_as_a_document(sandbox, caplog):
    workbook = sandbox / "spend.xlsx"
    workbook.write_bytes(b"PK\x03\x04" + b"0" * 64)
    adapter = _prose_delivery_adapter("Saved it to /workspace/spend.xlsx for you.")

    await _deliver(adapter, caplog)

    assert adapter.documents == [str(workbook.resolve())]


@pytest.mark.asyncio
async def test_prose_route_logs_intent_then_confirmation(sandbox, caplog):
    workbook = sandbox / "spend.xlsx"
    workbook.write_bytes(b"PK\x03\x04" + b"0" * 64)
    adapter = _prose_delivery_adapter(f"Saved it to {workbook} for you.")

    messages = await _deliver(adapter, caplog)

    assert adapter.documents == [str(workbook.resolve())]
    intent = [i for i, m in enumerate(messages) if "Delivering 1 local file attachment(s)" in m]
    done = [i for i, m in enumerate(messages) if "Delivered local file" in m]
    assert intent, "the prose route must log its intent, as the MEDIA route already does"
    assert done, "a successful send must be confirmed, or a hung upload leaves no trace"
    assert intent[0] < done[0]


@pytest.mark.asyncio
async def test_failed_send_is_not_confirmed(sandbox, caplog):
    workbook = sandbox / "spend.xlsx"
    workbook.write_bytes(b"PK\x03\x04" + b"0" * 64)
    adapter = _prose_delivery_adapter(f"Saved it to {workbook} for you.")

    async def refuse(chat_id, file_path, **kwargs):
        return SendResult(success=False, error="upload refused")

    adapter.send_document = refuse

    messages = await _deliver(adapter, caplog)

    assert any("Delivering 1 local file attachment(s)" in m for m in messages)
    assert not any("Delivered local file" in m for m in messages)
