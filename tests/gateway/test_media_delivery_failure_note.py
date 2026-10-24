"""A post-turn media upload that fails must be reported twice: once to the user, and once to the
model as a one-shot note on its next run.

Delivery happens after the agent's turn ended, so without the note the model believes a file it
attached was delivered and never retries — the Feishu 99992402 case, where the transcript said
"sent" and nothing arrived.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _NoteSink:
    """Stands in for the runner's sidecar-note store."""

    def __init__(self):
        self.staged: dict[str, list[str]] = {}

    def _set_pending_turn_sidecar_notes(self, session_key: str, notes: list[str]) -> None:
        self.staged[session_key] = list(notes)


class _MediaAdapter(BasePlatformAdapter):
    def __init__(self, *, fail: bool):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.FEISHU)
        self.fail = fail
        self.sent: list[str] = []
        self.gateway_runner = _NoteSink()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="om_notice")

    async def send_document(self, chat_id, file_path, **kwargs):
        if self.fail:
            return SendResult(success=False, error="[2200] Internal Error")
        return SendResult(success=True, message_id="om_doc")


def _event() -> MessageEvent:
    return MessageEvent(
        text="",
        source=SessionSource(platform=Platform.FEISHU, chat_id="oc_1", chat_type="dm"),
    )


async def _deliver(adapter: _MediaAdapter, path: str) -> None:
    # Session-key derivation has its own tests; this file is about what a failure records.
    adapter._source_session_key = lambda source: "sk-1"  # type: ignore[method-assign]
    await adapter._deliver_media_attachments(
        _event(), [(path, False)], [], force_document_attachments=False,
        human_delay=0.0, metadata={}, record_delivery=lambda result: None,
    )


@pytest.mark.asyncio
async def test_failed_media_stages_a_note_for_the_next_run(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    adapter = _MediaAdapter(fail=True)

    await _deliver(adapter, str(path))

    notes = adapter.gateway_runner.staged.get("sk-1") or []
    assert len(notes) == 1, notes
    assert "report.pdf" in notes[0]
    assert "[2200] Internal Error" in notes[0], "the platform's own reason must survive"
    assert "send it again" in notes[0]


@pytest.mark.asyncio
async def test_failed_media_also_warns_the_user(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    adapter = _MediaAdapter(fail=True)

    await _deliver(adapter, str(path))

    assert adapter.sent, "a failed attachment must not be silent for the user either"
    assert any("report.pdf" in message for message in adapter.sent)


@pytest.mark.asyncio
async def test_delivered_media_records_nothing(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    adapter = _MediaAdapter(fail=False)

    await _deliver(adapter, str(path))

    assert adapter.gateway_runner.staged == {}
    assert adapter.sent == []
