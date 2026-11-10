"""Queued-follow-up lane: the fallback first-response send and the normal completion send share one
"already delivered" source of truth (#81052).

With streaming disabled (the shipped default) there is never a stream consumer, so the queued
lane always sends the first turn's final itself before running the follow-up. When the lane then
hands the FIRST turn's result back (the follow-up's text was refused), the normal completion path
must not send that text — or its attachments — a second time, and must still send them when the
queued lane's own send was refused.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner, _run_with_agent

_SESSION_KEY = "agent:main:telegram:group:-1001:17585"


class _DocCaptureAdapter(ProgressCaptureAdapter):
    """Capture document uploads so a double MEDIA delivery is visible."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.documents = []

    async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                            reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.documents.append(file_path)
        return SendResult(success=True, message_id="doc-1")


class _RefusingDocCaptureAdapter(_DocCaptureAdapter):
    """Every send is refused (flood control, retries exhausted): a SendResult, not an exception."""

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        return SendResult(success=False, error="flood control", retryable=False)


def _agent_returning(text):
    class _Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            return {"final_response": text, "messages": [], "api_calls": 1}

    return _Agent


async def _none():
    return None


async def _run_with_refused_followup(monkeypatch, tmp_path, agent_cls, adapter_cls):
    """First turn answers, the queued follow-up's text is refused: the lane hands back the FIRST
    turn's result, which is the object the completion send then acts on."""
    monkeypatch.setattr(
        GatewayRunner, "_expand_inbound_context_references",
        lambda self, source, session_key, message_text: _none(),
    )
    return await _run_with_agent(
        monkeypatch, tmp_path, agent_cls, session_id="sess-refused-followup",
        pending_text="@file:/etc/shadow please", adapter_cls=adapter_cls,
    )


async def _completion_seam(adapter, agent_result, response):
    """Run the result through ``_hmwa_deliver_turn_response`` — the seam that decides whether the
    caller sends the body again. Returns the text the caller would send (None = suppressed)."""
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=adapter.platform, chat_id="-1001", chat_type="group", thread_id="17585",
    )
    event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")

    class _Entry:
        session_id = "s1"

    return await runner._hmwa_deliver_turn_response(
        event, source, _Entry(), _SESSION_KEY, None, agent_result, [], response, None, False,
    )


@pytest.mark.asyncio
async def test_queued_lane_delivers_text_and_media_exactly_once(monkeypatch, tmp_path):
    """Single delivery: the queued lane sends the text and uploads the attachment, and the
    completion seam re-sends neither."""
    doc = tmp_path / "report.pdf"
    doc.write_text("x", encoding="utf-8")
    final = f"answer 1\n\nMEDIA: {doc}"
    adapter, result = await _run_with_refused_followup(
        monkeypatch, tmp_path, _agent_returning(final), _DocCaptureAdapter,
    )

    assert [c["content"] for c in adapter.sent] == ["answer 1"]
    assert adapter.documents == [str(doc)]
    assert result["final_response"] == final

    assert await _completion_seam(adapter, result, final) is None
    assert len(adapter.sent) == 1
    assert adapter.documents == [str(doc)]


@pytest.mark.asyncio
async def test_refused_queued_send_leaves_the_completion_send_as_the_fallback(monkeypatch, tmp_path):
    """A refused queued send (``SendResult.success`` False) must NOT mark the turn delivered: the
    completion seam is the only remaining send, and suppressing it loses the answer entirely."""
    doc = tmp_path / "report.pdf"
    doc.write_text("x", encoding="utf-8")
    final = f"answer 1\n\nMEDIA: {doc}"
    adapter, result = await _run_with_refused_followup(
        monkeypatch, tmp_path, _agent_returning(final), _RefusingDocCaptureAdapter,
    )

    # Attempted (and retried) but never accepted by the platform.
    assert adapter.sent and {c["content"] for c in adapter.sent} == {"answer 1"}
    # The text never landed, so its attachments wait for the completion send too.
    assert adapter.documents == []
    assert not result.get("already_sent")

    assert await _completion_seam(adapter, result, final) == final
