"""Use the claimed prompt, not a pre-transcription snapshot, for card retirement."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import SessionSource, build_session_key
from tools import clarify_gateway as cg


@pytest.fixture(autouse=True)
def isolated_clarify_queue():
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
        cg._notify_cbs.clear()
    yield
    with cg._lock:
        cg._entries.clear()
        cg._session_index.clear()
        cg._notify_cbs.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "response", "result"),
    [("2", "D", ""), ("Let us do something else instead", cg.CANCELLED, None)],
)
async def test_reply_retires_the_prompt_selected_after_transcription(text, response, result):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="user")
    session_key = build_session_key(source)
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source)
    first = cg.register("first", session_key, "First?", ["A", "B"])
    second = cg.register("second", session_key, "Second?", ["C", "D"])
    retired = []

    class Adapter:
        def resume_typing_for_chat(self, chat_id):
            pass

        async def retire_clarify_card(self, clarify_id, content):
            retired.append(clarify_id)

    async def prepare_reply(inbound_event):
        # A button callback answers the head while text preparation is awaiting.
        assert cg.resolve_gateway_clarify(first.clarify_id, "A")
        return inbound_event.text

    runner = SimpleNamespace(
        _pending_event_audio_paths=lambda inbound_event: [],
        _prepare_clarify_reply_text=prepare_reply,
        _delivery_adapter_for=lambda inbound_source: Adapter(),
    )

    assert await GatewayInboundMixin._hm_clarify_reply(runner, event, source, session_key) == result
    assert first.response == "A"
    assert second.response == response
    assert second.event.is_set()
    assert retired == [second.clarify_id]
