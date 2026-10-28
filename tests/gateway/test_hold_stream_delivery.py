"""A ``hold_stream`` plugin's session shows the user only the rewritten final, on every chat shape.

Real ``GatewayRunner._run_agent`` + real ``AIAgent`` against a loopback model streaming a reply
that a ``transform_llm_output`` redactor rewrites. The adapters are recording transports (and the
real ``RelayAdapter`` over the stub connector for Slack's stream-is-the-message lane); the base
adapter's own ``_process_message_background`` runs the typing loop and the final send. With
``hold_stream=True`` no draft, edit, native stream frame, split head or commentary bubble ever
carries the raw text, the typing indicator runs while the reply is held, and the final goes out
once, rewritten. The same plugin without the declaration keeps today's stream (control).
"""

import os
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.redactor_plugin import REWRITE, raw_leaks, redactor_home, redactor_turn
from tests.gateway.test_run_progress_topics import _make_runner


class _Chat(BasePlatformAdapter):
    """Recording transport: every send/edit/frame lands in ``log`` as ``(kind, text)``."""

    def __init__(self, *, editable=True, limit=4096):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.SUPPORTS_MESSAGE_EDITING, self.MAX_MESSAGE_LENGTH = editable, limit
        self.log, self.typing, self._ids = [], 0, 0

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        for chunk in self.truncate_message(content, self.MAX_MESSAGE_LENGTH):  # what real adapters do
            self._ids += 1
            self.log.append(("send", chunk))
        return SendResult(success=True, message_id=f"m{self._ids}" if self.SUPPORTS_MESSAGE_EDITING else None)

    async def edit_message(self, chat_id, message_id, content, **_):
        if not self.SUPPORTS_MESSAGE_EDITING or len(content) > self.MAX_MESSAGE_LENGTH:
            return SendResult(success=False, error="cannot edit")
        self.log.append(("edit", content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.typing += 1

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _NativeStreamChat(_Chat):
    """WeCom's shape: no edits, a native cumulative stream that opens with an empty seed frame."""

    SUPPORTS_NATIVE_STREAMING = True

    def supports_native_streaming(self, chat_type=None, metadata=None):
        return True

    async def send_stream_frame(self, text, *, finalize=False, chat_id=None, reply_to=None, turn_id=None):
        self.log.append(("frame", text))
        return True


def _relay_slack():
    """Slack fronted by the relay: native ``chat.*Stream`` drafts ARE the message."""
    from gateway.relay.adapter import RelayAdapter
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_slack_dm_streaming import _slack_desc

    desc = _slack_desc(supports_draft_streaming=True)
    adapter = RelayAdapter(PlatformConfig(), desc, transport=StubConnector(desc))
    assert adapter.draft_stream_is_message is True
    return adapter


def _frames(chat):
    if hasattr(chat, "log"):
        return chat.log
    return [(frame["op"], frame.get("content") or frame.get("text") or "") for frame in chat._transport.sent]


CHATS = {
    "edits_in_place": lambda: _Chat(editable=True),
    "cannot_edit": lambda: _Chat(editable=False),
    "splits_long_replies": lambda: _Chat(editable=True, limit=60),
    "native_stream": lambda: _NativeStreamChat(editable=False),
    "relay_stream_is_the_message": _relay_slack,
}


async def _deliver(chat, *, hold, monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli import plugins

    home = Path(os.environ["HERMES_HOME"])
    platform = Platform.SLACK if chat.platform == Platform.RELAY else Platform.TELEGRAM
    source = SessionSource(platform=platform, chat_id="c1", chat_type="dm", user_id="u1")
    event = MessageEvent(text="show me the config", source=source, message_type=MessageType.TEXT)
    session_key = f"agent:main:{platform.value}:dm:c1"
    with FakeLLMServer(redactor_turn(delay_per_chunk=0.03)) as srv:
        redactor_home(home, srv.base_url, hold=hold)
        plugins._reset_plugin_managers_for_tests()
        monkeypatch.setattr(gateway_run, "_hermes_home", home)
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
            "api_key": "sk-fake-e2e", "base_url": srv.base_url, "provider": "custom", "api_mode": "chat_completions"})
        runner = _make_runner(chat)
        runner.adapters[platform] = chat
        runner.config.streaming = StreamingConfig(enabled=True, transport="auto", edit_interval=0.01,
                                                  buffer_threshold=1, cursor="")
        if hasattr(chat, "_capture_scope"):
            chat._capture_scope(event)

        async def handler(event):
            result = await runner._run_agent(message=event.text, context_prompt="", history=[], source=source,
                                             session_id="s1", session_key=session_key)
            assert result["final_response"] == REWRITE
            return None if result.get("already_sent") else result["final_response"]

        chat.set_message_handler(handler)
        try:
            await chat._process_message_background(event, session_key)
        finally:
            plugins._reset_plugin_managers_for_tests()
    return _frames(chat)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", CHATS)
async def test_a_held_reply_reaches_the_chat_only_as_the_rewrite(shape, monkeypatch):
    chat = CHATS[shape]()
    frames = await _deliver(chat, hold=True, monkeypatch=monkeypatch)

    assert raw_leaks(text for _, text in frames) == []
    # Only plain sends of the final: nothing streamed, so nothing to edit, seal or close.
    assert {kind for kind, _ in frames} - {"typing"} == {"send"}, frames
    assert " ".join(text for _, text in frames).count("[REDACTED]") == 2
    # The typing indicator is what the user sees while the reply is held.
    assert getattr(chat, "typing", 0) or ("typing", "") in frames


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["edits_in_place", "relay_stream_is_the_message"])
async def test_a_transform_that_did_not_ask_keeps_streaming(shape, monkeypatch):
    frames = await _deliver(CHATS[shape](), hold=False, monkeypatch=monkeypatch)

    assert raw_leaks(text for _, text in frames), frames  # the raw reply streamed, as before
