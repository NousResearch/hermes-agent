"""Real native/relay send paths and profile-scoped effective warning policy.

Only external Slack SDK/relay transport I/O is replaced; config, agent emission,
callback classification, presentation and adapter send/edit are production code.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.status_output import StatusOutputMixin
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.run import GatewayRunner, _load_gateway_config, _profile_runtime_scope
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.relay.stub_connector import StubConnector


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("thread_id", [None, "1700.1"])
@pytest.mark.parametrize("enabled", [False, True])
def test_warning_policy_reaches_slack_transport(tmp_path, monkeypatch, relay, thread_id, enabled):
    from gateway import run

    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(not enabled).lower()}}}")
    monkeypatch.setattr(run, "_hermes_home", tmp_path)
    source = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm", user_id="U1", thread_id=thread_id)
    metadata = {"thread_id": thread_id} if thread_id else {}
    pc = PlatformConfig(extra={"reply_in_thread": bool(thread_id)})
    if relay:
        desc = CapabilityDescriptor(contract_version=CONTRACT_VERSION, platform="slack", label="Slack",
                                    max_message_length=4000, supports_edit=True, supports_threads=True, supports_draft_streaming=False,
                                    markdown_dialect="mrkdwn", len_unit="chars", emoji="", platform_hint="", pii_safe=False)
        transport = StubConnector(desc)
        adapter = RelayAdapter(pc, desc, transport=transport)
        adapter._capture_scope(MessageEvent(text="hello", source=source, message_type=MessageType.TEXT))
        sent = transport.sent
    else:
        adapter = SlackAdapter(pc)
        client = AsyncMock()
        client.chat_postMessage.return_value = {"ok": True, "ts": "1700.2"}
        client.chat_update.return_value = {"ok": True, "ts": "1700.2"}
        adapter._app = SimpleNamespace(client=client)
        sent = []
        async def post(**kwargs):
            sent.append(kwargs)
            return {"ok": True, "ts": "1700.2"}
        client.chat_postMessage.side_effect = post
        client.chat_update.side_effect = post
    gateway = object.__new__(GatewayRunner)
    gateway.config = None
    gateway._adapter_for_source = lambda source: adapter
    gateway._thread_metadata_for_source = lambda source: metadata
    ctx = TurnContext(source=source, user_config=_load_gateway_config(), _run_still_current=lambda: True,
                      _status_adapter=adapter, _status_chat_id="D1", _status_thread_metadata=metadata)
    turn = TurnRunner(gateway, ctx)
    monkeypatch.setattr(run, "safe_schedule_threadsafe", lambda coro, *a, **k: asyncio.run(coro))
    agent = StatusOutputMixin()
    agent.suppress_status_output = True
    agent.status_callback = turn._status_callback_sync
    for _ in range(2):
        agent._emit_warning("⚠ Context is over the limit; compression blocked")
    assert len(sent) == (2 if enabled else 0)
    # A warning-looking real answer remains deliverable through the identical adapter.
    result = asyncio.run(adapter.send("D1", "⚠ The requested task failed", metadata=metadata))
    assert result.success
    assert len(sent) == (2 if enabled else 0) + 1
    if relay:
        assert sent[-1]["metadata"].get("thread_id") == thread_id
    else:
        assert sent[-1].get("thread_ts") == thread_id


def test_profile_config_isolated_for_callbacks_and_direct_warnings(tmp_path, monkeypatch):
    from gateway import run
    from gateway.warning_notifications import warning_notifications_enabled

    homes = [tmp_path / name for name in ("a", "b")]
    for home, value in zip(homes, ("true", "false")):
        home.mkdir()
        (home / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {value}}}")
    # The ambient default must never override the context-bound owner.
    monkeypatch.setattr(run, "_hermes_home", homes[1])
    source = SessionSource(platform=Platform.SLACK, chat_id="D1", chat_type="dm", user_id="U1")
    for home, expected in ((homes[0], False), (homes[1], True), (homes[0], False)):
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            config = _load_gateway_config()
            assert warning_notifications_enabled(source.platform, config) is expected
            from tests.gateway.test_warning_notifications import RecordingAdapter
            adapter = RecordingAdapter()
            adapter.send = AsyncMock()
            gateway = object.__new__(GatewayRunner)
            gateway._adapter_for_source = lambda source: adapter
            asyncio.run(gateway._hmwa_hygiene_notify(source, {}, "Compression failed", "failure"))
            assert adapter.send.await_count == int(expected)
