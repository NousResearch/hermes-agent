"""Composed delivery policy: real profile files, runner scopes and adapter transport."""
import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, ProcessingOutcome, SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from hermes_constants import get_hermes_home


class RecordingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.TELEGRAM)
        self.wire = []
        self.attempts = []
        self.completed = []
        self.fail_text_once = False
        self.fail_media = False

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.attempts.append((content, get_hermes_home()))
        if self.fail_text_once:
            self.fail_text_once = False
            return SendResult(success=False, error="invalid markup")
        self.wire.append((content, get_hermes_home(), metadata))
        return SendResult(success=True, message_id=f"sent-{len(self.wire)}")

    async def send_document(self, chat_id, file_path, **kwargs):
        self.attempts.append((file_path, get_hermes_home()))
        if self.fail_media:
            return SendResult(success=False, error="transport rejected attachment")
        return await super().send_document(chat_id, file_path, **kwargs)

    async def on_processing_complete(self, event, outcome):
        self.completed.append((event.text, outcome))


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    homes = {name: root / "profiles" / name for name in ("a", "b")}
    root.mkdir()
    # Opposite to the routed opt-in profile; ambient reads must not decide delivery.
    (root / "config.yaml").write_text("display: {suppress_warning_notifications: false}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    for home in homes.values():
        home.mkdir(parents=True)
    return root, homes


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("lane", ["text", "media", "error"])
async def test_post_handler_delivery_uses_owner_profile(profiles, monkeypatch, setting, lane, caplog):
    import agent.secret_scope as secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    root, homes = profiles
    for name, home in homes.items():
        value = setting if name == "a" else not bool(setting)
        config = "{}\n" if value is None else (
            f"display: {{suppress_warning_notifications: {str(value).lower()}}}\n")
        (home / "config.yaml").write_text(config)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True))
    adapter = RecordingAdapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    artifact = root / "result.pdf"
    artifact.write_bytes(b"test attachment")
    observed = []

    async def handler(event):
        # The real runner scope exits before BasePlatformAdapter delivers the result.
        with runner._media_delivery_scope_for_source(event.source):
            observed.append(get_hermes_home())
            if lane == "error":
                raise RuntimeError("provider detail retained in logs")
            if lane == "media":
                return f"MEDIA:{artifact}"
            return "⚠️ Error budget is the requested task result"

    adapter.set_message_handler(handler)
    for name in ("a", "b", "a"):
        suppressed = (setting is True) if name == "a" else not bool(setting)
        source = adapter.build_source(chat_id="42", user_id="42")
        source.profile = name
        event = MessageEvent(text=f"request-{name}", source=source, message_id=f"{lane}-{len(observed)}")
        adapter.wire.clear()
        adapter.attempts.clear()
        adapter.fail_text_once = lane == "text"
        adapter.fail_media = lane == "media"
        await adapter._process_message_background(event, f"agent:{name}:telegram:dm:42")
        assert get_hermes_home() == root
        assert event.internal is False
        assert event.allow_gateway_control is True
        assert observed[-1] == homes[name]
        if lane == "text":
            expected = "⚠️ Error budget is the requested task result"
            if not suppressed:
                expected = "(Response formatting failed, plain text:)\n\n" + expected
            assert [row[0] for row in adapter.wire] == [expected]
            assert adapter.completed[-1][1] is ProcessingOutcome.SUCCESS
        elif lane == "error":
            assert len(adapter.wire) == 1  # requested task still has a truthful outcome
            text = adapter.wire[0][0]
            if suppressed:
                assert text == "Sorry, I encountered an error."
            else:
                assert "provider detail retained in logs" in text
            assert adapter.completed[-1][1] is ProcessingOutcome.FAILURE
        else:
            assert bool(adapter.wire) is not suppressed
            assert adapter.completed[-1][1] is ProcessingOutcome.FAILURE
        assert all(home == homes[name] for _, home in adapter.attempts)
        assert all(row[1] == homes[name] for row in adapter.wire)
    assert len(observed) == 3
    assert {"text": "trying plain-text fallback", "media": "transport rejected attachment",
            "error": "provider detail retained in logs"}[lane] in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic_first", [False, True])
@pytest.mark.parametrize("setting", [None, False, True])
async def test_recursive_three_turn_chain_then_human(profiles, monkeypatch, diagnostic_first, setting):
    """Real runner recursion and final-event handoff, not a copied mute snapshot."""
    import gateway.run as gateway_run
    from gateway.session import SessionSource

    root, _ = profiles
    monkeypatch.setattr(gateway_run, "_hermes_home", root)
    (root / "config.yaml").write_text(
        "display: {tool_progress: off, streaming: false" +
        ("}" if setting is None else f", suppress_warning_notifications: {str(setting).lower()}}}")
    )
    from gateway.config import HomeChannel
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        enabled=True, home_channel=HomeChannel(platform=Platform.TELEGRAM, chat_id="42", name="test"))})
    runner = GatewayRunner(config)
    adapter = RecordingAdapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42")
    key = runner._session_key_for_source(source)
    events = [MessageEvent(
        text=f"turn-{i}", source=source, message_id=f"inbound-{i}",
        internal=(diagnostic_first if i % 2 == 0 else not diagnostic_first),
        metadata={"notification_category": "diagnostic"},
    ) for i in range(3)]
    # An untrusted category in metadata is deliberately present on human inputs.
    events.append(MessageEvent(text="next human", source=source, message_id="inbound-next",
                               metadata={"notification_category": "diagnostic"}))
    original_authority = [(e.internal, e.allow_gateway_control) for e in events]
    calls = []
    loop = asyncio.get_running_loop()

    class ScriptedAgent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            calls.append(message)
            index = len(calls) - 1
            if index < 2:
                ready = threading.Event()
                def enqueue():
                    adapter._pending_messages[key] = events[index + 1]
                    ready.set()
                loop.call_soon_threadsafe(enqueue)
                assert ready.wait(5), "event loop did not admit queued turn"
            return {"final_response": f"result-{index}", "messages": [], "api_calls": 1}

    fake = types.ModuleType("run_agent")
    fake.AIAgent = ScriptedAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"})
    # No platform credentials: authorize the fixture identity at the auth backend,
    # retaining admission, generation/lease lifecycle and queued-result handoff.
    authorized = []
    def authorize(source, **kwargs):
        authorized.append(source.user_id)
        return source.user_id == "42"
    runner._is_user_authorized_for_source = authorize
    adapter.set_message_handler(runner._handle_message)
    try:
        await adapter._process_message_background(events[0], key)
        assert len(calls) == 3, (calls, adapter.wire)
        expected = [f"result-{i}" for i in range(3)
                    if not (setting is True and original_authority[i][0])]
        assert [row[0] for row in adapter.wire] == expected
        await adapter._process_message_background(events[3], key)
        assert len(calls) == 4
        assert [row[0] for row in adapter.wire] == expected + ["result-3"]
        assert [(e.internal, e.allow_gateway_control) for e in events] == original_authority
        assert authorized[-1] == "42"  # next human still crosses authorization
        denied = MessageEvent(
            text="⚠️ diagnostic-looking human input", internal=False,
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="group", chat_type="group",
                                 user_id="denied"),
            metadata={"notification_category": "diagnostic", "_notification_reply_muted": True})
        before = list(adapter.wire)
        await adapter._process_message_background(denied, runner._session_key_for_source(denied.source))
        assert authorized[-1] == "denied"
        assert len(calls) == 4 and adapter.wire == before
        assert denied.internal is False and denied.allow_gateway_control is True
        assert get_hermes_home() == root
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("diagnostic", [False, True])
async def test_turn_error_preserves_diagnostic_bridge_and_failed_outcome(
        profiles, monkeypatch, setting, diagnostic, caplog):
    import logging
    from types import SimpleNamespace
    from agent.monitoring import gateway_health
    root, _ = profiles
    (root / "config.yaml").write_text(
        "{}" if setting is None else
        f"display: {{suppress_warning_notifications: {str(setting).lower()}}}")
    emitted = []
    monkeypatch.setattr(gateway_health.emitter, "get_emitter",
                        lambda: SimpleNamespace(emit=emitted.append))
    logger = logging.getLogger("gateway.platforms.base")
    bridge = gateway_health.GatewayDiagnosticLogHandler(profile="fixture", version="fixture")
    logger.addHandler(bridge)
    adapter = RecordingAdapter()
    async def handler(event):
        raise RuntimeError("same source failure")
    adapter.set_message_handler(handler)
    source = adapter.build_source(chat_id="42", user_id="42")
    event = MessageEvent(text="work", source=source, internal=diagnostic,
                         metadata={"notification_category": "diagnostic"})
    try:
        key = "agent:default:telegram:dm:42"
        assert adapter._start_session_processing(event, key)
        await adapter._session_tasks[key]
        assert adapter.completed[-1][1] is ProcessingOutcome.FAILURE
        assert "same source failure" in caplog.text
        errors = [e for e in emitted if e.name == "gateway.log.error"]
        assert len(errors) == 1
        assert errors[0].severity == "error" and errors[0].profile == "fixture"
        if diagnostic and setting is True:
            assert adapter.wire == []
        else:
            assert len(adapter.wire) == 1
            assert ("same source failure" in adapter.wire[0][0]) is (setting is not True)
        assert not adapter._active_sessions
    finally:
        logger.removeHandler(bridge)
        await adapter.cancel_background_tasks()
