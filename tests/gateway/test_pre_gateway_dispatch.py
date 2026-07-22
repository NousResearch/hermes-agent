"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


@pytest.mark.asyncio
async def test_hook_skip_short_circuits_dispatch(monkeypatch):
    """A plugin returning {'action': 'skip'} drops the message before auth."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_rewrite_replaces_event_text(monkeypatch):
    """A plugin returning {'action': 'rewrite', 'text': ...} mutates event.text."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen_text = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "REWRITTEN"}]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        seen_text["value"] = event.text
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("original"))

    assert seen_text.get("value") == "REWRITTEN"


@pytest.mark.asyncio
async def test_hook_allow_falls_through_to_auth(monkeypatch):
    """A plugin returning {'action': 'allow'} continues to normal dispatch."""
    _clear_auth_env(monkeypatch)
    # No allowed users set → auth fails → pairing flow triggers.
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "allow"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    # auth chain ran → pairing code was generated
    assert result is None
    runner.pairing_store.generate_code.assert_called_once()


@pytest.mark.asyncio
async def test_required_gate_rejects_an_unrelated_allow_before_agent_dispatch(monkeypatch):
    """Only the declared gate owner can approve a required ingress gate."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="unrelated", source="user"),
    ).register_hook("pre_gateway_dispatch", lambda **_kwargs: {"action": "allow"})
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    reached_agent = False

    async def capture(*_args):
        nonlocal reached_agent
        reached_agent = True
        return "unexpected"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    event = _make_event("guarded")
    event.required_dispatch_gate = "line-group-context"

    assert await runner._handle_message(event) is None
    assert reached_agent is False


@pytest.mark.asyncio
async def test_required_gate_accepts_only_matching_owner_and_keeps_enrichment(monkeypatch):
    """The declared owner may enrich only after an explicit gate approval."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()

    def approve(event, **_kwargs):
        event.channel_prompt = "approved context"
        event.auto_skill = ["research", "calendar"]
        return {"action": "approve", "gate": "line-group-context"}

    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        approve,
        gate_owner="line-group-context",
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    observed = {}

    async def capture(event, *_args):
        observed["prompt"] = event.channel_prompt
        observed["skills"] = event.auto_skill
        return "ok"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    event = _make_event("guarded")
    event.required_dispatch_gate = "line-group-context"
    event.platform_event_id = "signed-event-id"
    event.platform_event_timestamp_ms = 1_784_678_400_000

    assert await runner._handle_message(event) == "ok"
    assert observed == {
        "prompt": "approved context",
        "skills": ["research", "calendar"],
    }


@pytest.mark.asyncio
async def test_later_allow_callback_cannot_erase_gate_owner_enrichment(monkeypatch):
    """Isolated later hooks preserve owner context when they make no edits."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()

    def approve(event, **_kwargs):
        event.channel_prompt = "approved context"
        event.auto_skill = ["mochiwiz-bookkeeper"]
        return {"action": "approve", "gate": "line-group-context"}

    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook("pre_gateway_dispatch", approve, gate_owner="line-group-context")
    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="unrelated", source="user"),
    ).register_hook("pre_gateway_dispatch", lambda **_kwargs: {"action": "allow"})
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    observed = {}

    async def capture(event, *_args):
        observed["prompt"] = event.channel_prompt
        observed["skills"] = event.auto_skill
        return "ok"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    event = _make_event("guarded")
    event.required_dispatch_gate = "line-group-context"
    event.platform_event_id = "signed-event-id"
    event.platform_event_timestamp_ms = 1_784_678_400_000

    assert await runner._handle_message(event) == "ok"
    assert observed == {
        "prompt": "approved context",
        "skills": ["mochiwiz-bookkeeper"],
    }


@pytest.mark.asyncio
async def test_required_gate_rejects_owner_source_mutation(monkeypatch):
    """A hook cannot replace the adapter-authenticated sender or routing lane."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()

    def mutate_source(event, **_kwargs):
        event.source.user_id = "forged-user"
        return {"action": "approve", "gate": "line-group-context"}

    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        mutate_source,
        gate_owner="line-group-context",
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    reached_agent = False

    async def capture(*_args):
        nonlocal reached_agent
        reached_agent = True
        return "unexpected"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    event = _make_event("guarded")
    event.required_dispatch_gate = "line-group-context"
    event.platform_event_id = "signed-event-id"
    event.platform_event_timestamp_ms = 1_784_678_400_000

    assert await runner._handle_message(event) is None
    assert reached_agent is False


@pytest.mark.asyncio
async def test_guarded_reconstruction_keeps_the_one_core_issued_resolution(monkeypatch):
    """A trusted queue-style reconstruction does not invoke onboarding twice."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    approvals = 0

    def approve(**_kwargs):
        nonlocal approvals
        approvals += 1
        return {"action": "approve", "gate": "line-group-context"}

    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        approve,
        gate_owner="line-group-context",
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    event = _make_event("/queue original")
    event.required_dispatch_gate = "line-group-context"
    event.platform_event_id = "signed-event-id"
    event.platform_event_timestamp_ms = 1_784_678_400_000
    session_key = runner._session_key_for_source(event.source)
    resolved = await runner._resolve_gateway_ingress(event, session_key)
    assert resolved is not None
    reconstructed = runner._rebind_ingress_event(
        [resolved],
        dataclasses.replace(resolved, text="queued follow-up"),
        session_key,
    )
    assert reconstructed is not None

    async def capture(*_args):
        return "ok"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    assert await runner._handle_message(reconstructed) == "ok"
    assert approvals == 1


@pytest.mark.asyncio
async def test_guarded_busy_merge_does_not_resolve_a_third_time(monkeypatch):
    """Gateway's own busy fallback rebinds a guarded merged pending event."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    approvals = 0

    def approve(**_kwargs):
        nonlocal approvals
        approvals += 1
        return {"action": "approve", "gate": "line-group-context"}

    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        approve,
        gate_owner="line-group-context",
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    adapter._pending_messages = {}
    first = _make_event("first")
    second = _make_event("second")
    for index, event in enumerate((first, second), start=1):
        event.required_dispatch_gate = "line-group-context"
        event.platform_event_id = f"signed-event-{index}"
        event.platform_event_timestamp_ms = 1_784_678_400_000 + index
    session_key = runner._session_key_for_source(first.source)
    first_resolved = await runner._resolve_gateway_ingress(first, session_key)
    second_resolved = await runner._resolve_gateway_ingress(second, session_key)
    assert first_resolved is not None and second_resolved is not None
    adapter._pending_messages[session_key] = first_resolved

    merged = runner._merge_pending_ingress_event(
        adapter, session_key, second_resolved, merge_text=True
    )
    assert merged is not None
    assert merged.text == "first\nsecond"

    async def capture(*_args):
        return "ok"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    assert await runner._handle_message(merged) == "ok"
    assert approvals == 2


@pytest.mark.asyncio
async def test_guarded_resolution_is_consumed_before_a_second_runner_dispatch(monkeypatch):
    """The same core-issued guarded event cannot invoke the agent twice."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        lambda **_kwargs: {"action": "approve", "gate": "line-group-context"},
        gate_owner="line-group-context",
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    event = _make_event("guarded")
    event.required_dispatch_gate = "line-group-context"
    event.platform_event_id = "signed-event-id"
    event.platform_event_timestamp_ms = 1_784_678_400_000
    session_key = runner._session_key_for_source(event.source)
    resolved = await runner._resolve_gateway_ingress(event, session_key)
    assert resolved is not None
    dispatches = 0

    async def capture(*_args):
        nonlocal dispatches
        dispatches += 1
        return "ok"

    runner._handle_message_with_agent = capture  # noqa: SLF001
    assert await runner._handle_message(resolved) == "ok"
    assert await runner._handle_message(resolved) is None
    assert dispatches == 1


@pytest.mark.asyncio
async def test_guarded_busy_queue_transfers_a_fresh_dispatchable_child(monkeypatch):
    """Queueing a consumed busy event transfers—not reuses—its approval."""
    _clear_auth_env(monkeypatch)
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    PluginContext(
        manager=manager,
        manifest=PluginManifest(name="context", source="user"),
    ).register_hook(
        "pre_gateway_dispatch",
        lambda **_kwargs: {"action": "approve", "gate": "line-group-context"},
        gate_owner="line-group-context",
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    runner, adapter = _make_runner(Platform.WHATSAPP)
    adapter._pending_messages = {}
    event = _make_event("queued")
    event.required_dispatch_gate = "line-group-context"
    event.platform_event_id = "signed-event-id"
    event.platform_event_timestamp_ms = 1_784_678_400_000
    session_key = runner._session_key_for_source(event.source)
    resolved = await runner._resolve_gateway_ingress(event, session_key)
    assert resolved is not None
    assert runner._consume_guarded_ingress_resolution(resolved, session_key)

    runner._enqueue_fifo(session_key, resolved, adapter)

    queued = adapter._pending_messages[session_key]
    assert queued is not resolved
    assert runner._consume_guarded_ingress_resolution(queued, session_key)
    assert not runner._consume_guarded_ingress_resolution(queued, session_key)


@pytest.mark.asyncio
async def test_hook_exception_does_not_break_dispatch(monkeypatch):
    """A raising plugin hook does not break the gateway."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = None

    # Should not raise; falls through to auth chain.
    result = await runner._handle_message(_make_event("hi"))
    assert result is None


@pytest.mark.asyncio
async def test_internal_events_bypass_hook(monkeypatch):
    """Internal events (event.internal=True) skip the plugin hook entirely."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    event.internal = True

    # Even though the hook would say skip, internal events bypass it.
    await runner._handle_message(event)
    assert called["count"] == 0
