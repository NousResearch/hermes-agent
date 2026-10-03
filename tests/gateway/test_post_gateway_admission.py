"""Regression tests for the post-admission consuming gateway extension (#129958).

Exercises the real ``GatewayRunner._handle_message`` pipeline (not a
reference model) plus the pure ``decide_post_admission`` / context helpers.
Covers: rejected/negative admission, pause/drain, control priority, FIFO
and rescue anchor, multiplex isolation, PASS/CONSUMED/FAILED agent counts,
fail-closed malformed/exception/timeout/cancellation, receipt requirement,
destination binding and session-ownership lifecycle.
"""

import asyncio
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key

POST_HOOK = "post_gateway_admission"
PRE_HOOK = "pre_gateway_dispatch"


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


def _make_source(**over) -> SessionSource:
    base = dict(
        platform=Platform.WHATSAPP,
        user_id="15551234567@s.whatsapp.net",
        chat_id="15551234567@s.whatsapp.net",
        user_name="tester",
        chat_type="dm",
    )
    base.update(over)
    return SessionSource(**base)


def _make_event(text="hello", **over) -> MessageEvent:
    src = over.pop("source", None) or _make_source()
    return MessageEvent(text=text, message_id="m1", source=src, **over)


def _make_runner():
    from gateway.run import GatewayRunner

    config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)})
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    # Authorize + admit by default; individual tests override to deny.
    runner._is_user_authorized_for_source = lambda _s: True  # noqa: SLF001
    runner._admit_bot_message_for_source = lambda _s: True  # noqa: SLF001
    runner._delivery_adapter_for = lambda _s: adapter  # noqa: SLF001
    runner._intake_adapter_for = lambda _s: adapter  # noqa: SLF001
    runner._rescue_orphaned_overflow = lambda _k, _a: None  # noqa: SLF001
    runner._enqueue_fifo = MagicMock()  # noqa: SLF001
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")  # noqa: SLF001
    runner._run_post_turn_hooks = AsyncMock()  # noqa: SLF001
    return runner, adapter


def _allow_all(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")


def _post_only(monkeypatch, post_fn, calls=None):
    """Patch the plugin hook seam: pre hook allows, post hook delegates to *post_fn*.

    Both ``pre_gateway_dispatch`` and ``post_gateway_admission`` flow through
    the same ``ainvoke_hook`` seam; counting every call would conflate the
    pre-auth policy gate with the business consumer. This helper makes the
    pre hook a no-op allow and only counts/invokes for the post hook.
    """

    async def _hook(name, **kwargs):
        if name == PRE_HOOK:
            return []
        if name == POST_HOOK:
            if calls is not None:
                calls["n"] += 1
            return await post_fn(name, **kwargs) if asyncio.iscoroutinefunction(post_fn) else post_fn(name, **kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.ainvoke_hook", _hook)
    return _hook


# ── pure helpers ──────────────────────────────────────────────────────────

def test_decide_pass_consumed_failed():
    from gateway.run_inbound_consumer import decide_post_admission

    assert decide_post_admission([])[0] == "pass"
    assert decide_post_admission([None])[0] == "pass"
    assert decide_post_admission([{"action": "pass"}])[0] == "pass"
    assert decide_post_admission([{"action": "allow"}])[0] == "pass"

    d, reply, receipt, _ = decide_post_admission(
        [{"action": "consumed", "receipt": "r1", "reply": "ack"}]
    )
    assert (d, reply, receipt) == ("consumed", "ack", "r1")

    d, reply, _, _ = decide_post_admission([{"action": "failed", "reply": "oops"}])
    assert d == "failed" and reply == "oops"

    # First decisive wins; ambiguous config warns but stays deterministic.
    d, reply, receipt, _ = decide_post_admission(
        [
            {"action": "consumed", "receipt": "first", "reply": "one"},
            {"action": "consumed", "receipt": "second", "reply": "two"},
        ]
    )
    assert (d, receipt) == ("consumed", "first")


def test_decide_fail_closed_on_malformed():
    from gateway.run_inbound_consumer import decide_post_admission

    assert decide_post_admission([{"action": "weird"}])[0] == "failed"
    assert decide_post_admission(["oops"])[0] == "failed"
    assert decide_post_admission([{"action": "consumed"}])[0] == "failed"
    assert decide_post_admission([{"action": "consumed", "receipt": "  "}])[0] == "failed"
    assert decide_post_admission([{"action": "consumed", "receipt": "r", "reply": 123}])[0] == "failed"
    # Timeout path contributes {"action": "block"} → FAILED.
    assert decide_post_admission([{"action": "block", "message": "timeout"}])[0] == "failed"


def test_context_is_versioned_and_immutable():
    from gateway.run_inbound_consumer import (
        POST_ADMISSION_VERSION,
        build_post_admission_context,
    )

    event = _make_event("hi")
    ctx = build_post_admission_context(event, event.source, "k1", reply_anchor="m1")
    assert ctx["version"] == POST_ADMISSION_VERSION
    assert ctx["platform"] == "whatsapp"
    assert ctx["session_key"] == "k1"
    assert ctx["reply_anchor"] == "m1"
    assert isinstance(ctx, MappingProxyType)
    with pytest.raises(TypeError):
        ctx["platform"] = "evil"
    with pytest.raises(TypeError):
        ctx["conversation"]["chat_id"] = "evil"


# ── admission gates: zero consumers, zero business writes ─────────────────

@pytest.mark.asyncio
async def test_rejected_source_invokes_zero_consumers(monkeypatch):
    """RED from the issue: rejected source → zero consumers, zero business writes."""
    _clear_auth_env(monkeypatch)
    calls = {"n": 0}

    async def _post(name, **kwargs):
        return [{"action": "consumed", "receipt": "r", "reply": "ack"}]

    _post_only(monkeypatch, _post, calls)
    runner, _ = _make_runner()
    runner._is_user_authorized_for_source = lambda _s: False  # noqa: SLF001

    result = await runner._handle_message(_make_event("hello"))

    assert result is None
    assert calls["n"] == 0
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_negative_admission_each_invokes_zero_consumers(monkeypatch):
    _allow_all(monkeypatch)
    from gateway.config import Platform as _P

    async def _post(name, **kwargs):
        raise AssertionError("business consumer must not run")

    _post_only(monkeypatch, _post)

    # Internal / system event still runs the agent but never the consumer.
    runner, _ = _make_runner()
    internal = _make_event("sys")
    internal.internal = True
    await runner._handle_message(internal)
    runner._handle_message_with_agent.assert_awaited()

    # Bot-loop rejection.
    runner2, _ = _make_runner()
    runner2._admit_bot_message_for_source = lambda _s: False  # noqa: SLF001
    assert await runner2._handle_message(_make_event("hi")) is None
    runner2._handle_message_with_agent.assert_not_awaited()

    # Unserved profile route.
    runner3, _ = _make_runner()
    ev = _make_event("hi")
    ev.source.profile_route_rejected = True
    assert await runner3._handle_message(ev) is None
    runner3._handle_message_with_agent.assert_not_awaited()

    # Ignored Slack channel.
    runner4 = _make_runner()[0]
    import gateway.config as _cfg

    runner4.config = _cfg.GatewayConfig(
        platforms={_P.SLACK: _cfg.PlatformConfig(enabled=True, extra={"ignored_channels": ["C123"]})},
    )
    runner4.adapters = {_P.SLACK: SimpleNamespace(send=AsyncMock(), config=SimpleNamespace(extra={"ignored_channels": ["C123"]}))}
    runner4._delivery_adapter_for = lambda _s: runner4.adapters[_P.SLACK]  # noqa: SLF001
    runner4._intake_adapter_for = lambda _s: runner4.adapters[_P.SLACK]  # noqa: SLF001
    slack_src = SessionSource(
        platform=_P.SLACK, user_id="U1", chat_id="C123", chat_type="group",
    )
    assert await runner4._handle_message(MessageEvent(text="hi", message_id="m9", source=slack_src)) is None
    runner4._handle_message_with_agent.assert_not_awaited()


# ── pause / drain ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pause_and_drain_suppress_consumer_but_commands_still_route(monkeypatch):
    _allow_all(monkeypatch)
    calls = {"n": 0}

    async def _post(name, **kwargs):
        return [{"action": "consumed", "receipt": "r", "reply": "ack"}]

    _post_only(monkeypatch, _post, calls)
    runner, _ = _make_runner()
    runner._hm_estop_gate = lambda _e, _s, _i: "paused-notice"  # noqa: SLF001

    assert await runner._handle_message(_make_event("business")) == "paused-notice"
    assert calls["n"] == 0
    runner._handle_message_with_agent.assert_not_awaited()

    # Recognized control traffic still reaches its owner under pause.
    # /start is DB-free (unlike /status) so the bare test runner can route it.
    runner2, _ = _make_runner()
    runner2._hm_estop_gate = lambda _e, _s, _i: None  # noqa: SLF001
    result = await runner2._handle_message(_make_event("/start"))
    assert calls["n"] == 0  # commands never reach the business consumer
    runner2._handle_message_with_agent.assert_not_awaited()
    assert result is not None

    # External drain refuses new business without invoking the consumer.
    runner3, _ = _make_runner()
    runner3._external_drain_active = True
    assert "drain" in (await runner3._handle_message(_make_event("work")) or "").lower()
    assert calls["n"] == 0
    runner3._handle_message_with_agent.assert_not_awaited()


# ── control priority ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_control_traffic_never_reaches_consumer(monkeypatch):
    _allow_all(monkeypatch)
    calls = {"n": 0}

    async def _post(name, **kwargs):
        return [{"action": "consumed", "receipt": "r", "reply": "ack"}]

    _post_only(monkeypatch, _post, calls)

    # Slash command (/start is DB-free for the bare runner).
    runner, _ = _make_runner()
    await runner._handle_message(_make_event("/start"))
    assert calls["n"] == 0

    # Pending-reply intercept owns the message.
    runner2, _ = _make_runner()
    runner2._hm_pending_reply_intercepts = AsyncMock(return_value="")  # noqa: SLF001
    await runner2._handle_message(_make_event("answer"))
    assert calls["n"] == 0
    runner2._handle_message_with_agent.assert_not_awaited()

    # Running session steers/queues; busy ownership is core-decided.
    runner3, _ = _make_runner()
    key = build_session_key(_make_source())
    runner3._session_state(key).turn.agent = MagicMock()
    runner3._session_state(key).turn.agent._supports_active_turn_redirect = False
    runner3._effective_busy_input_mode = lambda _s: "interrupt"  # noqa: SLF001
    runner3._steer_running_agent = lambda *_a: False  # noqa: SLF001
    await runner3._handle_message(_make_event("steer me"))
    assert calls["n"] == 0
    runner3._handle_message_with_agent.assert_not_awaited()


# ── FIFO / rescue anchor ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_busy_messages_defer_and_rescue_presents_selected_anchor(monkeypatch):
    _allow_all(monkeypatch)
    seen = {}

    async def _post(name, **kwargs):
        ctx = kwargs.get("context", {})
        seen["event_id"] = ctx.get("event_id")
        seen["anchor"] = ctx.get("reply_anchor")
        return [{"action": "consumed", "receipt": "r1", "reply": "ack"}]

    _post_only(monkeypatch, _post)
    runner, _ = _make_runner()

    rescued_src = _make_source()
    rescued = MessageEvent(text="oldest orphan", message_id="old-1", source=rescued_src)
    incoming = _make_event("new arrival")
    runner._hm_rescue_orphaned_fifo = lambda e, s, i, k: (rescued, rescued_src, False)  # noqa: SLF001

    result = await runner._handle_message(incoming)
    assert result == "ack"
    assert seen["event_id"] == "old-1"
    runner._handle_message_with_agent.assert_not_awaited()


# ── PASS / CONSUMED / FAILED ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pass_runs_agent_once_consumed_and_failed_suppress(monkeypatch):
    _allow_all(monkeypatch)

    async def _pass(name, **kwargs):
        return [{"action": "pass"}]

    _post_only(monkeypatch, _pass)
    runner, _ = _make_runner()
    assert await runner._handle_message(_make_event("work")) == "agent-ok"
    runner._handle_message_with_agent.assert_awaited_once()

    async def _consumed(name, **kwargs):
        return [{"action": "consumed", "receipt": "req-1", "reply": "queued ✓"}]

    _post_only(monkeypatch, _consumed)
    runner2, _ = _make_runner()
    assert await runner2._handle_message(_make_event("follow up")) == "queued ✓"
    runner2._handle_message_with_agent.assert_not_awaited()

    async def _failed(name, **kwargs):
        return [{"action": "failed", "reply": "cannot queue right now"}]

    _post_only(monkeypatch, _failed)
    runner3, _ = _make_runner()
    assert await runner3._handle_message(_make_event("follow up")) == "cannot queue right now"
    runner3._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_fail_closed_variants_suppress_agent(monkeypatch):
    _allow_all(monkeypatch)

    async def _malformed(name, **kwargs):
        return [{"action": "definitely-not-real"}]

    _post_only(monkeypatch, _malformed)
    runner, _ = _make_runner()
    result = await runner._handle_message(_make_event("hi"))
    assert "failed" in result.lower()
    runner._handle_message_with_agent.assert_not_awaited()

    async def _boom(name, **kwargs):
        raise RuntimeError("consumer exploded")

    _post_only(monkeypatch, _boom)
    runner2, _ = _make_runner()
    result2 = await runner2._handle_message(_make_event("hi"))
    assert "failed" in result2.lower()
    assert "exploded" not in result2
    runner2._handle_message_with_agent.assert_not_awaited()

    # Timeout path surfaces {"action": "block"} from the bounded dispatcher.
    # Fail-closed means the agent is suppressed; the dispatcher-provided
    # message becomes the short user notice.
    async def _timeout(name, **kwargs):
        return [{"action": "block", "message": "timed out"}]

    _post_only(monkeypatch, _timeout)
    runner3, _ = _make_runner()
    result3 = await runner3._handle_message(_make_event("hi"))
    assert result3 == "timed out"
    runner3._handle_message_with_agent.assert_not_awaited()

    # Cancellation never starts inline execution.
    async def _cancel(name, **kwargs):
        raise asyncio.CancelledError()

    _post_only(monkeypatch, _cancel)
    runner4, _ = _make_runner()
    with pytest.raises(asyncio.CancelledError):
        await runner4._handle_message(_make_event("hi"))
    runner4._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_consumed_without_receipt_fails_closed(monkeypatch):
    _allow_all(monkeypatch)

    async def _hook(name, **kwargs):
        return [{"action": "consumed", "reply": "ack without commit"}]

    _post_only(monkeypatch, _hook)
    runner, _ = _make_runner()
    result = await runner._handle_message(_make_event("do work"))
    assert "failed" in result.lower()
    assert "without commit" not in result
    runner._handle_message_with_agent.assert_not_awaited()


# ── destination binding + lifecycle ───────────────────────────────────────

@pytest.mark.asyncio
async def test_destination_mutation_is_restored(monkeypatch):
    _allow_all(monkeypatch)
    from gateway.run_inbound_consumer import invoke_post_admission_hook

    async def _hook(name, **kwargs):
        event = kwargs.get("event")
        if name == POST_HOOK:
            event.source.chat_id = "EVIL-CHAT"
            event.source.platform = Platform.TELEGRAM
            event.text = "mutated text is allowed"
            return [{"action": "pass"}]
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.ainvoke_hook", _hook)
    runner, _ = _make_runner()
    event = _make_event("original")
    handled, reply, out_event, out_source = await invoke_post_admission_hook(
        runner, event, event.source, "k"
    )
    assert handled is False and reply is None
    assert out_source.chat_id == "15551234567@s.whatsapp.net"
    assert out_source.platform is Platform.WHATSAPP
    assert out_event.source.chat_id == "15551234567@s.whatsapp.net"
    assert out_event.source.platform is Platform.WHATSAPP


@pytest.mark.asyncio
async def test_failure_does_not_leak_session_ownership(monkeypatch):
    _allow_all(monkeypatch)
    calls = {"n": 0}

    async def _flaky(name, **kwargs):
        # Wrapper already counted this post-hook invocation in calls["n"].
        if calls["n"] == 1:
            raise RuntimeError("first turn fails")
        return [{"action": "pass"}]

    _post_only(monkeypatch, _flaky, calls)
    runner, _ = _make_runner()
    first = await runner._handle_message(_make_event("one"))
    assert "failed" in first.lower()
    second = await runner._handle_message(_make_event("two"))
    assert second == "agent-ok"
    assert runner._handle_message_with_agent.await_count == 1


@pytest.mark.asyncio
async def test_context_carries_routed_profile_and_anchor(monkeypatch):
    _allow_all(monkeypatch)
    seen = {}

    async def _hook(name, **kwargs):
        seen.update(kwargs.get("context", {}))
        return [{"action": "pass"}]

    _post_only(monkeypatch, _hook)
    runner, _ = _make_runner()
    src = _make_source(profile="satellite")
    await runner._handle_message(MessageEvent(text="hi", message_id="mid-7", source=src))
    assert seen["profile"] == "satellite"
    assert seen["event_id"] == "mid-7"
    assert seen["reply_anchor"] == "mid-7"
    runner._handle_message_with_agent.assert_awaited_once()


def test_hook_registered_in_valid_hooks_and_fail_closed():
    """The hook is a valid registration and times out fail-closed."""
    from hermes_cli.plugins import VALID_HOOKS
    from hermes_cli.plugins_dispatch import _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS

    assert "post_gateway_admission" in VALID_HOOKS
    assert "post_gateway_admission" in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS


def test_profile_scoped_managers_isolate_consumers(tmp_path, monkeypatch):
    """Two homes with real discovery never cross consumers (A→B→A)."""
    import textwrap

    import hermes_yaml as yaml
    from hermes_cli import plugins as plugins_mod

    def _home(name, marker):
        home = tmp_path / name
        plugindir = home / "plugins" / "consumer"
        plugindir.mkdir(parents=True)
        (plugindir / "plugin.yaml").write_text(
            yaml.safe_dump({"name": "consumer", "version": "0.1.0", "description": "test"}),
            encoding="utf-8",
        )
        (plugindir / "__init__.py").write_text(
            "def register(ctx):\n"
            f"    ctx.register_hook('post_gateway_admission', lambda **kw: {{'action': 'consumed', 'receipt': '{marker}', 'reply': '{marker}'}})\n",
            encoding="utf-8",
        )
        cfg = {"plugins": {"enabled": ["consumer"]}}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return home

    home_a, home_b = _home("homeA", "A"), _home("homeB", "B")
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        mgr_a = plugins_mod.get_plugin_manager()
        mgr_a.discover_and_load(force=True)
        res_a = mgr_a.invoke_hook("post_gateway_admission", event="e")
        assert res_a == [{"action": "consumed", "receipt": "A", "reply": "A"}]

        monkeypatch.setenv("HERMES_HOME", str(home_b))
        mgr_b = plugins_mod.get_plugin_manager()
        assert mgr_b is not mgr_a
        mgr_b.discover_and_load(force=True)
        res_b = mgr_b.invoke_hook("post_gateway_admission", event="e")
        assert res_b == [{"action": "consumed", "receipt": "B", "reply": "B"}]

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert plugins_mod.get_plugin_manager() is mgr_a
        assert mgr_a.invoke_hook("post_gateway_admission", event="e") == [
            {"action": "consumed", "receipt": "A", "reply": "A"}
        ]
    finally:
        plugins_mod._reset_plugin_managers_for_tests()
