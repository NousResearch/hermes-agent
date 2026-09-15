"""Regression tests for session context during gateway plugin slash commands."""

import asyncio
from pathlib import Path

import pytest

from gateway.session import build_session_key
from gateway.session_context import (
    async_delivery_supported,
    clear_session_vars,
    get_session_env,
    session_history_delivery_supported,
    set_session_vars,
)
from tests.gateway.test_unknown_command import _make_event, _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("async_handler", [False, True])
async def test_plugin_command_handler_has_inbound_session_context_and_restores_it(
    monkeypatch, tmp_path, async_handler,
):
    from gateway.run import GatewayRunner
    from hermes_cli import plugins as plugins_mod
    from agent.runtime_cwd import resolve_context_cwd

    runner = _make_runner()
    runner._draining = False
    runner._set_session_env = GatewayRunner._set_session_env.__get__(runner)
    event = _make_event("/observe value")
    expected_key = build_session_key(event.source)
    observed = []

    outer_tokens = set_session_vars(
        platform="outer-platform",
        source="outer-source",
        chat_id="outer-chat",
        chat_type="outer-type",
        chat_name="outer-name",
        thread_id="outer-thread",
        user_id="outer-user",
        user_id_alt="outer-user-alt",
        user_name="outer-user-name",
        scope_id="outer-scope",
        session_key="outer-key",
        session_id="outer-session",
        ui_session_id="outer-ui-session",
        message_id="outer-message",
        profile="outer-profile",
        browser_control_principal="outer-principal",
        browser_control_transport_family="outer-transport",
        cron_session="1",
        parent_chat_id="outer-parent-chat",
        async_delivery=False,
        session_history_delivery="1",
        cwd=str(tmp_path),
    )

    session_names = (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_SOURCE",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_CHAT_TYPE",
        "HERMES_SESSION_CHAT_NAME",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_USER_ID_ALT",
        "HERMES_SESSION_USER_NAME",
        "HERMES_SESSION_SCOPE_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
        "HERMES_UI_SESSION_ID",
        "HERMES_SESSION_MESSAGE_ID",
        "HERMES_SESSION_PROFILE",
        "HERMES_BROWSER_CONTROL_PRINCIPAL",
        "HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY",
        "HERMES_CRON_SESSION",
        "HERMES_SESSION_PARENT_CHAT_ID",
    )
    outer_context = {name: get_session_env(name) for name in session_names}

    def observe():
        observed.append({
            "key": get_session_env("HERMES_SESSION_KEY"),
            "platform": get_session_env("HERMES_SESSION_PLATFORM"),
            "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
            "user_id": get_session_env("HERMES_SESSION_USER_ID"),
        })
        return "observed"

    async def async_observe(_args):
        await asyncio.sleep(0)
        return observe()

    def sync_observe(_args):
        return observe()

    monkeypatch.setattr(plugins_mod, "get_plugin_commands", lambda: {"observe": {}})
    monkeypatch.setattr(
        plugins_mod,
        "get_plugin_command_handler",
        lambda name: (async_observe if async_handler else sync_observe) if name == "observe" else None,
    )

    try:
        handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
            event, event.source, "observe", expected_key,
        )

        assert (handled, result, command) == (True, "observed", "observe")
        assert observed == [{
            "key": expected_key,
            "platform": "telegram",
            "chat_id": "c1",
            "user_id": "u1",
        }]
        assert {name: get_session_env(name) for name in session_names} == outer_context
        assert async_delivery_supported() is False
        assert session_history_delivery_supported() is True
        assert resolve_context_cwd() == Path(tmp_path)
    finally:
        clear_session_vars(outer_tokens)


@pytest.mark.asyncio
async def test_plugin_command_exception_restores_outer_session_context(monkeypatch, tmp_path):
    from gateway.run import GatewayRunner
    from hermes_cli import plugins as plugins_mod
    from agent.runtime_cwd import resolve_context_cwd

    runner = _make_runner()
    runner._draining = False
    runner._set_session_env = GatewayRunner._set_session_env.__get__(runner)
    event = _make_event("/explode")
    observed = []
    outer_tokens = set_session_vars(
        platform="outer-platform",
        session_key="outer-key",
        async_delivery=False,
        session_history_delivery="1",
        cwd=str(tmp_path),
    )

    def explode(_args):
        observed.append(get_session_env("HERMES_SESSION_KEY"))
        raise RuntimeError("plugin boom")

    monkeypatch.setattr(plugins_mod, "get_plugin_commands", lambda: {"explode": {}})
    monkeypatch.setattr(
        plugins_mod, "get_plugin_command_handler", lambda name: explode if name == "explode" else None,
    )

    try:
        result = await runner._hm_dispatch_quick_and_plugin_commands(
            event, event.source, "explode", build_session_key(event.source),
        )

        assert observed == [build_session_key(event.source)]
        assert result == (False, None, "explode")
        assert get_session_env("HERMES_SESSION_KEY") == "outer-key"
        assert get_session_env("HERMES_SESSION_PLATFORM") == "outer-platform"
        assert async_delivery_supported() is False
        assert session_history_delivery_supported() is True
        assert resolve_context_cwd() == Path(tmp_path)
    finally:
        clear_session_vars(outer_tokens)
