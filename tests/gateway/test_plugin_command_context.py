"""Synthetic integration events against actual GatewayRunner, never live users."""
import asyncio
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_runner(tmp_path, monkeypatch):
    home = tmp_path / "engineering-fixture"
    home.mkdir()
    (home / "config.yaml").write_text("plugins:\n  enabled: []\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111,222")
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "false")
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "")
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(home)
    from hermes_cli import plugins
    from gateway import session_context as sc
    from gateway.config import Platform
    from gateway.session import SessionStore
    import gateway.run as gateway_run
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    path = Path(gateway_run.__file__).parents[1] / "tests/gateway/test_gateway_command_dispatch_minimal.py"
    spec = importlib.util.spec_from_file_location("native_minimal_runner_fixture", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    runner, adapter = helper._make_runner()
    runner.config.platforms[Platform.DISCORD] = runner.config.platforms[Platform.TELEGRAM]
    runner.adapters = {Platform.DISCORD: adapter}
    adapter.authorization_is_upstream = False
    adapter.enforces_own_access_policy = False
    adapter.supports_async_delivery = True
    runner._is_user_authorized = gateway_run.GatewayRunner._is_user_authorized.__get__(runner)
    runner._pairing_store = None
    runner.pairing_store = None
    runner.session_store = SessionStore(home / "sessions", runner.config)
    manager = plugins.PluginManager(scope_key=str(home))
    manager._discovered = True
    monkeypatch.setitem(plugins._plugin_managers_by_home, home, manager)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    ctx = plugins.PluginContext(plugins.PluginManifest(name="native-fixture", key="native-fixture"), manager)
    sc.reset_session_vars()
    yield SimpleNamespace(runner=runner, adapter=adapter, manager=manager, ctx=ctx, home=home, sc=sc)
    manager.unload()
    sc.reset_session_vars()
    reset_hermes_home_override(token)


def incoming(text="/probe-native inspect", user="111", chat="333", message="444"):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource
    return MessageEvent(text=text, message_type=MessageType.COMMAND,
                        source=SessionSource(platform=Platform.DISCORD, user_id=user,
                                             chat_id=chat, chat_type="group", message_id=message),
                        message_id=message, internal=False)


@pytest.mark.asyncio
async def test_authenticated_ingress_overrides_inheritance_and_restores_context(native_runner, monkeypatch):
    f = native_runner
    from hermes_cli.plugin_commands import get_command_context
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "foreign-daemon")
    f.sc.set_session_vars(user_id="inherited-sibling", session_id="foreign-session")
    event = incoming()
    entry = f.runner.session_store.get_or_create_session(event.source)
    async def handler(raw, *, context=None):
        with pytest.raises(FrozenInstanceError):
            context.user_id = "forged"
        await asyncio.sleep(0)
        assert get_command_context() is context
        return json.dumps({"actor": context.user_id, "key": context.session_key,
                           "session": context.session_id, "message": context.message_id,
                           "bound_actor": f.sc.get_session_env("HERMES_SESSION_USER_ID"),
                           "bound_session": f.sc.get_session_env("HERMES_SESSION_ID")})
    f.ctx.register_command("probe-native", handler, contextual=True)
    result = json.loads(await f.runner._handle_message(event))
    assert result == {"actor": "111", "key": entry.session_key, "session": entry.session_id,
                      "message": "444", "bound_actor": "111", "bound_session": entry.session_id}
    assert get_command_context() is None
    assert f.sc._SESSION_USER_ID.get() is f.sc._UNSET


@pytest.mark.asyncio
async def test_denied_actor_and_slash_policy_never_execute_plugin(native_runner):
    f = native_runner
    calls = []
    def handler(raw, *, context=None):
        calls.append(context)
        raise AssertionError("must not execute")
    f.ctx.register_command("probe-native", handler, contextual=True)
    assert await f.runner._handle_message(incoming(user="999")) is None
    f.runner._check_slash_access = lambda source, command: "admin-only"
    assert await f.runner._handle_message(incoming()) == "admin-only"
    assert calls == []
    assert not (f.home / "plugin-data").exists()


@pytest.mark.asyncio
async def test_command_error_is_fail_closed_and_concurrent_identity_is_task_local(native_runner):
    f = native_runner
    from hermes_cli.plugin_commands import get_command_context
    barrier = asyncio.Event()
    seen = []
    async def handler(raw, *, context=None):
        seen.append(context.user_id)
        if len(seen) == 2:
            barrier.set()
        await barrier.wait()
        assert f.sc.get_session_env("HERMES_SESSION_USER_ID") == context.user_id
        assert get_command_context() is context
        if context.user_id == "111":
            raise RuntimeError("secret=DO_NOT_DISCLOSE")
        return context.user_id
    f.ctx.register_command("probe-native", handler, contextual=True)
    results = await asyncio.gather(f.runner._handle_message(incoming()),
                                  f.runner._handle_message(incoming(user="222", chat="555", message="666")))
    assert "failed" in results[0] and "DO_NOT_DISCLOSE" not in results[0]
    assert results[1] == "222"
    assert get_command_context() is None


@pytest.mark.asyncio
async def test_contextual_command_is_reviewed_before_busy_queue_but_legacy_is_not(native_runner):
    f = native_runner
    from hermes_cli.commands import should_bypass_active_session
    from hermes_cli.plugin_commands import PluginAgentDispatch
    f.ctx.register_command("probe-native", lambda raw, context=None: PluginAgentDispatch("Reviewed operation"), contextual=True)
    f.ctx.register_command("legacy-probe", lambda raw: "legacy:" + raw)
    assert should_bypass_active_session("probe-native")
    assert not should_bypass_active_session("legacy-probe")
    f.runner._busy_input_mode = "queue"
    f.runner._is_session_running = lambda key: True
    event = incoming()
    key = f.runner._session_key_for_source(event.source)
    from gateway.run import _AGENT_PENDING_SENTINEL
    f.runner._running_agents[key] = _AGENT_PENDING_SENTINEL
    assert await f.runner._handle_message(event) is None
    pending = f.adapter._pending_messages[key]
    assert pending is event and not pending.is_command()
    assert json.loads(pending.text.split("\n")[-1]) == {"command": "probe-native", "args": "inspect"}
    assert pending.source.user_id == "111" and pending.message_id == "444"
    f.runner._is_session_running = lambda key: False
    f.runner._running_agents.clear()
    assert await f.runner._handle_message(incoming("/legacy-probe x")) == "legacy:x"


@pytest.mark.asyncio
async def test_native_command_deny_hook_runs_before_dispatch(native_runner):
    f = native_runner
    from hermes_cli.plugin_commands import PluginAgentDispatch
    f.ctx.register_command("probe-native", lambda raw, context=None: PluginAgentDispatch("must not run"), contextual=True)
    f.runner.hooks.emit_collect.return_value = [{"decision": "deny", "message": "blocked by policy"}]
    result = await f.runner._handle_message(incoming())
    assert result == "blocked by policy"
    assert f.runner._running_agents == {}

@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["clarify", "update"])
async def test_command_is_not_consumed_as_pending_prompt_answer(native_runner, monkeypatch, prompt):
    from unittest.mock import Mock
    from gateway.run import _AGENT_PENDING_SENTINEL
    from hermes_cli.plugin_commands import PluginAgentDispatch
    from tools import clarify_gateway

    f = native_runner
    f.ctx.register_command("probe-native", lambda raw, context=None: PluginAgentDispatch("Reviewed operation"), contextual=True)
    event = incoming()
    key = f.runner._session_key_for_source(event.source)
    f.runner._running_agents[key] = _AGENT_PENDING_SENTINEL
    attempted_answer = Mock(return_value=clarify_gateway.TEXT_RESOLVED)
    if prompt == "clarify":
        monkeypatch.setattr(clarify_gateway, "get_pending_for_session", lambda *args, **kwargs: SimpleNamespace(clarify_id="pending"))
        monkeypatch.setattr(clarify_gateway, "attempt_text_response_for_session", attempted_answer)
    else:
        f.runner._update_prompt_pending[key] = True
        (f.home / ".update_prompt.json").write_text("{}")
    assert await f.runner._handle_message(event) is None
    assert f.adapter._pending_messages[key] is event
    assert not event.is_command()
    attempted_answer.assert_not_called()
    if prompt == "update":
        assert (f.home / ".update_response").read_text() == ""
