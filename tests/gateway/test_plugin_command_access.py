from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from tests.gateway.test_slash_access_dispatch import _make_runner as _access_runner

ADMIN = "111"
NON_ADMIN = "999"


def _make_source(
    *,
    platform: Platform = Platform.TELEGRAM,
    user_id: str = NON_ADMIN,
    chat_type: str = "dm",
    chat_id: str = "chat-1",
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name=f"name-{user_id}",
        chat_type=chat_type,
    )


def _make_event(text: str, source: SessionSource) -> MessageEvent:
    return MessageEvent(text=text, source=source, message_id="m1")


def _make_runner(*, platform_extra=None, platform=Platform.TELEGRAM):
    runner = _access_runner(platform_extra=platform_extra, platform=platform)
    runner._draining = False
    runner._run_agent = AsyncMock(return_value=None)
    return runner


def _install_plugin_command(monkeypatch, name: str) -> list:
    from hermes_cli import plugins as plugins_mod

    invocations: list = []

    def handler(args: str):
        invocations.append(args)
        return "PLUGIN-COMMAND-RAN"

    entries = {
        name: {
            "handler": handler,
            "description": f"{name} command",
            "plugin": "test-plugin",
            "args_hint": "",
        }
    }
    monkeypatch.setattr(plugins_mod, "get_plugin_commands", lambda: entries)
    monkeypatch.setattr(
        plugins_mod,
        "get_plugin_command_handler",
        lambda n: entries[n]["handler"] if n in entries else None,
    )
    return invocations


@pytest.mark.asyncio
async def test_non_admin_denied_for_hyphenated_plugin_command_typed_underscored(
    monkeypatch,
):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(
        platform_extra={"allow_admin_from": [ADMIN], "user_allowed_commands": []}
    )

    result = await runner._handle_message(
        _make_event("/my_cmd", _make_source(user_id=NON_ADMIN))
    )

    assert result is not None
    assert "⛔" in result
    assert "/my-cmd is admin-only here" in result
    assert invocations == []
    assert "PLUGIN-COMMAND-RAN" not in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_non_admin_denied_for_plugin_command_typed_exactly(monkeypatch):
    invocations = _install_plugin_command(monkeypatch, "mycmd")
    runner = _make_runner(
        platform_extra={"allow_admin_from": [ADMIN], "user_allowed_commands": []}
    )

    result = await runner._handle_message(
        _make_event("/mycmd", _make_source(user_id=NON_ADMIN))
    )

    assert result is not None
    assert "⛔" in result
    assert invocations == []


@pytest.mark.asyncio
async def test_denial_holds_when_the_plugin_command_takes_arguments(monkeypatch):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(
        platform_extra={"allow_admin_from": [ADMIN], "user_allowed_commands": []}
    )

    result = await runner._handle_message(
        _make_event("/my_cmd 10m", _make_source(user_id=NON_ADMIN))
    )

    assert "⛔" in result
    assert invocations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_admin_runs_plugin_command_when_gating_enabled(monkeypatch, typed):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(
        platform_extra={"allow_admin_from": [ADMIN], "user_allowed_commands": []}
    )

    result = await runner._handle_message(
        _make_event(typed, _make_source(user_id=ADMIN))
    )

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
@pytest.mark.parametrize("listed", ["my-cmd", "my_cmd"])
async def test_non_admin_runs_plugin_command_listed_in_user_allowed(
    monkeypatch, typed, listed
):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": [ADMIN],
            "user_allowed_commands": [listed],
        }
    )

    result = await runner._handle_message(
        _make_event(typed, _make_source(user_id=NON_ADMIN))
    )

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_underscored_allowlist_entry_does_not_admit_other_commands(
    monkeypatch, typed
):
    invocations = _install_plugin_command(monkeypatch, "other-cmd")
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": [ADMIN],
            "user_allowed_commands": ["my_cmd"],
        }
    )

    result = await runner._handle_message(
        _make_event(typed.replace("my", "other"), _make_source(user_id=NON_ADMIN))
    )

    assert "⛔" in result
    assert invocations == []


@pytest.mark.asyncio
async def test_backward_compat_no_admin_list_means_no_gating(monkeypatch):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(platform_extra={})

    result = await runner._handle_message(
        _make_event("/my_cmd", _make_source(user_id="anyone"))
    )

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]


@pytest.mark.asyncio
async def test_group_only_gating_leaves_dm_plugin_commands_unrestricted(monkeypatch):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(platform_extra={"group_allow_admin_from": [ADMIN]})

    result = await runner._handle_message(
        _make_event("/my_cmd", _make_source(user_id="anyone", chat_type="dm"))
    )

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]


@pytest.mark.asyncio
async def test_group_scope_gating_denies_non_admin_plugin_command(monkeypatch):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(
        platform_extra={
            "group_allow_admin_from": [ADMIN],
            "group_user_allowed_commands": [],
        }
    )

    result = await runner._handle_message(
        _make_event(
            "/my_cmd",
            _make_source(user_id=NON_ADMIN, chat_type="group", chat_id="g-1"),
        )
    )

    assert "⛔" in result
    assert invocations == []


def _hook_returns(runner, *results):
    emit_collect = AsyncMock(return_value=list(results))
    runner.hooks.emit_collect = emit_collect
    return emit_collect


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_command_hook_deny_blocks_the_plugin_command(monkeypatch, typed):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    _hook_returns(runner, {"decision": "deny", "message": "HOOK-DENIED"})

    result = await runner._handle_message(_make_event(typed, _make_source()))

    assert result == "HOOK-DENIED"
    assert invocations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_command_hook_deny_without_message_still_blocks(monkeypatch, typed):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    _hook_returns(runner, {"decision": "deny"})

    result = await runner._handle_message(_make_event(typed, _make_source()))

    assert "blocked by a hook" in result
    assert invocations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_command_hook_handled_short_circuits_plugin_dispatch(monkeypatch, typed):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    _hook_returns(runner, {"decision": "handled", "message": "HOOK-HANDLED"})

    result = await runner._handle_message(_make_event(typed, _make_source()))

    assert result == "HOOK-HANDLED"
    assert invocations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_command_hook_fires_under_the_registered_name(monkeypatch, typed):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    emit_collect = _hook_returns(runner)

    result = await runner._handle_message(_make_event(f"{typed} 10m", _make_source()))

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == ["10m"]

    hook_names = [call.args[0] for call in emit_collect.await_args_list]
    assert "command:my-cmd" in hook_names

    ctx = next(
        call.args[1]
        for call in emit_collect.await_args_list
        if call.args[0] == "command:my-cmd"
    )
    assert ctx["command"] == "my-cmd"
    assert ctx["raw_command"] == typed.lstrip("/")
    assert ctx["args"] == "10m"


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_hook_without_a_decision_leaves_dispatch_alone(monkeypatch, typed):
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    _hook_returns(runner, {}, {"decision": "allow"}, "not-a-dict")

    result = await runner._handle_message(_make_event(typed, _make_source()))

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]
