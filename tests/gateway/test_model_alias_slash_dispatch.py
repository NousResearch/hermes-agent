"""Regression coverage for PR #59606 — /<alias> must enter the canonical
/model dispatch path on the gateway, not bypass per-platform slash access
control or the ``command:model`` hook.

Drives the real ``GatewayRunner._handle_message`` path with a stub
session store. Uses the same ``object.__new__`` runner construction
pattern as test_slash_access_dispatch.py so we exercise the real alias
branch and hook site, not a re-implementation in the test.

Coverage:
  - /sonnet (a real MODEL_ALIASES entry) is dispatched as /model sonnet
  - The per-platform slash access-control gate fires for /sonnet under
    the same policy as /model.
  - The ``command:model`` hook fires for /sonnet with canonical
    command="model" and raw_command="sonnet".
  - A non-admin denied of /model is also denied of /sonnet; an admin
    bypasses both.
  - Hook deny decision short-circuits the switch.
  - /sonnet does NOT pre-empt a user-defined quick_command named sonnet.
  - Unknown /<x> still gets the unknown-command notice when x is not an
    alias.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(
    *,
    platform: Platform = Platform.DISCORD,
    user_id: str = "user1",
    chat_type: str = "dm",
    chat_id: str = "c1",
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


def _make_runner(*, platform_extra: dict | None = None,
                 platform: Platform = Platform.DISCORD):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                token="***",
                extra=platform_extra or {},
            )
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    session_entry = SessionEntry(
        session_key="agent:main:discord:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_sources = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    # The real _handle_model_command needs heavy gateway plumbing; tests
    # stub it so we can assert dispatch happened without actually
    # mutating the live model state.
    runner._handle_model_command = AsyncMock(return_value="model-handled")
    return runner


# -----------------------------------------------------------------------
# 1. Alias routes through the same /model handler a typed /model uses.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_routes_through_handle_model_command():
    """Typing /sonnet must invoke _handle_model_command exactly as
    typing /model sonnet would."""
    runner = _make_runner(platform_extra={})  # no admin gating
    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )
    assert result == "model-handled"
    runner._handle_model_command.assert_awaited_once()
    # The event passed to _handle_model_command must be the rewritten
    # /model sonnet so the handler's own arg parsing matches a typed
    # /model sonnet path.
    event_arg = runner._handle_model_command.await_args.args[0]
    assert event_arg.text.lower().startswith("/model"), (
        f"alias /sonnet must be rewritten to /model sonnet before "
        f"reaching the handler, got event.text={event_arg.text!r}"
    )
    assert "sonnet" in event_arg.text.lower()


# -----------------------------------------------------------------------
# 2. Per-platform slash access-control gate fires for /<alias>.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_denied_for_non_admin_when_model_not_allowed():
    """Non-admin who can't run /model must also be denied /sonnet — the
    alias must NOT bypass the access-control gate."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["status"],  # /model is NOT listed
        }
    )
    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="999"))
    )
    assert result is not None
    assert "⛔" in result
    assert "model is admin-only here" in result
    # /sonnet was denied → _handle_model_command never fired
    runner._handle_model_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_alias_allowed_for_non_admin_when_model_allowed():
    """When /model IS in user_allowed_commands, /sonnet must be allowed
    too (alias and canonical are indistinguishable from the gate's POV)."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["status", "model"],
        }
    )
    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="999"))
    )
    assert result == "model-handled"
    runner._handle_model_command.assert_awaited_once()


@pytest.mark.asyncio
async def test_alias_admin_bypasses_gate():
    """An admin can switch models via /sonnet the same way they can via
    /model sonnet."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="111"))
    )
    assert result == "model-handled"
    runner._handle_model_command.assert_awaited_once()


# -----------------------------------------------------------------------
# 3. command:model hook fires for /<alias> with canonical name.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_fires_command_model_hook():
    """hooks.emit_collect('command:model', ...) must be called when an
    alias is dispatched, so handlers can't tell /sonnet from /model sonnet.
    """
    runner = _make_runner(platform_extra={})
    # Capture the actual call args to assert on hook context shape.
    captured: list[tuple] = []

    async def _capture_emit(name, ctx):
        captured.append((name, ctx))
        return []

    runner.hooks.emit_collect = _capture_emit

    await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )

    hook_names = [name for name, _ in captured]
    assert "command:model" in hook_names, (
        f"command:model hook must fire for /sonnet, "
        f"got hook names: {hook_names}"
    )
    # Find the command:model hook call and verify context shape
    model_hook = next(
        ctx for name, ctx in captured if name == "command:model"
    )
    assert model_hook["command"] == "model"
    assert model_hook["raw_command"] == "sonnet"
    assert model_hook["args"] == "sonnet"


@pytest.mark.asyncio
async def test_alias_hook_deny_blocks_switch():
    """A hook that returns decision='deny' for command:model must block
    the alias dispatch the same way it blocks /model sonnet."""
    runner = _make_runner(platform_extra={})
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "deny", "message": "blocked-by-test"}]
    )
    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )
    assert result == "blocked-by-test"
    runner._handle_model_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_alias_hook_handled_short_circuits():
    """A hook returning decision='handled' must stop the switch without
    calling _handle_model_command."""
    runner = _make_runner(platform_extra={})
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "handled", "message": "handled-by-test"}]
    )
    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )
    assert result == "handled-by-test"
    runner._handle_model_command.assert_not_awaited()


# -----------------------------------------------------------------------
# 4. Priority preservation — alias must NOT pre-empt higher-priority commands.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_does_not_shadow_quick_command(monkeypatch):
    """If a user defines a quick_command named sonnet, that command wins
    over the model alias (preserves existing priority rules).

    We avoid exercising the actual subprocess sink (printf/sh) and
    instead stub the lower-level ``asyncio.create_subprocess_shell``
    so the test runs cross-platform — only the dispatch priority
    matters here, not shell semantics.
    """
    import asyncio as _asyncio

    runner = _make_runner(platform_extra={})
    runner.config.quick_commands = {
        "sonnet": {"type": "exec", "command": "echo quick-wins"}
    }

    class _FakeProc:
        async def communicate(self):
            return (b"quick-wins", b"")

    async def _fake_shell(*_args, **_kwargs):
        return _FakeProc()

    monkeypatch.setattr(_asyncio, "create_subprocess_shell", _fake_shell)

    result = await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )
    assert result == "quick-wins"
    runner._handle_model_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_alias_does_not_shadow_running_agent_fastpath():
    """When an agent is running and a fast-path-eligible command is
    typed under an alias name, the fast-path semantics must be
    preserved (alias only intercepts unknown commands, not built-ins)."""
    runner = _make_runner(platform_extra={})
    # /restart is a fast-path command — its alias 'r' (if any) would win
    # via _resolve_cmd. The alias check only runs for unrecognized
    # commands, so /restart never enters the alias branch.
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")
    src = _make_source(user_id="anyone")
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0  # not stale
    result = await runner._handle_message(_make_event("/restart", src))
    assert result == "restart-handled"
    runner._handle_model_command.assert_not_awaited()


# -----------------------------------------------------------------------
# 5. Unknown command behavior unchanged.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_still_warns_when_not_alias():
    """A typed /<x> where x is neither a built-in, plugin, skill, nor
    a model alias must still hit the unknown-command warning."""
    runner = _make_runner(platform_extra={})
    # /xyzzy-nope is not in COMMAND_REGISTRY, not a plugin, not a skill,
    # not a model alias.
    result = await runner._handle_message(
        _make_event("/xyzzy-nope", _make_source(user_id="anyone"))
    )
    assert result is not None
    assert "Unknown command" in result
    runner._handle_model_command.assert_not_awaited()


# -----------------------------------------------------------------------
# 6. Edge case — alias with extra args.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_with_provider_flag_dispatches_with_args():
    """Typing /sonnet --provider foo must reach _handle_model_command
    with the rewritten event including both the alias name and the
    --provider flag, so the handler's flag parsing works identically to
    /model sonnet --provider foo."""
    runner = _make_runner(platform_extra={})
    await runner._handle_message(
        _make_event("/sonnet --provider openrouter", _make_source(user_id="anyone"))
    )
    runner._handle_model_command.assert_awaited_once()
    event_arg = runner._handle_model_command.await_args.args[0]
    assert event_arg.text.lower().startswith("/model")
    assert "sonnet" in event_arg.text.lower()
    assert "--provider openrouter" in event_arg.text.lower()


# -----------------------------------------------------------------------
# 7. command:model hook rewrite decision — must mirror canonical /model
#    rewrite protocol field-for-field.
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alias_hook_rewrite_routes_to_handle_model_command():
    """A hook returning decision='rewrite' with command_name and raw_args
    must rewrite event.text, re-resolve command/canonical, and then
    dispatch to _handle_model_command with the rewritten args. This
    mirrors the canonical /model rewrite branch at gateway/run.py:9456
    (and the test_unknown_command.py::test_command_hook_rewrite_routes_to_plugin
    contract) so alias and /model have the same command:model hook
    lifecycle.
    """
    runner = _make_runner(platform_extra={})
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {
                "decision": "rewrite",
                "command_name": "opus",  # alias-style rewrite within model namespace
                "raw_args": "--provider openrouter",
            }
        ]
    )

    await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )

    # Hook fires once, on the original alias — NOT re-fired after rewrite
    # (one decision per turn, matches canonical behavior).
    runner.hooks.emit_collect.assert_awaited_once()
    hook_name = runner.hooks.emit_collect.await_args.args[0]
    assert hook_name == "command:model"

    # _handle_model_command receives the rewritten event.text: /opus --provider openrouter
    runner._handle_model_command.assert_awaited_once()
    event_arg = runner._handle_model_command.await_args.args[0]
    assert event_arg.text.lower().startswith("/opus")
    assert "--provider openrouter" in event_arg.text.lower()


@pytest.mark.asyncio
async def test_alias_hook_rewrite_with_empty_command_is_noop():
    """If the rewrite decision returns an empty command_name, the alias
    dispatcher treats it as a no-op (continue), exactly like the canonical
    branch at gateway/run.py:9458-9460."""
    runner = _make_runner(platform_extra={})
    runner.hooks.emit_collect = AsyncMock(
        return_value=[{"decision": "rewrite", "command_name": "", "raw_args": "ignored"}]
    )

    await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )

    # Empty command_name → continue → fall through to dispatch with the
    # alias's pre-rewrite event.text (still /model sonnet).
    runner._handle_model_command.assert_awaited_once()
    event_arg = runner._handle_model_command.await_args.args[0]
    assert event_arg.text.lower().startswith("/model")
    assert "sonnet" in event_arg.text.lower()


@pytest.mark.asyncio
async def test_alias_hook_rewrite_strips_leading_slash_from_command_name():
    """The rewrite protocol (canonical run.py:9457-9459) strips any leading
    slash from command_name before concatenating. Alias must do the same."""
    runner = _make_runner(platform_extra={})
    runner.hooks.emit_collect = AsyncMock(
        return_value=[
            {
                "decision": "rewrite",
                "command_name": "/opus",  # leading slash must be stripped
                "raw_args": "",
            }
        ]
    )

    await runner._handle_message(
        _make_event("/sonnet", _make_source(user_id="anyone"))
    )

    runner._handle_model_command.assert_awaited_once()
    event_arg = runner._handle_model_command.await_args.args[0]
    # No double-slash: /opus, not //opus
    assert event_arg.text.startswith("/opus"), (
        f"rewrite target must have exactly one leading slash, "
        f"got event_arg.text={event_arg.text!r}"
    )