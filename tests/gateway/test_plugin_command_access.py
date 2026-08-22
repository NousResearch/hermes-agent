"""Plugin slash commands must obey the same per-platform slash-access policy
as built-in and quick commands.

The cold-path gate in ``_handle_message`` resolves access on the name the user
typed. Plugin dispatch, however, normalizes underscores to hyphens before
looking the handler up — Telegram's command menu surfaces a hyphenated plugin
command as ``/my_cmd``, and that underscored form is what the adapter sends
back. So for every hyphenated plugin command the typed name (``my_cmd``) is not
a known command while the dispatched name (``my-cmd``) is, and the command runs
unchecked: with ``allow_admin_from`` configured, a non-admin can invoke it.

These tests drive the real ``GatewayRunner._handle_message`` seam and mirror the
expectations already pinned for built-ins and quick commands in
``test_slash_access_dispatch.py``: gate off unless the operator configured it,
admins unrestricted, ``user_allowed_commands`` honoured, everyone else denied
before the handler runs.

The ``command:<canonical>`` hook is the second decision point keyed off the same
resolved name — it can return ``{"decision": "deny"}`` — so it had the identical
bypass, and the last section here pins both spellings through it.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource

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


def _make_runner(*, platform_extra: dict | None = None,
                 platform: Platform = Platform.TELEGRAM):
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
        session_key="agent:main:telegram:dm:chat-1",
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
    # A denied command must never reach the agent either.
    runner._run_agent = AsyncMock(return_value=None)
    return runner


def _install_plugin_command(monkeypatch, name: str) -> list:
    """Register *name* as a plugin slash command; return its invocation log."""
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


# ---------------------------------------------------------------------------
# Denial — the gate applies, and applies before the handler runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_admin_denied_for_hyphenated_plugin_command_typed_underscored(
    monkeypatch,
):
    """The Telegram-menu form of a hyphenated plugin command is the live
    bypass: the typed name never resolves, so nothing gated it."""
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
    """The exactly-typed form must stay denied too."""
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


# ---------------------------------------------------------------------------
# Allow — admins, listed users, and the un-configured default
# ---------------------------------------------------------------------------


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
    """Either spelling of the allowlist entry admits either typed form.

    The registration is hyphenated, but Telegram's command menu shows the
    command as ``/my_cmd`` — so that underscored form is what an operator
    copies into ``user_allowed_commands``. The policy folds ``_`` to ``-`` on
    both sides, so the four combinations of listed × typed all resolve to the
    same command.
    """
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
    """Folding must not widen the allowlist beyond the command it names."""
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
    """Unless the operator configured ``allow_admin_from`` for the scope, the
    policy is disabled and every authorized user runs every plugin command."""
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner(platform_extra={})

    result = await runner._handle_message(
        _make_event("/my_cmd", _make_source(user_id="anyone"))
    )

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]


@pytest.mark.asyncio
async def test_group_only_gating_leaves_dm_plugin_commands_unrestricted(monkeypatch):
    """Admin lists are scope-specific: a group-scoped list must not gate DMs."""
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


# ---------------------------------------------------------------------------
# command:<canonical> hook — the second decision point on the same name
# ---------------------------------------------------------------------------
#
# The hook fires only ``if is_gateway_known_command(canonical)``, the same
# predicate the access gate used, so the underscored spelling skipped it too:
# a handler returning ``{"decision": "deny"}`` was silently ignored for
# /my_cmd while it was honoured for /my-cmd. Both spellings must now reach it,
# and must reach it under the *registered* name so one hook registration works
# regardless of which spelling the platform sends.


def _hook_returns(runner, *results):
    """Make the ``command:`` hook return *results*; return the emit_collect mock."""
    emit_collect = AsyncMock(return_value=list(results))
    runner.hooks.emit_collect = emit_collect
    return emit_collect


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_command_hook_deny_blocks_the_plugin_command(monkeypatch, typed):
    """A deny hook must stop dispatch for either spelling."""
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
    """``handled`` means the hook did the work — core dispatch must not."""
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    _hook_returns(runner, {"decision": "handled", "message": "HOOK-HANDLED"})

    result = await runner._handle_message(_make_event(typed, _make_source()))

    assert result == "HOOK-HANDLED"
    assert invocations == []


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["/my_cmd", "/my-cmd"])
async def test_command_hook_fires_under_the_registered_name(monkeypatch, typed):
    """One registration, ``command:my-cmd``, serves both spellings.

    The hook name and ``ctx["command"]`` carry the registered (hyphenated)
    name so a handler need not know which spelling the platform sent, while
    ``ctx["raw_command"]`` preserves what the user actually typed.
    """
    _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    emit_collect = _hook_returns(runner)

    await runner._handle_message(_make_event(f"{typed} 10m", _make_source()))

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
    """Telemetry-style hooks that return nothing must not change behaviour."""
    invocations = _install_plugin_command(monkeypatch, "my-cmd")
    runner = _make_runner()
    _hook_returns(runner, {}, {"decision": "allow"}, "not-a-dict")

    result = await runner._handle_message(_make_event(typed, _make_source()))

    assert result == "PLUGIN-COMMAND-RAN"
    assert invocations == [""]
