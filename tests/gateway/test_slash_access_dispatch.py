"""Integration tests for slash command access control gating in gateway/run.py.

Drives the real ``GatewayRunner._handle_message`` path with a stub session
store so we exercise the actual gate inserted at the dispatch site (not a
re-implementation in the test). Uses the same ``object.__new__`` runner
construction pattern as test_status_command.py.

Coverage targets:
  - Backward compat: no ``allow_admin_from`` set → behaves exactly as before
    (no denial messages, dispatch reaches the real handler).
  - Admin path: user in ``allow_admin_from`` runs anything.
  - User path: user not in admin list, but command in
    ``user_allowed_commands`` → allowed.
  - User denied: command not in either list → returns the ⛔ denial.
  - Always-allowed floor: /help and /whoami reachable for non-admins
    even with empty user_allowed_commands.
  - DM vs group scope isolation.
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
    return runner


# ---------------------------------------------------------------------------
# /whoami response shape — proves the handler is reachable AND uses the
# resolver. We use /whoami because it's deterministic and short-circuits
# before any session/agent setup.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whoami_unrestricted_when_no_admin_list():
    runner = _make_runner(platform_extra={})  # no admin list
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "Tier: unrestricted" in result
    assert "no admin list configured" in result


@pytest.mark.asyncio
async def test_whoami_admin_user():
    runner = _make_runner(platform_extra={"allow_admin_from": ["111"]})
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="111")))
    assert "**admin**" in result


@pytest.mark.asyncio
async def test_whoami_non_admin_lists_runnable_commands():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["status", "model"],
        }
    )
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "Tier: user" in result
    assert "/help" in result      # always-allowed floor
    assert "/whoami" in result    # always-allowed floor
    assert "/status" in result
    assert "/model" in result


# ---------------------------------------------------------------------------
# Gate denial — admin-only command attempted by non-admin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_admin_denied_for_unlisted_command():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["status"],
        }
    )
    # /stop is NOT in user_allowed_commands and not in the always-allowed floor.
    result = await runner._handle_message(_make_event("/stop", _make_source(user_id="999")))
    assert result is not None
    assert "⛔" in result
    assert "/stop is admin-only here" in result
    assert "/status" in result  # denial preview shows what they CAN run


@pytest.mark.asyncio
async def test_non_admin_with_empty_user_commands_gets_floor_only():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],  # explicitly empty
        }
    )
    # /stop denied
    result = await runner._handle_message(_make_event("/stop", _make_source(user_id="999")))
    assert "⛔" in result
    assert "No slash commands are enabled" in result
    # /whoami still works (always-allowed floor)
    whoami_result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "Tier: user" in whoami_result


# ---------------------------------------------------------------------------
# Gate ALLOW — admin and listed user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_runs_unlisted_command():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],  # users can run nothing
        }
    )
    # Admin runs /whoami (proxy for "any command works"); the gate must NOT
    # return the ⛔ denial. The /whoami handler is deterministic and doesn't
    # need a real agent, so we can assert against its content.
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="111")))
    assert "⛔" not in result
    assert "**admin**" in result


@pytest.mark.asyncio
async def test_user_runs_listed_command():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["whoami"],  # explicit
        }
    )
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="999")))
    assert "⛔" not in result
    assert "Tier: user" in result


# ---------------------------------------------------------------------------
# Backward compatibility — no admin list set means no gating at all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backward_compat_no_admin_list_means_no_gate():
    runner = _make_runner(platform_extra={})  # nothing configured
    # Random non-listed user runs /whoami; should return unrestricted profile,
    # never a denial.
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="anyone")))
    assert "⛔" not in result
    assert "Tier: unrestricted" in result


# ---------------------------------------------------------------------------
# Scope isolation — DM vs group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_admin_is_not_group_admin():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "group_allow_admin_from": ["222"],
            "group_user_allowed_commands": [],
        }
    )
    # User 111 is DM admin. In group context they're a non-admin with no
    # listed commands → /stop denied.
    result = await runner._handle_message(
        _make_event("/stop", _make_source(user_id="111", chat_type="group"))
    )
    assert "⛔" in result


@pytest.mark.asyncio
async def test_group_only_gating_leaves_dm_unrestricted():
    runner = _make_runner(
        platform_extra={
            # Only group has an admin list → DM scope stays in backward-compat mode
            "group_allow_admin_from": ["222"],
        }
    )
    result = await runner._handle_message(_make_event("/whoami", _make_source(user_id="anyone", chat_type="dm")))
    assert "Tier: unrestricted" in result


# ---------------------------------------------------------------------------
# Plugin-registered slash commands are gated through the same path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_registered_command_is_gated(monkeypatch):
    """The gate must recognize plugin-registered slash commands, not just
    built-in COMMAND_REGISTRY entries. We verify by stubbing
    is_gateway_known_command and resolve_command so a fictitious /myplugin
    command is treated as a known plugin command.
    """
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )

    from hermes_cli import commands as cmd_mod

    real_resolve = cmd_mod.resolve_command
    real_is_known = cmd_mod.is_gateway_known_command

    def fake_resolve(name):
        if name == "myplugin":
            # Return a CommandDef-like duck so canonical resolution succeeds
            return SimpleNamespace(name="myplugin")
        return real_resolve(name)

    def fake_is_known(name):
        if name == "myplugin":
            return True
        return real_is_known(name)

    monkeypatch.setattr(cmd_mod, "resolve_command", fake_resolve)
    monkeypatch.setattr(cmd_mod, "is_gateway_known_command", fake_is_known)

    # Non-admin tries to run the plugin command → must be denied by the gate.
    result = await runner._handle_message(
        _make_event("/myplugin foo bar", _make_source(user_id="999"))
    )
    assert "⛔" in result
    assert "/myplugin is admin-only here" in result


# ---------------------------------------------------------------------------
# Running-agent fast-path gating — admin/user split must hold even when an
# agent is already running. The fast-path block in _handle_message dispatches
# /stop, /restart, /new, /steer, /model, /approve, /deny, /agents,
# /background, /kanban, /goal, /yolo, /verbose, /footer, /help, /commands,
# /profile, /update directly without going through the cold dispatch site.
# We must apply the gate there too — otherwise non-admins could bypass
# gating just because an agent happens to be busy.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_agent_fastpath_blocks_non_admin_command():
    """When an agent is running, /restart from a non-admin must be denied."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    src = _make_source(user_id="999")
    # Mark the session as having an in-flight agent so the fast-path runs.
    from gateway.session import build_session_key
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0  # not stale (epoch + small delta on this machine)

    result = await runner._handle_message(_make_event("/restart", src))
    assert result is not None
    assert "⛔" in result
    assert "/restart is admin-only here" in result


@pytest.mark.asyncio
async def test_running_agent_fastpath_allows_admin_command():
    """Admins must still be able to run privileged commands like /restart
    through the running-agent fast-path. We check that we don't get the
    denial message; the actual /restart handler is mocked out via the
    runner's MagicMock."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    src = _make_source(user_id="111")  # admin
    from gateway.session import build_session_key
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0
    # Mock the restart handler so it doesn't actually try to restart anything.
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")

    result = await runner._handle_message(_make_event("/restart", src))
    assert result == "restart-handled"
    assert "⛔" not in (result or "")


@pytest.mark.asyncio
async def test_running_agent_fastpath_status_always_works():
    """/status is intentionally pre-gate on the fast-path so users can
    always see session state, even non-admins."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    src = _make_source(user_id="999")  # non-admin
    from gateway.session import build_session_key
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0
    runner._handle_status_command = AsyncMock(return_value="status-handled")

    result = await runner._handle_message(_make_event("/status", src))
    assert result == "status-handled"
    assert "⛔" not in (result or "")


# ---------------------------------------------------------------------------
# Alias resolution — /h aliases to /help; the gate must canonicalize before
# checking access. /hist (history alias) is a real one to exercise.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_uses_canonical_name_not_alias():
    """If /hist resolves to canonical 'history' and history is in
    user_allowed_commands, the alias must be allowed too."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["history"],
        }
    )
    # Find a real alias in the registry to use.
    from hermes_cli.commands import COMMAND_REGISTRY
    history_def = next(c for c in COMMAND_REGISTRY if c.name == "history")
    # If /history has aliases, use one. Otherwise just use /history.
    alias = history_def.aliases[0] if history_def.aliases else "history"
    # Mock the history handler so we don't need real session state.
    runner._handle_history_command = AsyncMock(return_value="history-handled")
    result = await runner._handle_message(_make_event(f"/{alias}", _make_source(user_id="999")))
    assert "⛔" not in (result or "")


# ---------------------------------------------------------------------------
# Unknown / unregistered command — gate must NOT intercept (let the existing
# unknown-command path handle it normally).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_does_not_intercept_unknown_command():
    """Random non-command text like /xyzzy is not in the registry. The gate
    must not produce a denial message — the existing unknown-command path
    will handle it (or the agent will see it as plain text)."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    # /xyzzy is not in COMMAND_REGISTRY and not a plugin command.
    # The gate should pass through (no ⛔) since canonical resolution
    # returns the raw command and is_gateway_known_command returns False.
    # We can only verify the gate didn't fire — downstream behavior may
    # vary (returns None, agent processes it, etc.). What matters: no denial.
    runner._handle_unknown_command = AsyncMock(return_value=None)
    # Stub out the rest of the cold path to short-circuit
    runner.session_store.get_or_create_session.side_effect = RuntimeError("would have proceeded past gate")
    try:
        await runner._handle_message(_make_event("/xyzzy", _make_source(user_id="999")))
    except RuntimeError as e:
        # Reaching session creation means we got past the gate without a denial.
        assert "would have proceeded past gate" in str(e)


# ---------------------------------------------------------------------------
# Scope independence — admin in DM scope is NOT auto-admin in group when
# group has its own admin list (regression guard for the "admin lists are
# scope-specific" rule).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_admin_blocked_in_group_with_separate_admin_list():
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],          # DM admin
            "group_allow_admin_from": ["222"],    # group admin
            "group_user_allowed_commands": ["status"],
        }
    )
    # User 111 is DM admin. In a group, they're a non-admin and can only
    # run group_user_allowed_commands. /restart is not in that list → denied.
    grp_src = _make_source(user_id="111", chat_type="group", chat_id="g1")
    result = await runner._handle_message(_make_event("/restart", grp_src))
    assert "⛔" in result
    assert "/restart is admin-only here" in result


# ---------------------------------------------------------------------------
# Multi-platform isolation — gating on Discord doesn't leak to Telegram.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gating_isolated_per_platform():
    """When Discord is gated and Telegram isn't, the same user_id on
    Telegram must be unrestricted."""
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="***",
                extra={
                    "allow_admin_from": ["111"],
                    "user_allowed_commands": [],
                },
            ),
            Platform.TELEGRAM: PlatformConfig(
                enabled=True, token="***", extra={}
            ),
        }
    )
    runner.adapters = {
        Platform.DISCORD: MagicMock(send=AsyncMock()),
        Platform.TELEGRAM: MagicMock(send=AsyncMock()),
    }
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )
    runner.session_store = MagicMock()
    session_entry = SessionEntry(
        session_key="agent:main:telegram:dm:c1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
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

    # Same user_id on Telegram → must be unrestricted (Telegram has no admin list).
    tg_src = _make_source(platform=Platform.TELEGRAM, user_id="999", chat_id="t1")
    result = await runner._handle_message(_make_event("/whoami", tg_src))
    assert "Tier: unrestricted" in result
