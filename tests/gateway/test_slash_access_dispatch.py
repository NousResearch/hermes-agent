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


# ---------------------------------------------------------------------------
# Backward compatibility — no admin list set means no gating at all
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scope isolation — DM vs group
# ---------------------------------------------------------------------------


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
async def test_non_admin_denied_for_unlisted_quick_command_exec():
    """A non-admin must not reach the quick_commands exec sink for a command
    that isn't in user_allowed_commands. Regression for #44727 — quick
    commands are never in the gateway registry, so the early gate skips them;
    the sink gate must catch them."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    runner.config.quick_commands = {
        "limits": {"type": "exec", "command": "printf quick-command-bypass-confirmed"}
    }

    result = await runner._handle_message(
        _make_event("/limits", _make_source(user_id="999"))
    )

    assert result is not None
    assert "⛔" in result
    assert "/limits is admin-only here" in result
    assert "quick-command-bypass-confirmed" not in result


@pytest.mark.asyncio
async def test_admin_runs_quick_command_when_gating_enabled():
    """An admin runs the quick command even under an enabled gate with an
    empty user_allowed_commands list."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": [],
        }
    )
    runner.config.quick_commands = {
        "limits": {"type": "exec", "command": "printf quick-command-admin"}
    }

    result = await runner._handle_message(
        _make_event("/limits", _make_source(user_id="111"))
    )

    assert result == "quick-command-admin"


# ---------------------------------------------------------------------------
# Running-agent fast-path gating — admin/user split must hold even when an
# agent is already running. The fast-path block in _handle_message dispatches
# /stop, /restart, /new, /steer, /model, /approve, /deny, /agents,
# /bg, /btw, /kanban, /goal, /yolo, /verbose, /footer, /help, /commands,
# /profile, /update directly without going through the cold dispatch site.
# We must apply the gate there too — otherwise non-admins could bypass
# gating just because an agent happens to be busy.
# ---------------------------------------------------------------------------


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
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0
    # Mock the restart handler so it doesn't actually try to restart anything.
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")

    result = await runner._handle_message(_make_event("/restart", src))
    assert result == "restart-handled"
    assert "⛔" not in (result or "")


# ---------------------------------------------------------------------------
# Alias resolution — /h aliases to /help; the gate must canonicalize before
# checking access. /hist (history alias) is a real one to exercise.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Unknown / unregistered command — gate must NOT intercept (let the existing
# unknown-command path handle it normally).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scope independence — admin in DM scope is NOT auto-admin in group when
# group has its own admin list (regression guard for the "admin lists are
# scope-specific" rule).
# ---------------------------------------------------------------------------


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
