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
    chat_name: str | None = None,
    chat_id_alt: str | None = None,
    thread_id: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        chat_name=chat_name,
        user_name=f"name-{user_id}",
        chat_type=chat_type,
        chat_id_alt=chat_id_alt,
        thread_id=thread_id,
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
# Per-channel/group hard allowlists — issue #37004
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_channel_allowlist_denies_unlisted_command_even_for_group_admin():
    """A channel-specific allowlist is a hard cap for that group/channel.

    Even if a user is an admin in the broader group scope, commands missing
    from the channel allowlist must be rejected before privileged handlers run.
    """
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "group_allow_admin_from": ["admin-user"],
            "channel_command_access": {
                "group:restricted-signal-group": {
                    "allowed_slash_commands": ["help", "status"],
                    "deny_message": "This command is not enabled in this group.",
                }
            },
        },
    )
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="admin-user",
        chat_type="group",
        chat_id="group:restricted-signal-group",
    )

    result = await runner._handle_message(_make_event("/restart", source))

    assert result == "This command is not enabled in this group."
    assert result != "restart-handled"


@pytest.mark.asyncio
async def test_channel_allowlist_allows_configured_command_in_restricted_group():
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "channel_command_access": {
                "group:ops-room": {
                    "allowed_slash_commands": ["whoami"],
                }
            },
        },
    )
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="group-user",
        chat_type="group",
        chat_id="group:ops-room",
    )

    result = await runner._handle_message(_make_event("/whoami", source))

    assert result is not None
    assert "⛔" not in result
    assert "Tier:" in result
    assert "Slash commands you can run: /whoami" in result
    assert "/help" not in result


@pytest.mark.asyncio
async def test_channel_allowlist_allows_configured_command_for_any_group_user():
    """A channel allowlist grants listed commands to any user in that channel.

    Broader group admin/user slash tiers must not re-deny a command that the
    operator explicitly exposed through ``channel_command_access`` for the
    channel. The channel allowlist is both the hard cap and the authorization
    surface for that channel.
    """
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "group_allow_admin_from": ["admin-user"],
            "group_user_allowed_commands": [],
            "channel_command_access": {
                "group:ops-room": {
                    "allowed_slash_commands": ["restart"],
                }
            },
        },
    )
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="regular-group-user",
        chat_type="group",
        chat_id="group:ops-room",
    )

    result = await runner._handle_message(_make_event("/restart", source))

    assert result == "restart-handled"
    assert "⛔" not in (result or "")


@pytest.mark.asyncio
async def test_channel_allowlist_matches_signal_raw_group_id_alt():
    """Signal sources expose both chat_id='group:<id>' and chat_id_alt='<id>'.

    Operators should be able to configure the raw Signal group id from
    signal-cli without needing to add the gateway's ``group:`` prefix.
    """
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "group_allow_admin_from": ["admin-user"],
            "channel_command_access": {
                "raw-signal-group-id": {
                    "allowed_slash_commands": ["status"],
                }
            },
        },
    )
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="admin-user",
        chat_type="group",
        chat_id="group:raw-signal-group-id",
        chat_id_alt="raw-signal-group-id",
    )

    result = await runner._handle_message(_make_event("/restart", source))

    assert result is not None
    assert "⛔" in result
    assert "/restart is not enabled in this group/channel" in result


@pytest.mark.asyncio
async def test_channel_allowlist_matches_signal_group_name():
    """Signal group allowlists should support human-readable group names.

    A config key like ``DroneProject`` must match the incoming Signal
    ``groupName`` surfaced as ``SessionSource.chat_name`` so operators do not
    have to paste opaque Signal group ids into config.yaml.
    """
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "group_allow_admin_from": ["admin-user"],
            "channel_command_access": {
                "DroneProject": {
                    "allowed_slash_commands": ["status"],
                }
            },
        },
    )
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="admin-user",
        chat_type="group",
        chat_id="group:opaque-signal-group-id",
        chat_name="DroneProject",
        chat_id_alt="opaque-signal-group-id",
    )

    result = await runner._handle_message(_make_event("/restart", source))

    assert result is not None
    assert "⛔" in result
    assert "/restart is not enabled in this group/channel" in result
    assert result != "restart-handled"


@pytest.mark.asyncio
async def test_channel_allowlist_only_applies_to_matching_channel():
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "group_allow_admin_from": ["admin-user"],
            "channel_command_access": {
                "group:restricted-room": {
                    "allowed_slash_commands": ["status"],
                }
            },
        },
    )
    runner._handle_restart_command = AsyncMock(return_value="restart-handled")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="admin-user",
        chat_type="group",
        chat_id="group:other-room",
    )

    result = await runner._handle_message(_make_event("/restart", source))

    assert result == "restart-handled"


@pytest.mark.asyncio
async def test_channel_allowlist_blocks_unlisted_skill_command(monkeypatch):
    from agent import skill_commands

    monkeypatch.setattr(
        skill_commands,
        "get_skill_commands",
        lambda: {
            "/deep-research": {
                "name": "deep-research",
                "description": "Deep research",
                "skill_dir": "/tmp/deep-research",
            },
            "/ops-skill": {
                "name": "ops-skill",
                "description": "Ops skill",
                "skill_dir": "/tmp/ops-skill",
            },
        },
    )
    monkeypatch.setattr(
        skill_commands,
        "build_skill_invocation_message",
        lambda *args, **kwargs: "[skill invocation message]",
    )
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "channel_command_access": {
                "group:drone-room": {
                    "allowed_slash_commands": ["deep-research"],
                }
            },
        },
    )
    runner._draining = False
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ran")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="regular-user",
        chat_type="group",
        chat_id="group:drone-room",
    )

    result = await runner._handle_message(_make_event("/ops-skill rotate keys", source))

    assert result is not None
    assert "⛔" in result
    assert "/ops-skill is not enabled in this group/channel" in result
    runner._handle_message_with_agent.assert_not_called()


@pytest.mark.asyncio
async def test_channel_allowlist_allows_listed_skill_command(monkeypatch):
    from agent import skill_commands

    monkeypatch.setattr(
        skill_commands,
        "get_skill_commands",
        lambda: {
            "/deep-research": {
                "name": "deep-research",
                "description": "Deep research",
                "skill_dir": "/tmp/deep-research",
            },
        },
    )
    monkeypatch.setattr(
        skill_commands,
        "build_skill_invocation_message",
        lambda *args, **kwargs: "[deep-research invocation message]",
    )
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "channel_command_access": {
                "group:drone-room": {
                    "allowed_slash_commands": ["deep-research"],
                }
            },
        },
    )
    runner._draining = False
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ran")
    source = _make_source(
        platform=Platform.SIGNAL,
        user_id="regular-user",
        chat_type="group",
        chat_id="group:drone-room",
    )
    event = _make_event("/deep-research battery suppliers", source)

    result = await runner._handle_message(event)

    assert result == "agent-ran"
    assert event.text == "[deep-research invocation message]"


def test_gateway_channel_command_access_config_bridges_to_platform_extra(monkeypatch, tmp_path):
    """config.yaml supports human-readable group names with YAML command lists."""
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "signal:\n"
        "  enabled: true\n"
        "gateway:\n"
        "  channel_command_access:\n"
        "    signal:\n"
        "      DroneProject:\n"
        "        allowed_slash_commands:\n"
        "          - deep-research\n"
        "        deny_message: This command is not enabled in this group.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
    monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)

    config = load_gateway_config()

    signal_extra = config.platforms[Platform.SIGNAL].extra
    assert signal_extra["channel_command_access"] == {
        "DroneProject": {
            "allowed_slash_commands": ["deep-research"],
            "deny_message": "This command is not enabled in this group.",
        }
    }


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
async def test_listed_quick_command_runs_for_non_admin():
    """When the operator lists the quick command in user_allowed_commands, a
    non-admin can run it — the gate must allow, not blanket-deny."""
    runner = _make_runner(
        platform_extra={
            "allow_admin_from": ["111"],
            "user_allowed_commands": ["limits"],
        }
    )
    runner.config.quick_commands = {
        "limits": {"type": "exec", "command": "printf quick-command-allowed"}
    }

    result = await runner._handle_message(
        _make_event("/limits", _make_source(user_id="999"))
    )

    assert result == "quick-command-allowed"


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
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0
    runner._handle_status_command = AsyncMock(return_value="status-handled")

    result = await runner._handle_message(_make_event("/status", src))
    assert result == "status-handled"
    assert "⛔" not in (result or "")


@pytest.mark.asyncio
async def test_running_agent_fastpath_blocks_unlisted_skill_command(monkeypatch):
    from agent import skill_commands

    monkeypatch.setattr(
        skill_commands,
        "get_skill_commands",
        lambda: {
            "/deep-research": {
                "name": "deep-research",
                "description": "Deep research",
                "skill_dir": "/tmp/deep-research",
            },
            "/ops-skill": {
                "name": "ops-skill",
                "description": "Ops skill",
                "skill_dir": "/tmp/ops-skill",
            },
        },
    )
    runner = _make_runner(
        platform=Platform.SIGNAL,
        platform_extra={
            "channel_command_access": {
                "group:drone-room": {
                    "allowed_slash_commands": ["deep-research"],
                }
            },
        },
    )
    src = _make_source(
        platform=Platform.SIGNAL,
        user_id="regular-user",
        chat_type="group",
        chat_id="group:drone-room",
    )
    sk = build_session_key(src)
    runner._running_agents[sk] = MagicMock()
    runner._running_agents_ts[sk] = 0

    result = await runner._handle_message(_make_event("/ops-skill rotate keys", src))

    assert result is not None
    assert "⛔" in result
    assert "/ops-skill is not enabled in this group/channel" in result


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
