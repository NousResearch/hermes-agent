"""Tests for the read-only /dobby command center slice."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.dobby_commands import handle_dobby_command, render_dobby_status
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, SUBCOMMANDS, resolve_command


SECRET_TOKEN = "sk-secret-012345678901234567890123456789"
WEBHOOK_SECRET = "whsec_secret012345678901234567890123456789"


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _strict_webhook_route() -> dict:
    return {
        "secret": WEBHOOK_SECRET,
        "prompt": "event",
        "require_signature": True,
        "signature_algorithm": "hmac-sha256",
        "signature_header": "X-Dobby-Signature",
        "timestamp_header": "X-Dobby-Timestamp",
        "replay_window_seconds": 300,
    }


def _make_config() -> GatewayConfig:
    return GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token=SECRET_TOKEN,
                extra={
                    "allowed_users": ["123"],
                    "allowed_channels": ["456"],
                },
            ),
            Platform.WEBHOOK: PlatformConfig(
                enabled=True,
                extra={"routes": {"dobby": _strict_webhook_route()}},
            ),
        }
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = _make_config()
    runner.adapters = {
        Platform.DISCORD: SimpleNamespace(_allowed_user_ids={"123"}, _allowed_role_ids=set()),
    }
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._draining = False
    runner._session_key_for_source = lambda source: build_session_key(source)
    runner._is_user_authorized = lambda _source: True
    return runner


def test_dobby_command_is_registered_for_gateway():
    cmd = resolve_command("dobby")

    assert cmd is not None
    assert cmd.gateway_only is True
    assert cmd.subcommands == ("status", "help")
    assert "dobby" in GATEWAY_KNOWN_COMMANDS
    assert SUBCOMMANDS["/dobby"] == ["status", "help"]


@pytest.mark.asyncio
async def test_dobby_status_dispatch_is_read_only_and_redacted():
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("/dobby reached agent"))

    result = await runner._handle_message(_make_event("/dobby status"))

    assert "Dobby Package Status" in result
    assert "- Enabled platforms: discord, webhook" in result
    assert "- Discord allowlist: present" in result
    assert "- Webhook strict policy: present" in result
    assert "- Pending:" in result
    assert "research" in result
    assert SECRET_TOKEN not in result
    assert WEBHOOK_SECRET not in result
    runner._run_agent.assert_not_called()
    runner.hooks.emit.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()
    runner.session_store.append_to_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()


@pytest.mark.asyncio
async def test_dobby_status_bypasses_running_agent_without_interrupt():
    runner = _make_runner()
    running_agent = MagicMock()
    runner._running_agents[build_session_key(_make_source())] = running_agent
    runner._run_agent = AsyncMock(side_effect=AssertionError("/dobby reached agent"))

    result = await runner._handle_message(_make_event("/dobby status"))

    assert "Dobby Package Status" in result
    running_agent.interrupt.assert_not_called()
    runner._run_agent.assert_not_called()
    assert runner._pending_messages == {}


@pytest.mark.asyncio
async def test_dobby_status_bypasses_hooks_and_stale_agent_cleanup():
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    stale_agent = MagicMock()
    stale_agent.get_activity_summary.return_value = {"seconds_since_activity": 999999}
    runner._running_agents[session_key] = stale_agent
    runner._running_agents_ts[session_key] = time.time() - 999999
    runner._release_running_agent_state = MagicMock(
        side_effect=AssertionError("/dobby status released running-agent state")
    )
    runner._run_agent = AsyncMock(side_effect=AssertionError("/dobby reached agent"))

    result = await runner._handle_message(_make_event("/dobby status"))

    assert "Dobby Package Status" in result
    runner.hooks.emit.assert_not_called()
    runner._release_running_agent_state.assert_not_called()
    runner._run_agent.assert_not_called()
    assert runner._running_agents[session_key] is stale_agent


@pytest.mark.asyncio
async def test_dobby_help_uses_strict_read_only_gateway_path():
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("/dobby reached agent"))

    result = await runner._handle_message(_make_event("/dobby help"))

    assert "Dobby command center" in result
    runner.hooks.emit.assert_not_called()
    runner._run_agent.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
async def test_dobby_status_does_not_read_profile_from_environment():
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("/dobby reached agent"))

    with patch(
        "hermes_cli.profiles.get_active_profile_name",
        side_effect=AssertionError("/dobby read profile from environment"),
    ):
        result = await runner._handle_message(_make_event("/dobby status"))

    assert "- Profile: unknown" in result
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_dobby_unsupported_subcommand_fails_closed_without_agent():
    runner = _make_runner()
    runner._run_agent = AsyncMock(side_effect=AssertionError("/dobby reached agent"))

    result = await runner._handle_message(_make_event("/dobby deploy production"))

    assert result == "Dobby subcommand `deploy` is not implemented yet. Try `/dobby status`."
    runner._run_agent.assert_not_called()
    runner.session_store.get_or_create_session.assert_not_called()
    runner.session_store.append_to_transcript.assert_not_called()
    runner.session_store.update_session.assert_not_called()


def test_dobby_status_reports_safe_available_cron_count(tmp_path):
    package_root = tmp_path / "dobby-package"
    hermes_home = tmp_path / "home"
    package_root.mkdir()
    hermes_home.mkdir()
    config = {
        "platforms": {
            "discord": {
                "enabled": True,
                "extra": {"allowed_users": ["123"], "allowed_channels": ["456"]},
            },
            "webhook": {
                "enabled": True,
                "extra": {"routes": {"dobby": _strict_webhook_route()}},
            },
        },
        "browser": {"enabled": False},
        "memory": {"memory_enabled": False, "user_profile_enabled": False, "provider": ""},
        "honcho": {},
        "external_memory_providers": {"enabled": False},
    }

    result = render_dobby_status(
        config=config,
        package_root=package_root,
        hermes_home=hermes_home,
        profile_name="default",
        cron_count=2,
    )

    assert "- Readiness: ready" in result
    assert "- Browser integration: disabled" in result
    assert "- Memory/Honcho boundary: native off; Honcho off" in result
    assert "- Cron jobs: 2" in result
    assert SECRET_TOKEN not in result
    assert WEBHOOK_SECRET not in result


def test_dobby_status_marks_unimplemented_features_pending():
    result = handle_dobby_command("status", config=_make_config(), profile_name="default")

    assert "- Pending:" in result
    assert "quota" in result
    assert "reminders" in result
    assert "attachment review" in result
    assert "repo helper" in result


def test_dobby_status_does_not_create_session_entries():
    runner = _make_runner()

    result = handle_dobby_command(
        "status",
        config=runner.config,
        adapters=runner.adapters,
        profile_name="default",
    )

    assert "Dobby Package Status" in result
    runner.session_store.get_or_create_session.assert_not_called()
