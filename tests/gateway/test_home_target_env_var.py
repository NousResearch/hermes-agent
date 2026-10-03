"""Regression tests for /sethome env-var resolution.

The `/sethome` command writes to a platform's home-target env var. Two platforms
don't follow the `{PLATFORM}_HOME_CHANNEL` convention: matrix uses
`MATRIX_HOME_ROOM` and email uses `EMAIL_HOME_ADDRESS`. Before PR #12698
`/sethome` hardcoded the `_HOME_CHANNEL` suffix, so Matrix and Email saves went
to env vars nothing read on startup — the home channel appeared to set
successfully but was lost on every new gateway session.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.run import _home_target_env_var, _home_thread_env_var


def test_matrix_home_target_env_var_uses_home_room():
    assert _home_target_env_var("matrix") == "MATRIX_HOME_ROOM"


def test_email_home_target_env_var_uses_home_address():
    assert _home_target_env_var("email") == "EMAIL_HOME_ADDRESS"


def test_telegram_home_target_env_var_uses_home_channel():
    assert _home_target_env_var("telegram") == "TELEGRAM_HOME_CHANNEL"


def test_discord_home_target_env_var_uses_home_channel():
    assert _home_target_env_var("discord") == "DISCORD_HOME_CHANNEL"


def test_unknown_platform_home_target_env_var_falls_back_to_home_channel():
    assert _home_target_env_var("custom") == "CUSTOM_HOME_CHANNEL"


def test_case_insensitive_platform_name():
    assert _home_target_env_var("MATRIX") == "MATRIX_HOME_ROOM"
    assert _home_target_env_var("Email") == "EMAIL_HOME_ADDRESS"


def test_home_thread_env_var_uses_home_target_name_plus_thread_id():
    assert _home_thread_env_var("discord") == "DISCORD_HOME_CHANNEL_THREAD_ID"
    assert _home_thread_env_var("matrix") == "MATRIX_HOME_ROOM_THREAD_ID"
    assert _home_thread_env_var("email") == "EMAIL_HOME_ADDRESS_THREAD_ID"


@pytest.mark.asyncio
async def test_active_config_backed_email_home_suppresses_sethome_notice(
    monkeypatch,
):
    """The onboarding branch must inspect the runner's effective config."""
    from gateway import run as gateway_run
    from gateway.config import GatewayConfig, HomeChannel, Platform
    from tests.gateway.test_slash_access_dispatch import (
        _make_event,
        _make_runner,
        _make_source,
    )

    class ReachedAfterSetHomeBranch(Exception):
        pass

    runner = _make_runner(platform=Platform.EMAIL)
    runner.config.platforms[Platform.EMAIL].home_channel = HomeChannel(
        platform=Platform.EMAIL,
        chat_id="andy@example.com",
        name="Andy",
    )
    runner._deliver_platform_notice = AsyncMock()
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=ReachedAfterSetHomeBranch
    )

    monkeypatch.delenv("EMAIL_HOME_ADDRESS", raising=False)
    monkeypatch.setattr(gateway_run, "load_gateway_config", GatewayConfig)

    source = _make_source(platform=Platform.EMAIL, chat_id="andy@example.com")
    with pytest.raises(ReachedAfterSetHomeBranch):
        await runner._handle_message(_make_event("hello", source))

    runner._deliver_platform_notice.assert_not_awaited()
