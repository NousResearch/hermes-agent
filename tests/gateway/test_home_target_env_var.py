"""Regression tests for /sethome env-var resolution.

The `/sethome` command writes to a platform's home-target env var. Two platforms
don't follow the `{PLATFORM}_HOME_CHANNEL` convention: matrix uses
`MATRIX_HOME_ROOM` and email uses `EMAIL_HOME_ADDRESS`. Before PR #12698
`/sethome` hardcoded the `_HOME_CHANNEL` suffix, so Matrix and Email saves went
to env vars nothing read on startup — the home channel appeared to set
successfully but was lost on every new gateway session.
"""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.run import _home_target_env_var, _home_thread_env_var, _should_prompt_home_channel


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


def test_plugin_platform_can_suppress_home_channel_prompt(monkeypatch):
    from gateway.platform_registry import PlatformEntry, platform_registry

    previous_entry = platform_registry.get("linear")
    platform_registry.register(PlatformEntry(
        name="linear",
        label="Linear",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
        suppress_home_channel_prompt=True,
    ))
    monkeypatch.delenv("LINEAR_HOME_CHANNEL", raising=False)
    try:
        source = SimpleNamespace(
            platform=Platform("linear"),
            chat_id="agentSession:as_123",
            chat_type="dm",
            user_id="linear-user",
        )
        assert _should_prompt_home_channel(source) is False
    finally:
        if previous_entry is None:
            platform_registry.unregister("linear")
        else:
            platform_registry.register(previous_entry)


def test_regular_platform_without_home_channel_gets_prompt(monkeypatch):
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="user-1",
    )

    assert _should_prompt_home_channel(source) is True
