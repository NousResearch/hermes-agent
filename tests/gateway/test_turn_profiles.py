from __future__ import annotations

from gateway.turn_profiles import resolve_turn_profile


def test_discord_default_is_conversation_tools() -> None:
    resolved = resolve_turn_profile(
        platform="discord",
        prompt="Mi volt a debug markerem?",
        current_enabled_toolsets=["hermes-cli"],
        env={},
    )

    assert resolved.turn_profile == "conversation_tools"
    assert resolved.enabled_toolsets == ("conversation_tools",)
    assert resolved.reason_code == "DISCORD_DEFAULT_CONVERSATION"


def test_cli_local_default_unchanged() -> None:
    resolved = resolve_turn_profile(
        platform="cli",
        prompt="run tests",
        current_enabled_toolsets=["hermes-cli"],
        env={},
    )

    assert resolved.cli_local_unchanged is True
    assert resolved.enabled_toolsets == ("hermes-cli",)


def test_heavy_command_selects_structural_bundle() -> None:
    web = resolve_turn_profile(
        platform="discord",
        prompt="/heavy web https://example.com",
        current_enabled_toolsets=["hermes-cli"],
        env={},
    )
    code = resolve_turn_profile(
        platform="discord",
        prompt="/heavy code fix tests",
        current_enabled_toolsets=["hermes-cli"],
        env={},
    )

    assert web.turn_profile == "heavy_web"
    assert web.enabled_toolsets == ("heavy_web",)
    assert code.turn_profile == "heavy_code"
    assert code.enabled_toolsets == ("heavy_code",)


def test_url_mention_is_candidate_not_auto_fetch() -> None:
    resolved = resolve_turn_profile(
        platform="discord",
        prompt="Jegyezd meg ezt az URL-t: https://example.com/repo",
        current_enabled_toolsets=["hermes-cli"],
        env={},
    )

    assert resolved.turn_profile == "conversation_tools"
    assert resolved.url_attachment_candidate_only is True
    assert resolved.explicit_heavy is False


def test_rollback_override_forces_heavy_without_redeploy() -> None:
    resolved = resolve_turn_profile(
        platform="discord",
        prompt="Mi volt a markerem?",
        current_enabled_toolsets=["web", "file"],
        env={"HERMES_DISCORD_TURN_PROFILE": "heavy"},
    )

    assert resolved.turn_profile == "heavy_work"
    assert resolved.enabled_toolsets == ("file", "web")
    assert resolved.rollback_override_active is True


def test_conversation_direct_override_has_no_tools() -> None:
    resolved = resolve_turn_profile(
        platform="discord",
        prompt="Mi volt a markerem?",
        current_enabled_toolsets=["hermes-cli"],
        env={"HERMES_DISCORD_CONVERSATION_DIRECT": "1"},
    )

    assert resolved.turn_profile == "conversation_direct"
    assert resolved.enabled_toolsets == ()
