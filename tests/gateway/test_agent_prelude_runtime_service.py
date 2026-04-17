from types import SimpleNamespace

from gateway.agent_prelude_runtime_service import (
    append_discord_voice_channel_context,
    build_agent_start_hook_context,
)
from gateway.config import Platform


def test_append_discord_voice_channel_context_only_for_discord_with_context():
    adapter = SimpleNamespace(get_voice_channel_context=lambda guild_id: "VC state")

    result = append_discord_voice_channel_context(
        "base",
        platform=Platform.DISCORD,
        guild_id=123,
        adapter=adapter,
    )

    assert result == "base\n\nVC state"


def test_append_discord_voice_channel_context_skips_when_not_available():
    result = append_discord_voice_channel_context(
        "base",
        platform=Platform.QQ_NAPCAT,
        guild_id=123,
        adapter=SimpleNamespace(get_voice_channel_context=lambda guild_id: "VC state"),
    )

    assert result == "base"


def test_build_agent_start_hook_context_truncates_message():
    context = build_agent_start_hook_context(
        platform=Platform.QQ_NAPCAT,
        user_id="179033731",
        session_id="sess-1",
        message_text="x" * 800,
    )

    assert context["platform"] == "qq_napcat"
    assert context["user_id"] == "179033731"
    assert context["session_id"] == "sess-1"
    assert len(context["message"]) == 500
