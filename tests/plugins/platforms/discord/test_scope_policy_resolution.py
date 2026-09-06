import pytest

from plugins.platforms.discord.policy import resolve_scope_policy, validate_scope_policies


RAW = {
    "version": 1,
    "platform": {"defaults": {"require_mention": False, "allow_humans": True}},
    "guilds": {
        "10": {
            "defaults": {"allow_bots": False},
            "channels": {"20": {"require_mention": True, "conversation_trust": "private"}},
            "threads": {"30": {"require_mention": False, "conversation_trust": "full_trusted"}},
        }
    },
}


def test_fieldwise_channel_and_thread_inheritance_and_explicit_false():
    channel = resolve_scope_policy(RAW, "10", "20")
    assert channel.require_mention is True
    assert channel.allow_humans is True
    assert channel.allow_bots is False
    assert channel.conversation_trust == "private"
    assert channel.sources["require_mention"] == "channel:20"

    thread = resolve_scope_policy(RAW, "10", "30", "20")
    assert thread.require_mention is False
    assert thread.allow_humans is True
    assert thread.allow_bots is False
    assert thread.conversation_trust == "full_trusted"
    assert thread.sources["conversation_trust"] == "thread:30"


def test_missing_scope_is_legacy_none_and_unknown_guild_uses_platform_defaults():
    assert resolve_scope_policy(None, "10", "20").require_mention is None
    unknown = resolve_scope_policy(RAW, "999", "20")
    assert unknown.require_mention is False
    assert unknown.allow_humans is True
    assert unknown.allow_bots is None


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"guilds": {"not-an-id": {}}}, "numeric Discord ID"),
        ({"guilds": {"123": {"conversation_trust": "public"}}}, "unknown key"),
        ({"platform": {"defaults": {"require_mention": "false"}}}, "must be a boolean"),
        ({"platform": {"defaults": {"conversation_trust": "owner"}}}, "must be one of"),
        ({"platform": {"defaults": {"unknown": True}}}, "unknown field"),
    ],
)
def test_schema_rejects_invalid_values_deterministically(raw, message):
    with pytest.raises(ValueError, match=message):
        validate_scope_policies(raw)
