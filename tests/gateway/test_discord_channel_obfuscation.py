"""Discord channel obfuscation (Aug 2026 privacy change, HTTP omission mandatory Nov 16 2026):
channels a bot lacks VIEW_CHANNEL on arrive named "___hidden___" with flag 1 << 17 set.
Enumeration sites must skip them. #90154"""

from types import SimpleNamespace

from gateway.platforms.helpers import (
    DISCORD_CHANNEL_OBFUSCATED_FLAG,
    DISCORD_OBFUSCATED_CHANNEL_NAME,
    is_discord_channel_obfuscated,
)


def _channel(name="general", flag_value=0, cid=123):
    return SimpleNamespace(id=cid, name=name, flags=SimpleNamespace(value=flag_value))


def test_flag_or_sentinel_name_marks_obfuscated():
    assert is_discord_channel_obfuscated(_channel(name="secret", flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG | (1 << 4)))
    # Older discord.py builds may not surface the flag bit; the name is the fallback signal.
    assert is_discord_channel_obfuscated(_channel(name=DISCORD_OBFUSCATED_CHANNEL_NAME))
    assert not is_discord_channel_obfuscated(_channel())
    assert not is_discord_channel_obfuscated(SimpleNamespace(id=1, name="ok"))  # no flags attribute


def test_build_discord_filters_hidden_channels(monkeypatch):
    from gateway import channel_directory as cd

    guild = SimpleNamespace(
        name="TestGuild",
        text_channels=[
            _channel(name="general", cid=1),
            _channel(name="secret", flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG, cid=2),
            _channel(name=DISCORD_OBFUSCATED_CHANNEL_NAME, cid=3),
        ],
        forums=[
            _channel(name="forum-open", cid=4),
            _channel(name="forum-secret", flag_value=DISCORD_CHANNEL_OBFUSCATED_FLAG, cid=5),
        ],
    )
    adapter = SimpleNamespace(_client=SimpleNamespace(guilds=[guild]))
    monkeypatch.setattr(cd, "_build_from_sessions", lambda platform: [])

    channels = cd._build_discord(adapter)

    assert {c["id"]: c["type"] for c in channels} == {"1": "channel", "4": "forum"}


def test_guild_channel_attributes_match_the_installed_discord_py():
    """The stubs above are SimpleNamespace, so a typo'd attribute name would satisfy
    them while returning nothing from the real library -- which is exactly how
    `forum_channels` (py-cord's name, never discord.py's) shipped and silently hid
    every forum from the directory. Pin the names against the installed client."""
    import discord

    for attr in ("text_channels", "forums"):
        assert hasattr(discord.Guild, attr), f"discord.Guild has no {attr!r}"
