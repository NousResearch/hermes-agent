"""Unit tests for the Discord adapter's opt-in outbound mention resolution.

Exercises the real method on an init-bypassed adapter instance so the test
rides the module's real import path (it would fail if adapter.py stopped
importing re/os), without reading or exec'ing the adapter source.

The second class covers the config.yaml path end to end: setting
``discord.resolve_outbound_mentions`` has to actually reach the adapter, not
just exist in the defaults. Declaring a setting that never reaches its call
site is the failure mode these tests exist to catch.
"""
import asyncio

import pytest

from gateway.config import Platform
from plugins.platforms.discord.adapter import DiscordAdapter, _apply_yaml_config


class _Member:
    def __init__(self, id, display_name=None, name=None, global_name=None):
        self.id = id
        self.display_name = display_name
        self.name = name
        self.global_name = global_name


class _Guild:
    def __init__(self, members):
        self.members = members


class _Channel:
    def __init__(self, guild):
        self.guild = guild


def _run(monkeypatch, content, flag="true"):
    # Bypass the heavy __init__ (needs a live client); the method under test
    # reads no instance state, only the env flag and channel.guild.members.
    adapter = object.__new__(DiscordAdapter)
    guild = _Guild([
        _Member(200, display_name="Support Bot", name="supportbot"),
        _Member(300, display_name="Alice", name="alice"),
        _Member(400, display_name="Al", name="al"),
    ])
    ch = _Channel(guild)
    # monkeypatch so the flag never leaks into unrelated tests.
    if flag is None:
        monkeypatch.delenv("DISCORD_RESOLVE_MENTIONS", raising=False)
    else:
        monkeypatch.setenv("DISCORD_RESOLVE_MENTIONS", flag)
    return asyncio.run(adapter._resolve_outbound_mentions(content, ch))


@pytest.mark.parametrize("content,expected", [
    ("@Support Bot can you take this?", "<@200> can you take this?"),   # multi-word name
    ("@support bot pls", "<@200> pls"),                                 # case-insensitive
    ("hey @Alice and @Support Bot", "hey <@300> and <@200>"),           # multiple, longest-first
    ("<@200> already tagged", "<@200> already tagged"),                 # already a real mention
    ("mail me@example.com", "mail me@example.com"),                     # not a mention (word-char before @)
    ("@Nobody here", "@Nobody here"),                                   # unknown name untouched
])
def test_resolves_when_enabled(monkeypatch, content, expected):
    assert _run(monkeypatch, content, "true") == expected


def test_noop_when_disabled(monkeypatch):
    assert _run(monkeypatch, "@Support Bot hi", "false") == "@Support Bot hi"


def test_noop_when_unset(monkeypatch):
    assert _run(monkeypatch, "@Support Bot hi", None) == "@Support Bot hi"


def test_shorter_name_still_resolves_alone(monkeypatch):
    assert _run(monkeypatch, "ping @Al now", "true") == "ping <@400> now"


class TestConfigYamlPath:
    """``discord.resolve_outbound_mentions`` in config.yaml must reach the adapter.

    The setting is the documented knob; ``DISCORD_RESOLVE_MENTIONS`` is the
    internal bridge ``_apply_yaml_config`` writes, the same arrangement
    ``discord.approval_mentions`` already uses.
    """

    def test_config_true_enables_resolution_end_to_end(self, monkeypatch):
        """Set it in config, and a written @Name comes out as a real mention."""
        monkeypatch.delenv("DISCORD_RESOLVE_MENTIONS", raising=False)

        _apply_yaml_config({}, {"resolve_outbound_mentions": True})

        adapter = object.__new__(DiscordAdapter)
        ch = _Channel(_Guild([_Member(200, display_name="Support Bot")]))
        out = asyncio.run(adapter._resolve_outbound_mentions("@Support Bot ping", ch))
        assert out == "<@200> ping", (
            "discord.resolve_outbound_mentions: true did not reach the adapter — "
            "the setting exists but is not wired to the send path"
        )

    def test_config_false_leaves_text_inert(self, monkeypatch):
        monkeypatch.delenv("DISCORD_RESOLVE_MENTIONS", raising=False)

        _apply_yaml_config({}, {"resolve_outbound_mentions": False})

        adapter = object.__new__(DiscordAdapter)
        ch = _Channel(_Guild([_Member(200, display_name="Support Bot")]))
        out = asyncio.run(adapter._resolve_outbound_mentions("@Support Bot ping", ch))
        assert out == "@Support Bot ping"

    def test_absent_from_config_is_off(self, monkeypatch):
        """Default off — an existing deployment sees no change until it opts in."""
        monkeypatch.delenv("DISCORD_RESOLVE_MENTIONS", raising=False)

        _apply_yaml_config({}, {})

        import os
        assert os.getenv("DISCORD_RESOLVE_MENTIONS") is None

    def test_platform_extra_also_carries_the_setting(self, monkeypatch):
        """platforms.discord.extra is the multiplexed-profile source of truth."""
        monkeypatch.delenv("DISCORD_RESOLVE_MENTIONS", raising=False)

        _apply_yaml_config(
            {"platforms": {"discord": {"extra": {"resolve_outbound_mentions": True}}}},
            {},
        )

        import os
        assert os.getenv("DISCORD_RESOLVE_MENTIONS") == "true"


class _SentChannel:
    """Minimal stand-in for a Discord text channel that records what it was sent."""

    def __init__(self, guild):
        self.guild = guild
        self.id = 99
        self.sent = []

    async def send(self, content=None, **kwargs):
        self.sent.append(content)
        return type("Msg", (), {"id": 12345})()


class _ForumChannel(_SentChannel):
    """Forum parent: rejects .send(), takes a starter post via create_thread()."""

    def __init__(self, guild):
        super().__init__(guild)
        self.created = []

    async def create_thread(self, name=None, content=None, **kwargs):
        self.created.append((name, content))
        thread = _SentChannel(self.guild)
        return type("Thread", (), {"id": 777, "thread": thread})()


def _adapter_for_send(monkeypatch, channel):
    """An adapter wired just enough to run send() against *channel*."""
    adapter = object.__new__(DiscordAdapter)
    # ``name`` is a read-only property derived from ``platform``; set that.
    adapter.platform = Platform.DISCORD
    adapter._client = type("C", (), {"get_channel": lambda _s, _i: channel})()
    adapter._reply_to_mode = "off"
    adapter.MAX_MESSAGE_LENGTH = 2000
    adapter.format_message = lambda c: c
    adapter.truncate_message = lambda c, _n: [c]
    adapter._record_discord_response = lambda **_kw: None
    monkeypatch.setenv("DISCORD_RESOLVE_MENTIONS", "true")
    return adapter


class TestSendCallSite:
    """send() must actually invoke the resolver — a method nothing calls is dead code.

    These are the tests that fail if the call site is dropped from send(); the
    unit tests above would all still pass, because they invoke the resolver
    directly.
    """

    def test_send_delivers_a_real_mention(self, monkeypatch):
        ch = _SentChannel(_Guild([_Member(200, display_name="Support Bot")]))
        adapter = _adapter_for_send(monkeypatch, ch)

        asyncio.run(adapter.send("99", "@Support Bot please look"))

        assert ch.sent == ["<@200> please look"], (
            "send() delivered the raw @Name — the resolver is not on the send path"
        )

    def test_forum_starter_post_is_resolved_too(self, monkeypatch):
        """The forum branch returns early; it must not skip mention resolution."""
        ch = _ForumChannel(_Guild([_Member(200, display_name="Support Bot")]))
        adapter = _adapter_for_send(monkeypatch, ch)
        adapter._is_forum_parent = lambda _c: True

        asyncio.run(adapter.send("99", "@Support Bot please look"))

        assert ch.created, "no forum thread was created — test proves nothing"
        _name, starter = ch.created[0]
        assert starter == "<@200> please look", (
            "the forum thread starter kept the raw @Name — resolution happens "
            "after the forum branch returns, so forum posts stay inert"
        )


class _EditableMessage:
    def __init__(self):
        self.edits = []

    async def edit(self, content=None, **_kw):
        self.edits.append(content)


class _EditChannel(_SentChannel):
    def __init__(self, guild, msg):
        super().__init__(guild)
        self._msg = msg

    def get_partial_message(self, _mid):
        # Upstream switched edit_message to a partial message (no API fetch).
        return self._msg

    async def fetch_message(self, _mid):
        return self._msg


def _adapter_for_edit(monkeypatch, channel):
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter._client = type("C", (), {"get_channel": lambda _s, _i: channel})()
    adapter.MAX_MESSAGE_LENGTH = 2000
    adapter.format_message = lambda c: c
    adapter.truncate_message = lambda c, _n: [c]
    adapter._last_overflow_preview = {}
    adapter._record_discord_response = lambda **_kw: None
    monkeypatch.setenv("DISCORD_RESOLVE_MENTIONS", "true")
    return adapter


class TestStreamingEditPath:
    """A streamed reply is delivered by edit_message(), not send().

    Covering only send() left every streaming response shipping inert @Name
    text, which is most of what the gateway actually delivers.
    """

    def test_final_edit_resolves_the_mention(self, monkeypatch):
        msg = _EditableMessage()
        ch = _EditChannel(_Guild([_Member(200, display_name="Support Bot")]), msg)
        adapter = _adapter_for_edit(monkeypatch, ch)

        asyncio.run(adapter.edit_message("99", "5", "@Support Bot please look", finalize=True))

        assert msg.edits == ["<@200> please look"], (
            "the final streamed edit kept the raw @Name — streaming delivery "
            "never gets a real mention"
        )

    def test_mid_stream_edit_is_left_alone(self, monkeypatch):
        """Partial text must not be resolved.

        While "@Alice" is still streaming it passes through "@Al", which is a
        different real member here. Resolving then would edit in a ping for the
        wrong person; the finalize pass delivers the correct one.
        """
        msg = _EditableMessage()
        ch = _EditChannel(
            _Guild([_Member(300, display_name="Alice"), _Member(400, display_name="Al")]),
            msg,
        )
        adapter = _adapter_for_edit(monkeypatch, ch)

        asyncio.run(adapter.edit_message("99", "5", "hey @Al", finalize=False))

        assert msg.edits == ["hey @Al"], "a partial name was resolved mid-stream"

    def test_overflow_split_path_receives_resolved_content(self, monkeypatch):
        """_edit_overflow_split re-formats the same content, so it must see it resolved."""
        msg = _EditableMessage()
        ch = _EditChannel(_Guild([_Member(200, display_name="Support Bot")]), msg)
        adapter = _adapter_for_edit(monkeypatch, ch)
        captured = {}

        async def _split(_channel, _msg, _mid, content):
            captured["content"] = content
            return None

        adapter._edit_overflow_split = _split
        adapter.MAX_MESSAGE_LENGTH = 5  # force the oversized branch

        asyncio.run(adapter.edit_message("99", "5", "@Support Bot please look", finalize=True))

        assert captured.get("content") == "<@200> please look", (
            "the overflow-split path was handed the raw @Name, so long streamed "
            "replies stay inert"
        )


class TestMemberIntent:
    """guild.members is empty without the privileged Server Members intent.

    Without gating the intent on this opt-in, turning the setting on resolves
    nothing and gives no clue why.
    """

    def _intent_enabled(self, monkeypatch, *, flag, allowed=(), roles=()):
        """Drive the real condition connect() uses, not a copy of it."""
        from plugins.platforms.discord.adapter import _needs_members_intent

        if flag is None:
            monkeypatch.delenv("DISCORD_RESOLVE_MENTIONS", raising=False)
        else:
            monkeypatch.setenv("DISCORD_RESOLVE_MENTIONS", flag)
        return _needs_members_intent(allowed, roles)

    def test_opt_in_requests_the_members_intent(self, monkeypatch):
        assert self._intent_enabled(monkeypatch, flag="true") is True

    def test_off_by_default_keeps_the_intent_unrequested(self, monkeypatch):
        assert self._intent_enabled(monkeypatch, flag=None) is False
        assert self._intent_enabled(monkeypatch, flag="false") is False

    def test_wildcard_allowlist_still_does_not_pull_the_intent(self, monkeypatch):
        """The migrate-from-OpenClaw wildcard path must stay unaffected."""
        assert self._intent_enabled(monkeypatch, flag=None, allowed=["*"]) is False

    def test_named_allowlist_and_roles_still_request_it(self, monkeypatch):
        """The pre-existing reasons must survive the new clause."""
        assert self._intent_enabled(monkeypatch, flag=None, allowed=["someuser"]) is True
        assert self._intent_enabled(monkeypatch, flag=None, roles=["12345"]) is True
        assert self._intent_enabled(monkeypatch, flag=None, allowed=["12345"]) is False
