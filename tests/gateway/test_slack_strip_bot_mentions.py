"""``slack.strip_bot_mentions``: whether the agent sees its own mention.

Default ``True`` deletes the bot's own ``<@U…>`` token before the agent reads
the message (the historical behaviour, pinned by
``test_slack.py::test_channel_mention_strips_bot_id``). ``False`` keeps it,
rendered as ``@BotName`` like any other participant's, so an explicit tag is
distinguishable from a thread-routed wake-up. Routing and command parsing are
unaffected either way.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import (
    SlackAdapter,
    _apply_yaml_config,
    _ThreadContextCache,
)


@pytest.fixture(autouse=True)
def _clean_env():
    """``_apply_yaml_config`` writes ``os.environ`` directly — restore it."""
    saved = os.environ.get("SLACK_STRIP_BOT_MENTIONS")
    os.environ.pop("SLACK_STRIP_BOT_MENTIONS", None)
    yield
    os.environ.pop("SLACK_STRIP_BOT_MENTIONS", None)
    if saved is not None:
        os.environ["SLACK_STRIP_BOT_MENTIONS"] = saved


def make_adapter(extra=None):
    return SlackAdapter(PlatformConfig(enabled=True, token="***", extra=extra or {}))


def delivery_adapter(strip, bot_name="TestBot", team_names=None):
    """Adapter wired to capture what ``_handle_slack_message`` delivers.

    Everything the trigger path would fetch from Slack is stubbed; the routing
    and text-building code under test is the real thing.
    """
    adapter = make_adapter({"strip_bot_mentions": strip})
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_bot_user_ids["T123"] = "U_BOT"
    adapter._bot_display_name = bot_name
    adapter._team_bot_names = dict(team_names or {})
    adapter._running = True
    adapter.handle_message = AsyncMock()
    adapter._has_active_session_for_thread = lambda **_: False

    async def _no_thread_context(**_):
        return ""

    async def _no_parent_text(**_):
        return ""

    async def _no_thread_images(**_):
        return [], []

    async def _resolve_user_name(user_id, chat_id="", team_id=""):
        return {"U_USER": "Nikita"}.get(user_id, "")

    adapter._fetch_thread_context = _no_thread_context
    adapter._fetch_thread_parent_text = _no_parent_text
    adapter._collect_thread_root_images = _no_thread_images
    adapter._resolve_user_name = _resolve_user_name
    return adapter


def thread_adapter(strip):
    """Like :func:`delivery_adapter`, but with the real parent-text lookup.

    The thread tests exercise ``_fetch_thread_parent_text`` itself, so its
    delivery-path stub is dropped (the cache is primed instead of Slack).
    """
    adapter = delivery_adapter(strip=strip)
    del adapter._fetch_thread_parent_text
    return adapter


def slack_event(text, ts="1234567890.000001", thread_ts=None, team="T123", **extra):
    event = {
        "type": "message",
        "channel": "C123",
        "channel_type": "channel",
        "team": team,
        "user": "U_USER",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    event.update(extra)
    return event


def delivered(adapter):
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0]


# The Slack composer mirrors the flat text in a rich_text block on every
# message; for a bare mention that mirror is the mention alone.
MENTION_ONLY_BLOCKS = [
    {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_section",
                "elements": [{"type": "user", "user_id": "U_BOT"}],
            }
        ],
    }
]


class TestFlagResolution:
    """``config.extra`` → ``SLACK_STRIP_BOT_MENTIONS`` → default ``True``."""

    def test_defaults_to_stripping(self):
        assert make_adapter()._slack_strip_bot_mentions() is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "OFF"])
    def test_env_turns_stripping_off(self, value, monkeypatch):
        monkeypatch.setenv("SLACK_STRIP_BOT_MENTIONS", value)
        assert make_adapter()._slack_strip_bot_mentions() is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "", "maybe"])
    def test_unrecognised_env_keeps_stripping(self, value, monkeypatch):
        """Explicit-false parsing: only a known negative disables the strip."""
        monkeypatch.setenv("SLACK_STRIP_BOT_MENTIONS", value)
        assert make_adapter()._slack_strip_bot_mentions() is True

    def test_config_extra_beats_env(self, monkeypatch):
        monkeypatch.setenv("SLACK_STRIP_BOT_MENTIONS", "false")
        assert make_adapter({"strip_bot_mentions": True})._slack_strip_bot_mentions() is True

    def test_config_extra_string_forms(self):
        assert (
            make_adapter({"strip_bot_mentions": "false"})._slack_strip_bot_mentions()
            is False
        )
        assert (
            make_adapter({"strip_bot_mentions": "true"})._slack_strip_bot_mentions()
            is True
        )

    def test_config_extra_bool_false(self):
        assert (
            make_adapter({"strip_bot_mentions": False})._slack_strip_bot_mentions()
            is False
        )


class TestYamlEnvBridge:
    """``config.yaml`` is the canonical surface; the env var is its mirror."""

    def test_bridges_config_yaml_to_env(self):
        _apply_yaml_config({}, {"strip_bot_mentions": False})
        assert os.environ["SLACK_STRIP_BOT_MENTIONS"] == "false"
        assert make_adapter()._slack_strip_bot_mentions() is False

    def test_does_not_overwrite_an_explicit_env_var(self, monkeypatch):
        monkeypatch.setenv("SLACK_STRIP_BOT_MENTIONS", "false")
        _apply_yaml_config({}, {"strip_bot_mentions": True})
        assert os.environ["SLACK_STRIP_BOT_MENTIONS"] == "false"

    def test_absent_key_leaves_env_alone(self):
        """A config.yaml without the key must not pin the mirror either way."""
        _apply_yaml_config({}, {})
        assert "SLACK_STRIP_BOT_MENTIONS" not in os.environ


class TestRenderOwnMention:
    """``_render_own_mention``: the token becomes ``@Name`` where it stood."""

    def test_renders_in_place(self):
        assert (
            SlackAdapter._render_own_mention("hey <@U_BOT> look", "U_BOT", "TestBot")
            == "hey @TestBot look"
        )

    def test_renders_the_labelled_form(self):
        """Slack also delivers ``<@U123|name>`` (legacy / some clients)."""
        assert (
            SlackAdapter._render_own_mention("<@U_BOT|yana> hi", "U_BOT", "TestBot")
            == "@TestBot hi"
        )

    def test_mention_only_text_is_not_emptied(self):
        assert (
            SlackAdapter._render_own_mention("<@U_BOT>", "U_BOT", "TestBot")
            == "@TestBot"
        )

    def test_unknown_name_leaves_the_raw_token(self):
        """An unresolved id still beats a deleted mention."""
        assert (
            SlackAdapter._render_own_mention("<@U_BOT> ping", "U_BOT", "")
            == "<@U_BOT> ping"
        )

    def test_other_participants_are_untouched(self):
        assert (
            SlackAdapter._render_own_mention("<@U_OTHER> ping", "U_BOT", "TestBot")
            == "<@U_OTHER> ping"
        )

    @pytest.mark.parametrize("name", ["Te\\st", "Bot\\1", "\\g<0>"])
    def test_regex_metacharacters_in_the_name_are_literal(self, name):
        """A display name is user data — it must never be read as a template."""
        assert (
            SlackAdapter._render_own_mention("<@U_BOT> ping", "U_BOT", name)
            == f"@{name} ping"
        )


class TestTriggerMessageText:
    """What the agent actually receives for an explicitly mentioned message."""

    @pytest.mark.asyncio
    async def test_default_strips_the_mention(self):
        adapter = delivery_adapter(strip=True)

        await adapter._handle_slack_message(slack_event("<@U_BOT> what's up?"))

        assert delivered(adapter).text == "what's up?"

    @pytest.mark.asyncio
    async def test_flag_off_keeps_the_mention_in_place(self):
        adapter = delivery_adapter(strip=False)

        await adapter._handle_slack_message(slack_event("hey <@U_BOT> look"))

        event = delivered(adapter)
        assert event.text == "hey @TestBot look"
        assert "<@U_BOT>" not in event.text

    @pytest.mark.asyncio
    async def test_flag_off_mention_only_message(self):
        adapter = delivery_adapter(strip=False)

        await adapter._handle_slack_message(slack_event("<@U_BOT>"))

        assert delivered(adapter).text == "@TestBot"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip,expected", [(True, ""), (False, "@TestBot")])
    async def test_mention_only_message_with_blocks_is_not_duplicated(
        self, strip, expected
    ):
        """Real Slack messages carry a rich_text mirror of their own text.

        Block dedupe deletes ``<@bot_uid>`` from both sides to bridge the strip
        below, which reduces a mention-only block to the empty string — read as
        new content and appended, so the mention would arrive twice.
        """
        adapter = delivery_adapter(strip=strip)

        await adapter._handle_slack_message(
            slack_event("<@U_BOT>", blocks=MENTION_ONLY_BLOCKS)
        )

        assert delivered(adapter).text == expected

    @pytest.mark.asyncio
    async def test_flag_off_uses_the_per_workspace_name(self):
        """Multi-workspace: the bot's handle is team-scoped, not global."""
        adapter = delivery_adapter(
            strip=False, bot_name="TestBot", team_names={"T123": "WorkspaceBot"}
        )

        await adapter._handle_slack_message(slack_event("<@U_BOT> ping"))

        assert delivered(adapter).text == "@WorkspaceBot ping"

    @pytest.mark.asyncio
    async def test_flag_off_without_a_known_name_keeps_the_mention(self):
        """Before connect resolves a handle: never delete, never crash."""
        adapter = delivery_adapter(strip=False, bot_name=None)

        await adapter._handle_slack_message(slack_event("<@U_BOT> ping"))

        text = delivered(adapter).text
        assert "ping" in text
        assert text != "ping", "the mention must not be dropped"

    @pytest.mark.asyncio
    async def test_flag_off_edited_message_follows_the_flag(self):
        """``message_changed`` is normalized into the same trigger path."""
        adapter = delivery_adapter(strip=False)

        await adapter._handle_slack_message(
            {
                "subtype": "message_changed",
                "channel": "C123",
                "channel_type": "channel",
                "team": "T123",
                "ts": "1234567890.000001",
                "message": {
                    "text": "<@U_BOT> take another look",
                    "user": "U_USER",
                    "channel": "C123",
                    "ts": "1234567890.000001",
                    "edited": {"user": "U_USER", "ts": "1234567899.000001"},
                },
            }
        )

        assert delivered(adapter).text == "@TestBot take another look"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_unmentioned_thread_wakeup_is_delivered_unchanged(self, strip):
        """The asymmetry is the signal: no tag ⇒ nothing added to the text."""
        adapter = delivery_adapter(strip=strip)
        adapter._register_mentioned_thread("100.000", team_id="T123")

        await adapter._handle_slack_message(
            slack_event("and then we ship", ts="101.000", thread_ts="100.000")
        )

        assert delivered(adapter).text == "and then we ship"


class TestThreadContext:
    """History must read the same way as the message that woke the bot."""

    _PARENT = {"ts": "100.000", "user": "U_USER", "text": "<@U_BOT> check this"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip,expected", [(True, ""), (False, "@TestBot")])
    async def test_mention_only_parent_with_blocks_is_not_duplicated(
        self, strip, expected
    ):
        """Same dedupe contract on the thread-history path."""
        adapter = delivery_adapter(strip=strip)

        _content, parent_text = await adapter._format_thread_context(
            [
                {
                    "ts": "100.000",
                    "user": "U_USER",
                    "text": "<@U_BOT>",
                    "blocks": MENTION_ONLY_BLOCKS,
                }
            ],
            thread_ts="100.000",
            current_ts="101.000",
            team_id="T123",
            channel_id="C123",
        )

        assert parent_text == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "strip,expected", [(True, "check this"), (False, "@TestBot check this")]
    )
    async def test_context_lines_follow_the_flag(self, strip, expected):
        adapter = delivery_adapter(strip=strip)

        content, parent_text = await adapter._format_thread_context(
            [dict(self._PARENT)],
            thread_ts="100.000",
            current_ts="101.000",
            team_id="T123",
            channel_id="C123",
        )

        assert parent_text == expected
        assert expected in content
        assert "<@U_BOT>" not in content

    @pytest.mark.asyncio
    async def test_keeping_the_mention_does_not_duplicate_the_message(self):
        """Block content is compared against the text as written.

        Slack mirrors an authored message into a ``rich_text`` block carrying
        the raw token, so comparing it against the rendered ``@BotName`` would
        append the same message a second time.
        """
        adapter = delivery_adapter(strip=False)
        parent = {
            "ts": "100.000",
            "user": "U_USER",
            "text": "hey <@U_BOT> look at this",
            "blocks": [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {"type": "text", "text": "hey "},
                                {"type": "user", "user_id": "U_BOT"},
                                {"type": "text", "text": " look at this"},
                            ],
                        }
                    ],
                }
            ],
        }

        _content, parent_text = await adapter._format_thread_context(
            [parent],
            thread_ts="100.000",
            current_ts="101.000",
            team_id="T123",
            channel_id="C123",
        )

        assert parent_text == "hey @TestBot look at this"
        assert "<@U_BOT>" not in parent_text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_root_mention_detection_still_sees_the_raw_token(self, strip):
        """The wake check greps for ``<@id>`` — the flag must not rewrite it."""
        adapter = thread_adapter(strip)
        adapter._thread_context_cache["C123:100.000:T123"] = _ThreadContextCache(
            content="",
            parent_text="already rendered",
            messages=[dict(self._PARENT)],
        )

        text = await adapter._fetch_thread_parent_text(
            channel_id="C123",
            thread_ts="100.000",
            team_id="T123",
            strip_bot_mention=False,
        )

        assert "<@U_BOT>" in text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_root_mention_detection_survives_a_cold_cache(self, strip):
        """Same contract on the fetch path — that's the #24848 case.

        The wake check runs during routing, before any thread-context fetch has
        primed the cache, so after a restart the detector always lands here.
        """
        adapter = thread_adapter(strip)
        client = AsyncMock()
        client.conversations_replies.return_value = {
            "messages": [dict(self._PARENT)]
        }
        adapter._get_client = lambda *_a, **_kw: client

        text = await adapter._fetch_thread_parent_text(
            channel_id="C123",
            thread_ts="100.000",
            team_id="T123",
            strip_bot_mention=False,
        )

        assert "<@U_BOT>" in text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_parent_wake_check_wakes_the_bot_in_both_states(self, strip):
        """End of the chain: a mention in the root still wakes on a plain reply."""
        adapter = thread_adapter(strip)
        adapter._thread_context_cache["C123:100.000:T123"] = _ThreadContextCache(
            content="",
            parent_text="already rendered",
            messages=[dict(self._PARENT)],
        )

        assert await adapter._should_wake_on_unmentioned_message(
            event_thread_ts="100.000",
            channel_id="C123",
            user_id="U_USER",
            is_thread_reply=True,
            team_id="T123",
        )


class TestCommandsUnaffected:
    """Command parsing runs off its own variable — both flag states dispatch."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_slash_command_behind_a_mention(self, strip):
        adapter = delivery_adapter(strip=strip)

        await adapter._handle_slack_message(slack_event("<@U_BOT> /status"))

        event = delivered(adapter)
        assert event.text == "/status"
        assert event.is_command()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("strip", [True, False])
    async def test_bang_command_behind_a_mention(self, strip):
        adapter = delivery_adapter(strip=strip)

        await adapter._handle_slack_message(slack_event("<@U_BOT> !new"))

        event = delivered(adapter)
        assert event.text == "/new"
        assert event.is_command()
