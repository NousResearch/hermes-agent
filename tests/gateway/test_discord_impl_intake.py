"""Tests for Discord intake and preflight helpers."""

from types import SimpleNamespace

from gateway.platforms.discord_impl import intake as discord_intake


class FakeForumChannel:
    def __init__(self, channel_id=1, name="forum", guild_name="Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)
        self.type = 15


class FakeTextChannel:
    def __init__(self, channel_id=1, name="general", guild_name="Hermes Server"):
        self.id = channel_id
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)


class FakeThread:
    def __init__(self, channel_id=1, name="thread", parent=None, guild_name="Hermes Server"):
        self.id = channel_id
        self.name = name
        self.parent = parent
        self.parent_id = getattr(parent, "id", None)
        self.guild = getattr(parent, "guild", None) or SimpleNamespace(name=guild_name)


def test_should_filter_bot_message_matches_policy():
    assert discord_intake.should_filter_bot_message(False, "none", False) is False
    assert discord_intake.should_filter_bot_message(True, "none", False) is True
    assert discord_intake.should_filter_bot_message(True, "mentions", False) is True
    assert discord_intake.should_filter_bot_message(True, "mentions", True) is False
    assert discord_intake.should_filter_bot_message(True, "all", False) is False
    assert discord_intake.should_filter_bot_message(True, "weird", False) is False


def test_should_skip_for_mention_matches_gate_conditions():
    assert discord_intake.should_skip_for_mention(True, False, False, False) is True
    assert discord_intake.should_skip_for_mention(True, True, False, False) is False
    assert discord_intake.should_skip_for_mention(True, False, True, False) is False
    assert discord_intake.should_skip_for_mention(True, False, False, True) is False
    assert discord_intake.should_skip_for_mention(False, False, False, False) is False


def test_strip_mention_removes_both_mention_forms():
    assert discord_intake.strip_mention("<@123> hello", 123) == "hello"
    assert discord_intake.strip_mention("<@!123> hello", 123) == "hello"
    assert discord_intake.strip_mention("before <@123> and <@!123> after", 123) == "before  and  after"


def test_classify_message_type_prefers_commands_and_attachment_types():
    assert discord_intake.classify_message_type("/status", []) == "command"
    assert discord_intake.classify_message_type("hello", [SimpleNamespace(content_type="image/png")]) == "photo"
    assert discord_intake.classify_message_type("hello", [SimpleNamespace(content_type="video/mp4")]) == "video"
    assert discord_intake.classify_message_type("hello", [SimpleNamespace(content_type="audio/ogg")]) == "audio"
    assert discord_intake.classify_message_type("hello", [SimpleNamespace(content_type="application/pdf")]) == "document"
    assert discord_intake.classify_message_type("hello", [SimpleNamespace(content_type=None)]) == "text"


def test_get_parent_channel_id_prefers_parent_object():
    parent = SimpleNamespace(id=222)
    channel = SimpleNamespace(parent=parent, parent_id=333)

    assert discord_intake.get_parent_channel_id(channel) == "222"


def test_get_parent_channel_id_falls_back_to_parent_id():
    channel = SimpleNamespace(parent=None, parent_id=333)

    assert discord_intake.get_parent_channel_id(channel) == "333"


def test_is_forum_parent_checks_discord_class_and_type(monkeypatch):
    monkeypatch.setattr(
        discord_intake,
        "discord",
        SimpleNamespace(ForumChannel=FakeForumChannel),
        raising=False,
    )

    assert discord_intake.is_forum_parent(FakeForumChannel()) is True
    assert discord_intake.is_forum_parent(SimpleNamespace(type=15)) is True
    assert discord_intake.is_forum_parent(SimpleNamespace(type=0)) is False
    assert discord_intake.is_forum_parent(None) is False


def test_format_thread_chat_name_includes_forum_context():
    forum = FakeForumChannel(name="support-forum")
    thread = FakeThread(name="Forum topic", parent=forum)

    assert (
        discord_intake.format_thread_chat_name(thread, discord_intake.is_forum_parent)
        == "Hermes Server / support-forum / Forum topic"
    )


def test_format_thread_chat_name_formats_regular_threads():
    parent = FakeTextChannel(name="general")
    thread = FakeThread(name="Follow-up", parent=parent)

    assert (
        discord_intake.format_thread_chat_name(thread, discord_intake.is_forum_parent)
        == "Hermes Server / #general / Follow-up"
    )
