from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import (
    DiscordAdapter,
    _discord_allow_bots_mode,
    _discord_allowed_bot_ids,
    _is_bot_dialogue_closure_only,
)


def test_bot_dialogue_closure_only_detects_ack_and_stop_phrases():
    assert _is_bot_dialogue_closure_only("👍")
    assert _is_bot_dialogue_closure_only("OK!")
    assert _is_bot_dialogue_closure_only("了解です〜。")
    assert _is_bot_dialogue_closure_only("（承知しました。）")
    assert _is_bot_dialogue_closure_only("（ここで止めます）")
    assert _is_bot_dialogue_closure_only("この会話は終了済みとして扱います。")


def test_bot_dialogue_closure_only_allows_substantive_reply():
    text = "了解です。追加で確認すると、実装前にログと設定を見たほうが安全です。"
    assert not _is_bot_dialogue_closure_only(text)


def test_bot_dialogue_closure_only_allows_short_substantive_ack():
    assert not _is_bot_dialogue_closure_only("了解しました。ログを確認します。")
    assert not _is_bot_dialogue_closure_only("承知しました。対応します。")


def test_bot_dialogue_closure_only_blocks_gateway_status_notices():
    assert _is_bot_dialogue_closure_only(
        "⚡ Interrupting current task (iteration 1/90). I'll respond to your message shortly."
    )
    assert _is_bot_dialogue_closure_only(
        "Operation interrupted: waiting for model response (16.5s elapsed)."
    )


def test_bot_dialogue_closure_only_does_not_block_questions():
    assert not _is_bot_dialogue_closure_only("大丈夫ですか？")
    assert not _is_bot_dialogue_closure_only("確認しましたか?")


def test_peer_bot_guard_blocks_after_configured_limit(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "2")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_RESET_SECONDS", "900")
    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    author = SimpleNamespace(id=456)

    def msg(content):
        return SimpleNamespace(content=content, channel=channel, author=author)

    assert adapter._accept_peer_bot_message(msg("1つ目の実質的な返答です。"))
    assert adapter._accept_peer_bot_message(msg("2つ目の実質的な返答です。"))
    assert not adapter._accept_peer_bot_message(msg("3つ目なので止まるべきです。"))


def test_peer_bot_guard_blocks_closure_without_incrementing_limit(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "2")
    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    author = SimpleNamespace(id=456)
    message = SimpleNamespace(content="承知しました。", channel=channel, author=author)

    assert not adapter._accept_peer_bot_message(message)
    assert adapter._bot_dialogue_counts["123"][1] == 0


def test_peer_bot_guard_allows_attachment_only_messages(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "2")
    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    author = SimpleNamespace(id=456)
    attachment = SimpleNamespace(filename="handoff.png", content_type="image/png")
    message = SimpleNamespace(
        content="",
        channel=channel,
        author=author,
        attachments=[attachment],
    )

    assert adapter._accept_peer_bot_message(message)
    assert adapter._bot_dialogue_counts["123"][1] == 1


def test_peer_bot_guard_allows_closure_text_when_attachment_present(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "2")
    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    author = SimpleNamespace(id=456)
    attachment = SimpleNamespace(filename="result.txt", content_type="text/plain")
    message = SimpleNamespace(
        content="完了しました。",
        channel=channel,
        author=author,
        attachments=[attachment],
    )

    assert adapter._accept_peer_bot_message(message)
    assert adapter._bot_dialogue_counts["123"][1] == 1


def test_peer_bot_guard_keys_threads_by_thread_channel_id(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "1")
    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    adapter = DiscordAdapter(PlatformConfig())
    author = SimpleNamespace(id=456)
    thread_a = SimpleNamespace(id=1001, parent_id=999)
    thread_b = SimpleNamespace(id=1002, parent_id=999)

    def msg(content, channel):
        return SimpleNamespace(content=content, channel=channel, author=author)

    assert adapter._accept_peer_bot_message(msg("thread A first substantive reply", thread_a))
    assert not adapter._accept_peer_bot_message(msg("thread A second substantive reply", thread_a))
    assert adapter._accept_peer_bot_message(msg("thread B first substantive reply", thread_b))


def test_peer_bot_guard_invalid_env_values_fall_back_safely(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "not-a-bool")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "1")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_RESET_SECONDS", "not-a-number")
    monkeypatch.delenv("DISCORD_ALLOWED_BOTS", raising=False)
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    author = SimpleNamespace(id=456)

    def msg(content):
        return SimpleNamespace(content=content, channel=channel, author=author)

    assert adapter._accept_peer_bot_message(msg("1つ目の実質的な返答です。"))
    assert not adapter._accept_peer_bot_message(msg("2つ目なので止まるべきです。"))


def test_discord_allow_bots_invalid_value_is_not_treated_as_all(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "true")
    assert _discord_allow_bots_mode() == "none"

    monkeypatch.setenv("DISCORD_ALLOW_BOTS", " ALL ")
    assert _discord_allow_bots_mode() == "all"


def test_discord_allowed_bot_ids_parse_comma_separated_allowlist(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_BOTS", " 111,222 ,, 333 ")
    assert _discord_allowed_bot_ids() == {"111", "222", "333"}


def test_peer_bot_guard_blocks_bots_not_in_allowlist(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_BOTS", "111,222")
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    unlisted_author = SimpleNamespace(id=999)
    message = SimpleNamespace(content="実質的な返答です。", channel=channel, author=unlisted_author)

    assert not adapter._accept_peer_bot_message(message)
    assert adapter._bot_dialogue_counts["123"][1] == 0


def test_peer_bot_guard_allows_bots_in_allowlist(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_GUARD", "true")
    monkeypatch.setenv("DISCORD_ALLOWED_BOTS", "111,222")
    monkeypatch.setenv("DISCORD_BOT_DIALOGUE_MAX_MESSAGES", "2")
    adapter = DiscordAdapter(PlatformConfig())
    channel = SimpleNamespace(id=123, parent_id=None)
    listed_author = SimpleNamespace(id=222)
    message = SimpleNamespace(content="実質的な返答です。", channel=channel, author=listed_author)

    assert adapter._accept_peer_bot_message(message)
    assert adapter._bot_dialogue_counts["123"][1] == 1
