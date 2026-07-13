"""Behavior contracts for configurable Telegram DM topic titles."""

from gateway.telegram_topic_titles import (
    TelegramTopicTitleOptions,
    dedupe_telegram_topic_title,
    resolve_telegram_topic_title_contexts,
    sanitize_telegram_topic_title,
    telegram_topic_title_options,
)


def test_default_options_preserve_readable_titles():
    options = telegram_topic_title_options({})

    assert options == TelegramTopicTitleOptions(style="readable", compact_max_words=2)
    assert sanitize_telegram_topic_title(
        "  Build   Telegram Topic UX  ", options=options
    ) == "Build Telegram Topic UX"


def test_nested_compact_options_are_parsed_and_word_count_is_clamped():
    assert telegram_topic_title_options(
        {"dm_topic_titles": {"style": "COMPACT", "compact_max_words": "3"}}
    ) == TelegramTopicTitleOptions(style="compact", compact_max_words=3)
    assert telegram_topic_title_options(
        {"dm_topic_titles": {"style": "compact", "compact_max_words": 0}}
    ).compact_max_words == 1
    assert telegram_topic_title_options(
        {"dm_topic_titles": {"style": "compact", "compact_max_words": 99}}
    ).compact_max_words == 6
    assert telegram_topic_title_options(
        {"dm_topic_titles": {"style": "compact", "compact_max_words": True}}
    ).compact_max_words == 2
    assert telegram_topic_title_options(
        {
            "dm_topic_titles": {
                "style": "compact",
                "generic_titles": [" נושא   חדש ", ""],
            }
        }
    ).generic_titles == ("נושא חדש",)


def test_invalid_style_fails_safe_to_readable():
    options = telegram_topic_title_options(
        {"dm_topic_titles": {"style": "semantic", "compact_max_words": 2}}
    )

    assert options.style == "readable"
    assert sanitize_telegram_topic_title("Keep This Readable", options=options) == "Keep This Readable"


def test_compact_titles_keep_unicode_tokens_in_original_order():
    options = TelegramTopicTitleOptions(style="compact", compact_max_words=2)

    assert sanitize_telegram_topic_title("Telegram topic reliability", options=options) == "telegram-topic"
    assert sanitize_telegram_topic_title("אבטחת Hermes Desktop מרחוק", options=options) == "אבטחת-hermes"
    assert sanitize_telegram_topic_title("مراجعة أمان تيليجرام", options=options) == "مراجعة-أمان"
    assert sanitize_telegram_topic_title("会話 タイトル 設定", options=options) == "会話-タイトル"


def test_compact_word_count_is_configurable():
    options = TelegramTopicTitleOptions(style="compact", compact_max_words=3)

    assert sanitize_telegram_topic_title("Release planning dashboard polish", options=options) == "release-planning-dashboard"


def test_compact_generic_or_empty_title_uses_first_message_fallback():
    options = TelegramTopicTitleOptions(
        style="compact",
        compact_max_words=2,
        generic_titles=("נושא חדש",),
    )

    assert sanitize_telegram_topic_title(
        "Hermes Agent", options=options, fallback_text="Compare keyboard switches for typing"
    ) == "compare-keyboard"
    assert sanitize_telegram_topic_title(
        "New Topic", options=options, fallback_text="Release planning dashboard"
    ) == "release-planning"
    assert sanitize_telegram_topic_title(
        "נושא חדש", options=options, fallback_text="בדיקת פריסת מקלדת"
    ) == "בדיקת-פריסת"
    assert sanitize_telegram_topic_title(
        "", options=options, fallback_text="مراجعة أمان تيليجرام"
    ) == "مراجعة-أمان"


def test_compact_without_any_tokens_uses_existing_safe_fallback():
    options = TelegramTopicTitleOptions(style="compact", compact_max_words=2)

    assert sanitize_telegram_topic_title("---", options=options, fallback_text="!!!") == "Hermes Chat"


def test_readable_title_retains_existing_length_contract():
    options = TelegramTopicTitleOptions(style="readable", compact_max_words=2)

    result = sanitize_telegram_topic_title("x" * 200, options=options)

    assert len(result) == 120
    assert result.endswith("...")


def test_compact_title_and_dedupe_suffix_stay_within_length_contract():
    options = TelegramTopicTitleOptions(style="compact", compact_max_words=2)

    compact = sanitize_telegram_topic_title("x" * 200, options=options)
    duplicate = dedupe_telegram_topic_title(compact, [compact])

    assert len(compact) == 120
    assert len(duplicate) == 120
    assert duplicate.endswith("2")


def test_compact_title_retains_unicode_letters_and_digits_but_drops_punctuation():
    options = TelegramTopicTitleOptions(style="compact", compact_max_words=4)

    assert sanitize_telegram_topic_title(
        "RAG / MCP: design_v2!!!", options=options
    ) == "rag-mcp-design-v2"


def test_dedupe_is_case_insensitive_and_appends_first_available_integer():
    assert dedupe_telegram_topic_title("daily-progress", []) == "daily-progress"
    assert dedupe_telegram_topic_title("Daily-Progress", ["daily-progress"]) == "Daily-Progress2"
    assert dedupe_telegram_topic_title(
        "daily-progress", ["daily-progress", "DAILY-PROGRESS2", "daily-progress4"]
    ) == "daily-progress3"


def test_context_resolver_processes_full_ordered_set_with_per_chat_dedupe():
    contexts = [
        {"chat_id": "100", "session_id": "a", "title": "Daily Progress Checkin"},
        {"chat_id": "100", "session_id": "b", "title": "Daily Progress Update"},
        {"chat_id": "200", "session_id": "c", "title": "Daily Progress Review"},
    ]
    options = TelegramTopicTitleOptions(style="compact", compact_max_words=2)

    resolved = resolve_telegram_topic_title_contexts(
        contexts,
        options=options,
        title_overrides={"a": "Daily Progress Retrospective"},
    )

    assert resolved == ["daily-progress", "daily-progress2", "daily-progress"]
    assert len(resolved) == len(contexts)
