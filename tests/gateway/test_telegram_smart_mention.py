from gateway.platforms.telegram_smart_mention import (
    DEFAULT_SMART_MENTION_SYSTEM_PROMPT,
    SmartMentionConfig,
    build_smart_mention_messages,
    format_recent_context_for_agent,
    normalize_smart_mention_config,
    parse_smart_mention_response,
)


def test_normalize_smart_mention_config_accepts_boolean_shortcut():
    config = normalize_smart_mention_config(True)

    assert config.enabled is True
    assert config.system_prompt == DEFAULT_SMART_MENTION_SYSTEM_PROMPT


def test_normalize_smart_mention_config_clamps_numeric_fields_and_trims_prompt():
    config = normalize_smart_mention_config(
        {
            "enabled": "yes",
            "system_prompt": "  classify only  ",
            "include_recent_context": "off",
            "recent_context_messages": 999,
            "recent_context_max_chars": -1,
            "pass_recent_context_to_agent": "true",
            "min_confidence": 1.5,
            "max_tokens": 1,
            "log_decisions": "false",
            "log_message_text": "on",
            "on_error": " escalate ",
        }
    )

    assert config.enabled is True
    assert config.system_prompt == "classify only"
    assert config.include_recent_context is False
    assert config.recent_context_messages == 50
    assert config.recent_context_max_chars == 0
    assert config.pass_recent_context_to_agent is True
    assert config.min_confidence == 1.0
    assert config.max_tokens == 16
    assert config.log_decisions is False
    assert config.log_message_text is True
    assert config.on_error == "escalate"


def test_build_smart_mention_messages_includes_metadata_media_and_recent_context():
    config = SmartMentionConfig(
        system_prompt="classify",
        recent_context_messages=2,
        recent_context_max_chars=1000,
    )

    messages = build_smart_mention_messages(
        config=config,
        current_text="Hermes, summarize the incident",
        recent_context=[
            {"sender": "Alice", "text": "old context"},
            {"sender": "Bob", "text": "deploy failed", "media": "log"},
            {"sender": "Cara", "text": "rollback started"},
        ],
        media_metadata={"type": "photo", "width": 1280, "height": 720, "empty": ""},
        bot_username="hermes_bot",
        chat_id="-100",
        thread_id="42",
    )

    assert messages[0] == {"role": "system", "content": "classify"}
    user_payload = messages[1]["content"]
    assert "bot_username: @hermes_bot" in user_payload
    assert "chat_id: -100" in user_payload
    assert "thread_id: 42" in user_payload
    assert "height: 720" in user_payload
    assert "type: photo" in user_payload
    assert "width: 1280" in user_payload
    assert "old context" not in user_payload
    assert "- Bob [log]: deploy failed" in user_payload
    assert "- Cara: rollback started" in user_payload
    assert "Hermes, summarize the incident" in user_payload


def test_build_smart_mention_messages_limits_recent_nonblank_context_items():
    config = SmartMentionConfig(recent_context_messages=2)

    messages = build_smart_mention_messages(
        config=config,
        current_text="Hermes, what changed?",
        recent_context=[
            {"sender": "Alice", "text": "first useful context"},
            {"sender": "Bob", "text": "second useful context"},
            {"sender": "Noisy", "text": "   "},
        ],
    )

    user_payload = messages[1]["content"]
    assert "first useful context" in user_payload
    assert "second useful context" in user_payload
    assert "Noisy" not in user_payload


def test_build_smart_mention_messages_omits_context_when_disabled():
    config = SmartMentionConfig(include_recent_context=False)

    messages = build_smart_mention_messages(
        config=config,
        current_text="can you check this?",
        recent_context=[{"sender": "Alice", "text": "previous request"}],
    )

    assert "Recent context:\n(none)" in messages[1]["content"]
    assert "previous request" not in messages[1]["content"]


def test_build_smart_mention_messages_truncates_context_without_truncating_current_message():
    current_text = "Hermes, decide whether this production alert needs escalation"
    config = SmartMentionConfig(
        recent_context_messages=3,
        recent_context_max_chars=40,
    )

    messages = build_smart_mention_messages(
        config=config,
        current_text=current_text,
        recent_context=[
            {"sender": "Alice", "text": "A" * 80},
            {"sender": "Bob", "text": "B" * 80},
        ],
    )

    user_payload = messages[1]["content"]
    recent_context_block = user_payload.split("Recent context:\n", 1)[1].split("\n\nCurrent message:", 1)[0]
    current_message_block = user_payload.split("Current message:\n", 1)[1]
    assert recent_context_block.endswith("...")
    assert len(recent_context_block) <= config.recent_context_max_chars
    assert current_message_block == current_text


def test_parse_smart_mention_response_accepts_fenced_json_and_aliases():
    classification = parse_smart_mention_response(
        """```json
        {"respond": true, "confidence": 0.91, "reason": "direct ask"}
        ```"""
    )

    assert classification.should_respond is True
    assert classification.confidence == 0.91
    assert classification.reason == "direct ask"


def test_parse_smart_mention_response_extracts_json_from_noisy_model_output():
    classification = parse_smart_mention_response(
        'I would classify it as: {"should_respond": true, "confidence": "0.74", "reason": "asks Hermes"}'
    )

    assert classification.should_respond is True
    assert classification.confidence == 0.74
    assert classification.reason == "asks Hermes"


def test_parse_smart_mention_response_rejects_malformed_text_and_clamps_confidence():
    malformed = parse_smart_mention_response("yes, answer this")
    clamped = parse_smart_mention_response('{"process": true, "confidence": 3.5}')

    assert malformed.should_respond is False
    assert malformed.confidence == 0.0
    assert clamped.should_respond is True
    assert clamped.confidence == 1.0


def test_format_recent_context_for_agent_formats_and_truncates_context():
    text = format_recent_context_for_agent(
        [
            {"sender": "Alice", "text": "deploy failed", "media": "log"},
            {"sender": "Bob", "text": "please inspect the traceback"},
        ],
        max_chars=36,
    )

    assert text.startswith("Telegram recent group context")
    assert "- Alice [log]: deploy failed" in text
    assert text.endswith("The current message is the user request.")
    assert "..." in text


def test_format_recent_context_for_agent_returns_empty_for_blank_context():
    text = format_recent_context_for_agent(
        [
            {"sender": "Alice", "text": " "},
            {"sender": "Bob"},
        ],
        max_chars=100,
    )

    assert text == ""
