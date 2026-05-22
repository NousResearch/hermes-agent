from gateway.ambient_voice_policy import AmbientVoicePolicy, VoiceContext


def living_room_context(**overrides):
    values = {
        "source": "assistant_final",
        "platform": "discord",
        "channel_id": "home",
        "chat_id": "chat-1",
        "thread_id": None,
        "source_message_id": "msg-1",
        "input_modality": "voice",
        "output_device": "living_room",
        "profile": "eon",
        "explicit_spoken_request": False,
        "is_private_context": False,
        "config_scope": "living_room_default",
    }
    values.update(overrides)
    return VoiceContext(**values)


def evaluate(text, **context_overrides):
    return AmbientVoicePolicy().evaluate(text, living_room_context(**context_overrides))


def test_secret_like_tokens_are_suppressed_without_raw_metadata():
    synthetic_token = "sk-" + "s" * 16

    decision = evaluate(f"The token is {synthetic_token}")

    assert decision.allowed is False
    assert decision.suppressed is True
    assert decision.text == ""
    assert "secret_like" in decision.reasons
    assert decision.classifiers["secret_like"] is True
    assert synthetic_token not in str(decision.to_metadata())


def test_slack_dash_tokens_are_suppressed_without_raw_metadata():
    synthetic_token = "xoxb-" + "a" * 12 + "-" + "b" * 12

    decision = evaluate(f"The workspace bot token is {synthetic_token}")

    assert decision.allowed is False
    assert decision.text == ""
    assert decision.classifiers["secret_like"] is True
    assert "secret_like" in decision.reasons
    assert synthetic_token not in str(decision.to_metadata())


def test_prefixed_env_secret_assignments_are_suppressed_without_raw_metadata():
    synthetic_value = "hms_" + "c" * 18
    secret_assignment = "SLACK_BOT_TOKEN=" + synthetic_value

    decision = evaluate(f"I found {secret_assignment} in the environment")

    assert decision.allowed is False
    assert decision.text == ""
    assert decision.classifiers["secret_like"] is True
    assert "secret_like" in decision.reasons
    assert secret_assignment not in str(decision.to_metadata())


def test_code_fences_and_inline_code_are_stripped_and_can_result_in_silence():
    decision = evaluate("```python\nprint('do not speak code')\n``` and `rm -rf /tmp/example`")

    assert decision.allowed is False
    assert decision.suppressed is True
    assert decision.text == ""
    assert decision.classifiers["code"] is True
    assert "code_stripped" in decision.reasons


def test_stack_traces_and_command_logs_are_suppressed():
    decision = evaluate(
        "$ pytest tests/gateway/test_ambient_voice_policy.py -q\n"
        "Traceback (most recent call last):\n"
        "  File \"/Users/brenno/.hermes/hermes-agent/gateway/run.py\", line 10, in run\n"
        "FAILED tests/gateway/test_ambient_voice_policy.py::test_policy"
    )

    assert decision.allowed is False
    assert decision.suppressed is True
    assert decision.classifiers["stack_trace"] is True
    assert decision.classifiers["command_log"] is True
    assert "stack_trace" in decision.reasons
    assert "/Users/brenno" not in str(decision.to_metadata())


def test_raw_file_paths_are_removed_from_allowed_speech():
    decision = evaluate("I wrote the report to /Users/brenno/main/aegis/report.md and verified it.")

    assert decision.allowed is True
    assert decision.sanitized is True
    assert decision.classifiers["raw_path"] is True
    assert "raw_path_stripped" in decision.reasons
    assert "/Users/brenno" not in decision.text
    assert decision.text == "I wrote the report to a local file and verified it."


def test_sensitive_financial_or_trading_topics_are_suppressed_in_living_room_by_default():
    decision = evaluate("Your trading PnL and portfolio drawdown look risky today.")

    assert decision.allowed is False
    assert decision.suppressed is True
    assert decision.classifiers["sensitive_topic"] is True
    assert "sensitive_topic" in decision.reasons


def test_sensitive_topics_can_be_acknowledged_when_explicit_and_private_but_not_spoken_in_full():
    decision = evaluate(
        "Your trading PnL and portfolio drawdown look risky today.",
        output_device="airpods",
        config_scope="airpods_private",
        explicit_spoken_request=True,
        is_private_context=True,
    )

    assert decision.allowed is True
    assert decision.sanitized is True
    assert decision.text == "I can discuss that sensitive topic in this private voice context."
    assert "sensitive_topic_summarized" in decision.reasons


def test_long_responses_are_bounded_to_one_breath():
    decision = evaluate(
        "I completed the implementation and ran the targeted tests. "
        "Here are several additional details that should not be spoken aloud in the room. "
        "A third sentence should also be omitted."
    )

    assert decision.allowed is True
    assert decision.truncated is True
    assert decision.text == "I completed the implementation and ran the targeted tests."
    assert decision.original_chars > len(decision.text)
    assert "bounded_to_one_sentence" in decision.reasons


def test_policy_metadata_contains_only_safe_reason_codes_and_booleans():
    decision = evaluate("AWS_SECRET_ACCESS_KEY=fake-value-for-policy-check at ~/main/private.txt")

    metadata = decision.to_metadata()

    assert metadata == {
        "allowed": False,
        "sanitized": True,
        "truncated": False,
        "suppressed": True,
        "reason_codes": ("secret_like", "raw_path_stripped", "empty_after_sanitization"),
        "rule_profile": "living_room_default",
        "blocked_secret_like": True,
        "dropped_code": False,
        "blocked_stack_trace": False,
        "dropped_tool_logs": False,
        "dropped_paths": True,
        "blocked_sensitive_topic": False,
    }
