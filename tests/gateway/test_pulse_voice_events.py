import json
from types import SimpleNamespace

from gateway.pulse_voice_events import (
    completion_voice_text,
    publish_completion_voice_out,
    publish_generated_ack_voice_out,
    publish_voice_event,
    publish_voice_out,
    summarize_final_voice_response,
    voice_events_path,
    voice_out_path,
    voice_safe_text,
)


UNSAFE_SECRET = "sk-test_1234567890abcdef1234567890abcdef"


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_voice_safe_text_removes_code_media_and_bounds_length():
    text = """
    Done. MEDIA:/tmp/audio.mp3

    ```python
    print('do not speak code')
    ```
    [[audio_as_voice]] Extra words that should not matter.
    """

    assert voice_safe_text(text) == "Done."


def test_voice_safe_text_skips_obvious_tool_log_lines():
    text = """
    $ pytest tests/gateway/test_pulse_voice_events.py -q
    FAILED tests/gateway/test_pulse_voice_events.py::test_old_ack
    Traceback (most recent call last):
    File \"gateway/run.py\", line 16774, in _run_agent
    I removed the canned ack path and kept Discord text unchanged.
    """

    assert voice_safe_text(text) == "I removed the canned ack path and kept Discord text unchanged."


def test_publish_voice_out_writes_canonical_schema_and_legacy_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out(
        "ack",
        "Got it. I’ll wire the voice path.",
        session_id="s1",
        source_message_id="m1",
        source="generated_ack",
        derived_from="turn_start",
        voice_profile="eon",
    )

    canonical = _jsonl(voice_out_path())
    legacy = _jsonl(voice_events_path())
    assert canonical == legacy
    assert canonical[0]["kind"] == "ack"
    assert canonical[0]["text"] == "Got it."
    assert canonical[0]["max_seconds"] == 2
    assert canonical[0]["session_id"] == "s1"
    assert canonical[0]["source_message_id"] == "m1"
    assert canonical[0]["schema_version"] == 2
    assert canonical[0]["source"] == "generated_ack"
    assert canonical[0]["derived_from"] == "turn_start"
    assert canonical[0]["voice_profile"] == "eon"
    assert canonical[0]["policy"] == {
        "allowed": True,
        "sanitized": True,
        "truncated": False,
        "suppressed": False,
        "rule_profile": "living_room_default",
        "reason_codes": ["bounded_to_one_sentence"],
        "classifiers": {
            "dropped_code": False,
            "dropped_tool_logs": False,
            "dropped_paths": False,
            "blocked_secret_like": False,
            "blocked_sensitive_topic": False,
            "blocked_stack_trace": False,
        },
    }


def test_publish_voice_out_adds_default_v2_source_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out("completion", "I finished the task and verified the checks.")

    [event] = _jsonl(voice_out_path())
    assert event["schema_version"] == 2
    assert event["source"] == "completion"
    assert event["derived_from"] == "pulse_voice_candidate"
    assert event["voice_profile"] == "eon"


def test_publish_voice_out_drops_malicious_metadata_from_canonical_and_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    malicious_values = [
        "debug_raw=/Users/brenno/.hermes/.env",
        "API_KEY=hk_test_1234567890abcdef",
        "Traceback (most recent call last): File /Users/brenno/app.py",
        "user said my portfolio exposure is private",
    ]

    publish_voice_out(
        "completion",
        "Safe completion.",
        session_id="safe-session-1",
        source_message_id="msg-1",
        debug_raw=malicious_values[0],
        raw_path=malicious_values[0],
        user_content=malicious_values[3],
        error=malicious_values[2],
        trace=malicious_values[2],
        path="/Users/brenno/.hermes/.env",
        extra_secret=malicious_values[1],
    )

    canonical = _jsonl(voice_out_path())
    legacy = _jsonl(voice_events_path())
    assert canonical == legacy
    payload = json.dumps(canonical[0], ensure_ascii=False)
    assert canonical[0]["session_id"] == "safe-session-1"
    assert canonical[0]["source_message_id"] == "msg-1"
    for raw in malicious_values:
        assert raw not in payload
    assert "/Users/brenno" not in payload
    assert "hk_test_1234567890abcdef" not in payload
    assert "debug_raw" not in canonical[0]
    assert "raw_path" not in canonical[0]
    assert "user_content" not in canonical[0]
    assert "extra_secret" not in canonical[0]


def test_publish_voice_out_drops_secret_shaped_allowlisted_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    secret_values = {
        "session_id": "xoxb-[REDACTED]",
        "chat_id": "hf_[REDACTED]",
        "channel_id": "glpat-[REDACTED]",
        "thread_id": "hk_[REDACTED]",
        "source_message_id": "rk_live_[REDACTED]",
    }

    publish_voice_out("completion", "Safe completion.", **secret_values)

    [event] = _jsonl(voice_out_path())
    payload = json.dumps(event, ensure_ascii=False)
    for key, raw in secret_values.items():
        assert key not in event
        assert raw not in payload


def test_publish_voice_out_routes_candidate_through_ambient_policy(tmp_path, monkeypatch):
    import gateway.pulse_voice_events as pulse_voice_events

    calls = []

    class FakeAmbientPolicy:
        def evaluate(self, text, context):
            calls.append((text, context))
            return SimpleNamespace(
                allowed=True,
                text="Policy approved.",
                sanitized=True,
                truncated=False,
                suppressed=False,
                max_seconds=6,
                reasons=("policy_checked",),
                classifiers={
                    "code": False,
                    "command_log": False,
                    "raw_path": False,
                    "secret_like": False,
                    "sensitive_topic": False,
                    "stack_trace": False,
                },
                rule_profile="airpods_private",
            )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pulse_voice_events, "_AMBIENT_POLICY", FakeAmbientPolicy())

    publish_voice_out(
        "question",
        "Original candidate. More detail must stay out.",
        output_device="airpods",
        voice_profile="eon",
    )

    [event] = _jsonl(voice_out_path())
    assert calls
    assert calls[0][0] == "Original candidate."
    assert calls[0][1].source == "question"
    assert calls[0][1].output_device == "airpods"
    assert event["text"] == "Policy approved."
    assert event["max_seconds"] == 6
    assert event["policy"]["rule_profile"] == "airpods_private"
    assert event["policy"]["reason_codes"] == ["bounded_to_one_sentence", "policy_checked"]


def test_publish_voice_out_suppresses_secrets_without_raw_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out("completion", f"The token is {UNSAFE_SECRET}; keep it safe.")

    assert not voice_out_path().exists()
    assert not voice_events_path().exists()


def test_voice_safe_text_suppresses_inline_key_and_password_prose():
    assert voice_safe_text("API key: hk_test_1234567890abcdef; keep it safe.") == ""
    assert voice_safe_text("The temporary password is example-pass-1234 for setup.") == ""


def test_publish_voice_out_suppresses_bearer_and_aws_style_tokens(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out("completion", "Bearer hk_test_1234567890abcdef is configured.")
    publish_voice_out("completion", "Using AWS key AKIAIOSFODNN7EXAMPLE for the demo.")

    assert not voice_out_path().exists()
    assert not voice_events_path().exists()


def test_publish_voice_out_suppresses_paths_and_code_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out("completion", "```python\nprint('/Users/brenno/.hermes/.env')\n```")
    publish_voice_out("completion", "`print('do not speak inline code')`")
    publish_completion_voice_out("```python\nprint('done')\n```")

    assert not voice_out_path().exists()
    assert completion_voice_text("```python\nprint('done')\n```") == ("completion", "")
    assert voice_safe_text("`print('do not speak inline code')`") == ""


def test_publish_voice_out_suppresses_stack_traces_and_sensitive_topics(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out(
        "error",
        'Traceback (most recent call last):\n  File "/Users/brenno/app.py", line 1, in <module>\nRuntimeError: boom',
    )
    publish_voice_out("completion", "Your trading PnL and portfolio exposure are down 12% today.")

    assert not voice_out_path().exists()


def test_policy_metadata_uses_safe_reason_codes_not_raw_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out(
        "progress",
        "I reviewed /Users/brenno/main/project/secret.py and finished the task.",
    )
    publish_voice_out(
        "progress",
        r"I reviewed C:\Users\brenno\project\secret.py and finished the task.",
    )

    publish_voice_out(
        "progress",
        "I reviewed gateway/run.py and finished the task.",
    )

    events = _jsonl(voice_out_path())
    assert len(events) == 3
    for event in events:
        payload = json.dumps(event, ensure_ascii=False)
        assert "/Users/brenno" not in payload
        assert "C:\\Users" not in payload
        assert "secret.py" not in payload
        assert "gateway/run.py" not in payload
        assert event["text"] == "I reviewed and finished the task."
        assert event["policy"]["classifiers"]["dropped_paths"] is True
        assert event["policy"]["reason_codes"] == ["path_stripped", "bounded_to_one_sentence"]


def test_passed_policy_metadata_is_normalized_without_raw_content(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_out(
        "completion",
        "Safe completion.",
        policy={
            "allowed": True,
            "reason_codes": ["/Users/brenno/.hermes/.env", "safe_reason"],
            "classifiers": {"raw_path": "/Users/brenno/.hermes/.env", "dropped_paths": True},
        },
    )

    [event] = _jsonl(voice_out_path())
    payload = json.dumps(event, ensure_ascii=False)
    assert "/Users/brenno" not in payload
    assert event["policy"]["reason_codes"] == ["safe_reason", "bounded_to_one_sentence"]
    assert event["policy"]["classifiers"] == {
        "dropped_code": False,
        "dropped_tool_logs": False,
        "dropped_paths": True,
        "blocked_secret_like": False,
        "blocked_sensitive_topic": False,
        "blocked_stack_trace": False,
    }


def test_publish_generated_ack_voice_out_is_voice_only_and_metadata_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw_user_text = "look at /Users/brenno/.hermes/.env and token sk-test...cdef"

    publish_generated_ack_voice_out(
        raw_user_text,
        generator=lambda prompt, *, timeout_ms: "That voice bridge needs sharper timing.",
        session_id="safe-session-2",
        platform="discord",
        chat_id="chat-1",
        thread_id="thread-1",
        source_message_id="msg-2",
        input_modality="voice",
        output_device="room_audio",
        raw_user_text=raw_user_text,
        debug_path="/Users/brenno/.hermes/.env",
    )

    [event] = _jsonl(voice_out_path())
    payload = json.dumps(event, ensure_ascii=False)
    assert event["kind"] == "ack"
    assert event["text"] == "That voice bridge needs sharper timing."
    assert event["source"] == "generated_ack"
    assert event["derived_from"] == "turn_start"
    assert event["max_seconds"] == 2
    assert event["ack"]["method"] == "generated"
    assert event["ack"]["timeout_ms"] == 1000
    assert event["session_id"] == "safe-session-2"
    assert event["source_message_id"] == "msg-2"
    assert "raw_user_text" not in event
    assert "debug_path" not in event
    assert "/Users/brenno" not in payload
    assert "sk-test" not in payload
    assert raw_user_text not in payload


def test_publish_generated_ack_voice_out_silences_invalid_generated_ack(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_generated_ack_voice_out(
        "voice test",
        generator=lambda prompt, *, timeout_ms: "I’ll run the tests and verify it.",
    )

    assert not voice_out_path().exists()
    assert not voice_events_path().exists()


def test_no_canned_ack_phrase_generator_is_exported():
    import gateway.pulse_voice_events as pulse_voice_events

    assert not hasattr(pulse_voice_events, "turn_ack_text")
    assert not hasattr(pulse_voice_events, "_ACK_PHRASES")


def test_legacy_delta_events_are_not_published(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_event("delta", "This raw token stream must not be spoken.")

    assert not voice_out_path().exists()
    assert not voice_events_path().exists()


def test_legacy_commentary_maps_to_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_voice_event("commentary", "I’ll inspect that now. More details follow later.")

    [event] = _jsonl(voice_out_path())
    assert event["kind"] == "progress"
    assert event["text"] == "I’ll inspect that now."


def test_completion_voice_text_summarizes_executor_output():
    kind, text = completion_voice_text("I updated the bridge and ran the tests. Full details are in Discord.")

    assert kind == "completion"
    assert text == "I updated the bridge and ran the tests."


def test_generated_summary_timeout_exception_and_invalid_output_fall_back_deterministically():
    final_response = "I updated the bridge and ran 14 tests. Full details are in Discord."

    def timed_out(_final_response):
        raise TimeoutError("fake summarizer timeout")

    assert completion_voice_text(final_response, summarizer=timed_out) == (
        "completion",
        "I updated the bridge and ran 14 tests.",
    )
    assert completion_voice_text(final_response, summarizer=lambda _text: "MEDIA:/tmp/report.pdf") == (
        "completion",
        "I updated the bridge and ran 14 tests.",
    )
    assert completion_voice_text(final_response, summarizer=lambda _text: "Bridge deployed and 15 tests passed.") == (
        "completion",
        "I updated the bridge and ran 14 tests.",
    )


def test_generated_summary_is_accepted_only_when_aligned_and_safe():
    final_response = "I updated the bridge and verified 14 tests passed. Full details are in Discord."

    result = summarize_final_voice_response(
        final_response,
        summarizer=lambda _text: "Bridge updated and 14 tests passed.",
    )

    assert result.kind == "completion"
    assert result.text == "Bridge updated and 14 tests passed."
    assert result.source == "assistant_final"
    assert result.derived_from == "final_response"
    assert result.voice_profile == "eon"
    assert result.summarizer["mode"] == "hybrid"
    assert result.summarizer["method"] == "generated"
    assert result.summarizer["fallback_used"] is False
    assert result.summarizer["validation_failed"] is False
    if "timeout_ms" in result.summarizer:
        assert result.summarizer["timeout_ms"] == 1000
    assert result.policy["pre_sanitized"] is True
    assert result.policy["post_sanitized"] is True


def test_question_and_error_classification_preserve_final_response_intent_with_generated_summary():
    assert completion_voice_text(
        "Which profile should own this task?",
        summarizer=lambda _text: "Which profile should own this task?",
    ) == ("question", "Which profile should own this task?")

    assert completion_voice_text(
        "Which profile should own this task?",
        summarizer=lambda _text: "I need a profile choice.",
    ) == ("question", "Which profile should own this task?")

    assert completion_voice_text(
        "Error: provider timeout while generating the summary.",
        summarizer=lambda _text: "The summary is ready.",
    ) == ("error", "Error: provider timeout while generating the summary.")


def test_publish_completion_voice_out_marks_questions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_completion_voice_out("Which profile should own the worker task?")

    [event] = _jsonl(voice_out_path())
    assert event["kind"] == "question"
    assert event["text"] == "Which profile should own the worker task?"
    assert event["source"] == "assistant_final"
    assert event["derived_from"] == "final_response"
    assert event["voice_profile"] == "eon"
    assert event["summarizer"]["method"] == "deterministic"
    assert event["policy"]["allowed"] is True
    assert event["policy"]["classifiers"]["blocked_secret_like"] is False


def test_publish_completion_voice_out_silences_empty_safe_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_completion_voice_out("MEDIA:/tmp/report.pdf\n```python\nprint('nothing safe')\n```")

    assert not voice_out_path().exists()
    assert not voice_events_path().exists()


def test_publish_completion_voice_out_keeps_platform_text_out_of_voice_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    final_text = "I updated the bridge and verified 14 tests passed. Full Discord detail remains unchanged."

    publish_completion_voice_out(final_text, summarizer=lambda _text: "Bridge updated and 14 tests passed.")

    [event] = _jsonl(voice_out_path())
    assert final_text == "I updated the bridge and verified 14 tests passed. Full Discord detail remains unchanged."
    assert event["text"] == "Bridge updated and 14 tests passed."
    assert event["source"] == "assistant_final"
    assert event["derived_from"] == "final_response"
    assert event["voice_profile"] == "eon"
    assert event["summarizer"]["method"] == "generated"


def test_publish_completion_voice_out_preserves_safe_summary_failure_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    publish_completion_voice_out(
        "I updated the bridge and ran 14 tests. Full details are in Discord.",
        summarizer=lambda _text: "Bridge deployed and 15 tests passed.",
    )

    [event] = _jsonl(voice_out_path())
    assert event["summarizer"]["method"] == "deterministic"
    assert event["summarizer"]["fallback_used"] is True
    assert event["summarizer"]["validation_failed"] is True
    assert event["summarizer"]["reason"] == "generated_invalid_unsupported_number"
    assert "Bridge deployed" not in json.dumps(event["summarizer"])
