import time

from gateway.final_speech_summarizer import (
    FinalSpeechSummarizer,
    VoiceContext,
    VoiceSummaryResult,
)


class FakeGenerator:
    def __init__(self, result=None, *, delay=0.0, error=None):
        self.result = result
        self.delay = delay
        self.error = error
        self.calls = []

    def __call__(self, prompt, *, timeout_ms):
        self.calls.append((prompt, timeout_ms))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def test_deterministic_summary_uses_first_safe_sentence_without_canned_fallback():
    summarizer = FinalSpeechSummarizer(generator=None, mode="deterministic")

    result = summarizer.summarize(
        "I updated the bridge and ran 14 tests. Full details are in Discord.",
        VoiceContext(max_spoken_chars=180),
    )

    assert result == VoiceSummaryResult(
        kind="completion",
        text="I updated the bridge and ran 14 tests.",
        method="deterministic",
        policy={
            "pre_sanitized": True,
            "post_sanitized": True,
            "truncated": False,
            "blocked_sensitive_content": False,
            "dropped_tool_logs": False,
            "dropped_code": False,
            "dropped_media_tags": False,
            "dropped_paths": False,
        },
        reason=None,
    )


def test_generated_success_returns_short_room_safe_line_and_metadata_flags():
    generator = FakeGenerator("I updated the bridge and verified 14 tests pass.")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "I updated the bridge and verified 14 tests pass. Details: no regressions.",
        VoiceContext(timeout_ms=250, max_spoken_chars=180, voice_profile="eon"),
    )

    assert result.kind == "completion"
    assert result.text == "I updated the bridge and verified 14 tests pass."
    assert result.method == "generated"
    assert result.policy["pre_sanitized"] is True
    assert result.policy["post_sanitized"] is True
    assert result.policy["blocked_sensitive_content"] is False
    [call] = generator.calls
    assert call[1] == 250
    assert "You are Eon speaking aloud" in call[0]
    assert "Use only facts explicitly present" in call[0]


def test_generated_timeout_falls_back_to_deterministic_summary_without_material_delay():
    generator = FakeGenerator("I should arrive too late.", delay=0.2)
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    started = time.perf_counter()
    result = summarizer.summarize(
        "I updated the bridge and ran the tests. Extra details follow.",
        VoiceContext(timeout_ms=5, max_spoken_chars=180),
    )
    elapsed = time.perf_counter() - started

    assert result.method == "deterministic"
    assert result.text == "I updated the bridge and ran the tests."
    assert result.reason == "generated_timeout"
    assert elapsed < 0.08


def test_generated_invalid_unsupported_number_falls_back():
    generator = FakeGenerator("I updated the bridge and verified 99 tests pass.")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "I updated the bridge and verified tests pass.",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.method == "deterministic"
    assert result.text == "I updated the bridge and verified tests pass."
    assert result.reason == "generated_invalid: unsupported_number"


def test_generated_summary_can_use_safe_facts_after_first_sentence():
    generator = FakeGenerator("I updated the bridge and ran 14 tests.")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "Summary: bridge work is complete. Evidence: I updated the bridge and ran 14 tests.",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.method == "generated"
    assert result.text == "I updated the bridge and ran 14 tests."


def test_empty_safe_output_returns_silence_not_finished():
    summarizer = FinalSpeechSummarizer(generator=None, mode="deterministic")

    result = summarizer.summarize(
        "MEDIA:/tmp/report.pdf\n```python\nprint('nothing safe')\n```",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.kind == "completion"
    assert result.text == ""
    assert result.method == "silence"
    assert result.reason == "empty_safe_output"


def test_inline_code_only_final_response_returns_silence():
    summarizer = FinalSpeechSummarizer(generator=None, mode="deterministic")

    result = summarizer.summarize(
        "`print('nothing safe')`",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.text == ""
    assert result.method == "silence"
    assert result.reason == "empty_safe_output"


def test_question_kind_is_source_of_truth_even_with_generated_summary():
    generator = FakeGenerator("Should voice be enabled everywhere, or only at home?")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "Should voice be enabled everywhere, or only at home?",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.kind == "question"
    assert result.text.endswith("?")
    assert result.method == "generated"


def test_generated_summary_cannot_turn_question_into_statement():
    generator = FakeGenerator("Voice should be enabled only at home.")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "Should voice be enabled everywhere, or only at home?",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.kind == "question"
    assert result.text == "Should voice be enabled everywhere, or only at home?"
    assert result.method == "deterministic"
    assert result.reason == "generated_invalid: question_downgrade"


def test_generated_exception_falls_back_to_deterministic_summary():
    generator = FakeGenerator(error=RuntimeError("model unavailable"))
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "I updated the bridge and ran the tests. Extra details follow.",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.method == "deterministic"
    assert result.text == "I updated the bridge and ran the tests."
    assert result.reason == "generated_exception"


def test_error_kind_is_source_of_truth_even_with_generated_summary():
    generator = FakeGenerator("The provider timed out safely.")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "Error: provider timeout while generating the response.",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.kind == "error"
    assert result.method == "generated"
    assert result.text == "The provider timed out safely."


def test_generated_summary_cannot_downgrade_error_to_completion():
    generator = FakeGenerator("All set.")
    summarizer = FinalSpeechSummarizer(generator=generator, mode="hybrid")

    result = summarizer.summarize(
        "Error: provider timeout while generating the response.",
        VoiceContext(max_spoken_chars=180),
    )

    assert result.kind == "error"
    assert result.text == "Error: provider timeout while generating the response."
    assert result.method == "deterministic"
    assert result.reason == "generated_invalid: error_downgrade"


def test_validation_rejects_curly_apostrophe_future_promises_not_in_final_text():
    summarizer = FinalSpeechSummarizer(generator=None, mode="deterministic")

    valid, reason = summarizer.validate_generated_summary(
        "I’ll share that now.",
        "The bridge work is complete.",
        VoiceContext(),
    )

    assert valid is False
    assert reason == "future_promise"


def test_validation_rejects_paths_secrets_code_media_actions_and_future_promises():
    final = "I updated the bridge and ran tests."
    bad_outputs = [
        "I wrote /Users/brenno/.hermes/config.yaml.",
        "I used /opt/homebrew/bin/rtk.",
        r"I wrote C:\\Users\\brenno\\secret.txt.",
        "The token starts with sk-123...cdef.",
        "api key is abcdefghijk",
        "Bearer abcdefghijk",
        "```python\nprint('x')\n```",
        "MEDIA:/tmp/report.pdf is ready.",
        "I updated gateway/run.py and ran tests.",
        "I deployed the bridge and ran tests.",
        "I will update the bridge now.",
    ]
    summarizer = FinalSpeechSummarizer(generator=None, mode="deterministic")

    for output in bad_outputs:
        valid, reason = summarizer.validate_generated_summary(output, final, VoiceContext())
        assert valid is False, output
        assert reason
