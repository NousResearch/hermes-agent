import json
import time
from types import SimpleNamespace

import pytest


class FakeAckGenerator:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, prompt, *, timeout_ms):
        self.calls.append((prompt, timeout_ms))
        return self.result


def test_generated_ack_default_config_shape():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["pulse"]["voice"]["generated_ack"]
    assert cfg["mode"] == "generated"
    assert 800 <= cfg["timeout_ms"] <= 1500
    assert cfg["max_words"] == 12
    assert cfg["max_spoken_chars"] <= 120
    assert cfg["max_seconds"] == 2
    assert cfg["voice_profile"] == "eon"
    assert cfg["silence_on_failure"] is True
    assert "fallback" not in cfg or cfg["fallback"] in {None, "silence"}


def test_prompt_contract_contains_required_voice_only_constraints():
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    harness = GeneratedAckHarness(generator=lambda prompt, *, timeout_ms: "That voice path needs the bridge checked.")
    prompt = harness.build_prompt(AckContext(user_message="The voice path feels wrong."))

    assert "Generate a short natural spoken acknowledgement as Eon" in prompt
    assert "Maximum 12 words" in prompt
    assert "No canned acknowledgement phrases" in prompt
    assert "Do not promise completion" in prompt
    assert "This is room audio, not Discord text" in prompt
    assert "The voice path feels wrong" in prompt


def test_prompt_context_redacts_user_secrets_paths_urls_logs_and_media():
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    raw = """
    check /Users/brenno/.hermes/.env and gateway/generated_ack_harness.py
    ../private/config src/secrets/config
    https://example.com/private
    OPENAI_API_KEY=[REDACTED] HERMES_TOKEN=[REDACTED]
    token [REDACTED]
    password is [REDACTED]
    MEDIA:/tmp/voice.wav
    please inspect this output
    INFO provider warmup line
    2026-05-24T12:00:00Z ERROR provider leaked details
    [worker][WARN] provider leaked details
    ERROR provider leaked details
    x + y
    `x + y`
    select secret from users
    git status --short
    foo = bar
    print('secret')
    console.log('secret')
    def leaked_function():
        return 'secret'
    ```python
    print('secret')
    ```
    Traceback (most recent call last): boom
    -----BEGIN PRIVATE KEY-----
    abc123
    -----END PRIVATE KEY-----
    """
    prompt = GeneratedAckHarness().build_prompt(AckContext(user_message=raw))

    assert "User asked about technical content; sensitive details omitted." in prompt
    assert "/Users/brenno" not in prompt
    assert "gateway/generated_ack_harness.py" not in prompt
    assert "../private/config" not in prompt
    assert "src/secrets/config" not in prompt
    assert "https://example.com" not in prompt
    assert "OPENAI_API_KEY" not in prompt
    assert "HERMES_TOKEN" not in prompt
    assert "[REDACTED]" not in prompt
    assert "password is" not in prompt
    assert "MEDIA:/tmp/voice.wav" not in prompt
    assert "INFO provider warmup" not in prompt
    assert "2026-05-24T12:00:00Z" not in prompt
    assert "[worker][WARN]" not in prompt
    assert "ERROR provider leaked" not in prompt
    assert "x + y" not in prompt
    assert "select secret from users" not in prompt
    assert "git status --short" not in prompt
    assert "foo = bar" not in prompt
    assert "print('secret')" not in prompt
    assert "console.log('secret')" not in prompt
    assert "def leaked_function" not in prompt
    assert "return 'secret'" not in prompt
    assert "print('secret')" not in prompt
    assert "Traceback" not in prompt
    assert "PRIVATE KEY" not in prompt


def test_prompt_context_redacts_separatorless_short_secret_keywords():
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    for raw in (
        "token [REDACTED]",
        "api key [REDACTED]",
        "access key [REDACTED]",
        "secret [REDACTED]",
        "password [REDACTED]",
        "passwd [REDACTED]",
        "pwd [REDACTED]",
        "private key is [REDACTED]",
        "private key [REDACTED]",
        "ssh key [REDACTED]",
        "passphrase [REDACTED]",
        "credential [REDACTED]",
        "xoxb-[REDACTED]",
        "hf_[REDACTED]",
        "glpat-[REDACTED]",
        "hk_[REDACTED]",
        "rk_live_[REDACTED]",
        "pk_live_[REDACTED]",
    ):
        prompt = GeneratedAckHarness().build_prompt(AckContext(user_message=raw))
        assert "User asked about technical content; sensitive details omitted." in prompt
        assert raw not in prompt


def test_generated_ack_success_uses_injected_generator_and_timeout():
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    generator = FakeAckGenerator("Yeah, that voice path needs to feel like Eon.")
    harness = GeneratedAckHarness(generator=generator, mode="generated")
    result = harness.generate(AckContext(user_message="The voice path feels generic.", timeout_ms=900))

    assert result.method == "generated"
    assert result.text == "Yeah, that voice path needs to feel like Eon."
    assert result.reason is None
    assert generator.calls[0][1] == 900


class SlowAckGenerator:
    def __call__(self, prompt, *, timeout_ms):
        time.sleep(0.2)
        return "This arrives too late."


def test_generated_ack_timeout_returns_silence_without_material_delay():
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    harness = GeneratedAckHarness(generator=SlowAckGenerator(), mode="generated")
    started = time.perf_counter()
    result = harness.generate(AckContext(user_message="Check voice", timeout_ms=5))
    elapsed = time.perf_counter() - started

    assert result.method == "silence"
    assert result.text == ""
    assert result.reason == "generated_timeout"
    assert elapsed < 0.08


def test_generated_ack_exception_returns_silence_not_canned_fallback():
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    def boom(prompt, *, timeout_ms):
        raise RuntimeError("provider down")

    result = GeneratedAckHarness(generator=boom).generate(AckContext(user_message="Check voice"))

    assert result.method == "silence"
    assert result.text == ""
    assert result.reason == "generated_error"


def test_generated_ack_saturation_returns_silence_without_queueing(monkeypatch):
    import gateway.generated_ack_harness as generated_ack_harness
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    class SaturatedSlots:
        def acquire(self, *, blocking):
            assert blocking is False
            return False

    monkeypatch.setattr(generated_ack_harness, "_GENERATOR_SLOTS", SaturatedSlots())
    generator = FakeAckGenerator("This should never enqueue.")

    result = GeneratedAckHarness(generator=generator).generate(AckContext(user_message="Check voice"))

    assert result.method == "silence"
    assert result.text == ""
    assert result.reason == "generated_saturated"
    assert generator.calls == []


@pytest.mark.parametrize("candidate", [
    "I’m here.",
    "Got you.",
    "I’m on it.",
    "Let’s see.",
    "Sure, I can help with that.",
    "Okay.",
    "Processing your request.",
])
def test_generated_ack_rejects_canned_or_generic_phrases(candidate):
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    harness = GeneratedAckHarness(generator=lambda prompt, *, timeout_ms: candidate)
    result = harness.generate(AckContext(user_message="Voice test"))

    assert result.method == "silence"
    assert result.text == ""
    assert result.reason is not None
    assert result.reason.startswith("generated_invalid")


@pytest.mark.parametrize("candidate", [
    "I will fix that voice path now.",
    "I’ll run the tests and verify it.",
    "Done, the voice bridge is fixed.",
    "The token is [REDACTED].",
    "I found it in /Users/brenno/.hermes/config.yaml.",
    "Check gateway/generated_ack_harness.py next.",
    "Open ../private/config for the token.",
    "MEDIA:/tmp/voice.wav",
    "OPENAI_API_KEY=[REDACTED]",
    "token [REDACTED]",
    "api key [REDACTED]",
    "access key [REDACTED]",
    "secret [REDACTED]",
    "private key is [REDACTED]",
    "private key [REDACTED]",
    "ssh key [REDACTED]",
    "passphrase [REDACTED]",
    "credential [REDACTED]",
    "xoxb-[REDACTED]",
    "hf_[REDACTED]",
    "glpat-[REDACTED]",
    "hk_[REDACTED]",
    "rk_live_[REDACTED]",
    "pk_live_[REDACTED]",
    "password is [REDACTED]",
    "eyJ[REDACTED].[REDACTED].[REDACTED]",
    "-----BEGIN PRIVATE KEY----- [REDACTED]",
    "print('[REDACTED]')",
    "console.log('[REDACTED]')",
    "foo = bar",
    "INFO provider warmup line",
    "2026-05-24T12:00:00Z ERROR provider leaked details",
    "[worker][WARN] provider leaked details",
    "x + y",
    "`x + y`",
    "select secret from users",
    "git status --short",
    "def leaked_function():",
    "function test() { return secret; }",
    "First sentence. Second sentence.",
    "One two three four five six seven eight nine ten eleven twelve thirteen.",
])
def test_generated_ack_rejects_unsafe_or_overlong_candidates(candidate):
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    harness = GeneratedAckHarness(generator=lambda prompt, *, timeout_ms: candidate)
    result = harness.generate(AckContext(user_message="Voice test", max_words=12))

    assert result.method == "silence"
    assert result.text == ""
    assert result.reason is not None
    assert result.reason.startswith("generated_invalid")


def test_generated_ack_routes_candidate_through_ambient_policy(monkeypatch):
    import gateway.generated_ack_harness as generated_ack_harness
    from gateway.generated_ack_harness import AckContext, GeneratedAckHarness

    calls = []

    class FakeAmbientPolicy:
        def evaluate(self, text, context):
            calls.append((text, context))
            return SimpleNamespace(
                allowed=True,
                text="Policy approved voice line.",
                sanitized=True,
                truncated=False,
                suppressed=False,
                max_seconds=2,
                reasons=("policy_checked",),
                classifiers={"code": False, "command_log": False, "raw_path": False, "secret_like": False, "sensitive_topic": False, "stack_trace": False},
                rule_profile="living_room_default",
            )

    monkeypatch.setattr(generated_ack_harness, "_AMBIENT_POLICY", FakeAmbientPolicy())
    result = GeneratedAckHarness(generator=lambda prompt, *, timeout_ms: "A natural Eon voice line.").generate(
        AckContext(user_message="Voice test", platform="discord", output_device="speaker")
    )

    assert result.text == "Policy approved voice line."
    assert calls[0][1].source == "ack"
    assert calls[0][1].platform == "discord"
    assert calls[0][1].output_device == "speaker"
    assert result.policy["allowed"] is True
