from gateway.run import GatewayRunner


import pytest


@pytest.mark.parametrize(
    ("text", "expected_mode", "expected_text"),
    [
        ("light hello", "light", "hello"),
        ("light: hello", "light", "hello"),
        ("light, hello", "light", "hello"),
        ("light — hello", "light", "hello"),
        ("lite hello", "light", "hello"),
        ("лайт объясни коротко", "light", "объясни коротко"),
        ("лайт: объясни коротко", "light", "объясни коротко"),
        ("heavy hello", "heavy", "hello"),
        ("heavy: hello", "heavy", "hello"),
        ("heavy - hello", "heavy", "hello"),
        ("хеви проверь", "heavy", "проверь"),
        ("хэви, проверь", "heavy", "проверь"),
        ("  HeAvY: Check this", "heavy", "Check this"),
    ],
)
def test_parse_turn_reasoning_prefix_matches(text, expected_mode, expected_text):
    assert GatewayRunner._parse_turn_reasoning_prefix(text) == (
        expected_mode,
        expected_text,
    )


@pytest.mark.parametrize(
    "text",
    [
        "highlight this",
        "lightweight summary",
        "the word heavy inside",
        "/reasoning heavy",
        "heavy",
        "light:",
        "",
    ],
)
def test_parse_turn_reasoning_prefix_ignores_non_prefixes(text):
    assert GatewayRunner._parse_turn_reasoning_prefix(text) == (None, text)


def test_parse_turn_reasoning_prefix_matches_voice_transcript_wrapper():
    text = '[The user sent a voice message~ Here\'s what they said: "Light. Проверка быстрого ответа."]'

    assert GatewayRunner._parse_turn_reasoning_prefix(text) == (
        "light",
        "Проверка быстрого ответа.",
    )


def test_parse_turn_reasoning_prefix_matches_voice_transcript_wrapper_with_tail():
    text = (
        '[The user sent a voice message~ Here\'s what they said: "HEAVY. Проверка думающей модели."]'
        "\n\n[reply context]"
    )

    assert GatewayRunner._parse_turn_reasoning_prefix(text) == (
        "heavy",
        "Проверка думающей модели.\n\n[reply context]",
    )


def test_apply_turn_reasoning_override_light_is_low():
    assert GatewayRunner._apply_turn_reasoning_override(None, "light") == {
        "enabled": True,
        "effort": "low",
    }


def test_apply_turn_reasoning_override_heavy_is_xhigh():
    assert GatewayRunner._apply_turn_reasoning_override(
        {"enabled": True, "effort": "medium"},
        "heavy",
    ) == {"enabled": True, "effort": "xhigh"}


def test_apply_turn_reasoning_override_none_preserves_session_config():
    config = {"enabled": True, "effort": "medium"}
    assert GatewayRunner._apply_turn_reasoning_override(config, None) is config
