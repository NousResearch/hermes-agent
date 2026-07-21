from types import SimpleNamespace
from unittest.mock import patch

from gateway.tts_summary import prepare_tts_text


def _cfg(enabled=False, max_chars=700, max_tokens=180):
    return {
        "voice": {
            "tts_summary_enabled": enabled,
            "tts_summary_max_chars": max_chars,
            "tts_summary_max_tokens": max_tokens,
        }
    }


def test_prepare_tts_text_disabled_uses_stripped_original():
    with patch("hermes_cli.config.load_config", return_value=_cfg(enabled=False)), \
         patch("gateway.tts_summary._strip_markdown_for_tts", side_effect=lambda t: t.replace("**", "")) as strip, \
         patch("agent.auxiliary_client.call_llm") as call_llm:
        result = prepare_tts_text("**Full** text")

    assert result == "Full text"
    strip.assert_called_once_with("**Full** text")
    call_llm.assert_not_called()


def test_prepare_tts_text_enabled_uses_auxiliary_summary():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Short spoken summary."))]
    )
    with patch("hermes_cli.config.load_config", return_value=_cfg(enabled=True, max_chars=80)), \
         patch("agent.auxiliary_client.call_llm", return_value=response) as call_llm, \
         patch("gateway.tts_summary._strip_markdown_for_tts", side_effect=lambda t: t):
        result = prepare_tts_text("Long detailed response with code and logs.")

    assert result == "Short spoken summary."
    call_llm.assert_called_once()
    kwargs = call_llm.call_args.kwargs
    assert kwargs["task"] == "tts_summary"
    assert kwargs["max_tokens"] == 180


def test_prepare_tts_text_summary_failure_falls_back_to_original():
    with patch("hermes_cli.config.load_config", return_value=_cfg(enabled=True)), \
         patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("boom")), \
         patch("gateway.tts_summary._strip_markdown_for_tts", return_value="fallback text"):
        result = prepare_tts_text("Long detailed response")

    assert result == "fallback text"


def test_prepare_tts_text_clamps_summary():
    long_summary = "This is the first useful sentence. " + ("extra " * 80)
    with patch("hermes_cli.config.load_config", return_value=_cfg(enabled=True, max_chars=120)), \
         patch("agent.auxiliary_client.call_llm", return_value={"content": long_summary}), \
         patch("gateway.tts_summary._strip_markdown_for_tts", side_effect=lambda t: t):
        result = prepare_tts_text("Long detailed response")

    assert len(result) <= 121  # allow ellipsis if no boundary is available
    assert result.startswith("This is the first useful sentence.")
