"""``model.streaming`` config seeds the session's streaming decision (#72901).

The conversation loop prefers ``stream=True`` for every turn — subagents
included — for liveness health-checking (#3120). Self-hosted OpenAI-compatible
backends with broken streaming tool-call paths (e.g. vLLM
``--tool-call-parser qwen3_xml`` + reasoning parser) can leak tool-call markup
into plain text and return zero ``tool_calls``, silently no-oping delegated
tasks. ``model.streaming: false`` must seed ``_disable_streaming`` at agent
init so the whole session (parent and subagents) uses the non-streaming path.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_BASE = {
    "model": {
        "default": "test/model",
        "provider": "custom",
        "base_url": "http://127.0.0.1:9999/v1",
        "api_key": "x",
    }
}


def _build_agent(config):
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        return AIAgent(
            api_key="x",
            base_url="http://127.0.0.1:9999/v1",
            model="test/model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


@patch("run_agent.OpenAI")
def test_streaming_false_seeds_disable_streaming(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = _build_agent({"model": {**_BASE["model"], "streaming": False}})

    assert agent._disable_streaming is True


@patch("run_agent.OpenAI")
def test_streaming_absent_keeps_streaming_enabled(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = _build_agent(_BASE)

    assert agent._disable_streaming is False


@patch("run_agent.OpenAI")
def test_streaming_true_keeps_streaming_enabled(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = _build_agent({"model": {**_BASE["model"], "streaming": True}})

    assert agent._disable_streaming is False


@patch("run_agent.OpenAI")
def test_streaming_string_false_seeds_disable_streaming(mock_openai):
    """String falsy values ('false', '0') must also disable streaming —
    YAML users commonly quote booleans."""
    mock_openai.return_value = MagicMock()
    agent = _build_agent({"model": {**_BASE["model"], "streaming": "false"}})

    assert agent._disable_streaming is True


@patch("run_agent.OpenAI")
def test_streaming_zero_seeds_disable_streaming(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = _build_agent({"model": {**_BASE["model"], "streaming": 0}})

    assert agent._disable_streaming is True


@patch("run_agent.OpenAI")
def test_streaming_invalid_value_keeps_streaming_enabled(mock_openai):
    """Unrecognized values warn and keep the safe default (streaming on),
    rather than silently disabling or crashing init."""
    mock_openai.return_value = MagicMock()
    agent = _build_agent({"model": {**_BASE["model"], "streaming": "flase"}})

    assert agent._disable_streaming is False


@patch("run_agent.OpenAI")
def test_missing_model_section_keeps_streaming_enabled(mock_openai):
    mock_openai.return_value = MagicMock()
    agent = _build_agent({})

    assert agent._disable_streaming is False


@patch("run_agent.OpenAI")
def test_legacy_string_model_section_does_not_crash(mock_openai):
    """The top-level ``model`` key is a legacy string; init must not crash."""
    mock_openai.return_value = MagicMock()
    agent = _build_agent({"model": "test/model"})

    assert agent._disable_streaming is False


@patch("run_agent.OpenAI")
def test_streaming_false_applies_to_every_agent_built_from_config(mock_openai):
    """Delegate children are constructed through the same init, so any agent
    (parent or subagent) built under this config gets the escape hatch —
    covering the reported failure surface."""
    mock_openai.return_value = MagicMock()
    cfg = {"model": {**_BASE["model"], "streaming": False}}

    first = _build_agent(cfg)
    second = _build_agent(cfg)

    assert first._disable_streaming is True
    assert second._disable_streaming is True


@patch("run_agent.OpenAI")
def test_streaming_false_read_from_real_config_file(mock_openai):
    """End-to-end: a real config.yaml in HERMES_HOME (sandboxed per-test by
    conftest) with ``model.streaming: false`` must seed the flag through the
    actual config loader — not just the patched function."""
    mock_openai.return_value = MagicMock()
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: \"test/model\"\n"
        "  provider: \"custom\"\n"
        "  base_url: \"http://127.0.0.1:9999/v1\"\n"
        "  api_key: \"x\"\n"
        "  streaming: false\n",
        encoding="utf-8",
    )

    agent = AIAgent(
        api_key="x",
        base_url="http://127.0.0.1:9999/v1",
        model="test/model",
        provider="custom",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )

    assert agent._disable_streaming is True
