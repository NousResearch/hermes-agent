"""Tests for configurable agent retry backoff delays."""
from unittest.mock import patch

import pytest

from run_agent import AIAgent


def _make_agent(**agent_cfg):
    """Build an AIAgent with mocked config containing the given agent keys."""
    cfg = {"agent": dict(agent_cfg)}

    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def test_default_retry_delays():
    agent = _make_agent()
    assert agent._retry_base_delay == 5.0
    assert agent._retry_max_delay == 120.0
    assert agent._rate_limit_retry_base_delay == 2.0
    assert agent._rate_limit_retry_max_delay == 60.0


def test_retry_delays_honor_config_override():
    agent = _make_agent(
        retry_base_delay=1.5,
        retry_max_delay=45.0,
        rate_limit_retry_base_delay=0.5,
        rate_limit_retry_max_delay=15.0,
    )
    assert agent._retry_base_delay == 1.5
    assert agent._retry_max_delay == 45.0
    assert agent._rate_limit_retry_base_delay == 0.5
    assert agent._rate_limit_retry_max_delay == 15.0


def test_retry_delays_fall_back_on_invalid_value():
    agent = _make_agent(
        retry_base_delay="bad",
        retry_max_delay=None,
        rate_limit_retry_base_delay=-3,
        rate_limit_retry_max_delay="nope",
    )
    assert agent._retry_base_delay == 5.0
    assert agent._retry_max_delay == 120.0
    assert agent._rate_limit_retry_base_delay == 0.0
    assert agent._rate_limit_retry_max_delay == 60.0


def test_conversation_loop_uses_configured_retry_delays(monkeypatch):
    """The API retry path should pass agent config into jittered_backoff."""
    from agent import conversation_loop as conv_loop

    agent = _make_agent(retry_base_delay=7.0, retry_max_delay=99.0)
    captured = {}

    def _capture_backoff(attempt, *, base_delay, max_delay, jitter_ratio=0.5):
        captured["attempt"] = attempt
        captured["base_delay"] = base_delay
        captured["max_delay"] = max_delay
        return 0.0

    monkeypatch.setattr(conv_loop, "jittered_backoff", _capture_backoff)

    conv_loop.jittered_backoff(
        2,
        base_delay=agent._retry_base_delay,
        max_delay=agent._retry_max_delay,
    )

    assert captured == {
        "attempt": 2,
        "base_delay": 7.0,
        "max_delay": 99.0,
    }


def test_trajectory_compressor_retry_max_delay_from_yaml(tmp_path):
    from trajectory_compressor import CompressionConfig

    yaml_path = tmp_path / "compression.yaml"
    yaml_path.write_text(
        "summarization:\n"
        "  retry_delay: 3.5\n"
        "  retry_max_delay: 42.0\n",
        encoding="utf-8",
    )

    config = CompressionConfig.from_yaml(str(yaml_path))
    assert config.retry_delay == 3.5
    assert config.retry_max_delay == 42.0


@pytest.mark.parametrize(
    ("base_delay", "max_delay"),
    [
        (2.0, 60.0),
        (4.0, 90.0),
    ],
)
def test_trajectory_compressor_uses_configured_backoff(base_delay, max_delay):
    from trajectory_compressor import CompressionConfig, TrajectoryCompressor

    config = CompressionConfig(retry_delay=base_delay, retry_max_delay=max_delay)
    compressor = TrajectoryCompressor.__new__(TrajectoryCompressor)
    compressor.config = config

    delay = compressor.config.retry_max_delay
    assert delay == max_delay
    assert compressor.config.retry_delay == base_delay
