"""A failed config load hands ``_parse_compression_config`` ``{}``; every key must fall back to
DEFAULT_CONFIG, and an explicit ``threshold_tokens: null`` must stay the ratio-only opt-out.

Also covers the summarizer tool-arg bounds (``tool_arg_head_chars`` / ``tool_arg_min_chars``),
which must default to the historical constants and be overridable per deployment.
"""

from types import SimpleNamespace

import pytest

from agent.agent_init import _parse_compression_config
from hermes_cli.config import DEFAULT_CONFIG


def _agent():
    return SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)


@pytest.mark.parametrize(
    ("agent_cfg", "expected"),
    [
        ({}, DEFAULT_CONFIG["compression"]["threshold_tokens"]),  # config-load failure → shipped default
        ({"compression": {"threshold_tokens": None}}, None),  # explicit null → ratio-only opt-out
    ],
)
def test_threshold_tokens_default_and_null_opt_out(agent_cfg, expected):
    cs = _parse_compression_config(_agent(), agent_cfg)
    assert cs.threshold_tokens == expected
    assert cs.threshold == DEFAULT_CONFIG["compression"]["threshold"]


def test_tool_arg_bounds_default_to_historical_constants():
    """Absent config → the pre-existing hard-coded 1200 / 1500 behaviour is preserved."""
    assert DEFAULT_CONFIG["compression"]["tool_arg_head_chars"] == 1200
    assert DEFAULT_CONFIG["compression"]["tool_arg_min_chars"] == 1500

    cs = _parse_compression_config(_agent(), {})
    assert cs.tool_arg_head_chars == 1200
    assert cs.tool_arg_min_chars == 1500


def test_tool_arg_bounds_are_overridable():
    cs = _parse_compression_config(
        _agent(),
        {"compression": {"tool_arg_head_chars": 400, "tool_arg_min_chars": 900}},
    )
    assert cs.tool_arg_head_chars == 400
    assert cs.tool_arg_min_chars == 900


@pytest.mark.parametrize("bad", [-1, "-50", "not-a-number", None])
def test_tool_arg_bounds_floor_at_zero_and_survive_junk(bad):
    """A zeroed or unparseable value must not reach the compressor as a negative/None slice bound."""
    cs = _parse_compression_config(
        _agent(),
        {"compression": {"tool_arg_head_chars": bad, "tool_arg_min_chars": bad}},
    )
    assert cs.tool_arg_head_chars >= 0
    assert cs.tool_arg_min_chars >= 0
    assert isinstance(cs.tool_arg_head_chars, int)
    assert isinstance(cs.tool_arg_min_chars, int)
