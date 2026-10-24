"""Configurable source budgets for long-running agents (PR #142749 feedback)."""

from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor, pin_summary_route
from agent.auxiliary_task_config import session_auxiliary_scope
from agent.model_metadata import estimate_tokens_rough
from hermes_cli.config import atomic_config_write
from hermes_constants import get_hermes_home


@pytest.mark.parametrize("mode", ["lean", "legacy"])
@pytest.mark.parametrize("previous", ["", "old state", "previous-state " * 14000], ids=["fresh", "short", "long"])
def test_configured_budget_preserves_long_source_and_rolling_state(mode, previous):
    # The 160K character caps, fixed 40K ceiling, or a reserved but unused
    # 40% handoff share would each drop the middle decision in this fixture.
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "compression": {"summary_input_max_tokens": 150000},
    })
    compressor = ContextCompressor(model="local/test", config_context_length=262144,
                                   tail_mode=mode, quiet_mode=True)
    compressor._summary_model_context_length = 200000
    compressor._previous_summary = previous
    compressor._active_compression_telemetry = {}
    count = 70 if len(previous) > 100 else 100
    turns = [{"role": "user", "content": f"decision-{i:03d} " + "a" * 4900} for i in range(count)]
    prompts = []
    with patch.object(compressor, "_call_summary_llm", side_effect=lambda p, _: prompts.append(p) or "Summary"):
        assert compressor._generate_summary(turns)
    assert len(prompts) == 1
    prompt = prompts[0]
    assert all(turn["content"] in prompt for turn in turns)
    assert previous in prompt
    assert 100000 < estimate_tokens_rough(prompt) < 175000
    assert estimate_tokens_rough(prompt) + compressor.max_summary_tokens < 200000
    if mode == "lean":
        assert compressor._active_compression_telemetry["summary_input_omitted_chars"] == 0


@pytest.mark.parametrize("limit,main,aux,fallen_back,fallback,expected_window,max_source", [
    (None, 131072, None, False, False, 131072, 32768),
    (None, 1000000, None, False, False, 1000000, 40000),
    (12000, 131072, None, False, False, 131072, 12000),
    (10**12, 131072, None, False, False, 131072, 98304),
    (150000, 131072, 200000, False, False, 200000, 150000),
    (150000, 131072, 200000, True, False, 131072, 98304),
    (150000, 262144, 200000, False, "chain", 64000, 48000),
    (150000, 262144, 200000, False, "pin", 64000, 48000),
    (0, 131072, None, False, False, 131072, 32768),
    (-1, 131072, None, False, False, 131072, 32768),
    (True, 131072, None, False, False, 131072, 32768),
    ("150000", 131072, None, False, False, 131072, 32768),
    (1.5, 131072, None, False, False, 131072, 32768),
])
def test_configured_allowance_respects_defaults_routes_and_window_headroom(
    limit, main, aux, fallen_back, fallback, expected_window, max_source,
):
    atomic_config_write(get_hermes_home() / "config.yaml", {
        "compression": {"summary_input_max_tokens": limit},
    })
    compressor = ContextCompressor(model="local/test", config_context_length=main, quiet_mode=True)
    compressor._summary_model_context_length = aux
    compressor._summary_model_fallen_back = fallen_back
    route = {"provider": "custom", "model": "local/fallback", "base_url": "http://127.0.0.1:8081/v1"}
    with (session_auxiliary_scope({"compression": {"fallback_chain": [route] if fallback == "chain" else []}}),
          pin_summary_route(route if fallback == "pin" else None)):
        allowance, window = compressor._summary_input_allowance(6553, None, "", True)
        assert (allowance, window) == (max_source, expected_window)
        prompt = compressor._fit_summary_prompt("新" * 180000, 6553, None, "", True)
        assert estimate_tokens_rough(prompt) + 6553 < expected_window
        # Instructions/output consume the window even if an explicit setting is huge.
        with pytest.raises(ValueError, match="no room"):
            compressor._summary_input_allowance(expected_window, None, "", True)
