"""The opt-in exact ratio bypasses floors while preserving real input limits."""

import pytest

from agent.context_compressor import ContextCompressor


@pytest.mark.parametrize("window", [65_536, 1_000_000])
@pytest.mark.parametrize("reserve", [None, 8192])
def test_exact_ratio_uses_input_budget(window, reserve):
    comp = ContextCompressor(
        "exact-test", threshold_percent=0.70, exact_threshold=True,
        config_context_length=window, max_tokens=reserve, quiet_mode=True,
    )
    expected = int((window - (reserve or 0)) * 0.70)
    assert comp.threshold_percent == 0.70
    assert comp.threshold_tokens == expected
    assert not comp.should_compress(expected - 1)
    assert comp.should_compress(expected)
    assert comp.preview_threshold_tokens(comp.model, window) == expected


@pytest.mark.parametrize("reserve, expected", [(None, 55_705), (8192, 48_742)])
def test_flag_off_keeps_existing_small_window_floors(reserve, expected):
    comp = ContextCompressor(
        "legacy-test", threshold_percent=0.70, exact_threshold=False,
        config_context_length=65_536, max_tokens=reserve, quiet_mode=True,
    )
    assert comp.threshold_tokens == expected
    assert comp.threshold_percent == 0.75


def test_exact_ratio_recomputed_on_window_and_model_updates():
    comp = ContextCompressor(
        "exact-test", threshold_percent=0.70, exact_threshold=True,
        config_context_length=65_536, max_tokens=8192, quiet_mode=True,
    )
    assert comp.threshold_tokens == 40_140
    comp.context_length = 1_000_000
    assert comp.threshold_tokens == 694_265
    comp.update_model("other-test", 65_536)
    assert comp.threshold_percent == 0.70
    assert comp.threshold_tokens == 40_140


def test_exact_ratio_keeps_absolute_cap_and_durable_auxiliary_ceiling():
    comp = ContextCompressor(
        "exact-test", threshold_percent=0.70, exact_threshold=True,
        config_context_length=1_000_000, max_tokens=8192,
        threshold_tokens_cap=100_000, quiet_mode=True,
    )
    assert comp.threshold_tokens == 100_000
    assert comp.preview_threshold_tokens(comp.model, 1_000_000) == 100_000
    comp._aux_context_ceiling = 30_000
    comp._threshold_tokens = None
    assert comp.threshold_tokens == 30_000
    comp.update_model(comp.model, 65_536)
    assert comp._aux_context_ceiling == 30_000
    assert comp.threshold_tokens == 30_000
