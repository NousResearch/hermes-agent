from types import SimpleNamespace

from agent.compression_policy import (
    mid_turn_threshold_tokens,
    should_compress_at_turn_end,
    should_compress_mid_turn,
)


class StubCompressor:
    def __init__(self, *, threshold=80, context=100, max_tokens=0):
        self.threshold_tokens = threshold
        self.context_length = context
        self.max_tokens = max_tokens

    def should_compress(self, tokens):
        return tokens >= self.threshold_tokens


def _agent(*, deferred, emergency=0.92, compressor=None):
    return SimpleNamespace(
        compression_enabled=True,
        compression_defer_until_turn_end=deferred,
        compression_emergency_threshold=emergency,
        context_compressor=compressor or StubCompressor(),
    )


def test_historical_mode_still_compresses_at_soft_threshold_mid_turn():
    agent = _agent(deferred=False)
    assert mid_turn_threshold_tokens(agent) == 80
    assert should_compress_mid_turn(agent, 80) is True


def test_historical_mode_delegates_threshold_decision_to_compressor():
    agent = _agent(deferred=False)
    agent.context_compressor.should_compress = lambda _tokens: True
    assert should_compress_mid_turn(agent, 1) is True


def test_deferred_mode_does_not_interrupt_between_soft_and_emergency_thresholds():
    agent = _agent(deferred=True)
    assert mid_turn_threshold_tokens(agent) == 92
    assert should_compress_mid_turn(agent, 80) is False
    assert should_compress_mid_turn(agent, 91) is False
    assert should_compress_mid_turn(agent, 92) is True


def test_deferred_mode_compacts_at_completed_turn_soft_threshold():
    agent = _agent(deferred=True)
    assert should_compress_at_turn_end(agent, 79) is False
    assert should_compress_at_turn_end(agent, 80) is True


def test_turn_end_hook_is_opt_in():
    agent = _agent(deferred=False)
    assert should_compress_at_turn_end(agent, 100) is False


def test_emergency_ratio_uses_effective_input_window_after_output_reserve():
    compressor = StubCompressor(threshold=720, context=1000, max_tokens=100)
    agent = _agent(deferred=True, compressor=compressor)
    assert mid_turn_threshold_tokens(agent) == 828
    assert should_compress_mid_turn(agent, 827) is False
    assert should_compress_mid_turn(agent, 828) is True


def test_emergency_threshold_can_never_be_lower_than_soft_threshold():
    compressor = StubCompressor(threshold=95, context=100, max_tokens=0)
    agent = _agent(deferred=True, emergency=0.90, compressor=compressor)
    assert mid_turn_threshold_tokens(agent) == 95
    assert should_compress_mid_turn(agent, 94) is False
    assert should_compress_mid_turn(agent, 95) is True


def test_disabled_compression_blocks_both_timing_paths():
    agent = _agent(deferred=True)
    agent.compression_enabled = False
    assert should_compress_mid_turn(agent, 100) is False
    assert should_compress_at_turn_end(agent, 100) is False
