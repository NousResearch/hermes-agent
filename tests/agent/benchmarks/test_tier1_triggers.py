# tests/agent/benchmarks/test_tier1_triggers.py
from tests.agent.benchmarks.fixture_builders import make_neutral_session


def test_1_6_every_trigger_reachable_and_correctly_labeled(compressor_pair):
    _, c = compressor_pair  # with_flags has all triggers configured
    short = [{"role": "user", "content": "x"}]

    # token trigger
    c.last_prompt_tokens = c.threshold_tokens + 1
    assert c.should_compress() is True
    assert c._last_trigger == "token"
    c.last_prompt_tokens = 0

    # message trigger
    big = [{"role": "user", "content": str(i)} for i in range(c.message_threshold)]
    assert c.should_compress(prompt_tokens=0, messages=big) is True
    assert c._last_trigger == "message"

    # turn trigger (use few enough messages that message_threshold doesn't fire)
    user_only = [{"role": "user", "content": "x"} for _ in range(c.turn_threshold)]
    assert c.should_compress(prompt_tokens=0, messages=user_only) is True
    assert c._last_trigger == "turn"

    # no trigger
    assert c.should_compress(prompt_tokens=0, messages=short) is False
    assert c._last_trigger is None


def test_1_7_anti_thrashing_back_off_preserved(compressor_pair):
    """After 2 ineffective compactions, should_compress must back off
    even if a new (multi-trigger) condition would otherwise fire."""
    _, c = compressor_pair
    c._ineffective_compression_count = 2
    c.last_prompt_tokens = c.threshold_tokens + 1
    msgs = [{"role": "user", "content": str(i)} for i in range(c.message_threshold + 5)]

    assert c.should_compress(messages=msgs) is False
    assert c._last_trigger is None  # cleared by the back-off branch
