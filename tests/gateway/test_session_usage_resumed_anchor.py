"""A resumed session's context meter reads its anchor before the first new turn (#159070, site 2).

Right after a resume ``last_prompt_tokens`` is 0 and ``_turn_base_usage_anchor`` is unset until a
turn is built, so ``_context`` reported 0 (or the rough estimate) for a session whose real size is
on record: the restored ``_usage_anchor``, or the anchor persisted on the session row.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.usage_anchor import USAGE_ANCHOR_MODEL_CONFIG_KEY, capture_usage_anchor
from gateway.session_usage import _context

MSGS = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]


def _compressor(last_prompt_tokens: int = 0):
    return SimpleNamespace(last_prompt_tokens=last_prompt_tokens, context_length=1_048_576, compression_count=0,
                           _config_context_length=None)


def _db(anchor):
    return SimpleNamespace(
        get_session_model_config_value=lambda sid, key, default=None: anchor if key == USAGE_ANCHOR_MODEL_CONFIG_KEY
        else default)


def test_resumed_session_uses_the_restored_anchor_before_its_first_turn():
    agent = SimpleNamespace(_usage_anchor=capture_usage_anchor(850_000, 500, MSGS), _turn_base_usage_anchor=None,
                            _session_messages=MSGS)
    assert _context(agent, _compressor())["context_tokens"] == 850_500


def test_resumed_session_reads_the_persisted_anchor_before_its_first_turn():
    agent = SimpleNamespace(_usage_anchor=None, _turn_base_usage_anchor=None, _session_messages=MSGS,
                            _session_db=_db(capture_usage_anchor(850_000, 500, MSGS)), session_id="s1")
    assert _context(agent, _compressor())["context_tokens"] == 850_500


def test_a_stale_persisted_anchor_is_ignored():
    other = [{"role": "user", "content": "something else"}, {"role": "assistant", "content": "ok"}]
    agent = SimpleNamespace(_usage_anchor=None, _turn_base_usage_anchor=None, _session_messages=MSGS,
                            _session_db=_db(capture_usage_anchor(850_000, 500, other)), session_id="s1")
    assert _context(agent, _compressor())["context_tokens"] == 0


def test_a_live_reading_still_wins_over_the_fallback():
    agent = SimpleNamespace(_usage_anchor=capture_usage_anchor(850_000, 500, MSGS), _turn_base_usage_anchor=None,
                            _session_messages=MSGS)
    assert _context(agent, _compressor(last_prompt_tokens=12_345))["context_tokens"] == 12_345
