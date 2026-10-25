"""An "insufficient progress" verdict must not outlive the request it judged.

Pre-API preflight sets ``_preflight_compression_blocked`` when a compaction
pass fails to shrink the request enough to be worth repeating. That verdict is
a statement about one transcript. The turn keeps appending tool results after
it, and the flag stayed armed for the rest of the turn, so the new bulk was
never offered to the compressor.

Observed live on a 65,536-token window: a pass at ~40,297 tokens came back at
~41,063 (the summary cost more than the tool results it replaced), the gate
went dark, and the following calls ran at 43,309 -> 53,549 -> 60,126 -> 62,239
tokens with no further attempt, past the point where the model began returning
empty turns.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import (
    _compression_warrants_another_preflight_pass,
    _preflight_block_outlived_its_request,
)
from tests.agent.test_turn_context import _FakeAgent, _build


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def test_growth_past_the_materiality_margin_lifts_the_block():
    # The live incident: blocked at ~41,063, request later reached ~62,239.
    assert _preflight_block_outlived_its_request(41_063, 62_239)


def test_noise_sized_growth_keeps_the_block():
    """A few hundred tokens is not a different transcript."""
    assert not _preflight_block_outlived_its_request(41_063, 41_500)


def test_shrinking_request_keeps_the_block():
    assert not _preflight_block_outlived_its_request(41_063, 38_000)


def test_unknown_block_pressure_never_lifts_on_its_own():
    """0 means "armed by a path that did not record a size" — stay conservative."""
    assert not _preflight_block_outlived_its_request(0, 62_239)


def test_margin_matches_the_shrink_test_that_arms_the_block():
    """Re-arm on exactly the scale of change it took to keep the gate open.

    ``_compression_warrants_another_preflight_pass`` treats a 5% reduction as
    the smallest move worth another pass; growth uses the same 5%, so the two
    halves of the decision cannot drift apart.
    """
    threshold = 40_000
    base = 50_000
    just_under = int(base * 1.05)
    just_over = just_under + 1

    assert not _preflight_block_outlived_its_request(base, just_under)
    assert _preflight_block_outlived_its_request(base, just_over)

    # Mirror image: a 5% shrink is likewise the boundary for continuing.
    assert not _compression_warrants_another_preflight_pass(
        base, int(base * 0.95), threshold,
    )
    assert _compression_warrants_another_preflight_pass(
        base, int(base * 0.95) - 1, threshold,
    )


def _pressured_compressor():
    """Over-threshold compressor stub that opens the preflight threshold path."""
    return types.SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1,
        context_length=100_000,
        last_prompt_tokens=0,
        should_compress=lambda _tokens=None: True,
        should_compress_info=lambda _tokens=None: (True, None),
        should_defer_preflight_to_real_usage=lambda _t: False,
        get_active_compression_failure_cooldown=lambda: None,
    )


def test_arming_the_block_records_the_request_it_judged():
    """Without the recorded size the loop has nothing to compare growth against.

    A no-op compaction arms the blocker; the context must carry the request
    size that verdict was about, or the re-arm check can never fire.
    """
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent.context_compressor = _pressured_compressor()
    agent._emit_status = MagicMock()
    agent._compress_context = lambda messages, _system_message, **_kw: (
        messages, "SYSTEM",
    )

    ctx = _build(agent, conversation_history=[
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "older"},
    ])

    assert ctx.preflight_compression_blocked is True
    assert ctx.preflight_blocked_at_tokens > 0
