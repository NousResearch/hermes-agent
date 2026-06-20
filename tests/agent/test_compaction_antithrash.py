"""Phase C — anti-thrash counter reflects REQUEST-LEVEL effectiveness.

Regression for the compaction-thrash incident (2026-06-19). Root cause (proven
against the code, correcting the spec's original session-split premise):

``ContextCompressor.compress()`` computes its "savings" verdict from
``display_tokens`` (the pre-compaction REQUEST-level real prompt count, ~205K)
minus ``estimate_messages_tokens_rough(compressed)`` (a MESSAGES-ONLY post
estimate that excludes the ~30K of tool schemas + system prompt). The post side
is therefore understated by the schema overhead, so a compaction that did NOT
actually drop the real request below the trigger still reads as a >=10% "saving"
-> ``_ineffective_compression_count`` resets to 0 -> the anti-thrash guard in
``should_compress`` never reaches 2 -> the loop re-fires forever.

The fix makes the effectiveness verdict apples-to-apples at the REQUEST level
(the same level that decides re-trigger), owned by a single method the
done-site calls with the request-level pre/post estimates.

Invariant I2: after at most K=2 consecutive request-level-ineffective passes,
``should_compress`` backs off. A single ineffective pass increments the counter
by exactly 1 (no double-count).
"""

from __future__ import annotations

from agent.context_compressor import ContextCompressor


def _make_compressor(threshold_tokens: int = 204_000) -> ContextCompressor:
    cc = ContextCompressor(
        model="gpt-5.5",
        threshold_percent=0.75,
        quiet_mode=True,
        config_context_length=272_000,
        provider="openai-codex",
    )
    # Pin the trigger to a known value for the test.
    cc.threshold_tokens = threshold_tokens
    return cc


def test_request_level_ineffective_increments_when_post_stays_over_threshold() -> None:
    """A compaction whose REQUEST-level post estimate did not drop below the
    threshold is ineffective, even if messages-only savings looked fine."""
    cc = _make_compressor(threshold_tokens=204_000)
    assert cc._ineffective_compression_count == 0
    # pre 205K -> post 205K (request-level: still over the 204K trigger).
    cc.record_compaction_effectiveness(pre_request_tokens=205_072, post_request_tokens=205_883)
    assert cc._ineffective_compression_count == 1


def test_single_ineffective_pass_increments_by_exactly_one() -> None:
    """Pass-2 nit: one ineffective pass increments by exactly 1 (no double wire)."""
    cc = _make_compressor()
    cc.record_compaction_effectiveness(pre_request_tokens=205_072, post_request_tokens=297_723)
    assert cc._ineffective_compression_count == 1


def test_two_request_level_ineffective_passes_trip_backoff() -> None:
    cc = _make_compressor(threshold_tokens=204_000)
    cc.record_compaction_effectiveness(pre_request_tokens=205_000, post_request_tokens=205_500)
    cc.record_compaction_effectiveness(pre_request_tokens=205_500, post_request_tokens=206_000)
    assert cc._ineffective_compression_count == 2
    # Now the guard must back off even though we're over threshold.
    assert cc.should_compress(250_000) is False


def test_effective_request_level_pass_resets_counter() -> None:
    """A genuine compaction that drops the request below threshold resets the
    counter to 0 — the single reset point."""
    cc = _make_compressor(threshold_tokens=204_000)
    cc._ineffective_compression_count = 1
    # 490K -> 37K request-level: a real reduction below threshold.
    cc.record_compaction_effectiveness(pre_request_tokens=490_256, post_request_tokens=36_954)
    assert cc._ineffective_compression_count == 0


def test_counter_persists_across_compression_session_split() -> None:
    """The compressor instance survives the compaction-driven session split
    (same cached agent), so the counter persists across the rollover. The split
    notifies via on_session_start(boundary_reason='compression'), which the
    built-in compressor ignores — it must NOT reset the counter."""
    cc = _make_compressor()
    cc._ineffective_compression_count = 1
    # Simulate the compression-driven split notification.
    if hasattr(cc, "on_session_start"):
        cc.on_session_start("new_sid_after_split", boundary_reason="compression")
    assert cc._ineffective_compression_count == 1


def test_counter_resets_on_real_new_session() -> None:
    """A real /new (on_session_reset) clears the counter — the leak direction.
    Anti-thrash state must not bleed into a genuinely fresh session."""
    cc = _make_compressor()
    cc._ineffective_compression_count = 2
    cc.on_session_reset()
    assert cc._ineffective_compression_count == 0
