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


def _build_session(n_turns: int, words_per_turn: int = 20) -> list:
    """Build a multi-turn conversation with a system prompt."""
    base_text = " ".join(["alpha"] * words_per_turn)
    messages = [{"role": "system", "content": "You are a helpful agent."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"{base_text} (user turn {i})"})
        messages.append({"role": "assistant", "content": f"{base_text} (assistant turn {i})"})
    return messages


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


# ---------------------------------------------------------------------------
# Integration: the REQUEST-level estimate (incl. tool schemas) is what makes
# the verdict correct. This reproduces the exact bug class on the real estimate
# functions: a placeholder/partial summary shrinks the MESSAGES enough to look
# effective, while the full REQUEST (messages + ~30K tool schemas) stays over
# the trigger -> must count as ineffective.
# ---------------------------------------------------------------------------

def _fat_tool_schemas(n: int = 60) -> list:
    """Build a realistic ~30K-token tool-schema payload (the live agent ships
    50+ tools; their JSON schemas are the blind spot messages-only misses)."""
    tools = []
    for i in range(n):
        tools.append({
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "x" * 300,
                "parameters": {
                    "type": "object",
                    "properties": {
                        f"param_{j}": {"type": "string", "description": "y" * 80}
                        for j in range(6)
                    },
                },
            },
        })
    return tools


def test_request_level_estimate_catches_thrash_that_messages_only_misses() -> None:
    """End-to-end on the real estimate functions: a compaction whose MESSAGES
    shrank >=10% but whose full REQUEST stayed over threshold AND shed <10% is
    ineffective. This is the real thrash mechanism: the protected tail (bulky
    recent tool outputs) + system + tool schemas dominate, so summarizing the
    middle barely moves the request even though messages-only looks like a win.

    Exercises the same estimate_request_tokens_rough the done-site uses, not a
    hand-picked number.
    """
    from agent.model_metadata import (
        estimate_messages_tokens_rough,
        estimate_request_tokens_rough,
    )

    tools = _fat_tool_schemas(200)
    system = "s" * 40000  # large system prompt (skills, soul, etc.)

    # Small head + a modest summarizable middle + a bulky protected tail. The
    # tool schemas (~80K) + system (~9K) dominate the request, so summarizing
    # the middle shrinks the messages-only count a lot (the OLD bug's verdict)
    # but barely dents the full request.
    head = _build_session(2, words_per_turn=30)
    middle = _build_session(6, words_per_turn=300)  # the summarizable window
    big_tail = [
        {"role": "user", "content": "recent " * 2500},
        {"role": "assistant", "content": "recent reply " * 2500},
    ] * 3  # bulky protected tail
    pre_messages = head + middle + big_tail
    # Post: middle replaced by a short placeholder; head + big_tail preserved.
    post_messages = head + [
        {"role": "user", "content": "[CONTEXT COMPACTION - REFERENCE ONLY] summary unavailable."}
    ] + big_tail

    pre_req = estimate_request_tokens_rough(pre_messages, system_prompt=system, tools=tools)
    post_req = estimate_request_tokens_rough(post_messages, system_prompt=system, tools=tools)
    pre_mo = estimate_messages_tokens_rough(pre_messages)
    post_mo = estimate_messages_tokens_rough(post_messages)

    # Fixture sanity: messages-only saw a big "win" (the OLD bug path would have
    # reset the counter)…
    assert (pre_mo - post_mo) / pre_mo * 100 >= 10
    # …but the REQUEST barely moved (schemas + system + tail dominate) — <10%.
    assert (pre_req - post_req) / pre_req * 100 < 10

    # Pin the trigger just under the post REQUEST size: the real request is
    # still over threshold (thrash condition).
    cc = _make_compressor(threshold_tokens=post_req - 1)
    cc.record_compaction_effectiveness(
        pre_request_tokens=pre_req, post_request_tokens=post_req,
    )
    assert cc._ineffective_compression_count == 1

