"""C4 automatic compaction: property + liveness suite on seeded random sessions (good summarizer).

A real AIAgent + SessionDB drives seeded random sessions (tool-pair heavy with parallel calls, huge single
turns, image parts, long answers, single tool results larger than the whole compaction threshold) against
the recording fake provider. After EVERY turn and for EVERY request the shared invariants in ``_helpers``
are asserted:

* liveness  -- the turn returns within a bounded wall time and a bounded number of summarizer calls;
  a session left over the trigger surfaces a warning; a provider-side overflow ends the turn with an
  explicit error or recovers, within a bounded number of retries (never a silent loop);
* safety    -- no user message is lost from state.db, tool_call/tool_result pairs stay matched in the
  live rows AND in every request, the system prompt stays the session's, and the in-flight ask (for a
  cron run: the job prompt) reaches the model after every summary;
* persisted == sent -- state.db at request time is exactly the request's history prefix (+ only the
  in-flight turn), the live rows equal the host's in-memory history after each turn, and a fresh agent
  resumed from state.db continues with the same history.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.compaction._helpers import (
    GOOD_SUMMARY_TOKEN,
    MAX_SUMMARY_CALLS_PER_TURN,
    SEEDS,
    THRESHOLD_TOKENS,
    drive,
    estimate_prompt_tokens,
    generate_transcript,
    job_turn,
    warned,
)


@pytest.mark.parametrize("seed", SEEDS)
def test_good_summarizer_compacts_and_keeps_every_invariant(make_scenario, tmp_path, seed):
    sc = make_scenario("good")
    drive(sc, seed, tmp_path / "work")
    assert sc.summary_calls, "the seeded session never crossed the compaction trigger"
    assert sc.committed_compactions() > 0, "a good summary was produced but no compaction committed"
    # A committed good summary is what the model sees next.
    last = sc.turns[-1].main_requests[-1]
    assert any(GOOD_SUMMARY_TOKEN in str(m.get("content")) for m in last["messages"]), (
        "the committed summary is not in the request after compaction")
    # Ends below the trigger, or said so explicitly.
    if estimate_prompt_tokens(last) >= THRESHOLD_TOKENS:
        assert warned(sc), "session left over the trigger with no warning"


@pytest.mark.parametrize("platform", ("cron", "cli"))
def test_job_prompt_survives_mid_turn_compaction(make_scenario, tmp_path, platform):
    """One job prompt, many tool rounds: compaction fires mid-turn (repeatedly). The job prompt must
    still reach the model after every summary, every tool result the turn produced must stay paired,
    and the run must actually finish its script (#100818: the run "succeeded" but delivered nothing)."""
    sc = make_scenario("good", platform=platform)
    spec = job_turn(SEEDS[0], tmp_path / "work", rounds=14)
    rec = sc.run_turn(spec)
    assert sc.committed_compactions() > 0, "the job run never compacted mid-turn"
    assert rec.summary_calls <= MAX_SUMMARY_CALLS_PER_TURN * 2
    assert spec.marker in str(rec.result.get("final_response")), "the job run did not deliver its report"


# Overflow retries per turn: each rejected request may trigger one compaction retry, capped by the
# per-turn compression attempt budget (3). Anything past a small multiple is a retry loop.
MAX_OVERFLOW_REJECTIONS_PER_TURN = 6


@pytest.mark.parametrize("mode", ("empty", "good"))
def test_provider_overflow_terminates(make_scenario, tmp_path, mode):
    """The provider itself rejects over-window requests (the local-server shape: a real window smaller
    than the configured one). Recovery is bounded per turn and either succeeds or ends the turn with an
    explicit error -- never a silent loop -- and every invariant still holds afterwards."""
    sc = make_scenario(mode, window=THRESHOLD_TOKENS * 3 // 4)
    specs = generate_transcript(SEEDS[1], 10, tmp_path / "work", kinds=("tail_bomb", "tools", "huge_user"))
    for i, spec in enumerate(specs):
        before = sc.overflow_rejections
        rec = sc.run_turn(spec, allow_abort=True)
        assert sc.overflow_rejections - before <= MAX_OVERFLOW_REJECTIONS_PER_TURN, (
            f"turn {i}: {sc.overflow_rejections - before} over-window requests in one turn (retry loop)")
        assert rec.summary_calls <= MAX_SUMMARY_CALLS_PER_TURN, f"turn {i}: {rec.summary_calls} summarizer calls"
    assert sc.overflow_rejections, "the scenario never hit the provider window"
    if mode == "good":
        assert sc.committed_compactions() > 0
    else:
        assert sc.committed_compactions() == 0
