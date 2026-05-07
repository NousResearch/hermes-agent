# tests/agent/benchmarks/test_tier1_anchor.py
from tests.agent.benchmarks.fixture_builders import make_neutral_session


def test_1_8_anchor_makes_summary_start_at_assistant(compressor_pair, stub_summarizer):
    """Build a session where protect_first_n=3 lands on the third user
    message in a leading user-only run. With anchor on, the first
    summarized message must be an assistant turn, not a user turn."""
    baseline, with_flags = compressor_pair
    # Custom shape: leading 5 users, then alternating
    msgs = (
        [{"role": "system", "content": "sys"}]
        + [{"role": "user", "content": f"context {i}"} for i in range(5)]
        + [{"role": m, "content": f"x{i}"} for i in range(20)
           for m in ("assistant", "user")]
        + [{"role": "user", "content": "end"}]
    )

    # Inspect the slice the compressor will summarize. We hook by calling
    # _anchor_to_first_assistant directly — same logic the real path uses.
    # baseline: anchor disabled, so it returns whatever protect_first_n was.
    # with_flags: anchor enabled, slides to the first assistant.
    base_start = baseline._anchor_to_first_assistant(
        msgs, start_idx=baseline.protect_first_n, tail_start=len(msgs) - 5,
    )
    cand_start = with_flags._anchor_to_first_assistant(
        msgs, start_idx=with_flags.protect_first_n, tail_start=len(msgs) - 5,
    )
    assert msgs[base_start]["role"] in {"user", "system"}, (
        "Baseline should land at protect_first_n unchanged"
    )
    assert msgs[cand_start]["role"] == "assistant", (
        f"Anchor should slide to assistant; landed at {msgs[cand_start]['role']}"
    )
