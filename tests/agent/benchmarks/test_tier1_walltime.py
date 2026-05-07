# tests/agent/benchmarks/test_tier1_walltime.py
import time
import statistics

from agent.model_metadata import estimate_messages_tokens_rough
from tests.agent.benchmarks._report import record
from tests.agent.benchmarks.fixture_builders import make_loop_session


def _time_compress(c, msgs, n_runs=5):
    pre = estimate_messages_tokens_rough(msgs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        c.compress(msgs.copy(), current_tokens=pre)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), max(times)


def test_1_3_compress_walltime_within_budget(compressor_pair, stub_summarizer):
    """compress() with flags on must not exceed 1.20 × baseline median."""
    baseline, with_flags = compressor_pair
    msgs = make_loop_session(n_iterations=30, chars_per_read=4_000)

    med_b, _ = _time_compress(baseline, msgs)
    med_c, _ = _time_compress(with_flags, msgs)
    record("1.3", "median_compress_seconds", med_b, med_c, "s")

    assert med_c <= 1.20 * med_b, (
        f"compress() slowed by {med_c/med_b:.2%} (budget 1.20×)"
    )


def test_1_4_dedup_pass_alone_fast_on_long_session(compressor_pair):
    """_dedup_by_operation must be linear-ish; 1000-message session < 50ms."""
    _, with_flags = compressor_pair
    msgs = make_loop_session(n_iterations=125, chars_per_read=200)
    assert len(msgs) > 500
    t0 = time.perf_counter()
    with_flags._dedup_by_operation(msgs)
    elapsed = time.perf_counter() - t0
    record("1.4", "dedup_walltime_ms", 0.0, elapsed * 1000, "ms",
           note=f"{len(msgs)} messages")

    assert elapsed < 0.05, f"_dedup_by_operation took {elapsed*1000:.1f}ms (budget 50ms)"
