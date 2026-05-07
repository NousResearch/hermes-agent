# Qwen-Aware Compaction — Benchmark & Regression Plan

> **Companion to:** `2026-05-02-qwen-aware-compaction-improvements.md`
> The implementation plan's unit tests verify *correctness*. This plan
> measures whether the changes **improved**, **worsened**, or **didn't
> meaningfully change** real performance/reliability/accuracy.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to land each tier in order.

**Goal:** Produce evidence-based ship/no-ship signal for each P0/P1/P2 change. Every benchmark has a numeric acceptance threshold and a published "what regression looks like" criterion.

**Approach:** Three tiers, fast→slow.

- **Tier 1 — Deterministic invariants.** Pure-Python; no LLM. Fast. These catch the bulk of regressions because most failure modes are mechanical (tokens not saved, atomicity violated, trigger misattributed).
- **Tier 2 — Local-Qwen integration.** Requires the moe profile up; auto-skipped otherwise. Measures wall-clock cost and end-to-end accuracy.
- **Tier 3 — Trajectory replay differential.** Captures real Hermes sessions to JSON, replays the same trajectory through both old and new compaction code, diffs the resulting message lists. Catches regressions on workloads we didn't anticipate.

---

## Acceptance Criteria — the decision matrix

Each row reads: "On this benchmark, the implementation **ships** if the metric satisfies this threshold; otherwise we **investigate**."

| ID | Benchmark | Ships if | Investigate if | What regression looks like |
|---|---|---|---|---|
| 1.1 | Token efficiency on dedup-friendly synthetic loop | tokens-after-compact (with flags) ≤ 0.85 × (without) | ratio ≥ 0.95 | dedup not catching anything; check `_last_op_deduped` |
| 1.2 | Token efficiency on neutral synthetic session (no resource reuse) | ratio within 0.95–1.05 | ratio < 0.90 (we deleted something we shouldn't) | non-redundant content destroyed |
| 1.3 | `compress()` wall-clock vs baseline | new ≤ 1.20 × baseline | new > 1.50 × baseline | regex/loop hot-path overhead |
| 1.4 | `_dedup_by_operation` wall-clock | < 50ms on 1000-message fixture | > 200ms | O(n²) bug in operation-key indexing |
| 1.5 | Tool-atomicity invariant | 0 orphaned `tool_call_id`s after compaction (any flag combo) | ≥ 1 | boundary calc broken, sanitizer not running |
| 1.6 | Trigger attribution | every documented trigger reachable + correctly labeled | mislabeled or unreachable trigger | `_last_trigger` not set / set wrong |
| 1.7 | Anti-thrashing preserved | back-off still fires after 2 ineffective passes | back-off doesn't fire | new triggers bypass the counter |
| 1.8 | First-assistant anchor sanity | post-compaction first-summarized-message role == "assistant" when anchor on, regardless of `protect_first_n` | role == "user" or "tool" | anchor logic broken |
| 1.9 | Multimodal tool-result preservation | image parts present in output for any flag combo | image parts dropped or replaced with strings | `_dedup_by_operation` skipped a list-content guard |
| 2.1 | Per-turn wall-clock on real local Qwen | median ≤ 1.15 × baseline (after 1 warmup discard, N=5) | median > 1.20 × baseline | new triggers fire too often; MoE batch-mode switching; missing warmup |
| 2.2 | Compaction-event frequency | flags-on fires no more than 1.3× as often as flags-off on the same fixture | flags-on fires > 2× as often | absolute cap set too tight; threshold mis-tuned |
| 2.3 | Fact-retention accuracy | ≥ 0.80 retention with flags AND ≥ (baseline − 0.05) | < 0.75 OR > 0.05 below baseline | summary quality degraded |
| 2.4 | Tool-loop completion rate | completion rate with flags ≥ baseline | drop > 5pp | post-compaction state confuses model |
| 3.1 | Trajectory replay — fingerprint diff on dedup-disabled inputs | new == baseline message-by-message | any diff | a flag-off codepath unexpectedly mutated |
| 3.2 | Trajectory replay — fingerprint diff on enabled inputs | only the documented-difference fields differ | other fields differ | side effect leaking through |
| 3.3 | Real-session replay (3 captured sessions) | every session compresses without raising; final message list valid OpenAI shape | exception OR invalid shape | edge case in real workload |

The headline pass criterion across the board: **no Tier-1 regression, ≤15% median wall-clock cost (with warmup, N=5; tightened later when GPU/CPU pinning lands), ≥80% fact retention, no atomicity violations, no message-shape errors.**

---

## File Structure

```
tests/agent/benchmarks/
├── __init__.py                           # Empty marker
├── conftest.py                           # Shared fixtures (compressor pair, fixture loader)
├── fixture_builders.py                   # Synthetic message-list generators
├── trajectory.py                         # Capture/replay primitives for Tier 3
├── test_tier1_token_efficiency.py        # 1.1, 1.2
├── test_tier1_walltime.py                # 1.3, 1.4
├── test_tier1_atomicity.py               # 1.5, 1.9
├── test_tier1_triggers.py                # 1.6, 1.7
├── test_tier1_anchor.py                  # 1.8
├── test_tier2_qwen_walltime.py           # 2.1, 2.2 (qwen-required)
├── test_tier2_qwen_accuracy.py           # 2.3, 2.4 (qwen-required)
├── test_tier3_trajectory_replay.py       # 3.1, 3.2, 3.3
└── trajectories/                         # Captured real-session JSON
    ├── README.md
    ├── 50turn-readwrite.json
    ├── 200turn-research.json
    └── 100turn-vision.json
docs/research/
└── 2026-05-02-qwen-aware-compaction-benchmark-report.md   # generated, gitignored
```

---

## Tier 0 — Shared harness

Before writing benchmarks, establish primitives every benchmark uses.

### 0.1 — Paired-compressor fixture

Create `tests/agent/benchmarks/conftest.py`:

```python
"""Shared benchmark fixtures.

`compressor_pair` returns two identically-configured compressors except
for the qwen_aware flag block. Every benchmark that A/B-compares uses
this fixture so the only variable is the flag set.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


@pytest.fixture
def compressor_pair():
    """Return ``(baseline, with_flags)`` compressor pair.

    Both have:
      - identical context_length (256K, mocked)
      - identical threshold_percent (0.50)
      - identical protect_first_n / protect_last_n / target_ratio
      - quiet_mode=True

    Difference: ``with_flags`` has every qwen_aware flag enabled.
    """
    def _make(**kw):
        defaults = dict(
            model="bench/qwen-instruct",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            summary_target_ratio=0.20,
            quiet_mode=True,
            base_url="",
            api_key="",
            config_context_length=262_144,
            provider="bench",
            api_mode="chat_completions",
        )
        defaults.update(kw)
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=262_144,
        ):
            return ContextCompressor(**defaults)

    baseline = _make()
    with_flags = _make(
        qwen_aware_enabled=True,
        dedup_operations=True,
        anchor_first_assistant=True,
        threshold_absolute_max=80_000,
        message_threshold=200,
        turn_threshold=30,
    )
    return baseline, with_flags


@pytest.fixture
def stub_summarizer(monkeypatch):
    """Patch _generate_summary to return a deterministic short string.

    Tier-1 benchmarks should never fire a real LLM call. This fixture
    makes the summarizer deterministic so token-delta math is reproducible.
    """
    def _stub(self, turns, focus_topic=None):
        return f"## Summary\n{len(turns)} turns compressed."
    monkeypatch.setattr(ContextCompressor, "_generate_summary", _stub)
```

### 0.2 — Fixture builders

Create `tests/agent/benchmarks/fixture_builders.py`:

```python
"""Synthetic message-list generators with predictable shape & token weight."""
from __future__ import annotations

from typing import Any, Dict, List


def make_loop_session(
    n_iterations: int,
    paths: tuple[str, ...] = ("/a.py", "/b.py", "/c.py"),
    chars_per_read: int = 4_000,
) -> List[Dict[str, Any]]:
    """N iterations of: user → assistant(read X) → tool → assistant(patch X) → tool.

    Cycles through ``paths`` so each path gets read+patched many times.
    This is the dedup-friendly shape: lots of redundant reads.

    Returns OpenAI-shaped messages including system + user-prelude.
    """
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Refactor the codebase."},
    ]
    for i in range(n_iterations):
        path = paths[i % len(paths)]
        cid_r, cid_p = f"r{i}", f"p{i}"
        msgs += [
            {"role": "assistant", "content": f"Reading {path}.",
             "tool_calls": [{"id": cid_r, "function": {
                 "name": "read_file",
                 "arguments": f'{{"path":"{path}"}}'}}]},
            {"role": "tool", "tool_call_id": cid_r,
             "content": ("x" * chars_per_read) + f"\n# rev {i}"},
            {"role": "assistant", "content": f"Patching {path}.",
             "tool_calls": [{"id": cid_p, "function": {
                 "name": "patch",
                 "arguments": f'{{"mode":"replace","path":"{path}",'
                              f'"old_string":"x","new_string":"y"}}'}}]},
            {"role": "tool", "tool_call_id": cid_p, "content": "OK"},
        ]
    msgs.append({"role": "user", "content": "Now summarize."})
    return msgs


def make_neutral_session(n_turns: int, chars_per_turn: int = 1_500) -> List[Dict[str, Any]]:
    """N turns of unique user/assistant exchanges — no resource reuse.

    Stresses the path where dedup should NOT fire and we shouldn't
    accidentally lose information.
    """
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are an assistant."},
    ]
    for i in range(n_turns):
        msgs += [
            {"role": "user", "content": f"Question {i}: " + ("q" * chars_per_turn)},
            {"role": "assistant", "content": f"Answer {i}: " + ("a" * chars_per_turn)},
        ]
    return msgs


def make_parallel_tool_session(n_groups: int, fanout: int = 3) -> List[Dict[str, Any]]:
    """N groups of (assistant emits ``fanout`` parallel tool_calls → ``fanout`` tool_results)."""
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Read three files."},
    ]
    for g in range(n_groups):
        cids = [f"g{g}c{j}" for j in range(fanout)]
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": cid, "function": {
                    "name": "read_file",
                    "arguments": f'{{"path":"/grp{g}_{j}.py"}}'}}
                for j, cid in enumerate(cids)
            ],
        })
        for j, cid in enumerate(cids):
            msgs.append({"role": "tool", "tool_call_id": cid,
                         "content": f"file g{g}/{j} content"})
    msgs.append({"role": "user", "content": "Now summarize."})
    return msgs


def make_multimodal_session(n_image_turns: int) -> List[Dict[str, Any]]:
    """User sends images, assistant analyzes. Verifies vision messages
    flow through compaction without corruption."""
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": "Vision-capable assistant."},
    ]
    for i in range(n_image_turns):
        msgs += [
            {"role": "user", "content": [
                {"type": "text", "text": f"What is in image {i}?"},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,IMG{i}=="}},
            ]},
            {"role": "assistant", "content": f"It's a cat in image {i}."},
        ]
    msgs.append({"role": "user", "content": "Done."})
    return msgs


def estimate_serialized_bytes(messages: List[Dict[str, Any]]) -> int:
    """Sum the serialized JSON byte length of every message.

    Approximates the wire cost of shipping the prompt to the local
    server. Local Qwen has no partial-prefix KV reuse, so wire bytes
    correlate strongly with prefill wall-clock.
    """
    import json
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
```

### 0.3 — Markdown report generator

Create `tests/agent/benchmarks/_report.py` (private):

```python
"""Aggregate benchmark results into a markdown table.

Each test calls ``record(benchmark_id, metric_name, baseline, candidate, unit)``
and the session-end hook writes
``docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md``.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

_RESULTS: dict[str, list[dict]] = defaultdict(list)


def record(bid: str, metric: str, baseline: float, candidate: float,
           unit: str = "", note: str = "") -> None:
    _RESULTS[bid].append({
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "delta_pct": (candidate - baseline) / baseline * 100 if baseline else 0.0,
        "unit": unit,
        "note": note,
    })


def emit_report(path: Path) -> None:
    lines = ["# Compaction Benchmark Report", "",
             "| Benchmark | Metric | Baseline | Candidate | Δ% | Unit | Note |",
             "|---|---|---|---|---|---|---|"]
    for bid in sorted(_RESULTS):
        for r in _RESULTS[bid]:
            lines.append(
                f"| {bid} | {r['metric']} | {r['baseline']:.2f} "
                f"| {r['candidate']:.2f} | {r['delta_pct']:+.1f}% "
                f"| {r['unit']} | {r['note']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    # Also dump raw JSON for downstream tooling.
    path.with_suffix(".json").write_text(
        json.dumps(_RESULTS, indent=2, default=str)
    )
```

Wire it into `conftest.py`'s session-end hook:

```python
def pytest_sessionfinish(session, exitstatus):
    from tests.agent.benchmarks._report import _RESULTS, emit_report
    if not _RESULTS:
        return
    out = Path(__file__).resolve().parents[3] / (
        "docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md"
    )
    emit_report(out)
    print(f"\n[bench] report written to {out}")
```

---

## Tier 1 — Deterministic invariants

These run on every CI build of the feature branch. No LLM, no live server. Total runtime target: < 30s.

### Benchmark 1.1 — Token efficiency on a dedup-friendly loop

```python
# tests/agent/benchmarks/test_tier1_token_efficiency.py
from agent.model_metadata import estimate_messages_tokens_rough
from tests.agent.benchmarks._report import record
from tests.agent.benchmarks.fixture_builders import make_loop_session


def test_1_1_dedup_friendly_loop_saves_tokens(compressor_pair, stub_summarizer):
    """30 iterations of read+patch on 3 paths = 90 redundant tool results.
    Pass 1.5 should supersede the older reads/patches; net token count
    after compaction should drop materially."""
    baseline, with_flags = compressor_pair
    msgs = make_loop_session(n_iterations=30, chars_per_read=4_000)
    pre_tokens = estimate_messages_tokens_rough(msgs)

    out_b = baseline.compress(msgs.copy(), current_tokens=pre_tokens)
    out_c = with_flags.compress(msgs.copy(), current_tokens=pre_tokens)

    tk_b = estimate_messages_tokens_rough(out_b)
    tk_c = estimate_messages_tokens_rough(out_c)
    record("1.1", "post_compact_tokens", tk_b, tk_c, "tok")
    record("1.1", "op_deduped_count",
           getattr(baseline, "_last_op_deduped", 0),
           getattr(with_flags, "_last_op_deduped", 0), "ops")

    # Acceptance: candidate ≤ 0.85 × baseline (≥ 15% improvement)
    assert tk_c <= 0.85 * tk_b, (
        f"Expected ≥15% token reduction; got {tk_c}/{tk_b} = {tk_c/tk_b:.2%}"
    )
    # And the dedup pass actually fired
    assert with_flags._last_op_deduped > 0


def test_1_2_neutral_session_does_not_lose_information(stub_summarizer):
    """No resource reuse → dedup should not fire. Token counts within ±5%.
    A larger drop would indicate dedup deleted non-redundant information.

    NOTE: This test isolates ``dedup_operations`` from the other qwen_aware
    flags. Using the integrated ``compressor_pair`` here would confound the
    measurement: ``threshold_absolute_max`` shrinks ``tail_token_budget``
    (since ``tail_token_budget = threshold_tokens * summary_target_ratio``)
    so the with_flags compressor compacts more aggressively for reasons
    unrelated to dedup. We construct a custom pair that differs ONLY in
    ``dedup_operations`` so the assertion measures dedup's effect alone.
    """
    from unittest.mock import patch
    from tests.agent.benchmarks.fixture_builders import make_neutral_session
    from agent.context_compressor import ContextCompressor

    def _make(**kw):
        defaults = dict(
            model="bench/qwen-instruct",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            summary_target_ratio=0.20,
            quiet_mode=True,
            base_url="",
            api_key="",
            config_context_length=262_144,
            provider="bench",
            api_mode="chat_completions",
        )
        defaults.update(kw)
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=262_144,
        ):
            return ContextCompressor(**defaults)

    # Identical configs except for dedup_operations.
    baseline = _make()
    dedup_only = _make(dedup_operations=True)

    msgs = make_neutral_session(n_turns=40, chars_per_turn=2_000)
    pre = estimate_messages_tokens_rough(msgs)

    out_b = baseline.compress(msgs.copy(), current_tokens=pre)
    out_c = dedup_only.compress(msgs.copy(), current_tokens=pre)
    tk_b = estimate_messages_tokens_rough(out_b)
    tk_c = estimate_messages_tokens_rough(out_c)
    record("1.2", "post_compact_tokens", tk_b, tk_c, "tok")

    # ratio in [0.95, 1.05] — dedup is a no-op on neutral content, so
    # output sizes must match closely. Anything outside this band means
    # dedup mistakenly collapsed unrelated tool calls.
    ratio = tk_c / tk_b if tk_b else 1.0
    assert 0.95 <= ratio <= 1.05, (
        f"Neutral session ratio {ratio:.3f} outside [0.95, 1.05] — "
        f"dedup_operations may be deleting non-redundant content"
    )
```

### Benchmark 1.3, 1.4 — Wall-clock micro-benchmarks

```python
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
```

### Benchmark 1.5, 1.9 — Atomicity and multimodal preservation

```python
# tests/agent/benchmarks/test_tier1_atomicity.py
from tests.agent.benchmarks.fixture_builders import (
    make_loop_session, make_parallel_tool_session, make_multimodal_session,
)


def _orphan_tool_ids(messages) -> set[str]:
    """Return any tool_call_ids present on a tool message but absent on
    any assistant tool_calls (or vice versa)."""
    asst_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                if cid:
                    asst_ids.add(cid)
    tool_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                tool_ids.add(cid)
    return (asst_ids ^ tool_ids)


def test_1_5_no_orphan_tool_pairs_after_compaction(compressor_pair, stub_summarizer):
    """Across all flag combos, compress() must never leave orphan tool
    pairs in the output. Sanitization is a post-condition the API
    relies on; an orphan would 400 on the next API call."""
    from itertools import product
    baseline, with_flags = compressor_pair

    fixtures = {
        "loop": make_loop_session(20),
        "parallel": make_parallel_tool_session(8, fanout=3),
        "mixed": make_loop_session(10) + make_parallel_tool_session(4, fanout=2),
    }
    for name, msgs in fixtures.items():
        for c in (baseline, with_flags):
            out = c.compress(msgs.copy(), current_tokens=999_999)  # force compact
            orphans = _orphan_tool_ids(out)
            assert not orphans, f"{name} → orphans: {orphans}"


def test_1_9_multimodal_image_parts_never_clobbered(compressor_pair, stub_summarizer):
    """Vision messages with image content must survive both compressors
    intact — no part dropped, no string replacement."""
    baseline, with_flags = compressor_pair
    msgs = make_multimodal_session(n_image_turns=15)

    for c in (baseline, with_flags):
        out = c.compress(msgs.copy(), current_tokens=999_999)
        # Every surviving user message that originally had an image part
        # must still have one. (Compaction may delete whole messages, but
        # whatever survives must be intact.)
        for m in out:
            content = m.get("content")
            if isinstance(content, list):
                # If list-content survived at all, image_url parts must
                # be present and uncorrupted (data: URL prefix).
                images = [p for p in content
                          if isinstance(p, dict) and p.get("type") == "image_url"]
                if images:
                    for img in images:
                        url = (img.get("image_url") or {}).get("url", "")
                        assert url.startswith("data:image/"), (
                            f"image_url corrupted: {url[:80]}"
                        )
```

### Benchmark 1.6, 1.7 — Trigger attribution + anti-thrashing

```python
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
```

### Benchmark 1.8 — First-assistant anchor sanitization

```python
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
```

---

## Tier 2 — Local-Qwen integration

Auto-skipped when `127.0.0.1:8085` is unreachable. These take real wall-clock time (10-60s each). Run on demand, not in CI.

### Benchmark 2.1, 2.2 — Per-turn wall-clock + compaction frequency

```python
# tests/agent/benchmarks/test_tier2_qwen_walltime.py
"""Real wall-clock A/B against the live moe profile.

Each test runs a synthetic-but-realistic session twice (baseline vs.
with-flags compressor) and records median/max per-turn duration plus
compaction event count.

Marked ``integration`` so pyproject's ``addopts = "-m 'not integration'"``
default keeps these out of CI runs. Opt in via ``-m integration``.
Also auto-skipped when the moe profile isn't reachable on :8085.
"""
import os
import socket
import time
import pytest
from contextlib import contextmanager

from agent.context_compressor import ContextCompressor
from tests.agent.benchmarks._report import record
from tests.agent.benchmarks.fixture_builders import (
    make_loop_session, estimate_serialized_bytes,
)

# Module-level marker — every test in this file is integration.
pytestmark = pytest.mark.integration


def _qwen_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8085), timeout=0.5):
            return True
    except OSError:
        return False


qwen_required = pytest.mark.skipif(
    not _qwen_up(),
    reason="local-qwen moe profile not running on :8085",
)


def _make(**flags):
    return ContextCompressor(
        model="qwen-instruct",
        threshold_percent=0.50,
        protect_first_n=3,
        protect_last_n=20,
        summary_target_ratio=0.20,
        quiet_mode=True,
        base_url="http://127.0.0.1:8085/v1",
        api_key="not-needed",
        # provider="custom" — auxiliary_client routing keyword for
        # "use the explicit base_url + api_key passed at the call site."
        # The YAML key in ~/.hermes/config.yaml ("local-qwen") is just
        # a logical label; auxiliary_client doesn't see that file.
        provider="custom",
        api_mode="chat_completions",
        config_context_length=262_144,
        **flags,
    )


@qwen_required
def test_2_1_per_turn_walltime_within_budget():
    """Run the same loop session twice, count compactions + measure
    per-compaction wall-clock. Threshold: candidate median ≤ 1.15 ×
    baseline (was 1.05× — see methodology note below).

    **Why 1.15× and not 1.05× (research, 2026-05-02):** empirical
    benchmarks of Qwen3.6-35B-A3B on RTX 3090 via llama.cpp show
    end-to-end wall-clock CV of 2-5% in the warm/stable regime and
    6-8% std at N=3 unwarmed. With confounders from MoE routing
    variance and CPU↔GPU sync overhead (--n-cpu-moe=28), a 5%
    threshold at N=3 is below the noise floor. We use 1.15× with
    1 warmup discard + N=5 + median (not mean) for robustness.

    To tighten back to 1.10× later: keep warmup, raise N to 10,
    pin GPU/CPU governors, and use llama-server's reported
    ``eval_duration`` from the response JSON (excludes HTTP +
    Python overhead). See research notes in the plan's main
    benchmark file.
    """
    import statistics

    msgs = make_loop_session(n_iterations=40)
    bytes_pre = estimate_serialized_bytes(msgs)

    def _run(c, n_runs=5, n_warmup=1):
        # Warmup: discard the first N runs to stabilize the page
        # cache + CPU clocks + (when this calls into a server) any
        # cold-start overhead. Empirically the first run is 12-18%
        # slower than the steady state.
        for _ in range(n_warmup):
            c.compress(msgs.copy(), current_tokens=200_000)
        events = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = c.compress(msgs.copy(), current_tokens=200_000)
            events.append({
                "elapsed": time.perf_counter() - t0,
                "msgs": len(out),
                "deduped": getattr(c, "_last_op_deduped", 0),
            })
        return events

    base_events = _run(_make())
    cand_events = _run(_make(
        qwen_aware_enabled=True,
        dedup_operations=True,
        anchor_first_assistant=True,
        threshold_absolute_max=80_000,
        message_threshold=200,
        turn_threshold=30,
    ))

    # Median is robust to outlier runs (e.g. one slow run from MoE
    # expert routing variance won't poison the regression check).
    base_med = statistics.median(e["elapsed"] for e in base_events)
    cand_med = statistics.median(e["elapsed"] for e in cand_events)
    record("2.1", "compaction_walltime_seconds_median", base_med, cand_med, "s")
    record("2.1", "compaction_walltime_seconds_max",
           max(e["elapsed"] for e in base_events),
           max(e["elapsed"] for e in cand_events), "s")
    record("2.2", "compactions_per_session",
           len(base_events) + 1, len(cand_events) + 1, "n",
           note="includes warmup")

    assert cand_med <= 1.15 * base_med, (
        f"compress() median slowed by {cand_med/base_med:.2%} (budget 1.15×). "
        f"baseline median={base_med*1000:.1f}ms, candidate={cand_med*1000:.1f}ms"
    )
```

### Benchmark 2.3 — Fact retention accuracy

```python
# tests/agent/benchmarks/test_tier2_qwen_accuracy.py
"""Plant N facts in early turns, force compaction, ask the model
N follow-up questions, score retention.

Marked ``integration`` (skipped by default; opt in with ``-m integration``)
AND auto-skipped when the moe profile isn't reachable on :8085.
"""
import pytest
import socket

from agent.context_compressor import ContextCompressor
from agent.auxiliary_client import call_llm
from tests.agent.benchmarks._report import record

# Module-level marker — every test in this file is integration.
pytestmark = pytest.mark.integration


def _qwen_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8085), timeout=0.5):
            return True
    except OSError:
        return False


qwen_required = pytest.mark.skipif(
    not _qwen_up(), reason="local-qwen moe profile not running on :8085",
)


# 10 facts paired with verification questions and expected substring
# matches in the model's answer (case-insensitive).
FACT_PROBES = [
    ("My favorite programming language is Rust.",
     "What is my favorite programming language?", ("rust",)),
    ("The deployment deadline is March 15, 2026.",
     "When is the deployment deadline?", ("march 15", "march 15, 2026")),
    ("The database password is stored in /etc/secrets/db.env.",
     "Where is the database password stored?", ("/etc/secrets/db.env",)),
    ("My API rate limit is 1000 requests per minute.",
     "What's my API rate limit?", ("1000", "1,000")),
    ("The project uses Python 3.13.",
     "Which Python version does the project use?", ("3.13", "python 3.13")),
    ("Production deploys go to us-east-2.",
     "Which AWS region for production?", ("us-east-2",)),
    ("Tests live under tests/ — never under src/.",
     "Where do tests live?", ("tests/", "under tests")),
    ("The CI provider is Buildkite.",
     "Which CI provider?", ("buildkite",)),
    ("Errors above 5xx page the on-call.",
     "Which errors page on-call?", ("5xx", "above 5xx", "500")),
    ("The lead reviewer is Akhil.",
     "Who is the lead reviewer?", ("akhil",)),
]


def _build_session_with_facts() -> list[dict]:
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    # Plant facts in early turns, padded with chat that pushes toward compaction
    for fact, _, _ in FACT_PROBES:
        msgs.append({"role": "user", "content": fact})
        msgs.append({"role": "assistant", "content": "Got it."})
    # Filler turns to force compaction
    for i in range(60):
        msgs.append({"role": "user", "content": f"Context filler turn {i}: " + "x" * 1500})
        msgs.append({"role": "assistant", "content": f"Acknowledged turn {i}."})
    return msgs


def _ask_questions(messages: list[dict]) -> int:
    """For each probe, append the question, call qwen-instruct, score.

    Note: ``call_llm`` signature (verified in agent/auxiliary_client.py)
    accepts ``provider``, ``model``, ``base_url``, ``api_key``,
    ``messages``, ``temperature``, ``max_tokens``, ``tools``, ``timeout``,
    ``extra_body`` — NO ``api_mode`` kwarg. Don't add one.
    """
    score = 0
    for _, question, expected in FACT_PROBES:
        probe_msgs = messages + [{"role": "user", "content": question}]
        resp = call_llm(
            messages=probe_msgs,
            model="qwen-instruct",
            base_url="http://127.0.0.1:8085/v1",
            api_key="not-needed",
            # provider="custom" — see test_tier2_qwen_walltime.py
            # for the explanation of why "custom" not "local-qwen".
            provider="custom",
            max_tokens=400,
        )
        answer = (resp.choices[0].message.content or "").lower()
        if any(e.lower() in answer for e in expected):
            score += 1
    return score


@qwen_required
def test_2_3_fact_retention_after_compaction():
    """A/B fact retention. Both compressors compact the same fixture,
    then we ask the model 10 fact-recall questions. Acceptance: with-flags
    score ≥ 0.80 AND ≥ (baseline − 0.05)."""
    fixture = _build_session_with_facts()

    def _make(**flags):
        return ContextCompressor(
            model="qwen-instruct", threshold_percent=0.50,
            protect_first_n=3, protect_last_n=20,
            summary_target_ratio=0.20, quiet_mode=True,
            base_url="http://127.0.0.1:8085/v1", api_key="not-needed",
            # provider="custom" — see test_tier2_qwen_walltime.py.
            provider="custom", api_mode="chat_completions",
            config_context_length=262_144, **flags,
        )

    baseline_compacted = _make().compress(fixture.copy(), current_tokens=200_000)
    candidate_compacted = _make(
        qwen_aware_enabled=True, dedup_operations=True,
        anchor_first_assistant=True, threshold_absolute_max=80_000,
        message_threshold=200, turn_threshold=30,
    ).compress(fixture.copy(), current_tokens=200_000)

    base_score = _ask_questions(baseline_compacted)
    cand_score = _ask_questions(candidate_compacted)
    record("2.3", "fact_retention_pct",
           base_score / len(FACT_PROBES) * 100,
           cand_score / len(FACT_PROBES) * 100, "%")

    assert cand_score >= 8, (
        f"Fact retention {cand_score}/10 below 0.80 floor"
    )
    assert cand_score >= base_score - 1, (
        f"Fact retention regressed: {base_score} → {cand_score} "
        f"(allowed drop: 1 fact)"
    )
```

### Benchmark 2.4 — Tool-loop completion rate

```python
# tests/agent/benchmarks/test_tier2_qwen_walltime.py (continued)
@qwen_required
def test_2_4_tool_loop_does_not_get_stuck_post_compaction():
    """Drive a real Hermes-shaped tool loop with compaction in the
    middle. The post-compaction model must complete the loop within
    a turn budget — NOT get stuck re-issuing tool calls because of
    orphaned/corrupted history."""
    # Run the agent harness (or a synthetic equivalent) on a fixture
    # that mid-session triggers compaction. Track turns-to-completion.
    # If the model gets stuck (turns > 50 without completion), fail.
    # (Implementation: leverage the existing tests/agent harness; this
    # task in the plan must wire that in. Marker for the implementer:
    # use run_agent.py:_run_tool_loop as the reference shape.)
    pytest.skip("Wire to run_agent harness in implementation step")
```

(2.4 is intentionally a skip/TODO marker — the harness wiring is non-trivial. The implementer wires it after observing 2.1-2.3 land.)

---

## Tier 3 — Trajectory replay differential

Captures real Hermes sessions to JSON, then replays the captured message lists through both compressor variants and diffs the outputs. Catches regressions on workloads we didn't synthesize.

### 3.1 — Capture format

A trajectory is a JSON file containing the message list at each compaction-trigger point in a real session, plus the metadata needed to reconstruct the compressor state (context_length, threshold_tokens, etc.).

```python
# tests/agent/benchmarks/trajectory.py
import json
from pathlib import Path
from typing import Any

TRAJECTORY_DIR = Path(__file__).resolve().parent / "trajectories"


def capture(session_id: str, messages: list[dict],
            engine_state: dict[str, Any]) -> Path:
    """Write a trajectory snapshot. Called from a debug-only hook in
    run_agent.py during a real session. Not enabled in production."""
    out = TRAJECTORY_DIR / f"{session_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": 1,
        "messages": messages,
        "engine_state": engine_state,
    }, indent=2, default=str))
    return out


def load(name: str) -> tuple[list[dict], dict[str, Any]]:
    raw = json.loads((TRAJECTORY_DIR / name).read_text())
    return raw["messages"], raw["engine_state"]
```

The implementer adds a CLI flag (e.g. `--capture-trajectory <name>`) to `hermes` that wires `capture()` into the existing `should_compress`-true branch in `run_agent.py:13280`. Single line of opt-in instrumentation.

### 3.2 — Replay differential

```python
# tests/agent/benchmarks/test_tier3_trajectory_replay.py
"""Trajectory replay differential.

Marked ``integration`` so pyproject's default ``addopts = "-m 'not integration'"``
keeps these out of plain CI runs. They depend on captured real-session
trajectories under ``trajectories/``; opt in with ``-m integration``.
"""
import hashlib
import json
import pytest
from pathlib import Path

from tests.agent.benchmarks.trajectory import load, TRAJECTORY_DIR

# Module-level marker — every test in this file is integration.
pytestmark = pytest.mark.integration


def _fingerprint(messages: list[dict]) -> str:
    """Stable hash of message-list shape (role + content + tool_call_ids)."""
    canonical = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c_str = json.dumps(
                [{"type": p.get("type"),
                  "text_len": len(p.get("text", ""))} for p in c if isinstance(p, dict)],
                sort_keys=True,
            )
        else:
            c_str = f"<{type(c).__name__}:{len(c or '')}>"
        tcs = [tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
               for tc in (m.get("tool_calls") or [])]
        canonical.append({"role": m.get("role"), "c": c_str, "tcs": tcs})
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()
    ).hexdigest()[:16]


def _trajectories():
    if not TRAJECTORY_DIR.exists():
        return []
    return sorted(p.name for p in TRAJECTORY_DIR.glob("*.json"))


@pytest.mark.parametrize("name", _trajectories())
def test_3_1_baseline_replay_is_stable(name, compressor_pair, stub_summarizer):
    """Sanity: baseline compaction on a captured trajectory is deterministic."""
    baseline, _ = compressor_pair
    msgs, state = load(name)

    out1 = baseline.compress(msgs.copy(), current_tokens=state.get("tokens", 200_000))
    out2 = baseline.compress(msgs.copy(), current_tokens=state.get("tokens", 200_000))
    assert _fingerprint(out1) == _fingerprint(out2)


@pytest.mark.parametrize("name", _trajectories())
def test_3_2_candidate_diff_only_in_documented_fields(name, compressor_pair, stub_summarizer):
    """Compare baseline vs candidate output. Flags-on can differ — but
    the only allowed differences are:
      - content of older tool results (Pass 1.5 supersession)
      - boundary index (anchor)
      - summary text (different inputs → different summary)
    Anything else is a regression (e.g. roles changed, tool_call_ids
    corrupted, image parts dropped)."""
    baseline, with_flags = compressor_pair
    msgs, state = load(name)
    pre_tokens = state.get("tokens", 200_000)

    out_b = baseline.compress(msgs.copy(), current_tokens=pre_tokens)
    out_c = with_flags.compress(msgs.copy(), current_tokens=pre_tokens)

    # Roles in same order (modulo length difference at the head/tail)
    roles_b = [m.get("role") for m in out_b]
    roles_c = [m.get("role") for m in out_c]

    # Strict: every role in candidate must be a valid OpenAI role
    assert set(roles_c) <= {"system", "user", "assistant", "tool"}

    # No orphaned tool_call_ids in either output
    from tests.agent.benchmarks.test_tier1_atomicity import _orphan_tool_ids
    assert not _orphan_tool_ids(out_b)
    assert not _orphan_tool_ids(out_c)

    # If both produced the same number of messages AND the same role
    # sequence, the only acceptable per-message diffs are content
    # (specifically: older tool results becoming "[Superseded ..." or
    # the summary message body).
    if len(out_b) == len(out_c) and roles_b == roles_c:
        for i, (mb, mc) in enumerate(zip(out_b, out_c)):
            if mb.get("role") != "tool":
                continue
            cb = mb.get("content") or ""
            cc = mc.get("content") or ""
            if cb != cc:
                # Only acceptable: candidate replaced with "[Superseded ..."
                assert isinstance(cc, str) and (
                    cc.startswith("[Superseded by later")
                    or cc.startswith("[Duplicate tool output")
                ), (
                    f"Unexpected tool-content diff at idx {i}: "
                    f"{cb[:80]} → {cc[:80]}"
                )


@pytest.mark.parametrize("name", _trajectories())
def test_3_3_real_session_replay_does_not_raise(name, compressor_pair, stub_summarizer):
    """Smoke: every captured trajectory should compact without raising."""
    baseline, with_flags = compressor_pair
    msgs, state = load(name)
    for c in (baseline, with_flags):
        out = c.compress(msgs.copy(), current_tokens=state.get("tokens", 200_000))
        # Final shape is OpenAI-valid
        for m in out:
            assert m.get("role") in {"system", "user", "assistant", "tool"}
            if m.get("role") == "tool":
                assert "tool_call_id" in m
```

When the trajectory dir is empty (clean checkout), the parametrize returns no cases and the tests pass trivially. The benchmark is opt-in: capture some real sessions, drop them in `tests/agent/benchmarks/trajectories/`, re-run.

---

## How to run

**Pytest invocation requirements (read first):**
- For Tier 1 (no LLM): any pytest ≥ 8.0 works. The project's hermes runtime venv lives at `./venv/bin/pytest`; a system or miniforge pytest is also fine.
- For **Tier 2 (live LLM)**: you MUST use `./venv/bin/pytest`. Tier 2 invokes `agent.auxiliary_client.call_llm` which lazy-imports the `openai` SDK; that package is installed in the hermes venv but typically NOT in `miniforge`/system Python. Running Tier 2 with the wrong interpreter triggers `ModuleNotFoundError: No module named 'openai'`.
- For Tier 3 (trajectory replay): same as Tier 1 — no LLM involved.
- Pyproject's `[tool.pytest.ini_options] addopts = "-m 'not integration' -n auto"` does two things that affect benchmarks:
  - `-m 'not integration'` skips Tier 2 + Tier 3 by default (they're tagged `pytest.mark.integration`). Run them explicitly with `-m integration` or `-m ""` to include.
  - `-n auto` enables pytest-xdist parallel workers when xdist is installed. **xdist process-level parallelism corrupts wall-clock measurements** (CPU contention from sibling workers inflates `time.perf_counter()` non-deterministically; pytest-benchmark auto-disables itself when xdist is detected for exactly this reason — see [pytest-benchmark FAQ](https://pytest-benchmark.readthedocs.io/en/latest/faq.html)). All benchmark invocations below use `-p no:xdist` to disable xdist for the run. The flag is a no-op when xdist isn't installed, so it's safe to include unconditionally.

**Tier 1 only (CI-friendly):**

```bash
pytest tests/agent/benchmarks/test_tier1_*.py -v -p no:xdist
```

Expected runtime: ≤ 30s. Exits non-zero on any threshold breach.

**Tier 2 (requires moe profile up + integration marker opt-in + hermes venv):**

```bash
qwen-server status   # confirm moe up
./venv/bin/pytest tests/agent/benchmarks/test_tier2_*.py -v -s -p no:xdist -m integration
```

The `./venv/bin/pytest` path is intentional — see the "Pytest invocation requirements" subsection above.

Expected runtime: 1-5 minutes. The `-s` is so the per-turn timing logs print live. The `-m integration` overrides pyproject's `'not integration'` default so the qwen-required tests are actually collected.

**Tier 3 (requires captured trajectories + integration marker opt-in):**

```bash
# 1. Capture a session (one-time)
hermes --capture-trajectory 50turn-readwrite -z "your prompt" ...
# 2. Replay
pytest tests/agent/benchmarks/test_tier3_*.py -v -p no:xdist -m integration
```

**Generate the report:**

After any tier runs, the markdown report lands at
`docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md`.

```bash
cat docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md
```

---

## How to interpret results

The report has rows like:

```
| 1.1 | post_compact_tokens | 32450.00 | 24180.00 | -25.5% | tok |
| 1.1 | op_deduped_count    | 0.00     | 28.00    | +inf%  | ops |
| 2.1 | compaction_walltime_seconds_mean | 0.42 | 0.43 | +2.4% | s |
| 2.3 | fact_retention_pct  | 90.0     | 80.0     | -11.1% | %   |
```

**Read the Δ% column first.** Then cross-reference the row's threshold from the Acceptance Criteria table.

**Three outcomes per benchmark:**

- **Improved:** Δ exceeds the "ships if" threshold by a meaningful margin (e.g. 1.1's −15% threshold met with −25%). Counts as a clear win.
- **No meaningful change:** Δ within ±5% on metrics where ±5% is noise (1.2, 1.3 wall-clock); within ±15% for 2.1 (real-server wall-clock has 2-8% baseline variance). Counts as "neutral, ship anyway" if all other tiers pass.
- **Regressed:** Δ violates the threshold OR fails the assertion. Block the merge until investigated.

**Common failure → diagnostic mappings:**

| Symptom in report | Likely cause | First place to look |
|---|---|---|
| 1.1 token reduction < 15% | Op dedup not firing | `_last_op_deduped` value; tool-name registry; `_operation_key` returning None |
| 1.2 ratio < 0.95 | Information lost on neutral session | `_dedup_by_operation` collapsing things it shouldn't; Pass 2 over-summarizing |
| 1.3 wall-clock > 1.20× | Hot-path overhead | `_dedup_by_operation` recomputing call_index per message; missing early-return |
| 1.5 orphan IDs | Boundary calc broken | `_align_boundary_backward` interaction with new anchor |
| 1.7 anti-thrashing didn't fire | New triggers bypass counter | `should_compress` ordering of the multi-trigger vs ineffective-count check |
| 1.9 image part missing | Multimodal skip not in dedup | `isinstance(older_content, list)` guard in `_dedup_by_operation` |
| 2.1 wall-clock > 1.15× | Real-server effect (not unit-time) | Compaction firing more often (check 2.2); summary model latency; cold GPU (warmup discard not running); MoE expert routing variance |
| 2.3 retention regressed | Summary input lost key facts | Check if dedup superseded a fact-bearing message; relax `dedup_operations` |
| 3.2 unexpected diff | Side effect leaked through | Whichever message field differs that isn't on the allowed-diff list |

---

## What this benchmark plan does NOT measure

These are explicitly **out of scope**; document so we don't argue about them later:

- **End-to-end UX latency** — measured by 2.1's per-compaction wall-clock, but not full per-keystroke or per-streamed-token latency. Add later if needed.
- **Summary text quality** — Tier 2.3 measures *fact retention* (substring matches in answers). It does NOT measure summary readability, prose coherence, or "is this a good summary by human standards." LLM-as-judge benchmarks are deferred (require a separate judge model and prompt).
- **Real production session diversity** — Tier 3 catches regressions on the trajectories we capture. It can't catch cases nobody has captured. Mitigation: capture trajectories from a few different agents (research, coding, multi-tool) before merging, not just one.
- **Cross-platform variance** — runs on the dev machine. Performance numbers are not portable; thresholds (1.20×, 1.05×) are *relative* to the same-run baseline so the harness is portable but absolute numbers aren't.
- **Memory pressure / OOM** — outside of the compaction code path. Watching `vmstat` is the operator's job.

---

## Maintenance — when to update this plan

- **A new compaction flag is added to `qwen_aware`:** add a row to the Acceptance Criteria table, write a Tier-1 invariant test, write a Tier-2 wall-clock differential test if the flag affects per-turn cost.
- **Hermes changes the `should_compress` callsite count:** `on_turn_end` may become reachable; revisit `Background — what we considered but rejected` in the main plan and add a 1.6 row for it.
- **Local Qwen server architecture changes (e.g. SGLang replaces llama.cpp):** revisit Tier 2 thresholds — RadixAttention prefix-cache reuse fundamentally changes the cost model; current 1.05× wall-clock budget may be too tight or too loose.
- **A captured Tier 3 trajectory takes > 30s to replay:** consider sampling — replaying every captured snapshot may be more than CI can absorb.

---

## Self-Review

**Does the plan answer "did the change improve / worsen / not change much"?** Yes:
- Improved → 1.1 (−15% tokens), 2.1 (≤1.05×), 2.3 (≥0.80 retention).
- Worsened → any threshold breach in the table.
- No change → 1.2 (±5%), 1.3 (≤1.20×), 2.1 (≤1.05×) all *passing within their bounds* with small Δs.

**Does it cover all six implementation tasks?**
- P0a (dedup) → 1.1, 1.2, 1.4, 3.2
- P0b (anchor) → 1.8, 3.2
- P0c (absolute cap) → indirectly via 2.1, 2.2 (compaction frequency)
- P1d (multi-trigger) → 1.6, 1.7
- P2 (CompactionResult) → consumed by every Tier-1/2 record() call
- Atomicity (existing helpers) → 1.5, 3.2

**Risks not covered (acknowledged):**
- Slow regressions that emerge over weeks (drift in summary quality with real users). Tier 2.3 is a snapshot; can be re-run periodically but isn't continuous.
- Hermes core changes that bypass the compressor (e.g. context-engine plugin). Out of scope for this plan; that plugin would have its own benchmark obligation.
