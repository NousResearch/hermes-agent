#!/usr/bin/env python3
"""Controlled benchmark for guarded background compression (task 10).

Compares the four operating modes of the plan on REAL persistence machinery
(AIAgent + SessionDB + apply_prepared_candidate) with a deterministic
simulated summariser (a sleep standing in for the LLM call — no provider
traffic, no cost, reproducible):

  1. sync        — current behavior: the turn blocks for summarise + commit;
  2. shadow      — background candidate generated/validated, never applied:
                   the compaction pause is unchanged (sync fallback), but
                   foreground-turn latency during preparation is measured;
  3. background  — candidate pre-computed between turns; the perceived pause
                   is only the atomic apply (validate + merge + commit);
  4. fallback    — background worker fails; the synchronous path runs as if
                   the feature did not exist.

Approval gates (from the plan):
  - candidate apply p95 below 500 ms;
  - >= 90% reduction of the perceived compaction pause vs sync;
  - foreground-turn p95 during preparation within +10% of baseline;
  - candidate ready before the apply boundary in >= 95% of runs;
  - zero unexpected foreground errors ("provider" health);
  - zero message loss / duplication.

The apply boundary is placed after the preparation window has had time to
finish, mirroring the real threshold gap: preparation starts at 65% of the
context window and applies at 82% — tens of thousands of tokens (minutes of
conversation) later, while a real summarise takes ~40 s. The simulated ratio
is conservative relative to production.

Usage:
  python scripts/benchmark_async_compression.py
  python scripts/benchmark_async_compression.py --iterations 30 --summariser-ms 400
  python scripts/benchmark_async_compression.py --json
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUMMARY_MARKER = "[CONTEXT COMPACTION]"


def _p(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _make_messages(n_turns: int = 24) -> List[Dict[str, Any]]:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"benchmark message {i} " * 8})
        if i % 6 == 5:
            call_id = f"call_{i}"
            msgs.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": call_id, "type": "function",
                                "function": {"name": "read_file",
                                             "arguments": "{}"}}],
            })
            msgs.append({"role": "tool", "tool_call_id": call_id,
                         "name": "read_file", "content": f"tool output {i} " * 20})
    return msgs


def _make_candidate(messages, session_id, *, generation=1):
    from agent.async_context_compression import (
        PreparedCompressionCandidate,
        align_prefix_boundary,
        canonical_prefix_digest,
    )
    prefix_count = align_prefix_boundary(messages, len(messages) - 5)
    return PreparedCompressionCandidate(
        session_id=session_id,
        generation=generation,
        prefix_message_count=prefix_count,
        prefix_digest=canonical_prefix_digest(messages, prefix_count),
        prepared_messages=(
            {"role": "user", "content": f"{SUMMARY_MARKER} benchmark summary"},
            copy.deepcopy(messages[prefix_count - 1]),
        ),
        source_prompt_tokens=200_000,
        created_at_monotonic=time.monotonic(),
        created_at_turn=1,
        used_fallback=False,
        summary_error=None,
    )


def _foreground_turn(messages, turn_work_ms: float) -> float:
    """One simulated foreground turn: a little CPU + provider-ish wait."""
    t0 = time.monotonic()
    json.dumps(messages, default=repr)  # request-assembly-ish CPU work
    time.sleep(turn_work_ms / 1000.0)
    return (time.monotonic() - t0) * 1000.0


class _Bench:
    def __init__(self, iterations: int, summariser_ms: float,
                 turn_work_ms: float):
        import os
        os.environ.setdefault("OPENROUTER_API_KEY", "benchmark-offline-key")
        self.iterations = iterations
        self.summariser_ms = summariser_ms
        self.turn_work_ms = turn_work_ms
        self.errors: List[str] = []
        self.loss_or_dup = 0

    # -- per-iteration session/agent -------------------------------------

    def _fresh(self, db, tag: str, i: int):
        from run_agent import AIAgent
        sid = f"bench-{tag}-{i}"
        db.create_session(sid, "cli", model="test/model")
        agent = AIAgent(
            api_key="benchmark-offline-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=sid,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._compression_feasibility_checked = True
        agent.compression_in_place = True
        messages = _make_messages()
        agent._flush_messages_to_session_db(messages, [])
        return agent, sid, messages

    def _check_merge(self, cand, messages, new_messages) -> None:
        prepared_n = len(cand.prepared_messages)
        live_suffix = messages[cand.prefix_message_count:]
        got = new_messages[prepared_n:prepared_n + len(live_suffix)]
        ids = [id(m) for m in got]
        if (len(got) != len(live_suffix)
                or any(a is not b for a, b in zip(got, live_suffix))
                or len(ids) != len(set(ids))):
            self.loss_or_dup += 1

    # -- modes ------------------------------------------------------------

    def run_sync(self, db) -> Dict[str, List[float]]:
        """Current path: summarise (blocking) + commit, all in the turn."""
        from agent.conversation_compression import apply_prepared_candidate
        pauses = []
        for i in range(self.iterations):
            agent, sid, messages = self._fresh(db, "sync", i)
            try:
                t0 = time.monotonic()
                time.sleep(self.summariser_ms / 1000.0)  # blocking summarise
                cand = _make_candidate(messages, sid)
                result = apply_prepared_candidate(agent, cand, messages, "sys")
                pauses.append((time.monotonic() - t0) * 1000.0)
                if result is None:
                    self.errors.append(f"sync[{i}]: commit failed")
                else:
                    self._check_merge(cand, messages, result[0])
            finally:
                agent.close()
        return {"pause_ms": pauses}

    def run_background(self, db) -> Dict[str, Any]:
        """Prepared between turns; the pause is only the atomic apply."""
        from agent.async_context_compression import (
            BackgroundCompressionConfig,
            BackgroundCompressionController,
            CandidateState,
        )
        from agent.conversation_compression import apply_prepared_candidate
        pauses, turn_ms, ready_flags = [], [], []
        for i in range(self.iterations):
            agent, sid, messages = self._fresh(db, "bg", i)
            ctl = BackgroundCompressionController(
                BackgroundCompressionConfig.from_dict(
                    {"enabled": True, "shadow_only": False,
                     "min_frozen_messages": 2, "min_delta_tokens": 0}
                )
            )
            try:
                def prepare_fn(prefix):
                    time.sleep(self.summariser_ms / 1000.0)
                    return [
                        {"role": "user",
                         "content": f"{SUMMARY_MARKER} benchmark summary"},
                        copy.deepcopy(prefix[-1]),
                    ]

                assert ctl.try_start_preparation(
                    session_id=sid, messages=messages,
                    prefix_count=len(messages) - 5, current_turn=1,
                    source_prompt_tokens=200_000, prepare_fn=prepare_fn,
                )
                # Foreground keeps answering while the worker summarises —
                # the 65%→82% threshold gap gives ample conversation time.
                deadline = time.monotonic() + (self.summariser_ms / 1000.0) * 3
                while (ctl.state is CandidateState.PREPARING
                       and time.monotonic() < deadline):
                    turn_ms.append(_foreground_turn(messages, self.turn_work_ms))
                ready_flags.append(ctl.state is CandidateState.READY)

                # Apply boundary: perceived pause = validate + merge + commit.
                t0 = time.monotonic()
                cand = ctl.take_valid_candidate(
                    session_id=sid, messages=messages, current_turn=3
                )
                result = (
                    apply_prepared_candidate(
                        agent, cand, messages, "sys", controller=ctl
                    ) if cand is not None else None
                )
                pauses.append((time.monotonic() - t0) * 1000.0)
                if result is None:
                    self.errors.append(f"background[{i}]: apply failed")
                else:
                    self._check_merge(cand, messages, result[0])
            finally:
                ctl.shutdown(wait=True)
                agent.close()
        return {"pause_ms": pauses, "turn_ms": turn_ms,
                "ready_rate": (sum(ready_flags) / len(ready_flags))
                if ready_flags else 0.0}

    def run_shadow(self, db) -> Dict[str, Any]:
        """Candidate generated + validated, never applied; sync still pays."""
        from agent.async_context_compression import (
            BackgroundCompressionConfig,
            BackgroundCompressionController,
            CandidateState,
        )
        from agent.conversation_compression import apply_prepared_candidate
        pauses, turn_ms = [], []
        for i in range(self.iterations):
            agent, sid, messages = self._fresh(db, "shadow", i)
            ctl = BackgroundCompressionController(
                BackgroundCompressionConfig.from_dict(
                    {"enabled": True, "shadow_only": True,
                     "min_frozen_messages": 2, "min_delta_tokens": 0}
                )
            )
            try:
                def prepare_fn(prefix):
                    time.sleep(self.summariser_ms / 1000.0)
                    return [
                        {"role": "user",
                         "content": f"{SUMMARY_MARKER} shadow summary"},
                        copy.deepcopy(prefix[-1]),
                    ]

                assert ctl.try_start_preparation(
                    session_id=sid, messages=messages,
                    prefix_count=len(messages) - 5, current_turn=1,
                    source_prompt_tokens=200_000, prepare_fn=prepare_fn,
                )
                deadline = time.monotonic() + (self.summariser_ms / 1000.0) * 3
                while (ctl.state is CandidateState.PREPARING
                       and time.monotonic() < deadline):
                    turn_ms.append(_foreground_turn(messages, self.turn_work_ms))
                cand = ctl.take_valid_candidate(
                    session_id=sid, messages=messages, current_turn=3
                )
                if cand is not None:
                    ctl.record_shadow_validation(cand)
                # Shadow never applies: the compaction pause is the sync one.
                t0 = time.monotonic()
                time.sleep(self.summariser_ms / 1000.0)
                sync_cand = _make_candidate(messages, sid, generation=2)
                result = apply_prepared_candidate(
                    agent, sync_cand, messages, "sys"
                )
                pauses.append((time.monotonic() - t0) * 1000.0)
                if result is None:
                    self.errors.append(f"shadow[{i}]: sync commit failed")
            finally:
                ctl.shutdown(wait=True)
                agent.close()
        return {"pause_ms": pauses, "turn_ms": turn_ms}

    def run_fallback(self, db) -> Dict[str, List[float]]:
        """Worker fails → foreground never notices; sync path unchanged."""
        from agent.async_context_compression import (
            BackgroundCompressionConfig,
            BackgroundCompressionController,
        )
        from agent.conversation_compression import apply_prepared_candidate
        pauses = []
        for i in range(self.iterations):
            agent, sid, messages = self._fresh(db, "fb", i)
            ctl = BackgroundCompressionController(
                BackgroundCompressionConfig.from_dict(
                    {"enabled": True, "shadow_only": False,
                     "min_frozen_messages": 2, "min_delta_tokens": 0}
                )
            )
            try:
                def broken(prefix):
                    raise RuntimeError("summariser unavailable (benchmark)")

                assert ctl.try_start_preparation(
                    session_id=sid, messages=messages,
                    prefix_count=len(messages) - 5, current_turn=1,
                    source_prompt_tokens=200_000, prepare_fn=broken,
                )
                assert ctl.wait_until_settled(timeout=10.0)
                if ctl.take_valid_candidate(
                    session_id=sid, messages=messages, current_turn=2
                ) is not None:
                    self.errors.append(f"fallback[{i}]: failed worker "
                                       "produced a candidate")
                # Sync fallback pays the normal pause.
                t0 = time.monotonic()
                time.sleep(self.summariser_ms / 1000.0)
                cand = _make_candidate(messages, sid, generation=99)
                result = apply_prepared_candidate(agent, cand, messages, "sys")
                pauses.append((time.monotonic() - t0) * 1000.0)
                if result is None:
                    self.errors.append(f"fallback[{i}]: sync commit failed")
            finally:
                ctl.shutdown(wait=True)
                agent.close()
        return {"pause_ms": pauses}


def run_benchmark(iterations: int = 10, summariser_ms: float = 2000.0,
                  turn_work_ms: float = 20.0) -> Dict[str, Any]:
    from hermes_state import SessionDB

    bench = _Bench(iterations, summariser_ms, turn_work_ms)
    with tempfile.TemporaryDirectory(prefix="hermes-bench-") as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        # Baseline foreground-turn latency with NO background worker running.
        baseline_turns = [
            _foreground_turn(_make_messages(), turn_work_ms)
            for _ in range(max(10, iterations))
        ]
        sync = bench.run_sync(db)
        shadow = bench.run_shadow(db)
        background = bench.run_background(db)
        fallback = bench.run_fallback(db)
        db.close()

    sync_p95 = _p(sync["pause_ms"], 0.95)
    bg_p95 = _p(background["pause_ms"], 0.95)
    turn_baseline_p95 = _p(baseline_turns, 0.95)
    turn_prepare_p95 = _p(
        background["turn_ms"] + shadow["turn_ms"], 0.95
    )
    pause_reduction = 1.0 - (bg_p95 / sync_p95) if sync_p95 else 0.0

    report = {
        "params": {"iterations": iterations, "summariser_ms": summariser_ms,
                   "turn_work_ms": turn_work_ms},
        "modes": {
            "sync": {"pause_p50_ms": _p(sync["pause_ms"], 0.50),
                     "pause_p95_ms": sync_p95},
            "shadow": {"pause_p50_ms": _p(shadow["pause_ms"], 0.50),
                       "pause_p95_ms": _p(shadow["pause_ms"], 0.95)},
            "background": {"pause_p50_ms": _p(background["pause_ms"], 0.50),
                           "pause_p95_ms": bg_p95,
                           "ready_rate": background["ready_rate"]},
            "fallback": {"pause_p50_ms": _p(fallback["pause_ms"], 0.50),
                         "pause_p95_ms": _p(fallback["pause_ms"], 0.95)},
        },
        "foreground_turns": {
            "baseline_p95_ms": turn_baseline_p95,
            "during_prepare_p95_ms": turn_prepare_p95,
        },
        "gates": {},
        "errors": bench.errors,
        "loss_or_dup": bench.loss_or_dup,
    }
    turn_ratio = (
        turn_prepare_p95 / turn_baseline_p95 if turn_baseline_p95 else 1.0
    )
    report["gates"] = {
        "apply_p95_under_500ms": bg_p95 < 500.0,
        "pause_reduction_at_least_90pct": pause_reduction >= 0.90,
        "foreground_p95_within_10pct": turn_ratio <= 1.10,
        "candidate_ready_rate_at_least_95pct":
            background["ready_rate"] >= 0.95,
        "no_provider_errors": not bench.errors,
        "zero_loss_or_duplication": bench.loss_or_dup == 0,
    }
    report["pause_reduction"] = pause_reduction
    report["foreground_turn_ratio"] = turn_ratio
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--summariser-ms", type=float, default=2000.0,
                        help="simulated summariser latency. The real pause is "
                             "~40s (41.42s observed upstream); the 2s default "
                             "is a 20x-scaled-down CONSERVATIVE stand-in — "
                             "the measured reduction understates production.")
    parser.add_argument("--turn-work-ms", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_benchmark(args.iterations, args.summariser_ms,
                           args.turn_work_ms)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        m = report["modes"]
        print(f"iterations={args.iterations} summariser={args.summariser_ms}ms")
        print(f"  sync       pause p50={m['sync']['pause_p50_ms']:8.1f}ms "
              f"p95={m['sync']['pause_p95_ms']:8.1f}ms")
        print(f"  shadow     pause p50={m['shadow']['pause_p50_ms']:8.1f}ms "
              f"p95={m['shadow']['pause_p95_ms']:8.1f}ms  (unchanged by design)")
        print(f"  background pause p50={m['background']['pause_p50_ms']:8.1f}ms "
              f"p95={m['background']['pause_p95_ms']:8.1f}ms  "
              f"ready_rate={m['background']['ready_rate']:.0%}")
        print(f"  fallback   pause p50={m['fallback']['pause_p50_ms']:8.1f}ms "
              f"p95={m['fallback']['pause_p95_ms']:8.1f}ms")
        ft = report["foreground_turns"]
        print(f"  foreground turn p95: baseline={ft['baseline_p95_ms']:.1f}ms "
              f"during_prepare={ft['during_prepare_p95_ms']:.1f}ms "
              f"(ratio {report['foreground_turn_ratio']:.2f})")
        print(f"  perceived-pause reduction: {report['pause_reduction']:.1%}")
        print("  gates:")
        for gate, ok in report["gates"].items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {gate}")
        if report["errors"]:
            print(f"  errors: {report['errors']}")
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
