#!/usr/bin/env python
"""Phase-4 shakedown for the token-budgeted fresh tail (offline, no LLM).

Runs the REAL LCMEngine.compress() path against a mixed-density corpus in a
throwaway HERMES_HOME. Without a summary model the escalation ladder lands on
L3 deterministic truncation, so compaction is real (leaves form, tail is cut)
and free.

Arms:
  OFF — fresh_tail_token_budget_enabled=False (legacy fixed-count tail)
  ON  — enabled, budget derived from target_ratio × threshold

Gates (spec v0.4 AC-2/AC-9):
  - ON-arm first compaction keeps a tail > fresh_tail_count messages whose
    token mass ≈ budget
  - ON arm fires ≥2 compactions
  - cadence_ratio = OFF_interval_tokens / ON_interval_tokens ≤ 2.0
    (interval = mean ingested tokens between consecutive fires)

Usage:
  HERMES_HOME=/tmp/lcm-tail-shakedown PYTHONPATH=$PWD \
    venv/bin/python scripts/lcm_tail_budget_shakedown.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plugins.context_engine.lcm.config import LCMConfig  # noqa: E402
from plugins.context_engine.lcm.engine import LCMEngine  # noqa: E402
from plugins.context_engine.lcm.tokens import (  # noqa: E402
    count_message_tokens,
    count_messages_tokens,
)

CONTEXT_LENGTH = 200_000
CONTEXT_THRESHOLD = 0.75
TARGET_RATIO = 0.25
EXPECTED_BUDGET = int(TARGET_RATIO * CONTEXT_THRESHOLD * CONTEXT_LENGTH)  # 37,500


def _mixed_turn(i: int) -> list[dict]:
    """One synthetic 'turn': assistant row (~300 chars) + tool row (~1,400).

    Matches the measured live density ratio (tool rows ~4.5× assistant rows).
    """
    return [
        {"role": "assistant", "content": f"turn{i:05d} " + ("a" * 300)},
        {"role": "tool", "content": f"result{i:05d} " + ("t" * 1400)},
    ]


def run_arm(enabled: bool, home: Path, max_turns: int = 3000) -> dict:
    home.mkdir(parents=True, exist_ok=True)
    cfg = LCMConfig()
    cfg.fresh_tail_count = 32
    cfg.leaf_chunk_tokens = 20_000  # default DAG knobs
    cfg.context_threshold = CONTEXT_THRESHOLD
    cfg.target_ratio = TARGET_RATIO
    cfg.fresh_tail_max_tokens = 60_000
    cfg.fresh_tail_token_budget_enabled = enabled
    cfg.database_path = str(home / f"lcm-{'on' if enabled else 'off'}.db")

    eng = LCMEngine(config=cfg, hermes_home=str(home))
    eng.update_model(model="shakedown-model", context_length=CONTEXT_LENGTH)
    eng.on_session_start(f"shakedown-{'on' if enabled else 'off'}", platform="cli")

    msgs: list[dict] = [{"role": "system", "content": "shakedown system prompt"}]
    fires: list[dict] = []
    ingested_tokens = 0
    first_fire_tail: dict | None = None

    for i in range(max_turns):
        turn = _mixed_turn(i)
        msgs.extend(turn)
        ingested_tokens += count_messages_tokens(turn)
        est = count_messages_tokens(msgs)
        if not eng.should_compress(est):
            continue

        pre_len = len(msgs)
        out = eng.compress(msgs, current_tokens=est)
        status = eng._last_compression_status
        if status == "compacted":
            kept_tail_count = eng._last_fresh_tail_count
            fires.append(
                {
                    "turn": i,
                    "ingested_tokens_at_fire": ingested_tokens,
                    "pre_len": pre_len,
                    "post_len": len(out),
                    "kept_tail_count": kept_tail_count,
                    "protect_last_n": eng.protect_last_n,
                }
            )
            if first_fire_tail is None:
                tail = msgs[max(0, pre_len - kept_tail_count):]
                first_fire_tail = {
                    "count": kept_tail_count,
                    "token_mass": count_messages_tokens(tail),
                    "budget": eng._fresh_tail_token_budget,
                }
            msgs = out
            if len(fires) >= 3:
                break

    intervals = [
        fires[j + 1]["ingested_tokens_at_fire"] - fires[j]["ingested_tokens_at_fire"]
        for j in range(len(fires) - 1)
    ]
    return {
        "arm": "ON" if enabled else "OFF",
        "budget": eng._fresh_tail_token_budget,
        "fires": fires,
        "fire_count": len(fires),
        "mean_interval_tokens": (sum(intervals) / len(intervals)) if intervals else None,
        "first_fire_tail": first_fire_tail,
    }


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="lcm-tail-shakedown-"))
    os.environ["HERMES_HOME"] = str(scratch)
    print(f"scratch HERMES_HOME: {scratch}")

    try:
        off = run_arm(enabled=False, home=scratch / "off")
        on = run_arm(enabled=True, home=scratch / "on")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    report = {"off": off, "on": on, "expected_budget": EXPECTED_BUDGET, "gates": {}}
    g = report["gates"]

    # AC-1: derived budget
    g["budget_resolves"] = on["budget"] == EXPECTED_BUDGET

    # AC-2: dynamic tail wider than legacy, token mass ≈ budget
    fft = on["first_fire_tail"] or {}
    g["tail_wider_than_legacy"] = bool(fft) and fft["count"] > 32
    tolerance = 1_600  # one max-size mixed message
    g["tail_mass_near_budget"] = (
        bool(fft) and abs(fft["token_mass"] - fft["budget"]) <= tolerance
    )

    # Legacy arm sanity: kept tail == 32
    g["legacy_tail_is_32"] = all(f["kept_tail_count"] == 32 for f in off["fires"])

    # AC-9: >=2 ON fires; cadence_ratio = OFF/ON interval <= 2.0, >= 1.0 expected
    g["on_fires_ge_2"] = on["fire_count"] >= 2
    if on["mean_interval_tokens"] and off["mean_interval_tokens"]:
        ratio = off["mean_interval_tokens"] / on["mean_interval_tokens"]
        report["cadence_ratio"] = round(ratio, 3)
        g["cadence_ratio_le_2"] = ratio <= 2.0
        g["cadence_direction_expected"] = ratio >= 1.0
    else:
        g["cadence_ratio_le_2"] = False
        g["cadence_direction_expected"] = False

    print(json.dumps(report, indent=2))
    ok = all(g.values())
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} ({sum(g.values())}/{len(g)} gates)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
