#!/usr/bin/env python3
"""
Loop-until-done execution script.

Iterative discovery loop with hard caps enforced by the SCRIPT, not the model.
Each round spawns a discovery agent, collects findings, evaluates a stop
condition via LLM call, and repeats or stops.

Hard caps: 10 rounds, 100K tokens, 2 hours. The model CANNOT override these.

Usage:
    python loop_until_done.py \\
      --task "Investigate X" \\
      --stop-condition "no new findings for 2 rounds" \\
      --max-rounds 10 \\
      --max-tokens 100000 \\
      --max-time 7200
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────
STATE_DIR = Path.home() / ".hermes" / "state" / "loop_until_done"
DEFAULT_MAX_ROUNDS = 10
DEFAULT_MAX_TOKENS = 100_000
DEFAULT_MAX_TIME = 7_200  # 2 hours
DEFAULT_STOP_CONDITION = "no new findings for 2 rounds"


def hash_task(task: str) -> str:
    """Short hash of the task string for stable state file naming."""
    return hashlib.sha256(task.encode()).hexdigest()[:12]


def spawn_agent(goal: str) -> dict:
    """Spawn a Hermes agent for one discovery round. Returns parsed JSON result."""
    try:
        result = subprocess.run(
            [
                "hermes", "run",
                "--prompt", goal,
                "--no-interactive",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min per round
        )
        output = result.stdout.strip()
        # Try to parse as JSON; fall back to raw text
        if output.startswith("{"):
            return json.loads(output)
        return {"raw_output": output, "parsed": False}
    except subprocess.TimeoutExpired:
        return {"error": "Agent timed out after 300s", "parsed": False}
    except json.JSONDecodeError:
        return {"raw_output": result.stdout.strip() if result else "", "parse_error": True, "parsed": False}
    except Exception as exc:
        return {"error": str(exc), "parsed": False}


def evaluate_stop_condition(findings: list, condition: str) -> dict:
    """Evaluate whether the stop condition has been met via LLM call."""
    goal = (
        "Evaluate whether the following stop condition has been met "
        "for this discovery loop.\n\n"
        f"STOP CONDITION: {condition}\n\n"
        f"FINDINGS SO FAR ({len(findings)} rounds):\n"
    )
    for f in findings[-5:]:  # last 5 rounds only for context
        goal += f"  Round {f['round']}: {f.get('finding', f.get('raw_output', ''))[:200]}\n"

    goal += """
Respond with JSON only:
{
  "condition_met": true|false,
  "confidence": 0.0-1.0,
  "reasoning": "why you reached this conclusion"
}
"""
    return spawn_agent(goal)


def main():
    parser = argparse.ArgumentParser(description="Loop-until-done discovery agent")
    parser.add_argument("--task", required=True, help="The investigation task")
    parser.add_argument("--stop-condition", default=DEFAULT_STOP_CONDITION)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME)

    args = parser.parse_args()

    # ── Init state ──
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    task_hash = hash_task(args.task)
    state_file = STATE_DIR / f"{task_hash}.json"

    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"Resuming from round {state.get('total_rounds', 0) + 1}")
    else:
        state = {
            "task": args.task,
            "stop_condition": args.stop_condition,
            "findings": [],
            "total_rounds": 0,
            "total_tokens": 0,
            "started_at": time.time(),
            "aborted": False,
        }

    start_time = state["started_at"]

    # ── Main loop ──
    for round_num in range(state["total_rounds"] + 1, args.max_rounds + 1):
        # Hard cap checks
        elapsed = time.time() - start_time
        if elapsed > args.max_time:
            print(f"TIME CAP REACHED ({elapsed:.0f}s > {args.max_time}s). Stopping.")
            state["aborted"] = True
            state["abort_reason"] = "time_cap"
            break

        if state["total_tokens"] >= args.max_tokens:
            print(f"TOKEN CAP REACHED ({state['total_tokens']} >= {args.max_tokens}). Stopping.")
            state["aborted"] = True
            state["abort_reason"] = "token_cap"
            break

        print(f"\n═══ ROUND {round_num}/{args.max_rounds} ═══")

        # Build discovery prompt
        if round_num == 1:
            goal = f"Investigate the following and report your findings:\n\n{args.task}"
        else:
            prev = "\n".join(
                f"Round {f['round']}: {f.get('finding', f.get('raw_output', ''))[:300]}"
                for f in state["findings"][-3:]
            )
            goal = (
                f"Previous investigation rounds found:\n{prev}\n\n"
                f"Continue investigating: {args.task}\n"
                f"Focus on NEW findings not already covered. "
                f"If nothing new to report, state 'NO NEW FINDINGS' clearly."
            )

        finding = spawn_agent(goal)
        finding["round"] = round_num
        finding["timestamp"] = time.time()
        state["findings"].append(finding)
        state["total_rounds"] = round_num

        # Crude token estimate (4 chars ≈ 1 token)
        round_tokens = len(json.dumps(finding)) // 4
        state["total_tokens"] += round_tokens

        # Evaluate stop condition
        eval_result = evaluate_stop_condition(state["findings"], args.stop_condition)
        condition_met = eval_result.get("condition_met", False)

        # Save state to temp file
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        if condition_met:
            print(f"STOP CONDITION MET after round {round_num}.")
            break

    # ── Final output ──
    state["duration_seconds"] = round(time.time() - start_time, 1)
    state["stop_condition_met"] = not state.get("aborted", False)

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)

    print(f"\nDone. {state['total_rounds']} rounds, "
          f"{state['total_tokens']} tokens, "
          f"{state['duration_seconds']}s")
    print(f"State file: {state_file}")

    if state.get("aborted"):
        sys.exit(1)


if __name__ == "__main__":
    main()
