#!/usr/bin/env python3
"""A/B: baseline Hermes (Caduceus OFF) vs our system (Caduceus ON).

Each (scenario, arm) runs in its OWN subprocess for clean state and so the
agent's stdout noise never corrupts the structured result. A run emits one JSON
line prefixed `@@RESULT@@`. The orchestrator spawns all combos and prints a
side-by-side comparison of the Devin-discipline rubric PLUS model-agnostic
outcome signals (plan made? verified? artifacts produced? tool cost).

Modes:
  (orchestrate) python3 ab_compare.py --scenarios complex,trivial --max-iters 30
  (one arm)     python3 ab_compare.py --run --scenario complex --arm on --dir /sandbox/x
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import parity_eval as pe  # noqa: E402

# Bounded, self-contained tasks. "complex" deliberately does NOT say "run until
# green" (that blows the iteration budget); whether the agent VERIFIES anyway is
# exactly the Caduceus discriminator.
AB_SCENARIOS = {
    "complex": pe.Scenario(
        "live/complex", "complex",
        "In the current directory create a Python module `calc.py` with `add(a, b)` "
        "and `divide(a, b)` (raise ValueError on divide-by-zero), and write pytest "
        "tests for both in `test_calc.py`."),
    "multi": pe.Scenario(
        "live/multi", "multi",
        "In the current directory do two things: create `hello.py` that prints "
        "'hello', and create a `README.md` with a one-line description."),
    "trivial": pe.Scenario(
        "live/trivial", "trivial",
        "What does the `git status` command do? Answer in one sentence."),
}
ARTIFACTS = {"complex": ["calc.py", "test_calc.py"], "multi": ["hello.py", "README.md"], "trivial": []}


def _tool_seq(messages):
    seq = []
    for m in messages or []:
        if isinstance(m, dict):
            for tc in (m.get("tool_calls") or []):
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                if fn.get("name"):
                    seq.append(fn["name"])
    return seq


def _verified_after_write(seq):
    writes = [i for i, n in enumerate(seq) if n in ("write_file", "patch")]
    if not writes:
        return False
    return any(n in ("terminal", "execute_code") for n in seq[writes[-1] + 1:])


def do_run(scenario_key: str, arm_on: bool, scene_dir: str, max_iters: int) -> dict:
    """Run ONE arm in this process and return the metrics dict."""
    os.makedirs(scene_dir, exist_ok=True)
    os.chdir(scene_dir)
    from run_agent import AIAgent  # type: ignore
    model = os.environ.get("HERMES_AB_MODEL", "grok-4.3")
    provider = os.environ.get("HERMES_AB_PROVIDER", "xai-oauth")
    base_url = os.environ.get("HERMES_AB_BASE_URL", "https://api.x.ai/v1")
    t0 = time.time()
    agent = AIAgent(model=model, provider=provider, base_url=base_url, quiet_mode=True,
                    skip_memory=True, skip_context_files=True, max_iterations=max_iters)
    st = getattr(agent, "caduceus", None)
    if st is not None:
        st.activate() if arm_on else st.deactivate()
        agent._cached_system_prompt = None
    sc = AB_SCENARIOS[scenario_key]
    result = agent.run_conversation(sc.prompt)
    msgs = result.get("messages") if isinstance(result, dict) else []
    final = result.get("final_response", "") if isinstance(result, dict) else ""
    seq = _tool_seq(msgs)
    snaps = pe._extract_snapshots(msgs)
    ev = pe.Scenario(sc.name, sc.kind, sc.prompt, snapshots=snaps)
    ok, results = pe.evaluate(ev)
    names = ARTIFACTS.get(scenario_key, [])
    have = sum(1 for n in names if os.path.exists(os.path.join(scene_dir, n)))
    return {
        "scenario": scenario_key, "arm": "on" if arm_on else "off",
        "rubric_ok": ok,
        "rubric_fail": [r.rule for r in results if r.passed is False],
        "made_plan": len(snaps) > 0, "todo_writes": len(snaps),
        "verified": _verified_after_write(seq),
        "artifacts": f"{have}/{len(names)}" if names else "n/a",
        "tool_calls": len(seq), "tool_seq": seq,
        "final_len": len(final or ""), "elapsed": round(time.time() - t0, 1),
    }


def fmt(a: dict) -> str:
    rub = "PASS" if a["rubric_ok"] else "FAIL(" + ",".join(a["rubric_fail"]) + ")"
    return (f"plan={'Y' if a['made_plan'] else 'N'} writes={a['todo_writes']} "
            f"verified={'Y' if a['verified'] else 'N'} artifacts={a['artifacts']} "
            f"tools={a['tool_calls']} {a['elapsed']}s | rubric={rub}")


def orchestrate(scenarios, max_iters):
    import tempfile
    sandbox = tempfile.mkdtemp(prefix="caduceus_ab_")
    print("=== A/B: baseline Hermes (OFF) vs our system (ON) ===")
    print(f"model={os.environ.get('HERMES_AB_MODEL','grok-4.3')} sandbox={sandbox}\n")
    rows = {}
    for key in scenarios:
        print(f"#### {key}: {AB_SCENARIOS[key].prompt[:80]}...")
        for arm in ("off", "on"):
            d = os.path.join(sandbox, f"{key}__{arm}")
            os.makedirs(d, exist_ok=True)
            cmd = [sys.executable, os.path.abspath(__file__), "--run",
                   "--scenario", key, "--arm", arm, "--dir", d, "--max-iters", str(max_iters)]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=420,
                                   env={**os.environ})
                line = [l for l in p.stdout.splitlines() if l.startswith("@@RESULT@@")]
                if not line:
                    print(f"  {arm.upper():3}: ERROR (no result; stderr tail: {p.stderr.strip()[-160:]})")
                    continue
                a = json.loads(line[-1][len("@@RESULT@@"):])
                rows[(key, arm)] = a
                print(f"  {arm.upper():3}: {fmt(a)}")
            except subprocess.TimeoutExpired:
                print(f"  {arm.upper():3}: TIMEOUT")
        print()
    print(f"(sandbox: {sandbox})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--scenario")
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--dir")
    ap.add_argument("--scenarios", default="complex,trivial")
    ap.add_argument("--max-iters", type=int, default=30)
    args = ap.parse_args()
    if args.run:
        _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        sys.path.insert(0, os.environ.get("HERMES_AGENT_DIR", _repo))
        res = do_run(args.scenario, args.arm == "on", args.dir, args.max_iters)
        print("@@RESULT@@" + json.dumps(res))
        return 0
    return orchestrate([s.strip() for s in args.scenarios.split(",")], args.max_iters)


if __name__ == "__main__":
    raise SystemExit(main())
