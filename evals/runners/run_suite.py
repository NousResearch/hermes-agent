#!/usr/bin/env python3
"""Eval suite runner for Hermes Agent capability scoring.

Loads a suite YAML, runs each scenario against AIAgent, scores with rubric,
and outputs a JSON report.

Usage:
    python evals/runners/run_suite.py --suite orchestration [--provider openrouter] [--output reports/latest.json]
    python evals/runners/run_suite.py --suite cost_cache --deterministic-only
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_WORKTREE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_WORKTREE))
_EVALS_DIR = _WORKTREE / "evals"


def load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"ERROR loading {path}: {e}", file=sys.stderr)
        return {}


def load_rubric(suite_name: str):
    """Dynamically import a rubric module from evals/rubrics/<suite_name>.py."""
    rubric_path = _EVALS_DIR / "rubrics" / f"{suite_name}.py"
    if not rubric_path.exists():
        print(f"WARNING: No rubric at {rubric_path} — all scenarios will pass by default", file=sys.stderr)
        return None
    spec = importlib.util.spec_from_file_location(f"rubric_{suite_name}", rubric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_flat_overrides(overrides: dict) -> dict:
    """Convert dotted config_overrides keys into nested dicts.

    Suite YAML commonly uses flat keys like:
      delegation.max_concurrent_children: 3
      delegation.max_spawn_depth: 2

    Rubrics read nested maps (scenario['config_overrides']['delegation'][...]).
    Without this coercion, depth/cap checks silently fall back to defaults and
    orchestration scenarios produce false negatives.
    """
    if not isinstance(overrides, dict):
        return {}

    # Start from a deep copy so pre-nested structures are preserved.
    out = copy.deepcopy(overrides)

    for key, value in overrides.items():
        if not isinstance(key, str) or "." not in key:
            continue
        parts = [p for p in key.split(".") if p]
        if not parts:
            continue

        cur = out
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value

    return out


def _apply_config_overrides(overrides_raw: dict) -> Dict[str, Any]:
    """Temporarily apply scenario config_overrides to CLI_CONFIG.

    AIAgent and delegate_task load settings from runtime CLI_CONFIG first.
    The eval runner must inject suite overrides there so live runs actually use
    per-scenario caps (e.g. delegation.max_concurrent_children=3).

    Returns a snapshot token used by _restore_config_overrides().
    """
    try:
        from cli import CLI_CONFIG
    except Exception:
        return {}

    try:
        overrides = _coerce_flat_overrides(overrides_raw)
    except Exception:
        overrides = {}

    snapshot = {
        "had_delegation": "delegation" in CLI_CONFIG,
        "delegation": copy.deepcopy(CLI_CONFIG.get("delegation")),
        "had_agent": "agent" in CLI_CONFIG,
        "agent": copy.deepcopy(CLI_CONFIG.get("agent")),
    }

    # delegation block
    if isinstance(overrides.get("delegation"), dict):
        if not isinstance(CLI_CONFIG.get("delegation"), dict):
            CLI_CONFIG["delegation"] = {}
        CLI_CONFIG["delegation"].update(overrides["delegation"])

    # agent block
    if isinstance(overrides.get("agent"), dict):
        if not isinstance(CLI_CONFIG.get("agent"), dict):
            CLI_CONFIG["agent"] = {}
        CLI_CONFIG["agent"].update(overrides["agent"])

    return snapshot


def _restore_config_overrides(snapshot: Dict[str, Any]) -> None:
    """Restore CLI_CONFIG after a scenario run."""
    try:
        from cli import CLI_CONFIG
    except Exception:
        return

    if not isinstance(snapshot, dict):
        return

    if snapshot.get("had_delegation"):
        CLI_CONFIG["delegation"] = snapshot.get("delegation")
    else:
        CLI_CONFIG.pop("delegation", None)

    if snapshot.get("had_agent"):
        CLI_CONFIG["agent"] = snapshot.get("agent")
    else:
        CLI_CONFIG.pop("agent", None)


def run_scenario_live(scenario: dict, provider: str, model: str) -> dict:
    """Run a single scenario against a live AIAgent and return the result dict."""
    from run_agent import AIAgent

    config_overrides = scenario.get("config_overrides", {})
    overrides = _coerce_flat_overrides(config_overrides)

    enabled_toolsets = scenario.get("enabled_toolsets", ["terminal", "file", "delegation"])
    max_iterations = (
        overrides.get("agent", {}).get("max_iterations")
        or config_overrides.get("agent.max_iterations", 12)
    )
    skip_memory = scenario.get("skip_memory", True)

    skip_context = scenario.get("skip_context_files", True)
    system_msg = scenario.get("system_message", None)


    _cfg_snapshot = _apply_config_overrides(config_overrides)
    try:
        agent = AIAgent(
            provider=provider,
            model=model,
            enabled_toolsets=enabled_toolsets if isinstance(enabled_toolsets, list) else None,
            quiet_mode=True,
            save_trajectories=False,
            skip_context_files=skip_context,
            skip_memory=skip_memory,
            platform="cli",
            max_iterations=max_iterations,
        )

        try:
            result = agent.run_conversation(
                user_message=scenario["user_message"],
                system_message=system_msg,
            )
        except Exception as e:
            return {
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "final_response": "",
                "messages": [],
                "api_calls": 0,
            }

        if isinstance(result, dict):
            return {
                "final_response": result.get("final_response", "") or "",
                "messages": result.get("messages", []) or [],
                "api_calls": getattr(agent, "iteration_budget", None) and agent.iteration_budget.used or 0,
                "error": None,
            }
        return {
            "final_response": str(result),
            "messages": [],
            "api_calls": 0,
            "error": None,
        }
    finally:
        _restore_config_overrides(_cfg_snapshot)


def grade_scenario(scenario: dict, result: dict, rubric_module) -> dict:
    """Score a scenario using its rubric or pass_conditions."""
    if rubric_module and hasattr(rubric_module, "grade"):
        try:
            return rubric_module.grade(scenario, result)
        except Exception as e:
            return {"pass": False, "score": 0.0, "details": {"rubric_error": str(e)}}

    # Fallback: check pass_conditions directly
    conditions = scenario.get("pass_conditions", [])
    if not conditions:
        return {"pass": True, "score": 1.0, "details": {"note": "no conditions specified"}}

    checks_passed = 0
    details = {}
    for cond in conditions:
        ctype = cond.get("type", "")
        if ctype == "delegate_call_count":
            count = _count_delegate_calls(result.get("messages", []))
            min_val = cond.get("min", 1)
            details[f"delegate_calls"] = count
            if count >= min_val:
                checks_passed += 1
        elif ctype == "no_cache_break":
            breaks = _count_cache_breaks(result.get("messages", []))
            details["cache_breaks"] = breaks
            if breaks == 0:
                checks_passed += 1
        elif ctype == "response_contains":
            val = cond.get("value", "")
            if val.lower() in result.get("final_response", "").lower():
                checks_passed += 1
            details[f"contains_{val[:30]}"] = val.lower() in result.get("final_response", "").lower()
        elif ctype == "no_tool_error":
            has_error = _has_tool_error(result.get("messages", []))
            details["has_tool_error"] = has_error
            if not has_error:
                checks_passed += 1
        else:
            checks_passed += 1  # Unknown condition → pass by default

    score = checks_passed / len(conditions) if conditions else 1.0
    return {"pass": score >= 0.5, "score": score, "details": details}


def _count_delegate_calls(messages: list) -> int:
    """Count delegate_task tool calls in the message transcript."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") == "delegate_task":
                    count += 1
    return count


def _count_cache_breaks(messages: list) -> int:
    """Count evidence of prompt-cache breaks in the message transcript."""
    breaks = 0
    prev_system = None
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if prev_system is not None and content != prev_system:
                breaks += 1
            prev_system = content
    return breaks


def _has_tool_error(messages: list) -> bool:
    """Check if any tool result contains error indicators."""
    for msg in messages:
        if msg.get("role") == "tool":
            content = str(msg.get("content", ""))
            if "error" in content.lower() or "traceback" in content.lower():
                return True
    return False


def run_suite(
    suite_path: Path,
    provider: str = "openrouter",
    model: str = "anthropic/claude-haiku-4.5",
    output_path: Optional[Path] = None,
    deterministic_only: bool = False,
) -> dict:
    """Run a full eval suite and return the report dict."""
    suite = load_yaml(suite_path)
    suite_name = suite.get("name", suite_path.stem)
    scenarios = suite.get("scenarios", [])

    if not scenarios:
        print(f"WARNING: No scenarios found in {suite_path}", file=sys.stderr)
        return {"suite": suite_name, "error": "no scenarios", "total": 0, "passed": 0, "failed": 0}

    rubric = load_rubric(suite_name)
    results = []
    passed = 0
    failed = 0
    errored = 0
    skipped = 0

    for i, scenario in enumerate(scenarios):
        sid = scenario.get("id", f"S{i}")
        print(f"  [{i+1}/{len(scenarios)}] {sid}: {scenario.get('description', '')[:80]}", file=sys.stderr)

        t0 = time.time()

        # Explicit deterministic skip — scenario requires a live agent /
        # real filesystem / real HERMES_HOME. Counted as skipped, not fail.
        if deterministic_only and scenario.get("deterministic_skip"):
            reason = (
                scenario.get("deterministic_skip_reason")
                or scenario.get("deterministic_skip")
                or "requires live agent"
            )
            if scenario.get("deterministic_skip") is True:
                reason = scenario.get("deterministic_skip_reason") or "requires live agent"
            skipped += 1
            results.append({
                "id": sid,
                "pass": None,
                "score": None,
                "skipped": True,
                "details": {
                    "skipped": True,
                    "reason": str(reason),
                },
                "api_calls": 0,
                "duration_s": round(time.time() - t0, 2),
            })
            print(f"    ↷ skipped (deterministic): {reason}", file=sys.stderr)
            continue

        if deterministic_only:
            # Deterministic mode: grade structural invariants against embedded
            # mock transcripts (_mock_messages). No live API call.
            messages = scenario.get("_mock_messages", []) or []
            final_response = scenario.get("_mock_final_response")
            if final_response is None:
                # Prefer explicit mock; else last assistant content; else ""
                final_response = ""
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content")
                        if isinstance(content, str) and content.strip():
                            final_response = content
                            break
            result = {
                "final_response": final_response or "",
                "messages": messages,
                "api_calls": int(scenario.get("_mock_api_calls", 0) or 0),
                "api_call_snapshots": scenario.get("_mock_api_call_snapshots") or [],
                "error": None,
            }
        else:
            result = run_scenario_live(scenario, provider, model)

        elapsed = time.time() - t0

        if result.get("error"):
            errored += 1
            results.append({
                "id": sid,
                "pass": False,
                "score": 0.0,
                "details": {"error": result["error"]},
                "api_calls": result.get("api_calls", 0),
                "duration_s": round(elapsed, 2),
            })
            continue

        grade = grade_scenario(scenario, result, rubric)
        if grade["pass"]:
            passed += 1
        else:
            failed += 1

        results.append({
            "id": sid,
            "pass": grade["pass"],
            "score": grade["score"],
            "details": grade.get("details", {}),
            "api_calls": result.get("api_calls", 0),
            "duration_s": round(elapsed, 2),
        })

    graded = passed + failed + errored
    total = len(scenarios)
    pass_rate = passed / graded if graded > 0 else 0.0

    report = {
        "suite": suite_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "deterministic_only": deterministic_only,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errored": errored,
        "skipped": skipped,
        "pass_rate": round(pass_rate, 4),
        "scenarios": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report written to {output_path}", file=sys.stderr)

    return report


def _load_baseline(baseline_path: Path) -> Optional[dict]:
    """Load a baseline JSON for comparison."""
    if not baseline_path.exists():
        return None
    try:
        return json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compare_baseline(report: dict, baseline_path: Path) -> dict:
    """Compare a new report against a stored baseline. Returns diff dict."""
    baseline = _load_baseline(baseline_path)
    if baseline is None:
        return {"status": "no_baseline", "message": f"No baseline at {baseline_path}"}

    old_rate = baseline.get("pass_rate", 0)
    new_rate = report.get("pass_rate", 0)
    delta = new_rate - old_rate

    regressions = []
    old_scenarios = {s["id"]: s for s in baseline.get("scenarios", [])}
    for s in report.get("scenarios", []):
        old = old_scenarios.get(s["id"])
        if old and old.get("pass") and not s.get("pass"):
            regressions.append(s["id"])

    return {
        "status": "regression" if delta < -0.05 else ("improvement" if delta > 0.05 else "stable"),
        "baseline_pass_rate": old_rate,
        "current_pass_rate": new_rate,
        "delta": round(delta, 4),
        "regressions": regressions,
    }


def print_summary(report: dict) -> None:
    """Print a human-readable summary to stdout."""
    print(f"\n{'='*60}")
    print(f"Suite: {report['suite']}")
    print(f"Model: {report.get('model', 'unknown')} via {report.get('provider', 'unknown')}")
    print(f"Time:  {report['timestamp']}")
    print(f"{'='*60}")
    print(f"Total:   {report['total']}")
    print(f"Passed:  {report['passed']}")
    print(f"Failed:  {report['failed']}")
    print(f"Errors:  {report.get('errored', 0)}")
    print(f"Skipped: {report.get('skipped', 0)}")
    print(f"Rate:    {report['pass_rate']:.1%}")
    print(f"{'='*60}")

    for s in report.get("scenarios", []):
        if s.get("skipped"):
            status = "↷"
            score = "skip"
        else:
            status = "✅" if s.get("pass") else ("❌" if s.get("score", 0) == 0 else "⚠️")
            score = f"{s.get('score', 0):.2f}" if s.get("score") is not None else "?"
        print(f"  {status} {s['id']}: score={score}  api_calls={s.get('api_calls', '?')}  {s.get('duration_s', 0):.1f}s")
        if s.get("skipped") and s.get("details"):
            print(f"      reason: {s['details'].get('reason', '')}")
        elif not s.get("pass") and s.get("details"):
            for k, v in s["details"].items():
                print(f"      {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hermes Agent Eval Suite Runner")
    parser.add_argument("--suite", required=True, help="Suite name (e.g., orchestration, cost_cache)")
    parser.add_argument("--suites-dir", default=str(_EVALS_DIR / "suites"), help="Path to suites directory")
    parser.add_argument("--provider", default="openrouter", help="LLM provider (openrouter, anthropic, etc.)")
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5", help="Model name")
    parser.add_argument("--output", help="Output JSON path (default: evals/reports/<suite>.json)")
    parser.add_argument("--baseline", help="Baseline JSON path for comparison")
    parser.add_argument("--deterministic-only", action="store_true", help="Skip live API calls, structural checks only")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-scenario output")
    args = parser.parse_args()

    suite_path = Path(args.suites_dir) / f"{args.suite}.yaml"
    if not suite_path.exists():
        print(f"ERROR: Suite not found: {suite_path}", file=sys.stderr)
        sys.exit(1)

    output_path = None
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = _EVALS_DIR / "reports" / f"{args.suite}.json"

    report = run_suite(
        suite_path=suite_path,
        provider=args.provider,
        model=args.model,
        output_path=output_path,
        deterministic_only=args.deterministic_only,
    )

    print_summary(report)

    if args.baseline:
        diff = compare_baseline(report, Path(args.baseline))
        print(f"Baseline comparison: {diff['status']}  (Δ={diff['delta']:+.2%})")
        if diff.get("regressions"):
            print(f"Regressions: {', '.join(diff['regressions'])}")
            sys.exit(1)

    if report["pass_rate"] < 0.5:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
