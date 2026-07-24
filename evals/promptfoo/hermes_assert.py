#!/usr/bin/env python3
"""promptfoo Python assertion script — bridges to Hermes rubric modules.

promptfoo calls this script when an assertion is configured as:

    assert:
      - type: python
        value: file://evals/promptfoo/hermes_assert.py
        config:
          suite: orchestration
          condition: delegate_call_count
          min: 2

promptfoo passes the following via stdin as a JSON object::

    {
        "output": "<the provider output string — JSON from hermes_provider.py>",
        "prompt": "<the rendered prompt>",
        "vars": { <test case variables> },
        "config": { <assertion config> },
        "context": { <additional context> }
    }

This script:
  1. Parses the provider output (JSON string) into the agent result dict.
  2. Loads the scenario from `vars` (which mirrors the suite YAML fields).
  3. Imports the Hermes rubric module for the requested suite
     (e.g., evals/rubrics/orchestration.py).
  4. Calls the rubric's ``grade()`` function (or a specific condition
     evaluator if `config.condition` is set).
  5. Returns a JSON object to stdout in promptfoo's assertion format::

      {"pass": true|false, "score": 0.0-1.0, "reason": "..."}

Supported config keys:
  - suite:      the rubric module name (e.g., "orchestration", "cost_cache")
  - condition:  a specific pass_condition type to evaluate in isolation
                (e.g., "delegate_call_count", "plan_score", "no_tool_error")
                If omitted, the full ``grade()`` is called and all
                conditions are checked.
  - min:        numeric minimum (for delegate_call_count, plan_score, etc.)
  - max:        numeric maximum (for delegate_call_count, etc.)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_EVALS_DIR = _REPO_ROOT / "evals"


def _load_rubric(suite_name: str):
    """Dynamically import a rubric module from evals/rubrics/<suite>.py."""
    rubric_path = _EVALS_DIR / "rubrics" / f"{suite_name}.py"
    if not rubric_path.exists():
        raise FileNotFoundError(f"No rubric at {rubric_path}")
    spec = importlib.util.spec_from_file_location(f"rubric_{suite_name}", rubric_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_scenario(vars_dict: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a scenario dict from promptfoo test vars.

    The vars dict contains the same fields as the suite YAML scenario,
    plus `scenario_id` which maps to the scenario `id`.
    """
    scenario = {
        "id": vars_dict.get("scenario_id", vars_dict.get("id", "unknown")),
        "user_message": vars_dict.get("user_message", ""),
        "system_message": vars_dict.get("system_message"),
        "config_overrides": vars_dict.get("config_overrides", {}),
        "enabled_toolsets": vars_dict.get("enabled_toolsets", []),
        "skip_memory": vars_dict.get("skip_memory", True),
        "skip_context_files": vars_dict.get("skip_context_files", True),
        "pass_conditions": vars_dict.get("pass_conditions", []),
    }
    # Strip None values
    return {k: v for k, v in scenario.items() if v is not None}


def _evaluate_single_condition(
    rubric_module,
    condition_type: str,
    scenario: dict,
    metrics: dict,
    config: dict,
) -> Dict[str, Any]:
    """Evaluate a single pass_condition type from the rubric.

    Many Hermes rubrics (e.g., orchestration.py) expose per-condition
    evaluator functions named ``_cond_<type>`` and a ``compute_metrics``
    function.  We call those directly for targeted condition checks.
    """
    # Compute metrics if the rubric supports it
    metrics_fn = getattr(rubric_module, "compute_metrics", None)
    if metrics_fn is not None and not metrics:
        # Build a minimal result dict to compute metrics from
        result_dict = {"messages": [], "final_response": ""}
        metrics = metrics_fn(result_dict)

    # Look up the condition evaluator function
    # Convention: _cond_<type> for each condition type
    evaluator_name = f"_cond_{condition_type}"
    evaluator = getattr(rubric_module, evaluator_name, None)

    if evaluator is None:
        # Fallback: check if it's in a _CONDITIONS registry
        conditions_registry = getattr(rubric_module, "_CONDITIONS", {})
        evaluator = conditions_registry.get(condition_type)

    if evaluator is None:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"Unknown condition type: {condition_type!r}",
        }

    # Build the condition dict from config (min, max, value, etc.)
    cond = {"type": condition_type}
    for key in ("min", "max", "value", "rubric"):
        if key in config:
            cond[key] = config[key]

    try:
        ok, reason = evaluator(scenario, metrics, cond)
    except Exception as exc:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"evaluator_error: {type(exc).__name__}: {exc}",
        }

    return {
        "pass": bool(ok),
        "score": 1.0 if ok else 0.0,
        "reason": str(reason),
    }


def _evaluate_full_grade(
    rubric_module,
    scenario: dict,
    result: dict,
) -> Dict[str, Any]:
    """Call the rubric's full grade() function and return assertion result."""
    grade_fn = getattr(rubric_module, "grade", None)
    if grade_fn is None:
        return {
            "pass": False,
            "score": 0.0,
            "reason": "Rubric module has no grade() function",
        }

    try:
        grade_result = grade_fn(scenario, result)
    except Exception as exc:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"grade_error: {type(exc).__name__}: {exc}",
        }

    return {
        "pass": grade_result.get("pass", False),
        "score": grade_result.get("score", 0.0),
        "reason": json.dumps(grade_result.get("details", {}), ensure_ascii=False),
    }


def main() -> None:
    """Entry point — read stdin, evaluate, print result to stdout."""
    try:
        ctx = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"pass": False, "score": 0.0, "reason": f"stdin parse error: {exc}"}))
        sys.exit(1)

    output_str = ctx.get("output", "")
    vars_dict = ctx.get("vars", {})
    config = ctx.get("config", {})

    # Parse the provider output (JSON string) into the agent result dict
    try:
        result = json.loads(output_str) if output_str else {}
    except json.JSONDecodeError:
        # If output isn't JSON, treat it as a plain final_response string
        result = {"final_response": output_str, "messages": [], "error": None}

    suite_name = config.get("suite", vars_dict.get("suite", "orchestration"))
    condition_type = config.get("condition")

    try:
        rubric_module = _load_rubric(suite_name)
    except FileNotFoundError as exc:
        print(json.dumps({"pass": False, "score": 0.0, "reason": str(exc)}))
        sys.exit(0)

    scenario = _build_scenario(vars_dict)

    if condition_type:
        # Evaluate a single condition in isolation
        # Need to compute metrics from the actual result
        metrics = {}
        metrics_fn = getattr(rubric_module, "compute_metrics", None)
        if metrics_fn is not None:
            metrics = metrics_fn(result)

        assertion_result = _evaluate_single_condition(
            rubric_module, condition_type, scenario, metrics, config
        )
    else:
        # Full grade() call — checks all pass_conditions in the scenario
        assertion_result = _evaluate_full_grade(rubric_module, scenario, result)

    print(json.dumps(assertion_result, ensure_ascii=False))


if __name__ == "__main__":
    main()