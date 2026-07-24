#!/usr/bin/env python3
"""promptfoo custom provider bridge for the Hermes eval harness.

This script is invoked by promptfoo when configured with:
    providers:
      - exec: python evals/promptfoo/hermes_provider.py "{{prompt}}" "{{vars}}"

promptfoo passes:
  - argv[1]: the rendered prompt string (the user_message)
  - argv[2]: a JSON string of the test case `vars` dict

The script calls ``run_suite.run_scenario_live()`` which in turn invokes
``AIAgent.run_conversation()``, then returns the result as a JSON string
to stdout in the format promptfoo expects::

    {"output": "<json string of the agent result>"}

The agent result dict has the shape::

    {
        "final_response": str,
        "messages": [ {role, content, tool_calls?, ...}, ... ],
        "api_calls": int,
        "error": str | None,
    }

promptfoo will pass this JSON string as the ``output`` field to
downstream assertions (contains, llm-rubric, python file://, etc.).

Usage:
    # Called by promptfoo automatically — not run directly.
    # For manual testing:
    python evals/promptfoo/hermes_provider.py \\
        "What is 2+2?" \\
        '{"scenario_id":"O3","suite":"orchestration","user_message":"What is 2+2?",...}'
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — add the repo root to sys.path so we can import the evals harness
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> tuple[str, dict[str, Any]]:
    """Parse promptfoo's argv into (prompt, vars_dict)."""
    prompt = sys.argv[1] if len(sys.argv) > 1 else ""
    vars_json = sys.argv[2] if len(sys.argv) > 2 else "{}"

    # promptfoo may pass vars as a JSON string or a path to a JSON file
    try:
        vars_dict = json.loads(vars_json)
    except (json.JSONDecodeError, TypeError):
        # Fallback: try reading as a file path
        try:
            vars_dict = json.loads(Path(vars_json).read_text(encoding="utf-8"))
        except Exception:
            vars_dict = {}

    return prompt, vars_dict


def _build_scenario(vars_dict: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Build a scenario dict from promptfoo test vars + the rendered prompt.

    The vars dict mirrors the fields from evals/suites/<suite>.yaml.
    If user_message is in vars, use it; otherwise fall back to the prompt.
    """
    user_message = vars_dict.get("user_message") or prompt

    scenario = {
        "id": vars_dict.get("scenario_id", "unknown"),
        "user_message": user_message,
        "system_message": vars_dict.get("system_message"),
        "config_overrides": vars_dict.get("config_overrides", {}),
        "enabled_toolsets": vars_dict.get(
            "enabled_toolsets",
            ["terminal", "file", "delegation"],
        ),
        "skip_memory": vars_dict.get("skip_memory", True),
        "skip_context_files": vars_dict.get("skip_context_files", True),
    }

    # Remove None values so AIAgent uses its own defaults
    scenario = {k: v for k, v in scenario.items() if v is not None}
    return scenario


def _run(vars_dict: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Run the scenario and return the agent result dict."""
    # Lazy import — only needed when actually running, not for --help
    from evals.runners.run_suite import run_scenario_live

    scenario = _build_scenario(vars_dict, prompt)

    # Read provider/model from the promptfoo provider config (passed via
    # vars by the exec: wrapper, or from environment variables)
    provider = vars_dict.get("provider") or os.environ.get(
        "HERMES_EVAL_PROVIDER", "openrouter"
    )
    model = vars_dict.get("model") or os.environ.get(
        "HERMES_EVAL_MODEL", "anthropic/claude-haiku-4.5"
    )

    # Deterministic mode: use mock messages from the suite YAML
    if vars_dict.get("deterministic_only"):
        return {
            "final_response": "",
            "messages": vars_dict.get("_mock_messages", []),
            "api_calls": 0,
            "error": None,
        }

    result = run_scenario_live(scenario, provider, model)
    return result


def main() -> None:
    """Entry point — parse args, run scenario, print JSON to stdout."""
    prompt, vars_dict = _parse_args()

    try:
        result = _run(vars_dict, prompt)
        # promptfoo expects {"output": <string>}
        # We serialize the full result dict so assertions can inspect it
        output = json.dumps(result, ensure_ascii=False)
        print(json.dumps({"output": output}))
    except Exception as exc:
        # On error, return a structured error so assertions can detect it
        error_result = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "final_response": "",
            "messages": [],
            "api_calls": 0,
        }
        print(json.dumps({"output": json.dumps(error_result, ensure_ascii=False)}))
        # Non-zero exit so promptfoo marks the test as errored
        sys.exit(1)


if __name__ == "__main__":
    main()