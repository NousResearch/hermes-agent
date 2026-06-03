"""CLI helpers for the local Hermes eval lab.

This module is intentionally local-only: it loads YAML scenarios, runs them
against a deterministic fake backend by default, writes replayable JSONL,
and emits a markdown report. It does not call remote services or training
backends.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agent.eval_lab.reporting import write_markdown_report
from agent.eval_lab.runner import LocalEvalRunner
from agent.eval_lab.scenarios import load_scenarios
from agent.eval_lab.scoring import score_attempt
from agent.eval_lab.storage import EvalRunStorage


class DeterministicEchoBackend:
    """Safe default backend for smoke-testing the eval lab plumbing."""

    def run_conversation(self, user_message: str) -> dict[str, Any]:
        return {
            "final_response": f"Echo: {user_message}",
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": f"Echo: {user_message}"},
            ],
            "metadata": {"backend": "deterministic-echo"},
        }


def cmd_eval_lab_run(args: argparse.Namespace) -> int:
    """Run scenarios through the local deterministic eval-lab pipeline."""

    scenario_path = Path(getattr(args, "scenarios", "eval_scenarios/smoke.yaml"))
    run_id = str(getattr(args, "run_id", "local-smoke"))
    attempts = int(getattr(args, "attempts", 1))
    output_dir = getattr(args, "output_dir", None)

    scenarios = load_scenarios(scenario_path)
    storage = EvalRunStorage(run_id=run_id, base_dir=output_dir)
    runner = LocalEvalRunner(DeterministicEchoBackend())

    groups = []
    scores = []
    for scenario in scenarios:
        group = runner.run(scenario, attempt_count=attempts)
        storage.write_group(group)
        groups.append(group)
        for attempt in group.attempts:
            score = score_attempt(scenario, attempt)
            storage.write_score(score)
            scores.append(score)

    report_path = write_markdown_report(
        storage.run_dir / "report.md",
        run_id=run_id,
        groups=groups,
        scores=scores,
        artifact_paths=[str(storage.groups_path), str(storage.scores_path)],
    )

    print(f"Eval lab run complete: {run_id}")
    print(f"Run directory: {storage.run_dir}")
    print(f"Report: {report_path}")
    return 0


def cmd_eval_lab(args: argparse.Namespace) -> int:
    """Dispatch eval-lab subcommands."""

    command = getattr(args, "eval_lab_command", "run")
    if command == "run":
        return cmd_eval_lab_run(args)
    raise SystemExit(f"Unknown eval-lab command: {command}")
