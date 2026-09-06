"""Subprocess bridge for an operator-owned Evaluator policy.

The command is supplied by trusted host configuration, never by a turn request.
The bridge only returns a decision; it never performs platform delivery.
"""
from __future__ import annotations

import json
import subprocess
from typing import Any, Sequence


class SubprocessEvaluatorPolicy:
    """Call a fixed Evaluator adapter command with one completed response."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        policy: dict[str, Any],
        agent_configuration_id: str,
        evaluator_configuration_id: str,
        timeout_seconds: int = 30,
    ) -> None:
        if not command or not all(isinstance(arg, str) and arg for arg in command):
            raise ValueError("command must be a non-empty sequence of strings")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self._command = tuple(command)
        self._policy = dict(policy)
        self._agent_configuration_id = agent_configuration_id
        self._evaluator_configuration_id = evaluator_configuration_id
        self._timeout_seconds = timeout_seconds

    def __call__(self, *, final_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
        request = {
            "schema_version": 1,
            "request_id": str(metadata.get("turn_id") or "hermes-turn"),
            "final_text": final_text,
            "metadata": {
                **metadata,
                "agent_configuration_id": self._agent_configuration_id,
                "evaluator_configuration_id": self._evaluator_configuration_id,
            },
            "policy": self._policy,
        }
        try:
            completed = subprocess.run(
                list(self._command),
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self._timeout_seconds,
                check=False,
            )
            result = json.loads(completed.stdout)
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            raise RuntimeError(f"evaluator policy invocation failed: {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("evaluator policy must return a JSON object")
        return result
