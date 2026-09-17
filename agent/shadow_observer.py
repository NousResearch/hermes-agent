"""Opt-in, failure-isolated Hermes CLI Shadow observer.

The observer is deliberately independent of the evaluator implementation. It invokes
an explicitly configured local adapter after a turn has completed, never changes the
runtime result, and never logs the response body or adapter output. The adapter call
is bounded; its failure is recorded in return metadata and does not fail the turn.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

_LOG = logging.getLogger(__name__)


def _hash_identity(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    section = config.get("evaluation")
    if not isinstance(section, Mapping):
        return None
    shadow = section.get("shadow")
    return shadow if isinstance(shadow, Mapping) and shadow.get("enabled") is True else None


def _command(settings: Mapping[str, Any]) -> list[str] | None:
    value = settings.get("adapter_command")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    command = [item for item in value if isinstance(item, str) and item]
    return command if len(command) == len(value) and command else None


def observe_completed_response(
    result: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Observe one completed CLI response without changing ``result``.

    A malformed configuration, adapter timeout, or adapter failure is represented in
    the returned metadata and is never raised into the normal CLI turn. ``final_text``
    exists only in the subprocess input buffer and is not written by this module.
    """
    settings = _settings(config)
    if settings is None:
        return {"status": "disabled", "invoked": False, "persisted": False}
    if not isinstance(result, Mapping):
        return {"status": "inconclusive", "invoked": False, "persisted": False, "reason_code": "invalid_result"}

    command = _command(settings)
    output_value = settings.get("evidence_output")
    if command is None or not isinstance(output_value, str) or not output_value:
        return {"status": "inconclusive", "invoked": False, "persisted": False, "reason_code": "invalid_configuration"}

    if result.get("completed") is not True:
        return {"status": "inconclusive", "invoked": False, "persisted": False, "reason_code": "turn_not_completed"}

    final_text = result.get("final_response")
    if not isinstance(final_text, str) or not final_text:
        return {"status": "inconclusive", "invoked": False, "persisted": False, "reason_code": "empty_final_response"}

    rid = request_id or f"hermes-cli-{uuid.uuid4().hex}"
    policy = settings.get("policy")
    if not isinstance(policy, Mapping):
        policy = {"required_patterns": [], "forbidden_patterns": []}
    event = {
        "schema_version": 1,
        "event_type": "completed_response",
        "request_id": rid,
        "final_text": final_text,
        "metadata": {
            "agent_configuration_id": _hash_identity("hermes-cli-runtime"),
            "evaluator_configuration_id": _hash_identity("agent-level-shadow-evaluator"),
            "trigger_origin": "hermes",
            "execution_mode": "shadow",
            "runtime_surface": "cli",
        },
        "policy": dict(policy),
        "trace": {
            "tool_count": _bounded_count(result.get("tool_calls")),
            "step_count": _bounded_count(result.get("api_calls")),
        },
    }
    timeout = settings.get("timeout_seconds", 10)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = 10
    try:
        completed = subprocess.run(
            [*command, "--evidence-output", output_value],
            input=json.dumps(event, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.warning("CLI Shadow adapter failed: %s", type(exc).__name__)
        return {"status": "inconclusive", "invoked": True, "persisted": False, "reason_code": "adapter_failed"}

    try:
        decision = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError):
        decision = {}
    if not isinstance(decision, dict):
        decision = {}
    return {
        "status": decision.get("status", "inconclusive"),
        "invoked": True,
        "persisted": decision.get("evidence_persisted") is True,
        "adapter_exit_code": completed.returncode,
        "delivery_attempted": decision.get("delivery_attempted") is True,
        "side_effect_status": decision.get("side_effect_status", "unknown"),
    }
