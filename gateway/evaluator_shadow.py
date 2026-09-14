"""Hermes-to-Evaluator shadow adapter.

The adapter is opt-in: callers must construct it and assign it to
``GatewayRunner.pre_delivery_gate``. It never sends platform messages.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


class EvaluatorShadowAdapter:
    """Invoke the deterministic Evaluator CLI without changing Shadow delivery."""

    def __init__(
        self,
        evaluator_root: str | Path,
        *,
        agent_configuration_id: str,
        evaluator_configuration_id: str,
        evidence_output: Optional[str | Path] = None,
        required_patterns: Optional[list[str]] = None,
        forbidden_patterns: Optional[list[str]] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.evaluator_root = Path(evaluator_root).resolve()
        self.agent_configuration_id = agent_configuration_id
        self.evaluator_configuration_id = evaluator_configuration_id
        self.evidence_output = Path(evidence_output) if evidence_output else None
        self.required_patterns = list(required_patterns or [])
        self.forbidden_patterns = list(forbidden_patterns or [])
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _ref(value: Any) -> str:
        digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    async def __call__(self, result: dict[str, Any], turn_ctx: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._evaluate_sync, result, turn_ctx)

    def _evaluate_sync(self, result: dict[str, Any], turn_ctx: Any) -> dict[str, Any]:
        inbound_id = getattr(turn_ctx, "inbound_message_id", None)
        event_id = getattr(turn_ctx, "event_message_id", None)
        request_id = str(inbound_id or event_id or getattr(turn_ctx, "session_id", None) or "turn-unknown")
        source = getattr(turn_ctx, "source", None)
        final_text = result.get("final_response")
        request = {
            "schema_version": 1,
            "request_id": request_id,
            "final_text": final_text if isinstance(final_text, str) else "",
            "metadata": {
                "platform": str(getattr(getattr(source, "platform", None), "value", "") or ""),
                "chat_id_ref": self._ref(getattr(source, "chat_id", "")),
                "session_id_ref": self._ref(getattr(turn_ctx, "session_key", "")),
                "turn_id": request_id,
                "agent_configuration_id": self.agent_configuration_id,
                "evaluator_configuration_id": self.evaluator_configuration_id,
                "trigger_origin": "agent",
            },
            "policy": {
                "required_patterns": self.required_patterns,
                "forbidden_patterns": self.forbidden_patterns,
            },
        }
        command = [
            sys.executable,
            str(self.evaluator_root / "scripts" / "live_shadow_policy.py"),
        ]
        if self.evidence_output is not None:
            command.extend(["--evidence-output", str(self.evidence_output)])
        try:
            completed = subprocess.run(
                command,
                cwd=self.evaluator_root,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict):
                raise ValueError("adapter response must be an object")
            decision = payload.get("status")
            if decision not in {"passed", "blocked", "inconclusive"}:
                return {
                    "decision": "inconclusive",
                    "side_effect_status": "not_attempted",
                    "evidence_persisted": payload.get("evidence_persisted"),
                    "evidence_error_code": "evaluator_malformed_result",
                    "evidence_ref": payload.get("evidence_ref"),
                }
            expected_returncode = 0 if decision == "passed" else 1
            if completed.returncode != expected_returncode:
                return {
                    "decision": "inconclusive",
                    "side_effect_status": "not_attempted",
                    "evidence_persisted": payload.get("evidence_persisted"),
                    "evidence_error_code": "evaluator_returncode_mismatch",
                    "evidence_ref": payload.get("evidence_ref"),
                }
            return {
                "decision": decision,
                "side_effect_status": "not_attempted",
                "evidence_persisted": payload.get("evidence_persisted"),
                "evidence_error_code": payload.get("evidence_error_code"),
                "evidence_ref": payload.get("evidence_ref"),
            }
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            return {
                "decision": "inconclusive",
                "side_effect_status": "not_attempted",
                "evidence_persisted": False,
                "evidence_error_code": "evaluator_unavailable",
            }
