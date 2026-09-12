"""AIAgent-compatible child adapter for the output-only Antigravity worker."""

from __future__ import annotations

import secrets
import threading
import time
from typing import Any, Mapping

from agent.antigravity_worker import AntigravityResult, AntigravityWorker
from agent.gemini_route_receipts import GeminiReceiptStore


_ERROR_STATUS = {
    "timeout": "timeout",
    "cancelled": "cancelled",
    "malformed_envelope": "malformed",
    "invalid_response": "malformed",
    "schema_validation_failed": "malformed",
    "output_too_large": "oversized",
    "input_too_large": "oversized",
    "tool_action_blocked": "denied",
}


class AntigravityDelegateChild:
    """Present a bounded subprocess as the small child surface delegate_task uses."""

    def __init__(
        self,
        *,
        worker: AntigravityWorker,
        fallback_child: Any | None,
        store: GeminiReceiptStore,
        task_index: int,
        goal: str,
        context: str,
        output_schema: dict[str, Any] | None,
        output_contract: str,
        route_requested: str,
        route_reason: str,
        data_classification: str,
        requested_provider: str,
        requested_model: str,
        requested_effort: str,
        parent_session_id: str,
        parent_turn_id: str,
    ) -> None:
        self.worker = worker
        self.fallback_child = fallback_child
        self.store = store
        self.task_index = int(task_index)
        self.goal = goal
        self.context = context
        self.output_schema = output_schema
        self.output_contract = output_contract
        self.route_requested = route_requested
        self.route_reason = route_reason
        self.data_classification = data_classification
        self.requested_provider = requested_provider
        self.requested_model = requested_model
        self.requested_effort = requested_effort
        self.parent_session_id = parent_session_id
        self.parent_turn_id = parent_turn_id

        self.receipt_id = ""
        self.session_id = str(getattr(fallback_child, "session_id", "") or f"gemini-{secrets.token_hex(8)}")
        self.model = requested_model
        self._delegate_role = "leaf"
        self._subagent_id = getattr(fallback_child, "_subagent_id", None)
        self._parent_subagent_id = getattr(fallback_child, "_parent_subagent_id", None)
        self._delegate_saved_tool_names = list(
            getattr(fallback_child, "_delegate_saved_tool_names", []) or []
        )
        self._credential_pool = None
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.tool_progress_callback = getattr(fallback_child, "tool_progress_callback", None)
        self._activity_lock = threading.Lock()
        self._started = time.monotonic()
        self._status = "ready"
        self._closed = False

    def run_conversation(
        self,
        user_message: str,
        task_id: str | None = None,
        stream_callback=None,
    ) -> dict[str, Any]:
        del user_message, task_id, stream_callback
        with self._activity_lock:
            self._status = "preparing"
        try:
            self.receipt_id = self.store.prepare_attempt(
                parent_session_id=self.parent_session_id,
                parent_turn_id=self.parent_turn_id,
                child_session_id=self.session_id,
                task_index=self.task_index,
                route_requested=self.route_requested,
                route_decision="gemini",
                route_reason=self.route_reason,
                data_classification=self.data_classification,
                output_contract=self.output_contract,
                goal_text=self.goal,
                context_text=self.context,
                requested_provider=self.requested_provider,
                requested_model=self.requested_model,
                requested_effort=self.requested_effort,
            )
        except Exception:
            return self._fallback_or_failure(
                "Gemini route receipt could not be prepared",
                route="sol_after_receipt_error",
            )

        with self._activity_lock:
            self._status = "running"
        try:
            result = self.worker.run(
                goal=self.goal,
                context=self.context,
                output_schema=self.output_schema,
                on_process_started=lambda: self.store.mark_process_started(self.receipt_id),
            )
        except Exception as exc:
            result = AntigravityResult(
                status="failed",
                response=None,
                conversation_id=None,
                usage={},
                raw_envelope=None,
                exit_code=None,
                duration_ms=max(0, int((time.monotonic() - self._started) * 1000)),
                error_code="worker_exception",
                error_message=f"Antigravity worker raised {type(exc).__name__}",
            )

        fallback = result.status != "success" and self.fallback_child is not None
        terminal_status = (
            "completed"
            if result.status == "success"
            else _ERROR_STATUS.get(result.error_code or "", "failed")
        )
        try:
            self.store.complete_attempt(
                self.receipt_id,
                worker_status=terminal_status,
                response_text=result.response,
                process_exit_code=result.exit_code,
                duration_ms=result.duration_ms,
                conversation_id=result.conversation_id,
                usage=result.usage,
                raw_envelope=result.raw_envelope,
                fallback_used=fallback,
                error_code=result.error_code,
                error_message=result.error_message,
            )
        except Exception:
            return self._fallback_or_failure(
                "Gemini route result could not be recorded",
                route="sol_after_receipt_error",
            )

        if result.status == "success" and result.response:
            with self._activity_lock:
                self._status = "completed"
            return {
                "final_response": result.response,
                "completed": True,
                "api_calls": 1,
                "messages": [],
                "route": "gemini",
                "route_reason": self.route_reason,
                "receipt_id": self.receipt_id,
            }
        return self._fallback_or_failure(
            result.error_message or "Antigravity worker failed",
            route="gemini_then_sol" if fallback else "gemini",
            error_code=result.error_code,
        )

    def _fallback_or_failure(
        self,
        message: str,
        *,
        route: str,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        if self.fallback_child is not None:
            with self._activity_lock:
                self._status = "fallback"
            if self.tool_progress_callback is not None:
                self.fallback_child.tool_progress_callback = self.tool_progress_callback
            result = self.fallback_child.run_conversation(
                user_message=self.goal,
                task_id=f"fallback-{self.task_index}",
            )
            if not isinstance(result, dict):
                result = {
                    "final_response": "",
                    "completed": False,
                    "api_calls": 0,
                    "messages": [],
                    "error": "Sol fallback returned an invalid result",
                }
            result = dict(result)
            result["route"] = route
            result["route_reason"] = self.route_reason
            if self.receipt_id:
                result["receipt_id"] = self.receipt_id
            if error_code:
                result["gemini_error_code"] = error_code
            with self._activity_lock:
                self._status = "completed" if result.get("final_response") else "failed"
            return result

        with self._activity_lock:
            self._status = "failed"
        return {
            "final_response": "",
            "completed": False,
            "api_calls": 0,
            "messages": [],
            "error": message,
            "route": route,
            "route_reason": self.route_reason,
            "receipt_id": self.receipt_id or None,
            "gemini_error_code": error_code,
        }

    def get_activity_summary(self) -> dict[str, Any]:
        if self._status == "fallback" and self.fallback_child is not None:
            summary = getattr(self.fallback_child, "get_activity_summary", None)
            if callable(summary):
                inherited = summary()
                if isinstance(inherited, dict):
                    return inherited
        with self._activity_lock:
            status = self._status
        return {
            "api_call_count": 1 if status in {"completed", "failed"} else 0,
            "max_iterations": 1,
            "current_tool": "antigravity" if status == "running" else None,
            "last_activity_desc": f"Gemini route {status}",
        }

    def cancel(self) -> None:
        self.worker.cancel()
        if self.fallback_child is not None:
            interrupt = getattr(self.fallback_child, "request_hard_interrupt", None)
            if callable(interrupt):
                interrupt("Gemini-routed delegation cancelled")
            elif hasattr(self.fallback_child, "_interrupt_requested"):
                self.fallback_child._interrupt_requested = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.worker.close()
        if self.fallback_child is not None:
            close = getattr(self.fallback_child, "close", None)
            if callable(close):
                close()
