"""AIAgent-compatible child adapter for the output-only Antigravity worker."""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from contextlib import contextmanager
from typing import Any, Mapping

from agent.antigravity_worker import AntigravityResult, AntigravityWorker
from agent.gemini_route_receipts import GeminiReceiptStore
from agent.route_receipts import (
    RouteReceiptChild,
    RouteReceiptStore,
    route_usage_from_mapping,
)
from agent.interrupt_compat import request_hard_interrupt


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


class _TerminalCommitCancelled(RuntimeError):
    """Cancellation won before a terminal receipt transaction committed."""


def _text_evidence_metadata(
    text: str | None,
    sha256: str | None,
    byte_count: int | None,
) -> tuple[str | None, int | None]:
    """Return integrity metadata without retaining the source text."""
    if text is None:
        return sha256, byte_count
    encoded = text.encode("utf-8")
    return sha256 or hashlib.sha256(encoded).hexdigest(), (
        byte_count if byte_count is not None else len(encoded)
    )


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
        route_store: RouteReceiptStore | None = None,
        run_kind: str = "production",
    ) -> None:
        self.worker = worker
        self.fallback_child = fallback_child
        self.store = store
        self.route_store = route_store or RouteReceiptStore(store.path)
        self.run_kind = run_kind
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
        self._receipt_lock = threading.Lock()
        self._terminal_commit_lock = threading.Lock()
        self._terminal_owner: str | None = None
        self._admission_lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._cancel_shutdown_confirmed = threading.Event()
        self._execution_done = threading.Event()
        self._execution_done.set()
        self._execution_thread_id: int | None = None
        self._started = time.monotonic()
        self._status = "ready"
        self._closed = False

    def prepare_receipt(self) -> str:
        """Persist the pre-execution receipt before lifecycle publication."""
        if self.receipt_id:
            return self.receipt_id
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
            goal_text="",
            context_text="",
            requested_provider=self.requested_provider,
            requested_model=self.requested_model,
            requested_effort=self.requested_effort,
        )
        self.route_store.prepare_attempt(
            stable_call_id=self.receipt_id,
            parent_session_id=self.parent_session_id,
            parent_turn_id=self.parent_turn_id,
            child_session_id=self.session_id,
            task_index=self.task_index,
            run_kind=self.run_kind,
            route_requested=self.route_requested,
            route_decision="gemini",
            route_reason=self.route_reason,
            data_classification=self.data_classification,
            output_contract=self.output_contract,
            provider=self.requested_provider,
            model=self.requested_model,
        )
        if isinstance(self.fallback_child, RouteReceiptChild):
            self.fallback_child.bind_fallback_from(self.receipt_id)
        self._route_metadata = {
            "route": "gemini",
            "route_reason": self.route_reason,
            "worker_route": "gemini",
            "worker_provider": self.requested_provider,
            "worker_model_requested": self.requested_model,
            "route_receipt_id": self.receipt_id,
            "fallback_used": False,
        }
        return self.receipt_id

    def run_conversation(
        self,
        user_message: str,
        task_id: str | None = None,
        stream_callback=None,
    ) -> dict[str, Any]:
        if not self._execution_lock.acquire(blocking=False):
            return self._already_running_result()
        try:
            with self._admission_lock:
                if self._cancel_event.is_set():
                    return self._cancelled_result()
                self._execution_thread_id = threading.get_ident()
                self._execution_done.clear()
            try:
                return self._run_conversation(user_message, task_id, stream_callback)
            finally:
                with self._admission_lock:
                    self._execution_thread_id = None
                    self._execution_done.set()
        finally:
            self._execution_lock.release()

    def _run_conversation(
        self,
        user_message: str,
        task_id: str | None = None,
        stream_callback=None,
    ) -> dict[str, Any]:
        del user_message, task_id, stream_callback
        receipt_error = False
        with self._receipt_lock:
            if self._cancel_event.is_set():
                return self._cancelled_result()
            try:
                self.prepare_receipt()
            except Exception:
                receipt_error = True
        if receipt_error:
            return self._fallback_or_failure(
                "Gemini route receipt could not be prepared",
                route="sol_after_receipt_error",
            )
        with self._admission_lock:
            if self._cancel_event.is_set():
                self._record_cancelled_attempt()
                return self._cancelled_result()
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

        if self._cancel_event.is_set():
            self._record_cancelled_attempt()
            return self._cancelled_result()

        fallback = result.status != "success" and self.fallback_child is not None
        terminal_status = (
            "completed"
            if result.status == "success"
            else _ERROR_STATUS.get(result.error_code or "", "failed")
        )
        commit_fence = self._pending_commit_fence if fallback else self._result_commit_fence
        response_sha256, response_bytes = _text_evidence_metadata(
            result.response or result.output_excerpt,
            result.output_sha256,
            result.output_bytes,
        )
        route_usage = route_usage_from_mapping(result.usage)
        # The legacy table is compatibility-only. Its failure must not roll back
        # or suppress the authoritative route-neutral receipt, but its fence
        # still gives cancellation the same admission point as before.
        try:
            with self._receipt_lock:
                self.store.complete_attempt(
                    self.receipt_id,
                    worker_status=terminal_status,
                    response_text=None,
                    response_sha256=response_sha256,
                    response_bytes=response_bytes,
                    process_exit_code=result.exit_code,
                    duration_ms=result.duration_ms,
                    conversation_id=result.conversation_id,
                    usage=result.usage,
                    raw_envelope=None,
                    fallback_used=fallback,
                    error_code=result.error_code,
                    error_message=None,
                    commit_fence=commit_fence,
                )
        except _TerminalCommitCancelled:
            return self._cancelled_result()
        except Exception:
            pass

        try:
            with self._receipt_lock:
                if self._cancel_event.is_set():
                    return self._cancelled_result()
                with commit_fence():
                    self.route_store.complete_attempt(
                        self.receipt_id,
                        status=terminal_status,
                        terminal_route="gemini",
                        terminal_provider=self.requested_provider,
                        terminal_model=self.requested_model,
                        usage=route_usage,
                        usage_status=(
                            "complete" if route_usage is not None else "unavailable"
                        ),
                        fallback_used=fallback,
                        error_code=result.error_code,
                    )
        except _TerminalCommitCancelled:
            return self._cancelled_result()
        except Exception:
            if self._cancel_event.is_set():
                return self._cancelled_result()
            return self._fallback_or_failure(
                "Gemini route result could not be recorded",
                route="sol_after_receipt_error",
                worker_route="gemini",
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
                "worker_route": "gemini",
                "worker_provider": self.requested_provider,
                "worker_model_requested": self.requested_model,
                "route_receipt_id": self.receipt_id,
                "fallback_used": False,
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
        worker_route: str | None = None,
    ) -> dict[str, Any]:
        if self._cancel_event.is_set():
            self._record_cancelled_attempt()
            return self._cancelled_result()
        if self.fallback_child is not None:
            with self._activity_lock:
                self._status = "fallback"
            if self.tool_progress_callback is not None:
                self.fallback_child.tool_progress_callback = self.tool_progress_callback
            fallback_metadata = {
                "route": route,
                "route_reason": self.route_reason,
                "worker_route": "sol",
                "worker_provider": str(
                    getattr(self.fallback_child, "provider", "delegation-model")
                ),
                "worker_model_requested": str(
                    getattr(self.fallback_child, "model", "")
                ),
                "route_receipt_id": self.receipt_id or None,
                "fallback_used": True,
            }
            if error_code:
                fallback_metadata["gemini_error_code"] = error_code
            self._route_metadata = dict(fallback_metadata)
            with self._admission_lock:
                if self._cancel_event.is_set():
                    self._record_cancelled_attempt()
                    return self._cancelled_result()
            try:
                result = self.fallback_child.run_conversation(
                    user_message=self.goal,
                    task_id=f"fallback-{self.task_index}",
                )
            except Exception as exc:
                result = {
                    "final_response": "",
                    "completed": False,
                    "api_calls": 0,
                    "messages": [],
                    "error": f"Sol fallback raised {type(exc).__name__}",
                }
            if not isinstance(result, dict):
                result = {
                    "final_response": "",
                    "completed": False,
                    "api_calls": 0,
                    "messages": [],
                    "error": "Sol fallback returned an invalid result",
                }
            if self._cancel_event.is_set():
                return self._cancelled_result()
            result = dict(result)
            fallback_metadata["worker_provider"] = str(
                result.get("worker_provider")
                or getattr(self.fallback_child, "provider", "delegation-model")
            )
            fallback_metadata["worker_model_requested"] = str(
                result.get("worker_model_requested")
                or getattr(self.fallback_child, "model", "")
            )
            fallback_status = (
                "completed"
                if result.get("completed") is True and result.get("final_response")
                else "failed"
            )
            fallback_response = (
                str(result["final_response"])
                if isinstance(result.get("final_response"), str)
                and result.get("final_response")
                else None
            )
            fallback_response_sha256, fallback_response_bytes = _text_evidence_metadata(
                fallback_response, None, None
            )
            if self.receipt_id:
                if isinstance(self.fallback_child, RouteReceiptChild):
                    fallback_metadata["fallback_route_receipt_id"] = (
                        self.fallback_child.route_receipt_id
                    )
                try:
                    with self._receipt_lock:
                        if self._cancel_event.is_set():
                            return self._cancelled_result()
                        self.store.record_fallback_outcome(
                            self.receipt_id,
                            worker_route="sol",
                            provider=str(fallback_metadata["worker_provider"]),
                            model=str(fallback_metadata["worker_model_requested"]),
                            worker_status=fallback_status,
                            response_text=None,
                            response_sha256=fallback_response_sha256,
                            response_bytes=fallback_response_bytes,
                            error_code=(
                                None
                                if fallback_status == "completed"
                                else "sol_fallback_failed"
                            ),
                            commit_fence=self._result_commit_fence,
                        )
                except _TerminalCommitCancelled:
                    return self._cancelled_result()
                except Exception:
                    result["legacy_receipt_error"] = "fallback_outcome_not_recorded"
                if isinstance(self.fallback_child, RouteReceiptChild):
                    try:
                        with self._receipt_lock:
                            self.route_store.link_fallback(
                                self.receipt_id, self.fallback_child.route_receipt_id
                            )
                    except Exception:
                        if self._cancel_event.is_set():
                            return self._cancelled_result()
                        result["route_receipt_error"] = "fallback_link_not_recorded"
            if not self._claim_result_publication():
                return self._cancelled_result()
            self._route_metadata = dict(fallback_metadata)
            result.update(fallback_metadata)
            if self.receipt_id:
                result["receipt_id"] = self.receipt_id
            with self._activity_lock:
                self._status = "completed" if result.get("final_response") else "failed"
            return result

        if not self._claim_result_publication():
            return self._cancelled_result()
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
            "worker_route": worker_route or route,
            "worker_provider": self.requested_provider,
            "worker_model_requested": self.requested_model,
            "route_receipt_id": self.receipt_id or None,
            "fallback_used": False,
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

    def _record_cancelled_attempt(self) -> None:
        if not self._cancel_shutdown_confirmed.is_set():
            return
        with self._receipt_lock:
            if not self.receipt_id:
                return
            try:
                self.store.complete_attempt(
                    self.receipt_id,
                    worker_status="cancelled",
                    duration_ms=max(0, int((time.monotonic() - self._started) * 1000)),
                    fallback_used=False,
                    error_code="cancelled",
                    error_message="Gemini-routed delegation cancelled",
                    commit_fence=self._cancelled_commit_fence,
                )
                self.route_store.complete_attempt(
                    self.receipt_id,
                    status="cancelled",
                    terminal_route="gemini",
                    terminal_provider=self.requested_provider,
                    terminal_model=self.requested_model,
                    usage=None,
                    usage_status="unavailable",
                    error_code="cancelled",
                )
            except (KeyError, ValueError):
                pass
            except Exception:
                raise RuntimeError(
                    "delegation cancellation could not be confirmed"
                ) from None

    def _cancelled_result(self) -> dict[str, Any]:
        with self._activity_lock:
            self._status = "failed"
        return {
            "final_response": "",
            "completed": False,
            "api_calls": 0,
            "messages": [],
            "error": "Gemini-routed delegation cancelled",
            "route": "gemini",
            "route_reason": self.route_reason,
            "receipt_id": self.receipt_id or None,
            "gemini_error_code": "cancelled",
            "worker_route": "gemini",
            "worker_provider": self.requested_provider,
            "worker_model_requested": self.requested_model,
            "route_receipt_id": self.receipt_id or None,
            "fallback_used": False,
        }

    def _already_running_result(self) -> dict[str, Any]:
        return {
            "final_response": "",
            "completed": False,
            "api_calls": 0,
            "messages": [],
            "error": "Gemini-routed delegation already has an active run",
            "route": "gemini",
            "route_reason": self.route_reason,
            "receipt_id": self.receipt_id or None,
            "gemini_error_code": "already_running",
            "worker_route": "gemini",
            "worker_provider": self.requested_provider,
            "worker_model_requested": self.requested_model,
            "route_receipt_id": self.receipt_id or None,
            "fallback_used": False,
        }

    @contextmanager
    def _pending_commit_fence(self):
        """Commit non-final receipt state only while cancellation has not won."""
        with self._terminal_commit_lock:
            if self._terminal_owner == "cancelled":
                raise _TerminalCommitCancelled()
            yield

    @contextmanager
    def _result_commit_fence(self):
        """Atomically claim terminal result ownership at receipt commit."""
        with self._terminal_commit_lock:
            if self._terminal_owner == "cancelled":
                raise _TerminalCommitCancelled()
            if self._terminal_owner not in {None, "result"}:
                raise RuntimeError("invalid terminal owner")
            previous_owner = self._terminal_owner
            self._terminal_owner = "result"
            try:
                yield
            except BaseException:
                self._terminal_owner = previous_owner
                raise

    @contextmanager
    def _cancelled_commit_fence(self):
        """Serialize cancelled receipt terminalization with result commits."""
        with self._terminal_commit_lock:
            if self._terminal_owner != "cancelled":
                raise _TerminalCommitCancelled()
            yield

    def _claim_result_publication(self) -> bool:
        """Publish only for the terminal owner selected by the shared fence."""
        with self._terminal_commit_lock:
            if self._terminal_owner == "cancelled":
                return False
            if self._terminal_owner is None:
                self._terminal_owner = "result"
            return self._terminal_owner == "result"

    def cancel(self) -> None:
        with self._terminal_commit_lock:
            cancellation_won = self._terminal_owner != "result"
            if cancellation_won:
                self._terminal_owner = "cancelled"
                self._cancel_event.set()
        with self._admission_lock:
            execution_thread_id = self._execution_thread_id
        shutdown_failed = False
        try:
            self.worker.close()
        except Exception:
            shutdown_failed = True
        if self.fallback_child is not None:
            try:
                request_hard_interrupt(
                    self.fallback_child, "Gemini-routed delegation cancelled"
                )
            except Exception:
                pass
        same_execution_thread = execution_thread_id == threading.get_ident()
        execution_settled = same_execution_thread
        if not same_execution_thread:
            execution_settled = self._execution_done.wait(timeout=10.0)
            if not execution_settled:
                shutdown_failed = True
        if self.fallback_child is not None:
            if same_execution_thread and not self._execution_done.is_set():
                shutdown_failed = True
            elif execution_settled:
                close = getattr(self.fallback_child, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        shutdown_failed = True
                else:
                    shutdown_failed = True
        if shutdown_failed:
            raise RuntimeError("delegation cancellation could not be confirmed")
        if cancellation_won:
            self._cancel_shutdown_confirmed.set()
            self._record_cancelled_attempt()

    def interrupt(self, message: str | None = None) -> None:
        del message
        self.cancel()

    def hard_interrupt(self, message: str | None = None) -> None:
        del message
        self.cancel()

    def close(self) -> None:
        if self._closed:
            return
        self.cancel()
        self._closed = True
