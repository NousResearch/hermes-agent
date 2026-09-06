"""Midstream application-tool execution for direct GPT-6 Astra Responses turns.

The provider can finish an ``async`` function-call item before the Responses stream reaches its
terminal frame.  This module admits those items into the existing tool middleware while keeping
their assistant fragments durable. Tool-result rows settle in order at stream completion or when
steering requires a durable prefix. Ordinary providers never instantiate this internal coordinator.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from agent.tool_dispatch_helpers import _plan_tool_batch_segments
from tools.daemon_pool import DaemonThreadPoolExecutor
from tools.thread_context import propagate_context_to_thread


def is_direct_astra(agent: Any) -> bool:
    """True only for the exact official OpenAI Responses Astra route."""
    if getattr(agent, "api_mode", None) != "codex_responses":
        return False
    model = str(getattr(agent, "model", "") or "").strip().lower().rsplit("/", 1)[-1]
    if model != "gpt-6-astra":
        return False
    from utils import base_url_hostname

    return base_url_hostname(str(getattr(agent, "base_url", "") or "")) == "api.openai.com"


def provider_async_marker(tool_call: Any) -> bool:
    """Read the provider marker, not the registry's Python coroutine metadata."""
    if isinstance(tool_call, dict):
        return tool_call.get("async") is True or tool_call.get("async_") is True
    provider_data = getattr(tool_call, "provider_data", None) or {}
    return (
        getattr(tool_call, "async", False) is True
        or getattr(tool_call, "async_", False) is True
        or provider_data.get("async") is True
    )


@dataclass
class _AstraJob:
    call: Any
    parsed: Any
    index: int
    future: Any = None
    managed: Any = None
    duration: float = 0.0
    done: bool = False
    committed: bool = False


class AstraAsyncExecutor:
    """Admit, schedule, and commit provider-marked Astra application tools."""

    def __init__(self, agent: Any, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        self.agent = agent
        self.messages = messages
        self.effective_task_id = effective_task_id
        self.api_call_count = api_call_count
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._settlement_lock = threading.Lock()
        self._jobs: list[_AstraJob] = []
        self._reservations: list[str] = []
        self._pending_calls: dict[str, Any] = {}
        self._admitted_ids: set[str] = set()
        self._stream_closed = False
        self._closed = False
        self._finalized = False
        self._failed = False
        self._consumed = False
        self._aborted_results = False
        self._executor = DaemonThreadPoolExecutor(max_workers=8)

        from tools.terminal_tool_lifecycle import get_active_env

        active_env = get_active_env(effective_task_id)
        self._execution_cwd = Path(active_env.cwd) if active_env is not None and active_env.cwd else None

    @property
    def has_admitted(self) -> bool:
        with self._lock:
            return bool(self._jobs)

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._reservations or self._pending_calls)

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._failed

    @property
    def finalized(self) -> bool:
        with self._lock:
            return self._finalized

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def retire_empty(self) -> bool:
        """Close an executor that admitted no calls so its turn-owned state cannot be reused."""
        with self._lock:
            if self._jobs or self._reservations or self._pending_calls or self._closed:
                return False
            self._stream_closed = self._closed = True
        self._executor.shutdown(wait=True)
        return True

    def _call_id(self, call: Any, index: int) -> str:
        from agent.chat_completion_helpers import _assistant_tool_call_dict

        # The normal builder is the source of truth for Responses aliases and deterministic IDs.
        try:
            return str(_assistant_tool_call_dict(self.agent, call, index).get("id") or "").strip()
        except Exception:
            raw = getattr(call, "call_id", None) or getattr(call, "id", None)
            return str(raw or f"astra_async_{index}").strip()

    def _assistant_fragment(self, call: Any, index: int) -> dict:
        from agent.chat_completion_helpers import _assistant_tool_call_dict

        tc = _assistant_tool_call_dict(self.agent, call, index)
        return {"role": "assistant", "content": None, "tool_calls": [tc], "finish_reason": "tool_calls"}

    def _reservation_key(self, call: Any) -> str:
        if isinstance(call, dict):
            raw = call.get("call_id") or call.get("id") or call.get("response_item_id")
        else:
            raw = getattr(call, "call_id", None) or getattr(call, "id", None) or getattr(call, "response_item_id", None)
        return str(raw or "").strip()

    def reserve(self, call: Any) -> bool:
        """Record provider output order before arguments are complete or executable."""
        if not provider_async_marker(call):
            return False
        with self._lock:
            if self._closed or self._stream_closed:
                return False
            key = self._reservation_key(call)
            if not key or key in self._admitted_ids:
                return True
            if key not in self._reservations:
                self._reservations.append(key)
            return True

    def _drain_admissions_locked(self) -> None:
        while self._reservations:
            key = self._reservations[0]
            call = self._pending_calls.get(key)
            if call is None:
                return
            self._pending_calls.pop(key, None)
            self._reservations.pop(0)
            if not self._admit_one_locked(call):
                return

    def admit(self, call: Any) -> bool:
        """Queue one completed call and admit only the completed announced prefix."""
        if not provider_async_marker(call):
            return False
        with self._lock:
            if self._closed or self._stream_closed:
                return False
            key = self._reservation_key(call)
            if not key:
                key = f"anonymous:{len(self._reservations) + len(self._pending_calls)}"
            if key in self._admitted_ids:
                return True
            if key not in self._reservations:
                self._reservations.append(key)
            self._pending_calls[key] = call
            self._drain_admissions_locked()
            return key in self._admitted_ids or key in self._pending_calls

    def _admit_one_locked(self, call: Any) -> bool:
        index = len(self._jobs)
        if not getattr(call, "function", None):
            setattr(call, "function", SimpleNamespace(
                name=getattr(call, "name", ""), arguments=getattr(call, "arguments", "{}") or "{}",
            ))
        call_id = self._call_id(call, index)
        if call_id in self._admitted_ids:
            return True
        try:
            fragment = self._assistant_fragment(call, index)
        except Exception:
            self._failed = True
            return False
        self.messages.append(fragment)
        try:
            persisted = self.agent._flush_messages_to_session_db(self.messages) is True
        except Exception:
            persisted = False
        if not persisted:
            self.messages.pop()
            self._failed = True
            return False

        # Bind the canonical ID back to this transient streamed object so the existing parser,
        # middleware, and result pairing all see the same identity.
        if not getattr(call, "id", None):
            setattr(call, "id", call_id)
        if not getattr(call, "call_id", None):
            setattr(call, "call_id", call_id)
        from agent.tool_executor import _parse_tool_call

        try:
            parsed = _parse_tool_call(self.agent, call)
        except Exception:
            self._failed = True
            return False
        job = _AstraJob(call=call, parsed=parsed, index=index, done=parsed.parse_error is not None)
        self._jobs.append(job)
        self._admitted_ids.add(call_id)
        self.agent._session_messages = self.messages
        self._pump_locked()
        return True

    def _segments_locked(self) -> list[tuple[str, list[Any]]]:
        calls = [job.call for job in self._jobs]
        return _plan_tool_batch_segments(calls, execution_cwd=self._execution_cwd)

    def _submit_locked(self, job: _AstraJob) -> None:
        if job.future is not None or self._closed or job.parsed.parse_error is not None:
            return
        job.future = self._executor.submit(propagate_context_to_thread(self._run_job), job)
        job.future.add_done_callback(lambda future, current=job: self._job_done(current, future))

    def _pump_locked(self) -> None:
        """Start the earliest runnable segment; later segments remain barriers."""
        if self._closed:
            return
        segments = self._segments_locked()
        offset = 0
        for kind, calls in segments:
            indexes = range(offset, offset + len(calls))
            group = [self._jobs[index] for index in indexes]
            if any(not job.done for job in self._jobs[:offset]):
                return
            if kind == "parallel":
                for job in group:
                    self._submit_locked(job)
                if any(not job.done for job in group):
                    return
                offset += len(group)
                continue
            unstarted = next(
                (job for job in group if job.future is None and job.parsed.parse_error is None), None,
            )
            if unstarted is not None:
                self._submit_locked(unstarted)
                return
            if any(not job.done for job in group):
                return
            offset += len(group)

    def _run_job(self, job: _AstraJob):
        from agent.tool_executor import _resolve_sequential_dispatch, _run_sequential_call

        start = time.time()
        ref = job.parsed.ref(self.effective_task_id)
        dispatch = _resolve_sequential_dispatch(self.agent, ref, self.messages)
        managed, duration = _run_sequential_call(
            self.agent, dispatch, ref, scope_block=job.parsed.scope_block, messages=self.messages,
            remaining_calls=[], display_index=job.index + 1, tool_start_time=start,
        )
        return managed, duration

    def _job_done(self, job: _AstraJob, future: Any) -> None:
        try:
            managed, duration = future.result()
        except Exception as exc:
            from agent.tool_executor import _ManagedToolResult

            managed, duration = _ManagedToolResult(
                result=f"Error executing tool '{job.parsed.name}': {exc}", args=job.parsed.args,
                middleware_trace=job.parsed.middleware_trace, blocked=False, dispatched=False,
            ), 0.0
        with self._lock:
            job.managed, job.duration, job.done = managed, duration, True
            self._pump_locked()
            self._changed.notify_all()

    def _all_done_locked(self) -> bool:
        return all(job.done for job in self._jobs)

    @staticmethod
    def _canonical_call_id(value: Any) -> str:
        return str(value or "").split("|", 1)[0].strip()

    def _job_call_ids(self, job: _AstraJob) -> set[str]:
        ref = job.parsed.ref(self.effective_task_id)
        return {
            call_id for call_id in (
                self._canonical_call_id(ref.call_id),
                self._canonical_call_id(getattr(job.call, "call_id", None)),
                self._canonical_call_id(getattr(job.call, "id", None)),
            ) if call_id
        }

    def _required_jobs_locked(self, call_ids: Any) -> Optional[list[_AstraJob]]:
        required = {self._canonical_call_id(call_id) for call_id in call_ids or ()}
        required.discard("")
        if not required:
            return []
        positions = {
            call_id: index
            for index, job in enumerate(self._jobs)
            for call_id in self._job_call_ids(job)
        }
        if not required.issubset(positions):
            return None
        # Publishing in assistant-call order requires every earlier job.  The scheduler's
        # segmented barriers also require those predecessors to have completed first.
        return list(self._jobs[:max(positions[call_id] for call_id in required) + 1])

    def _wait_for_jobs(self, jobs: list[_AstraJob]) -> bool:
        while True:
            abort = False
            with self._lock:
                if all(job.committed for job in jobs):
                    return True
                if self._closed or self._failed:
                    return False
                if getattr(self.agent, "_interrupt_requested", False):
                    abort = True
                elif all(job.done for job in jobs):
                    return True
                else:
                    self._pump_locked()
                    self._changed.wait(timeout=0.1)
            if abort:
                self.abort_stream()
                return False

    def _publish_jobs(self, jobs: list[_AstraJob], *, finalize: bool = False) -> bool:
        """Persist one ordered job prefix, skipping rows already committed by settlement."""
        if not jobs:
            return True
        from agent.tool_executor import (
            _append_invalid_arguments_result, _budget_for_agent, _finalize_tool_batch, _publish_sequential_result,
        )

        selected = {id(job) for job in jobs}
        ordered = [job for job in self._jobs if id(job) in selected]
        budget = _budget_for_agent(self.agent)
        for index, job in enumerate(ordered):
            if job.committed:
                continue
            ref = job.parsed.ref(self.effective_task_id)
            message_count = len(self.messages)
            if job.parsed.parse_error is not None:
                published = _append_invalid_arguments_result(self.agent, self.messages, ref, job.parsed.parse_error)
            else:
                published = _publish_sequential_result(
                    self.agent, self.messages, ref, job.managed, tool_duration=job.duration,
                    index=job.index + 1, budget=budget,
                )
            if not published:
                self._failed = True
                del self.messages[message_count:]
                recovered = self._recover_uncommitted_results(ordered[index:])
                self.agent._session_messages = self.messages
                return recovered
            job.committed = True
        if finalize and not self._failed and self._jobs:
            _finalize_tool_batch(self.agent, self.messages, self.effective_task_id, len(self._jobs), budget)
        self.agent._session_messages = self.messages
        return True

    def settle_required(self, call_ids: Any) -> bool:
        """Wait for required calls and ordered predecessors, then durably publish them once.

        This is the PR3 pending-steer boundary.  It never exposes an in-memory managed result;
        callers may construct required input only after this returns ``True`` and read the
        canonical persisted tool rows from ``agent._session_messages``.
        """
        with self._settlement_lock:
            with self._lock:
                jobs = self._required_jobs_locked(call_ids)
            if jobs is None or not self._wait_for_jobs(jobs):
                return False
            published = self._publish_jobs(jobs)
            # A recovery notice is durable, but it is not the requested tool result.  Keep
            # pending continuation fail-closed after a canonical publication failure.
            return published and not self.failed

    def _recover_uncommitted_results(self, jobs: list[_AstraJob]) -> bool:
        from agent.replay_cleanup import _DANGLING_NOTICES, _orphan_recovery
        from agent.tool_dispatch_helpers import make_tool_result_message

        uncommitted = [job for job in jobs if not job.committed]
        for job in uncommitted:
            ref = job.parsed.ref(self.effective_task_id)
            if job.parsed.parse_error is not None:
                content = job.parsed.parse_error
                disposition = None
            else:
                disposition, content = _orphan_recovery(job.parsed.name, _DANGLING_NOTICES)
            self.messages.append(make_tool_result_message(
                job.parsed.name, content, ref.call_id, effect_disposition=disposition,
            ))
        if not uncommitted:
            return True
        try:
            persisted = self.agent._flush_messages_to_session_db(self.messages) is True
        except Exception:
            persisted = False
        if persisted:
            for job in uncommitted:
                job.committed = True
        self.agent._session_messages = self.messages
        return persisted

    def finish_stream(self, assistant_content: str = "", settled_calls: Any = None) -> bool:
        """Wait for all admitted jobs and publish results in original call order."""
        with self._settlement_lock:
            with self._lock:
                if self._closed:
                    return self._finalized
                if getattr(self.agent, "_interrupt_requested", False):
                    self.abort_stream()
                    return False
                # The assembler settles announced items that lack an output_item.done frame only when
                # it builds the terminal response.  Admit those calls before retiring reservations.
                for call in settled_calls or ():
                    if provider_async_marker(call):
                        self.admit(call)
                self._stream_closed = True
                self._drain_admissions_locked()
                self._pending_calls.clear()
                self._reservations.clear()
                self._pump_locked()
                while not self._all_done_locked():
                    if getattr(self.agent, "_interrupt_requested", False):
                        self.abort_stream()
                        return False
                    self._changed.wait(timeout=0.1)

            if assistant_content:
                from agent.chat_completion_helpers import build_assistant_message

                text_message = build_assistant_message(
                    self.agent, SimpleNamespace(content=assistant_content, tool_calls=[]), "tool_calls",
                )
                text_message.pop("tool_calls", None)
                self.messages.append(text_message)
                try:
                    if self.agent._flush_messages_to_session_db(self.messages) is not True:
                        self._failed = True
                except Exception:
                    self._failed = True
            if self._failed:
                recovered = self._recover_uncommitted_results(self._jobs)
                self._closed = True
                self._finalized = recovered
                self._executor.shutdown(wait=True)
                return recovered
            finalized = self._publish_jobs(self._jobs, finalize=True)
            self._closed = True
            self._finalized = finalized
            self._executor.shutdown(wait=True)
            return self._finalized

    def abort_stream(self) -> None:
        """Retire scheduling after interruption/stream failure without retrying handlers."""
        with self._lock:
            if self._closed:
                return
            self._stream_closed = True
            for job in self._jobs:
                if job.future is not None and not job.future.done():
                    job.future.cancel()
            self._pending_calls.clear()
            self._reservations.clear()
            if not self._aborted_results and self._jobs:
                self._recover_uncommitted_results(self._jobs)
                self.agent._session_messages = self.messages
                self._aborted_results = True
            self._closed = True
            self._changed.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def consume_response(self, assistant_message: Any) -> bool:
        """Tell the ordinary turn loop that this response's handlers already ran."""
        with self._lock:
            if not self._finalized or self._consumed or not self._jobs:
                return False
            self._consumed = True
            return True


__all__ = ["AstraAsyncExecutor", "is_direct_astra", "provider_async_marker"]
