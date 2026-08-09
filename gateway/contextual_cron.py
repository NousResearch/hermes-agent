"""Gateway-owned execution lane for opt-in same-session cron jobs.

The scheduler is deliberately kept out of SessionDB.  It hands an occurrence to
this bridge; the live gateway resolves the stable routing key, seals the exact
session-id admission in the cron ledger, and serializes the synthetic turn
behind any active user turn.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Optional

from gateway.session import (
    ContextualRouteInstanceMismatch,
    Platform,
    SessionSource,
    canonical_route_coordinates,
)

logger = logging.getLogger(__name__)

_CONTEXTUAL_KINDS = frozenset(
    {"notify", "no_action", "retryable", "rejected", "stale", "failure", "unknown"}
)


class ContextualCronGuardBusy(RuntimeError):
    """Compatibility seam for transient pre-awaitable adapter contention."""


class ContextualCronTranscriptConflict(RuntimeError):
    """A later transcript turn overtook a pending contextual outbox."""


@dataclass(frozen=True)
class ContextualCronOutcome:
    """Typed result returned from the gateway lane to the scheduler."""

    kind: str
    final_response: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _CONTEXTUAL_KINDS:
            raise ValueError(f"unknown contextual cron outcome: {self.kind}")

    @property
    def success(self) -> bool:
        return self.kind in {"notify", "no_action"}

    @classmethod
    def notify(cls, text: str) -> "ContextualCronOutcome":
        return cls("notify", final_response=str(text or ""))

    @classmethod
    def no_action(cls) -> "ContextualCronOutcome":
        return cls("no_action")

    @classmethod
    def rejected(cls, error: str) -> "ContextualCronOutcome":
        return cls("rejected", error=str(error))

    @classmethod
    def stale(cls, error: str) -> "ContextualCronOutcome":
        return cls("stale", error=str(error))

    @classmethod
    def failure(cls, error: str) -> "ContextualCronOutcome":
        return cls("failure", error=str(error))

    @classmethod
    def retryable(cls, error: str) -> "ContextualCronOutcome":
        return cls("retryable", error=str(error))

    @classmethod
    def unknown(cls, error: str) -> "ContextualCronOutcome":
        return cls("unknown", error=str(error))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ContextualCronQueueItem:
    job_id: str
    execution_id: str
    prompt: str
    session_key: str
    admitted_session_id: str
    source: Any
    future: asyncio.Future
    admitted_routing_revision: int = 0
    admitted_route_instance_id: Optional[str] = None
    admitted_binding_version: int = 1
    turn_lease_token: Any = None
    adapter: Any = None
    adapter_guard: Any = None
    transcript_session_id: Optional[str] = None
    transcript_entries: Optional[list[Dict[str, Any]]] = None
    transcript_base_message_count: Optional[int] = None
    transcript_base_revision: Optional[int] = None
    last_prompt_tokens: Optional[int] = None


class ContextualCronGateway:
    """Dedicated per-session FIFO owned by one live :class:`GatewayRunner`."""

    def __init__(
        self,
        runner: Any,
        *,
        seal_admission: Optional[Callable[..., bool]] = None,
        finish_admission: Optional[Callable[..., Any]] = None,
        load_execution: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        busy_poll_seconds: float = 0.02,
        transcript_retry_seconds: float = 0.1,
    ) -> None:
        self.runner = runner
        self._seal_admission: Callable[..., bool] = (
            seal_admission or self._default_seal_admission
        )
        # Persist the typed agent result before resolving the scheduler future.
        # The scheduler remains the sole owner of the external delivery claim.
        self._finish_admission = finish_admission or self._default_finish_admission
        self._load_execution = load_execution or self._default_load_execution
        self._busy_poll_seconds = max(0.001, float(busy_poll_seconds))
        self._transcript_retry_seconds = max(
            0.001, float(transcript_retry_seconds)
        )
        self._queues: Dict[str, Deque[ContextualCronQueueItem]] = defaultdict(deque)
        self._drainers: Dict[str, asyncio.Task] = {}
        self._execution_futures: Dict[str, asyncio.Future] = {}

    @staticmethod
    def _default_seal_admission(
        execution_id: str,
        session_key: str,
        session_id: str,
        routing_revision: int,
        route_instance_id: Optional[str] = None,
        binding_version: int = 1,
    ) -> bool:
        from cron.executions import seal_contextual_admission

        return bool(
            seal_contextual_admission(
                execution_id,
                session_key=session_key,
                admitted_session_id=session_id,
                admitted_routing_revision=routing_revision,
                admitted_route_instance_id=route_instance_id,
                admitted_binding_version=binding_version,
            )
        )

    def _seal_occurrence(
        self,
        execution_id: str,
        session_key: str,
        session_id: str,
        routing_revision: int,
        route_instance_id: Optional[str] = None,
        binding_version: int = 1,
    ) -> bool:
        """Call production v2 seal while retaining old test doubles."""
        try:
            params = inspect.signature(self._seal_admission).parameters.values()
            variadic = any(
                p.kind is inspect.Parameter.VAR_POSITIONAL for p in params
            )
            positional_count = sum(
                p.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
                for p in params
            )
            supports_binding_version = variadic or positional_count >= 6
            supports_route_instance = variadic or positional_count >= 5
            supports_revision = variadic or positional_count >= 4
        except (TypeError, ValueError):
            supports_binding_version = True
            supports_route_instance = True
            supports_revision = True
        if supports_binding_version:
            return bool(
                self._seal_admission(
                    execution_id,
                    session_key,
                    session_id,
                    routing_revision,
                    route_instance_id,
                    binding_version,
                )
            )
        if supports_route_instance:
            return bool(
                self._seal_admission(
                    execution_id,
                    session_key,
                    session_id,
                    routing_revision,
                    route_instance_id,
                )
            )
        if supports_revision:
            return bool(
                self._seal_admission(
                    execution_id,
                    session_key,
                    session_id,
                    routing_revision,
                )
            )
        return bool(self._seal_admission(execution_id, session_key, session_id))

    @staticmethod
    def _default_load_execution(execution_id: str) -> Optional[Dict[str, Any]]:
        from cron.executions import get_execution

        return get_execution(execution_id)

    @staticmethod
    def _default_finish_admission(
        execution_id: str,
        outcome: ContextualCronOutcome,
        item: Optional[ContextualCronQueueItem] = None,
    ) -> Any:
        from cron.executions import persist_contextual_agent_result

        return persist_contextual_agent_result(
            execution_id,
            outcome=outcome.kind,
            final_response=outcome.final_response,
            error=outcome.error,
            transcript_session_id=(
                item.transcript_session_id if item is not None else None
            ),
            transcript_entries=(item.transcript_entries if item is not None else None),
            transcript_base_message_count=(
                item.transcript_base_message_count if item is not None else None
            ),
            transcript_base_revision=(
                item.transcript_base_revision if item is not None else None
            ),
            transcript_last_prompt_tokens=(
                item.last_prompt_tokens if item is not None else None
            ),
        )

    def _finish_occurrence(
        self,
        item: ContextualCronQueueItem,
        outcome: ContextualCronOutcome,
    ) -> Any:
        """Pass the private transcript outbox while retaining old test doubles."""
        try:
            params = inspect.signature(self._finish_admission).parameters.values()
            supports_item = any(
                p.kind is inspect.Parameter.VAR_POSITIONAL for p in params
            ) or sum(
                p.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
                for p in params
            ) >= 3
        except (TypeError, ValueError):
            supports_item = True
        if supports_item:
            return self._finish_admission(item.execution_id, outcome, item)
        return self._finish_admission(item.execution_id, outcome)

    def _persisted_outcome(self, execution_id: str) -> Optional[ContextualCronOutcome]:
        """Replay a scheduler-owned terminal result after future eviction."""
        try:
            record = self._load_execution(execution_id)
        except Exception:
            logger.exception("Failed to read contextual cron execution %s", execution_id)
            return ContextualCronOutcome.unknown(
                "Could not read the durable contextual cron occurrence."
            )
        if not record:
            return None
        phase = str(record.get("phase") or "")
        has_durable_result = bool(record.get("result_json") and record.get("outcome"))
        if record.get("status") not in {"completed", "failed", "unknown"}:
            if has_durable_result and phase in {
                "agent_completed",
                "delivering",
                "retry_wait",
            }:
                pass
            # A previously sealed, non-terminal occurrence with no durable
            # result crossed an unknown-crash boundary. Never enqueue its model
            # side effect again.
            elif (
                phase == "admitted"
                and int(record.get("retry_count") or 0) > 0
                and not has_durable_result
            ):
                # Scheduler explicitly cleared a typed retryable result and
                # re-armed this same sealed occurrence. Admission is
                # idempotent; the concrete session/revision must still match.
                return None
            elif record.get("admitted_session_id") or record.get("session_key"):
                return ContextualCronOutcome.unknown(
                    "This contextual cron occurrence was already admitted, but "
                    "its in-process result is unavailable."
                )
            else:
                return None
        try:
            payload = json.loads(record.get("result_json") or "{}")
            return ContextualCronOutcome(
                str(payload.get("kind") or record.get("outcome") or "unknown"),
                final_response=str(payload.get("final_response") or ""),
                error=payload.get("error") or record.get("error"),
            )
        except Exception:
            return ContextualCronOutcome.unknown(
                str(record.get("error") or "Persisted contextual cron result is unreadable.")
            )

    def _peek_entry(self, session_key: str):
        store = self.runner.session_store
        peek = getattr(store, "peek_session_entry", None)
        if callable(peek):
            return peek(session_key)
        # Compatibility with an older SessionStore while the additive public
        # read helper rolls out.  This is still the live store, never SessionDB.
        return getattr(store, "_sessions", {}).get(session_key)

    def _validate_entry(
        self,
        session_key: str,
        admitted_session_id: Optional[str] = None,
        admitted_routing_revision: Optional[int] = None,
        admitted_route_instance_id: Optional[str] = None,
        authorization_source: Optional[SessionSource] = None,
        propagate_authorization_errors: bool = False,
    ):
        entry = self._peek_entry(session_key)
        if entry is None or not getattr(entry, "origin", None):
            return None, ContextualCronOutcome.rejected(
                "The target gateway session is missing; contextual cron never falls back to isolation."
            )
        if admitted_session_id is not None and entry.session_id != admitted_session_id:
            return None, ContextualCronOutcome.stale(
                "The target session was reset after this occurrence was admitted."
            )
        if (
            admitted_route_instance_id is not None
            and str(getattr(entry, "route_instance_id", ""))
            != admitted_route_instance_id
        ):
            return None, ContextualCronOutcome.stale(
                "The originating logical route changed after this occurrence was admitted."
            )
        if (
            admitted_routing_revision is not None
            and int(getattr(entry, "routing_revision", 0))
            != int(admitted_routing_revision)
        ):
            return None, ContextualCronOutcome.stale(
                "The target session route changed after this occurrence was admitted."
            )
        source = entry.origin
        key_builder = getattr(self.runner, "_session_key_for_source", None)
        if callable(key_builder):
            try:
                if key_builder(source) != session_key:
                    return None, ContextualCronOutcome.rejected(
                        "The stored source no longer owns the target session key."
                    )
            except Exception:
                return None, ContextualCronOutcome.rejected(
                    "The target source could not be revalidated."
                )
        try:
            authorized = bool(
                self.runner._is_user_authorized(authorization_source or source)
            )
        except Exception:
            if propagate_authorization_errors:
                raise
            authorized = False
        if not authorized:
            return None, ContextualCronOutcome.rejected(
                "The target session authorization is no longer valid."
            )
        return entry, None

    @staticmethod
    def _source_from_logical_binding(binding: Dict[str, Any]) -> SessionSource:
        """Reconstruct the immutable creator authority for a v2 occurrence."""
        return SessionSource(
            platform=Platform(str(binding.get("platform") or "")),
            chat_type=str(binding.get("chat_type") or "dm"),
            chat_id=str(binding.get("chat_id") or ""),
            thread_id=str(binding.get("thread_id") or "") or None,
            user_id=str(binding.get("user_id") or "") or None,
            scope_id=str(binding.get("scope_id") or "") or None,
            parent_chat_id=str(binding.get("parent_chat_id") or "") or None,
            user_id_alt=str(binding.get("user_id_alt") or "") or None,
            chat_id_alt=str(binding.get("chat_id_alt") or "") or None,
            profile=str(binding.get("profile") or "") or None,
        )

    @staticmethod
    def _logical_binding_rejection(entry, binding) -> Optional[ContextualCronOutcome]:
        """Require exact immutable conversation-route equality for v2 jobs."""
        route_instance_id = str(binding.get("route_instance_id") or "").strip()
        if not route_instance_id or entry.route_instance_id != route_instance_id:
            return ContextualCronOutcome.stale(
                "The originating logical conversation no longer exists."
            )
        source = entry.origin
        chat_type, chat_id, thread_id, parent_chat_id = canonical_route_coordinates(
            source
        )
        actual = {
            "profile": str(getattr(source, "profile", None) or ""),
            "platform": str(
                getattr(getattr(source, "platform", None), "value", "") or ""
            ),
            "chat_type": chat_type,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "scope_id": str(getattr(source, "scope_id", None) or ""),
            "parent_chat_id": parent_chat_id,
            "chat_id_alt": str(getattr(source, "chat_id_alt", None) or ""),
        }
        for field, current in actual.items():
            if str(binding.get(field) or "") != current:
                return ContextualCronOutcome.stale(
                    "The originating conversation's authenticated route changed."
                )
        return None

    async def dispatch(self, job: Dict[str, Any], *, execution_id: str) -> ContextualCronOutcome:
        """Seal and enqueue one contextual occurrence.

        Repeated dispatch of the same execution id shares one future in this
        process.  Durable terminal idempotency is enforced by the execution
        ledger's compare-and-set update.
        """
        if str(job.get("session_target") or "isolated").lower() != "current":
            return ContextualCronOutcome.rejected("Job is not a contextual cron job.")
        session_key = str(job.get("session_key") or "").strip()
        if not session_key:
            return ContextualCronOutcome.rejected(
                "Contextual cron job has no captured session key; no fallback was attempted."
            )
        prompt = str(job.get("prompt") or "").strip()
        if not prompt:
            return ContextualCronOutcome.rejected("Contextual cron job has no prompt.")

        # Reserve the durable occurrence before the first suspension. Event-loop
        # task scheduling makes setdefault linearizable here: exactly one owner
        # performs admission, while every concurrent duplicate joins its future.
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        shared = self._execution_futures.setdefault(execution_id, future)
        if shared is not future:
            return await asyncio.shield(shared)

        def finish_pre_admission(outcome: ContextualCronOutcome) -> ContextualCronOutcome:
            if not future.done():
                future.set_result(outcome)
            if self._execution_futures.get(execution_id) is future:
                self._execution_futures.pop(execution_id, None)
            return outcome

        persisted = self._persisted_outcome(execution_id)
        if persisted is not None:
            return finish_pre_admission(persisted)

        captured_session_id = None
        captured_routing_revision = None
        captured_route_instance_id = None
        captured_authorization_source: Optional[SessionSource] = None
        binding = job.get("context_binding")
        raw_binding_version = int(job.get("_contextual_binding_version") or 0)
        binding_version = raw_binding_version or (1 if binding is not None else 0)
        if binding is not None or raw_binding_version >= 1:
            from cron.contextual import (
                contextual_definition_route,
                contextual_definition_route_instance,
            )

            try:
                if binding_version == 2:
                    captured_route_instance_id = contextual_definition_route_instance(job)
                    assert isinstance(binding, dict)
                    captured_authorization_source = self._source_from_logical_binding(
                        binding
                    )
                elif binding_version == 1:
                    captured_session_id, captured_routing_revision = (
                        contextual_definition_route(job)
                    )
                else:
                    raise ValueError(
                        f"unsupported contextual binding version: {binding_version}"
                    )
            except (TypeError, ValueError) as exc:
                return finish_pre_admission(
                    ContextualCronOutcome.rejected(str(exc))
                )

        entry, rejection = self._validate_entry(
            session_key,
            admitted_session_id=captured_session_id,
            admitted_routing_revision=captured_routing_revision,
            authorization_source=captured_authorization_source,
        )
        if rejection is not None:
            return finish_pre_admission(rejection)
        assert entry is not None
        if binding_version == 2:
            assert isinstance(binding, dict)
            rejection = self._logical_binding_rejection(entry, binding)
            if rejection is not None:
                return finish_pre_admission(rejection)

        routing_revision = int(getattr(entry, "routing_revision", 0))
        try:
            linearize = getattr(
                self.runner.session_store,
                "seal_contextual_admission",
                None,
            )
            if callable(linearize):
                seal_resolved = (
                    lambda admitted_execution_id, admitted_key, admitted_session_id, admitted_revision: self._seal_occurrence(
                        admitted_execution_id,
                        admitted_key,
                        admitted_session_id,
                        admitted_revision,
                        captured_route_instance_id,
                        binding_version or 1,
                    )
                )
                try:
                    linearize_params = inspect.signature(linearize).parameters.values()
                    linearize_supports_route_instance = any(
                        parameter.kind is inspect.Parameter.VAR_POSITIONAL
                        for parameter in linearize_params
                    ) or sum(
                        parameter.kind
                        in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        }
                        for parameter in linearize_params
                    ) >= 4
                except (TypeError, ValueError):
                    linearize_supports_route_instance = True
                linearize_args = (
                    (
                        session_key,
                        execution_id,
                        seal_resolved,
                        captured_route_instance_id,
                    )
                    if linearize_supports_route_instance
                    else (session_key, execution_id, seal_resolved)
                )
                sealed_entry = await asyncio.to_thread(linearize, *linearize_args)
                sealed = sealed_entry is not None
                if sealed_entry is not None:
                    entry = sealed_entry
                    routing_revision = int(
                        getattr(sealed_entry, "routing_revision", 0)
                    )
            else:
                if binding_version == 2:
                    rejection = self._logical_binding_rejection(entry, binding)
                    if rejection is not None:
                        return finish_pre_admission(rejection)
                sealed = self._seal_occurrence(
                    execution_id,
                    session_key,
                    entry.session_id,
                    routing_revision,
                    captured_route_instance_id,
                    binding_version or 1,
                )
        except ContextualRouteInstanceMismatch as exc:
            return finish_pre_admission(ContextualCronOutcome.stale(str(exc)))
        except asyncio.CancelledError:
            finish_pre_admission(
                ContextualCronOutcome.unknown(
                    "Contextual cron admission was cancelled before enqueue."
                )
            )
            raise
        except Exception as exc:
            outcome = ContextualCronOutcome.unknown(
                f"Could not durably seal contextual cron admission: {exc}"
            )
            return finish_pre_admission(outcome)
        if not sealed:
            outcome = ContextualCronOutcome.unknown(
                "Contextual cron admission was already sealed or terminal."
            )
            return finish_pre_admission(outcome)

        item = ContextualCronQueueItem(
            job_id=str(job.get("id") or ""),
            execution_id=str(execution_id),
            prompt=prompt,
            session_key=session_key,
            admitted_session_id=entry.session_id,
            admitted_routing_revision=routing_revision,
            admitted_route_instance_id=captured_route_instance_id,
            admitted_binding_version=binding_version or 1,
            source=(
                captured_authorization_source
                if captured_authorization_source is not None
                else getattr(entry, "origin", None)
            ),
            future=future,
        )
        # Admission is durable before this append.  That ordering is the reset
        # boundary: reset-before-admission targets the new id; reset-after is stale.
        self._queues[session_key].append(item)
        if session_key not in self._drainers:
            self._drainers[session_key] = asyncio.create_task(
                self._drain(session_key),
                name=f"contextual-cron:{session_key}",
            )
        return await asyncio.shield(future)

    async def _run_queued_item(
        self, item: ContextualCronQueueItem
    ) -> ContextualCronOutcome:
        """Acquire adapter guard then resolved-session lease, revalidate, run."""
        adapter_for_source = getattr(self.runner, "_adapter_for_source", None)
        adapter = adapter_for_source(item.source) if callable(adapter_for_source) else None
        adapter_guard = None
        lease_registry = getattr(self.runner, "_turn_leases", None)
        lease_token = None
        try:
            if adapter is not None:
                acquire_guard = getattr(adapter, "acquire_contextual_cron_guard", None)
                if callable(acquire_guard):
                    pending_guard = acquire_guard(item.session_key)
                    adapter_guard = (
                        await pending_guard
                        if inspect.isawaitable(pending_guard)
                        else pending_guard
                    )
                    item.adapter = adapter
                    item.adapter_guard = adapter_guard

            if lease_registry is not None:
                lease_token = await lease_registry.acquire(
                    item.admitted_session_id,
                    owner_key=f"contextual-cron:{item.execution_id}",
                    generation=0,
                )
                if getattr(lease_token, "degraded", False):
                    outcome = ContextualCronOutcome.retryable(
                        "The live session turn lease timed out; the scheduled turn was not run."
                    )
                    await self._release_item_ownership(item)
                    return outcome
                item.turn_lease_token = lease_token
            else:
                # Lightweight providers/test doubles may lack the live registry.
                while bool(self.runner._contextual_cron_session_busy(item.session_key)):
                    await asyncio.sleep(self._busy_poll_seconds)

            while True:
                entry, rejection = self._validate_entry(
                    item.session_key,
                    admitted_session_id=item.admitted_session_id,
                    admitted_routing_revision=item.admitted_routing_revision,
                    admitted_route_instance_id=item.admitted_route_instance_id,
                    authorization_source=item.source,
                )
                if rejection is not None:
                    return rejection
                history, transcript_revision = (
                    await self.runner.async_session_store.load_transcript_with_fence_strict(
                        item.admitted_session_id
                    )
                )
                item.transcript_base_message_count = len(history)
                item.transcript_base_revision = transcript_revision
                try:
                    outcome = await self.runner._run_contextual_cron_turn(
                        item, entry, history
                    )
                except ContextualCronGuardBusy:
                    # Real adapters block in acquire_contextual_cron_guard above.
                    # Retain this bounded compatibility seam for older providers
                    # and deterministic doubles without terminally dropping the
                    # occurrence.
                    await asyncio.sleep(self._busy_poll_seconds)
                    continue
                if not isinstance(outcome, ContextualCronOutcome):
                    return ContextualCronOutcome.failure(
                        "Gateway contextual cron executor returned an invalid outcome."
                    )
                return outcome
        except BaseException:
            await self._release_item_ownership(item)
            raise

    async def _release_item_ownership(self, item: ContextualCronQueueItem) -> None:
        """Release the human-turn fence only after transcript application."""
        lease_token = item.turn_lease_token
        lease_registry = getattr(self.runner, "_turn_leases", None)
        adapter = item.adapter
        adapter_guard = item.adapter_guard
        item.turn_lease_token = None
        try:
            if (
                lease_token is not None
                and lease_registry is not None
                and not getattr(lease_token, "degraded", False)
            ):
                lease_registry.release(lease_token)
        finally:
            try:
                if adapter is not None and adapter_guard is not None:
                    release_guard = getattr(adapter, "release_contextual_cron_guard", None)
                    if callable(release_guard):
                        released = release_guard(item.session_key, adapter_guard)
                        if inspect.isawaitable(released):
                            release_task = asyncio.ensure_future(released)
                            try:
                                await asyncio.shield(release_task)
                            except asyncio.CancelledError:
                                await release_task
                                raise
            finally:
                item.adapter_guard = None
                item.adapter = None

    async def _drain(self, session_key: str) -> None:
        queue = self._queues[session_key]
        try:
            while queue:
                item = queue[0]
                try:
                    outcome = await self._run_queued_item(item)
                except (TimeoutError, ConnectionError) as exc:
                    outcome = ContextualCronOutcome.retryable(str(exc))
                except Exception as exc:
                    logger.exception("Contextual cron execution failed: %s", item.execution_id)
                    outcome = ContextualCronOutcome.failure(str(exc))

                try:
                    _, commit_rejection = self._validate_entry(
                        item.session_key,
                        admitted_session_id=item.admitted_session_id,
                        admitted_routing_revision=item.admitted_routing_revision,
                        admitted_route_instance_id=item.admitted_route_instance_id,
                        authorization_source=item.source,
                        propagate_authorization_errors=True,
                    )
                except Exception as auth_exc:
                    outcome = ContextualCronOutcome.retryable(
                        "Contextual cron authorization check failed before commit: "
                        f"{auth_exc}"
                    )
                    item.transcript_entries = None
                else:
                    if commit_rejection is not None:
                        outcome = commit_rejection
                        item.transcript_entries = None

                try:
                    persisted = self._finish_occurrence(item, outcome)
                    if inspect.isawaitable(persisted):
                        persisted = await persisted
                    if item.transcript_entries is not None:
                        if persisted is None:
                            raise RuntimeError(
                                "contextual result and transcript outbox were not persisted"
                            )
                        apply_transcript = getattr(
                            self.runner, "_apply_contextual_cron_transcript", None
                        )
                        if not callable(apply_transcript):
                            raise RuntimeError(
                                "gateway cannot apply the durable contextual transcript outbox"
                            )
                        retry_delay = self._transcript_retry_seconds
                        while True:
                            try:
                                authorized_for_apply = bool(
                                    self.runner._is_user_authorized(item.source)
                                )
                            except Exception:
                                outcome = ContextualCronOutcome.unknown(
                                    "Contextual transcript application was deferred because "
                                    "authorization could not be revalidated."
                                )
                                break
                            try:
                                if not authorized_for_apply:
                                    raise ContextualCronTranscriptConflict(
                                        "Contextual cron authorization was revoked "
                                        "before transcript application."
                                    )
                                applied = apply_transcript(item)
                                if inspect.isawaitable(applied):
                                    await applied
                                break
                            except ContextualCronTranscriptConflict as conflict:
                                from cron.executions import (
                                    get_execution,
                                    mark_contextual_transcript_conflict,
                                )

                                conflict_error = str(conflict)
                                while True:
                                    try:
                                        marked = await asyncio.to_thread(
                                            mark_contextual_transcript_conflict,
                                            item.execution_id,
                                            error=conflict_error,
                                        )
                                        if not marked:
                                            record = await asyncio.to_thread(
                                                get_execution, item.execution_id
                                            )
                                            if not record or record.get(
                                                "transcript_state"
                                            ) != "conflict":
                                                raise RuntimeError(
                                                    "contextual transcript conflict acknowledgement failed"
                                                )
                                        break
                                    except asyncio.CancelledError:
                                        raise
                                    except Exception:
                                        logger.exception(
                                            "Contextual transcript conflict acknowledgement deferred"
                                        )
                                        await asyncio.sleep(retry_delay)
                                        retry_delay = min(5.0, retry_delay * 2)
                                outcome = ContextualCronOutcome.unknown(conflict_error)
                                break
                            except asyncio.CancelledError:
                                raise
                            except Exception:
                                # Keep the owning turn fenced until the durable
                                # outbox is visible. Releasing it here would let
                                # a human turn overtake a completed hidden turn.
                                logger.exception(
                                    "Contextual transcript application deferred; retrying: %s",
                                    item.execution_id,
                                )
                                await asyncio.sleep(retry_delay)
                                retry_delay = min(5.0, retry_delay * 2)
                except Exception as exc:
                    logger.exception(
                        "Failed to persist contextual cron result: %s",
                        item.execution_id,
                    )
                    outcome = ContextualCronOutcome.unknown(
                        f"Could not durably persist contextual cron result: {exc}"
                    )

                await self._release_item_ownership(item)
                queue.popleft()
                if not item.future.done():
                    item.future.set_result(outcome)
                self._execution_futures.pop(item.execution_id, None)
        except asyncio.CancelledError:
            # Cancellation is a lane-wide shutdown boundary. Resolve the active
            # and every queued occurrence honestly, then preserve cancellation;
            # do not swallow it and continue executing later side effects.
            if queue:
                release_task = asyncio.create_task(
                    self._release_item_ownership(queue[0])
                )
                await asyncio.shield(release_task)
            unknown = ContextualCronOutcome.unknown(
                "Gateway stopped before contextual cron reached a terminal outcome."
            )
            while queue:
                pending = queue.popleft()
                if not pending.future.done():
                    pending.future.set_result(unknown)
                self._execution_futures.pop(pending.execution_id, None)
            raise
        finally:
            self._queues.pop(session_key, None)
            self._drainers.pop(session_key, None)

    def dispatch_from_scheduler(
        self,
        job: Dict[str, Any],
        *,
        execution_id: str,
        loop: asyncio.AbstractEventLoop,
        timeout: Optional[float] = None,
    ) -> ContextualCronOutcome:
        """Thread-safe blocking bridge used by the scheduler ticker thread."""
        future = asyncio.run_coroutine_threadsafe(
            self.dispatch(job, execution_id=execution_id), loop
        )
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            return ContextualCronOutcome.unknown(
                "Timed out waiting for the gateway contextual cron lane."
            )
