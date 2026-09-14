"""Metadata-only Slack Socket Mode intake observation.

This module deliberately ignores raw frame bodies.  It records only bounded,
HMAC-correlated envelope metadata and keeps Socket Mode available when the
local receipt ledger is degraded.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from gateway import slack_intake_ledger as _ledger

_MAX_RAW_FRAME_BYTES = 1024 * 1024
_DEFAULT_CONTEXT_CAP = 2048
_SAFE_DROP_REASONS = frozenset(
    {
        "unauthorized",
        "duplicate_event",
        "duplicate_ts",
        "invalid_event",
        "ignored_channel",
        "bot_message",
        "missing_mention",
        "message_deleted",
        "dm_disabled",
        "listener_cancelled",
        "exception_runtime_error",
        "exception_value_error",
        "exception_type_error",
        "exception_other",
    }
)
_EXCEPTION_REASONS = {
    RuntimeError: "exception_runtime_error",
    ValueError: "exception_value_error",
    TypeError: "exception_type_error",
}


def event_team_id(event: dict, body: Optional[dict] = None) -> str:
    """Share the adapter's event-first workspace policy with receipt intake.

    Event identity precedes the outer payload, then authorizations. Keeping
    this existing routing policy in one place avoids observing one workspace
    while dispatching another; it is not a Slack Connect topology assertion.
    """
    for payload in (event, body or {}):
        if not isinstance(payload, dict):
            continue
        team = payload.get("team_id") or payload.get("team")
        if isinstance(team, str) and team:
            return team
        if isinstance(team, dict) and team.get("id"):
            return str(team["id"])
    authorizations = (body or {}).get("authorizations") if isinstance(body, dict) else None
    for authorization in authorizations or []:
        if isinstance(authorization, dict) and authorization.get("team_id"):
            return str(authorization["team_id"])
    return ""


@dataclass(frozen=True)
class IntakeContext:
    """Pseudonymous bounded context carried across one intake lifecycle."""

    receipt_id: Optional[str]
    envelope_hash: Optional[str]
    event_hash: Optional[str]
    message_key_hash: Optional[str]
    persisted: bool
    degraded: bool
    duplicate: bool
    retry_attempt: int


@dataclass(frozen=True)
class _SyntheticTerminal:
    """Intended listener disposition ONLY, never evidence of durable acceptance."""
    state: str
    reason: Optional[str]
    related_receipt_id: Optional[str] = None
    persisted: bool = False
    failure_reason: str = "unknown_receipt"


class LedgerIntakeStore:
    """Async adapter over the durable ledger's off-event-loop APIs."""

    async def record_envelope(self, **metadata: Any) -> Any:
        return await _ledger.record_listener_received_safely(
            **metadata, stage="envelope_received"
        )

    def record_unavailable_envelope(self, **metadata: Any) -> None:
        _ledger.record_unavailable_envelope(**metadata)

    def record_unavailable_stage(
        self,
        receipt_id: Optional[str],
        *,
        stage: str,
        observed_at: float,
        reason: Optional[str] = None,
    ) -> None:
        _ledger.record_unavailable_stage(
            receipt_id, stage=stage, observed_at=observed_at, reason=reason
        )

    async def append_stage(
        self, receipt_id: Optional[str], *, stage: str, observed_at: float, reason: Optional[str] = None
    ) -> Any:
        if receipt_id is None:
            return None
        return await _ledger.append_stage_safely(
            receipt_id, stage=stage, observed_at=observed_at, reason=reason
        )

    async def mark_accepted(self, receipt_id: Optional[str], *, decided_at: float) -> Any:
        if receipt_id is None:
            return _SyntheticTerminal("accepted", None)
        outcome = await _ledger.mark_accepted_safely(
            receipt_id, decided_at=decided_at
        )
        return outcome.observation or _SyntheticTerminal("accepted", None, failure_reason=outcome.failure_reason or "persistence_failed")

    async def mark_dropped(
        self, receipt_id: Optional[str], *, reason: str, decided_at: float
    ) -> Any:
        if receipt_id is None:
            return _SyntheticTerminal("dropped", reason)
        outcome = await _ledger.mark_dropped_safely(
            receipt_id, reason=reason, decided_at=decided_at
        )
        return outcome.observation or _SyntheticTerminal("dropped", reason, failure_reason=outcome.failure_reason or "persistence_failed")


class SlackIntakeObserver:
    """Coordinate durable intake stages without retaining raw Slack IDs."""

    def __init__(
        self,
        *,
        store: Optional[Any] = None,
        clock: Callable[[], float] = time.time,
        max_contexts: int = _DEFAULT_CONTEXT_CAP,
    ) -> None:
        if not isinstance(max_contexts, int) or max_contexts < 1 or max_contexts > 10_000:
            raise ValueError("max_contexts must be between 1 and 10000")
        self._store = store if store is not None else LedgerIntakeStore()
        self._clock = clock
        self._max_contexts = max_contexts
        self._envelope_contexts: dict[str, Optional[IntakeContext]] = {}
        self._event_contexts: dict[str, Optional[IntakeContext]] = {}
        self._live_contexts: weakref.WeakValueDictionary[int, IntakeContext] = (
            weakref.WeakValueDictionary()
        )
        self._revoked_receipts: dict[str, None] = {}
        self._revocation_saturated = False
        self._stage_admission_lock = asyncio.Lock()
        self._active_listeners: dict[str, asyncio.Future] = {}

    def _remember(
        self,
        mapping: dict[str, Optional[IntakeContext]],
        key: str,
        context: IntakeContext,
    ) -> None:
        if key not in mapping or mapping[key] is not None:
            mapping[key] = context
        while len(mapping) > self._max_contexts:
            mapping.pop(next(iter(mapping)))

    def _remember_live_context(self, context: IntakeContext) -> None:
        if context.receipt_id is not None and context.message_key_hash is not None:
            self._live_contexts[id(context)] = context

    def _tombstone(self, mapping: dict[str, Optional[IntakeContext]], key: Optional[str]) -> None:
        if key is None:
            return
        mapping[key] = None
        while len(mapping) > self._max_contexts:
            mapping.pop(next(iter(mapping)))

    def _tombstone_receipt_envelopes(self, receipt_id: Optional[str]) -> None:
        if receipt_id is None:
            return
        for key, context in self._envelope_contexts.items():
            if context is not None and context.receipt_id == receipt_id:
                self._envelope_contexts[key] = None

    def _revoke_receipt(self, receipt_id: Optional[str]) -> None:
        if receipt_id is None or self._revocation_saturated:
            return
        if receipt_id in self._revoked_receipts:
            return
        if len(self._revoked_receipts) >= self._max_contexts:
            # Evicting a revoked digest would reactivate any IntakeContext
            # capability still held outside the bounded correlation maps.
            # Saturation instead closes only diagnostic stage persistence;
            # listener dispatch remains available and memory stays bounded.
            self._revocation_saturated = True
            return
        self._revoked_receipts[receipt_id] = None

    def _receipt_is_revoked(self, receipt_id: Optional[str]) -> bool:
        return self._revocation_saturated or (
            receipt_id is not None and receipt_id in self._revoked_receipts
        )

    async def _run_stage_admitted(
        self,
        operation: Callable[[], Awaitable[Any]],
        unavailable: Callable[[], Any],
    ) -> Any:
        # Diagnostic callbacks must never form an unbounded queue behind local
        # persistence. Their ``unavailable`` fallback records bounded metadata
        # and returns immediately while the admitted owner retains serialization.
        # There is no await between the lock check and acquire, so the task that
        # observes an unlocked gate owns the next admission.
        if self._stage_admission_lock.locked():
            return unavailable()
        await self._stage_admission_lock.acquire()
        try:
            owner = asyncio.create_task(operation())
            cancellation: Optional[asyncio.CancelledError] = None
            while not owner.done():
                try:
                    await asyncio.shield(owner)
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
            try:
                result = owner.result()
            except BaseException:
                if cancellation is not None:
                    raise cancellation from None
                raise
            if cancellation is not None:
                raise cancellation
            return result
        finally:
            self._stage_admission_lock.release()

    def _remember_envelope(self, key: str, context: IntakeContext) -> None:
        existing = self._envelope_contexts.get(key)
        if key not in self._envelope_contexts or (
            existing is not None and existing.receipt_id == context.receipt_id
        ):
            self._envelope_contexts[key] = context
        elif existing is not None:
            # Socket acknowledgements carry only the envelope ID. If separate
            # workspaces reuse one ID, neither receipt may safely own the ack.
            self._envelope_contexts[key] = None
        while len(self._envelope_contexts) > self._max_contexts:
            self._envelope_contexts.pop(next(iter(self._envelope_contexts)))

    async def envelope_received(
        self,
        *,
        workspace_id: str,
        envelope_id: str,
        event_id: str,
        event_type: str,
        channel_id: str,
        thread_id: Optional[str],
        message_id: str,
        retry_attempt: int = 0,
    ) -> IntakeContext:
        self._revoke_known_identity_conflict(
            workspace_id=workspace_id,
            event_id=event_id,
            channel_id=channel_id,
            message_id=message_id,
        )
        # Declare a known identity conflict before admission so captured stage
        # capabilities fail closed without queueing behind its persistence.
        return await self._run_stage_admitted(
            lambda: self._envelope_received_admitted(
                workspace_id=workspace_id,
                envelope_id=envelope_id,
                event_id=event_id,
                event_type=event_type,
                channel_id=channel_id,
                thread_id=thread_id,
                message_id=message_id,
                retry_attempt=retry_attempt,
            ),
            lambda: self._record_unavailable_envelope(
                workspace_id=workspace_id,
                envelope_id=envelope_id,
                event_id=event_id,
                event_type=event_type,
                channel_id=channel_id,
                thread_id=thread_id,
                message_id=message_id,
                retry_attempt=retry_attempt,
            ),
        )

    def _revoke_known_identity_conflict(
        self,
        *,
        workspace_id: str,
        event_id: str,
        channel_id: str,
        message_id: str,
    ) -> None:
        try:
            for name, value in (
                ("workspace_id", workspace_id),
                ("event_id", event_id),
                ("channel_id", channel_id),
                ("message_id", message_id),
            ):
                _ledger._validate_identifier(name, value)
            event_hash = _ledger._cached_digest(
                "slack-event", workspace_id + "\x1f" + event_id
            )
            message_hash = _ledger._cached_digest(
                "slack-message",
                workspace_id + "\x1f" + channel_id + "\x1f" + message_id,
            )
        except (OSError, TypeError, ValueError):
            return
        if event_hash is None or message_hash is None:
            return
        existing = self._event_contexts.get(event_hash)
        if existing is None:
            existing = next(
                (
                    context
                    for context in self._envelope_contexts.values()
                    if context is not None and context.receipt_id == event_hash
                ),
                None,
            )
        if existing is None:
            existing = next(
                (
                    context
                    for context in self._live_contexts.values()
                    if context.receipt_id == event_hash
                ),
                None,
            )
        if existing is None or existing.message_key_hash == message_hash:
            return
        self._revoke_receipt(existing.receipt_id)
        self._tombstone_receipt_envelopes(existing.receipt_id)
        self._tombstone(self._event_contexts, event_hash)

    def _record_unavailable_envelope(
        self,
        *,
        workspace_id: str,
        envelope_id: str,
        event_id: str,
        event_type: str,
        channel_id: str,
        thread_id: Optional[str],
        message_id: str,
        retry_attempt: int,
    ) -> IntakeContext:
        self._store.record_unavailable_envelope(
            workspace_id=workspace_id,
            event_id=event_id,
            transport_id=envelope_id,
            event_type=event_type,
            channel_id=channel_id,
            thread_id=thread_id,
            message_id=message_id,
            received_at=float(self._clock()),
        )
        try:
            for name, value in (
                ("workspace_id", workspace_id),
                ("event_id", event_id),
                ("envelope_id", envelope_id),
                ("channel_id", channel_id),
                ("message_id", message_id),
            ):
                _ledger._validate_identifier(name, value)
            receipt_id = _ledger._fallback_receipt_id(workspace_id, event_id)
            envelope_hash = _ledger._cached_digest(
                "slack-transport", workspace_id + "\x1f" + envelope_id
            )
            envelope_lookup_hash = _ledger._fallback_digest("transport", envelope_id)
            message_key_hash = _ledger._cached_digest(
                "slack-message",
                workspace_id + "\x1f" + channel_id + "\x1f" + message_id,
            )
        except (OSError, TypeError, ValueError):
            receipt_id = None
            envelope_hash = None
            envelope_lookup_hash = None
            message_key_hash = None
        context = IntakeContext(
            receipt_id=receipt_id,
            envelope_hash=envelope_hash,
            event_hash=receipt_id,
            message_key_hash=message_key_hash,
            persisted=False,
            degraded=True,
            duplicate=False,
            retry_attempt=max(0, min(int(retry_attempt), 1000)),
        )
        self._remember_live_context(context)
        if envelope_lookup_hash is not None:
            self._remember_envelope(envelope_lookup_hash, context)
        if receipt_id is not None:
            self._remember(self._event_contexts, receipt_id, context)
        return context

    async def _envelope_received_admitted(
        self,
        *,
        workspace_id: str,
        envelope_id: str,
        event_id: str,
        event_type: str,
        channel_id: str,
        thread_id: Optional[str],
        message_id: str,
        retry_attempt: int = 0,
    ) -> IntakeContext:
        observed_at = float(self._clock())
        outcome = await self._store.record_envelope(
            workspace_id=workspace_id,
            event_id=event_id,
            transport_id=envelope_id,
            event_type=event_type,
            channel_id=channel_id,
            thread_id=thread_id,
            message_id=message_id,
            received_at=observed_at,
        )
        observation = getattr(outcome, "observation", None)
        receipt_id = getattr(observation, "receipt_id", None)
        envelope_hash = getattr(observation, "transport_hash", None)
        envelope_lookup_hash = getattr(observation, "transport_lookup_hash", None)
        event_hash = receipt_id
        message_key_hash = getattr(observation, "message_key_hash", None)
        failure_reason = getattr(outcome, "failure_reason", None)
        if observation is None and failure_reason not in {
            "identity_conflict",
            "work_deadline_exceeded",
        }:
            try:
                for name, value in (
                    ("workspace_id", workspace_id),
                    ("event_id", event_id),
                    ("envelope_id", envelope_id),
                    ("channel_id", channel_id),
                    ("message_id", message_id),
                ):
                    _ledger._validate_identifier(name, value)
                receipt_id = _ledger._fallback_receipt_id(workspace_id, event_id)
                envelope_hash = _ledger._cached_digest(
                    "slack-transport", workspace_id + "\x1f" + envelope_id
                )
                envelope_lookup_hash = _ledger._fallback_digest(
                    "transport", envelope_id
                )
                event_hash = receipt_id
                message_key_hash = _ledger._cached_digest(
                    "slack-message",
                    workspace_id + "\x1f" + channel_id + "\x1f" + message_id,
                )
            except (OSError, TypeError, ValueError):
                receipt_id = None
                envelope_hash = None
                envelope_lookup_hash = None
                event_hash = None
                message_key_hash = None
        context = IntakeContext(
            receipt_id=receipt_id,
            envelope_hash=envelope_hash,
            event_hash=event_hash,
            message_key_hash=message_key_hash,
            persisted=bool(getattr(outcome, "persisted", False)),
            degraded=not bool(getattr(outcome, "persisted", False)),
            duplicate=bool(getattr(observation, "duplicate", False)),
            retry_attempt=max(0, min(int(retry_attempt), 1000)),
        )
        self._remember_live_context(context)
        if observation is None and failure_reason in {
            "identity_conflict",
            "work_deadline_exceeded",
        }:
            try:
                _ledger._validate_identifier("workspace_id", workspace_id)
                _ledger._validate_identifier("event_id", event_id)
                conflicting_event_hash = _ledger._cached_digest(
                    "slack-event", workspace_id + "\x1f" + event_id
                )
            except (OSError, TypeError, ValueError):
                conflicting_event_hash = None
            # The workspace-scoped event digest is also the deterministic
            # receipt ID. Use it directly: the bounded event map may have
            # evicted its context while an envelope context still survives.
            # A work deadline is an unknown durable outcome because a running
            # SQLite/fsync call can settle after the bounded caller returns.
            # Revoke conservatively before releasing stage admission so that
            # a late identity conflict cannot race stale captured contexts.
            self._revoke_receipt(conflicting_event_hash)
            self._tombstone_receipt_envelopes(conflicting_event_hash)
            self._tombstone(
                self._envelope_contexts,
                _ledger._fallback_digest("transport", envelope_id),
            )
            self._tombstone(self._event_contexts, conflicting_event_hash)
        if envelope_lookup_hash is not None:
            if (
                event_hash is not None
                and event_hash in self._event_contexts
                and self._event_contexts[event_hash] is None
            ):
                self._tombstone(self._envelope_contexts, envelope_lookup_hash)
            else:
                self._remember_envelope(envelope_lookup_hash, context)
        if event_hash is not None:
            self._remember(self._event_contexts, event_hash, context)
        return context

    def context_for_envelope(self, envelope_id: Any) -> Optional[IntakeContext]:
        try:
            digest = _ledger._fallback_digest("transport", envelope_id)
        except (OSError, TypeError, ValueError):
            return None
        return self._envelope_contexts.get(digest)

    def context_for_event(self, event_id: Any, *, workspace_id: Any = None) -> Optional[IntakeContext]:
        try:
            _ledger._validate_identifier("workspace_id", workspace_id)
            _ledger._validate_identifier("event_id", event_id)
            digest = _ledger._cached_digest("slack-event", workspace_id + "\x1f" + event_id)
        except (OSError, TypeError, ValueError):
            return None
        return self._event_contexts.get(digest)

    async def _append(
        self, context: IntakeContext, stage: str, *, reason: Optional[str] = None
    ) -> Any:
        return await self._run_stage_admitted(
            lambda: self._append_admitted(context, stage, reason=reason),
            lambda: self._record_unavailable_stage(context, stage, reason=reason),
        )

    def _record_unavailable_stage(
        self, context: IntakeContext, stage: str, *, reason: Optional[str] = None
    ) -> _SyntheticTerminal:
        if self._receipt_is_revoked(context.receipt_id):
            return _SyntheticTerminal(
                stage, reason, failure_reason="identity_conflict"
            )
        self._store.record_unavailable_stage(
            context.receipt_id,
            stage=stage,
            observed_at=float(self._clock()),
            reason=reason,
        )
        return _SyntheticTerminal(
            stage, reason, failure_reason="admission_busy"
        )

    async def _append_admitted(
        self, context: IntakeContext, stage: str, *, reason: Optional[str] = None
    ) -> Any:
        if self._receipt_is_revoked(context.receipt_id):
            return _SyntheticTerminal(
                stage, reason, failure_reason="identity_conflict"
            )
        return await self._store.append_stage(
            context.receipt_id,
            stage=stage,
            observed_at=float(self._clock()),
            **({"reason": reason} if reason is not None else {}),
        )

    async def acknowledged(self, context: IntakeContext) -> Any:
        return await self._append(context, "acknowledged")

    async def acknowledged_for_envelope(self, envelope_id: Any) -> None:
        context = self.context_for_envelope(envelope_id)
        if context is not None:
            try:
                await self.acknowledged(context)
            except Exception as exc:
                _ledger._note_persistence_failed(_ledger._classify_persistence_failure(exc), exc)

    async def listener_entered(self, context: IntakeContext) -> Any:
        return await self._append(context, "listener_entered")

    async def accepted(self, context: IntakeContext) -> Any:
        return await self._run_stage_admitted(
            lambda: self._accepted_admitted(context),
            lambda: self._record_unavailable_stage(context, "accepted"),
        )

    async def _accepted_admitted(self, context: IntakeContext) -> Any:
        if self._receipt_is_revoked(context.receipt_id):
            return _SyntheticTerminal(
                "accepted", None, failure_reason="identity_conflict"
            )
        return await self._store.mark_accepted(
            context.receipt_id, decided_at=float(self._clock())
        )

    async def dropped(self, context: IntakeContext, *, reason: str) -> Any:
        if not isinstance(reason, str) or reason not in _SAFE_DROP_REASONS:
            raise ValueError("drop reason is not in the fixed safe vocabulary")
        return await self._run_stage_admitted(
            lambda: self._dropped_admitted(context, reason=reason),
            lambda: self._record_unavailable_stage(context, "dropped", reason=reason),
        )

    async def _dropped_admitted(
        self, context: IntakeContext, *, reason: str
    ) -> Any:
        if self._receipt_is_revoked(context.receipt_id):
            return _SyntheticTerminal(
                "dropped", reason, failure_reason="identity_conflict"
            )
        return await self._store.mark_dropped(
            context.receipt_id, reason=reason, decided_at=float(self._clock())
        )

    async def run_listener(
        self, context: IntakeContext, listener: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        # Multiple envelopes can share one event receipt. Only its active
        # callback owns terminal disposition; cancelling a retry must not drop
        # the owner's receipt or cancel the owner's work. No task is spawned.
        receipt_id = context.receipt_id
        if receipt_id is None:
            return await self._run_owned_listener(context, listener, *args, **kwargs)
        active = self._active_listeners.get(receipt_id)
        if active is not None:
            result, error = await asyncio.shield(active)
            if error is not None:
                raise error
            return result
        active = asyncio.get_running_loop().create_future()
        self._active_listeners[receipt_id] = active
        try:
            result = await self._run_owned_listener(context, listener, *args, **kwargs)
        except BaseException as exc:
            # Store failure as a result so an owner without retries cannot
            # produce an unhandled-future-exception warning.
            active.set_result((None, exc))
            raise
        else:
            active.set_result((result, None))
            return result
        finally:
            del self._active_listeners[receipt_id]

    async def _run_owned_listener(
        self, context: IntakeContext, listener: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        try:
            try:
                await self.listener_entered(context)
            except Exception as exc:
                _ledger._note_persistence_failed("persistence_failed", exc)
            result = await listener(*args, **kwargs)
            if result is not None and (
                not isinstance(result, str) or result not in _SAFE_DROP_REASONS
            ):
                raise TypeError("listener result is not a fixed drop reason")
        except asyncio.CancelledError:
            await self._terminalize_without_replacing(context, "listener_cancelled")
            raise
        except Exception as exc:
            reason = next(
                (safe for kind, safe in _EXCEPTION_REASONS.items() if isinstance(exc, kind)),
                "exception_other",
            )
            await self._terminalize_without_replacing(context, reason)
            raise
        if result is not None:
            return await self.dropped(context, reason=result)
        return await self.accepted(context)

    async def _terminalize_without_replacing(self, context: IntakeContext, reason: str) -> None:
        try:
            await self.dropped(context, reason=reason)
        except (Exception, asyncio.CancelledError) as exc:
            # This method is called only while preserving an original listener
            # exception/cancellation. Secondary cancellation cannot replace it.
            _ledger._note_persistence_failed("persistence_failed", exc)

    async def observe_socket_message(self, message: Any, raw_message: Any) -> None:
        """Best-effort SDK callback; malformed frames never block Bolt dispatch."""
        try:
            if not isinstance(raw_message, str) or len(raw_message.encode("utf-8")) > _MAX_RAW_FRAME_BYTES:
                return
            if not isinstance(message, dict):
                return
            payload = message.get("payload")
            if not isinstance(payload, dict):
                return
            event = payload.get("event")
            if not isinstance(event, dict):
                return
            envelope_id = message.get("envelope_id")
            event_id = payload.get("event_id")
            workspace_id = event_team_id(event, payload)
            event_type = event.get("type")
            if event_type not in {"message", "app_mention"}:
                return
            if event.get("subtype") == "message_changed" and isinstance(event.get("message"), dict):
                changed = event["message"]
                channel_id = changed.get("channel") or event.get("channel")
                message_id = changed.get("ts")
                thread_id = changed.get("thread_ts")
            else:
                channel_id = event.get("channel")
                message_id = event.get("ts") or event.get("event_ts")
                thread_id = event.get("thread_ts")
            required = (workspace_id, envelope_id, event_id, event_type, channel_id, message_id)
            if not all(isinstance(value, str) and value for value in required):
                return
            retry_attempt = message.get("retry_attempt", payload.get("retry_attempt", 0))
            if not isinstance(retry_attempt, int) or isinstance(retry_attempt, bool):
                retry_attempt = 0
            await self.envelope_received(
                workspace_id=workspace_id,
                envelope_id=envelope_id,
                event_id=event_id,
                event_type=event_type,
                channel_id=channel_id,
                thread_id=thread_id if isinstance(thread_id, str) and thread_id else None,
                message_id=message_id,
                retry_attempt=retry_attempt,
            )
        except Exception as exc:
            _ledger._note_persistence_failed(_ledger._classify_persistence_failure(exc), exc)


def install_socket_observer(client: Any, observer: SlackIntakeObserver) -> None:
    """Install the exact slack_sdk 3.43.0 callback seam and post-send ack hook."""

    async def _message_listener(_client: Any, message: dict, raw_message: str) -> None:
        await observer.observe_socket_message(message, raw_message)

    listeners = getattr(client, "message_listeners")
    listeners.insert(0, _message_listener)
    original_send = client.send_socket_mode_response

    async def _send_socket_mode_response(response: Any) -> Any:
        result = await original_send(response)
        try:
            envelope_id = response.get("envelope_id") if isinstance(response, dict) else getattr(response, "envelope_id", None)
            await observer.acknowledged_for_envelope(envelope_id)
        except Exception as exc:
            _ledger._note_persistence_failed("persistence_failed", exc)
        return result

    client.send_socket_mode_response = _send_socket_mode_response
