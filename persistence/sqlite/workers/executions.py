"""The worker execution repository on SQLite: ``WorkerExecutionRepository`` and its six sibling worker protocols on
one object, registered as the store's ``worker_executions``.

Each verb builds the wire payload of its operation from its typed parameters, runs its transaction body
inside :func:`refused`, and converts the stored result to the contract record. Every receipted verb goes through the
one receipt transaction of :mod:`._receipt`; a verb whose payload carries open keyword keys checks that key set before
the store, so a malformed set answers ``invalid_params`` whatever the epoch.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional, Sequence, Unpack

from persistence.authority.admissions import AdmissionId
from persistence.authority.vocabulary import OwnerRule
from persistence.conversation.messages import ConversationPeer
from persistence.conversation.sessions import SessionLifecycleStatus, TitleSource
from persistence.core.errors import AuthorityRefusal, CompressionPublishRefused
from persistence.core.fences import AuthorityEpoch, ExecutionGeneration
from persistence.core.identity import JsonValue, MessageId, SessionId
from persistence.workers.compaction import (
    WorkerArchiveResult, WorkerCompaction, WorkerCompressionPublication, WorkerPublishResult,
)
from persistence.workers.compression import WorkerCooldown, WorkerCooldownSnapshot
from persistence.workers.compression_reads import WorkerLineageView
from persistence.workers.delegations import (
    WorkerDelegationChild, WorkerDelegationClaim, WorkerDelegationCompletion, WorkerDelegationDispatch,
    WorkerDelegationPayload, WorkerDelegationView, delegation_payload,
)
from persistence.workers.executions import (
    AdmissionWorkerScope, ChildExecution, ChildReservation, WorkerAuxiliaryUsage, WorkerExecutionId,
    WorkerExecutionRecord, WorkerKind, WorkerMainUsage, WorkerStatus, WorkerTurnLease,
)
from persistence.workers.sessions import (
    RowAnnotation, ToolPin, TranscriptAppendOptions, WorkerActivity, WorkerAppendResult, WorkerBillingRoute,
    WorkerMessage, WorkerSessionCreate, WorkerTitleResult,
)

from .._backend import SQLiteStore, register_repository
from ..authority._runtime import _worker_public, refused
from . import _gateway
from ._compression import worker_claim_origin, worker_retry_target
from ._receipt import mutate_worker_execution

__all__ = ["SQLiteWorkerExecutionRepository"]

_MAIN_USAGE_KEYS = WorkerMainUsage.__required_keys__ | WorkerMainUsage.__optional_keys__
_AUXILIARY_USAGE_KEYS = WorkerAuxiliaryUsage.__required_keys__ | WorkerAuxiliaryUsage.__optional_keys__
_CREATE_KEYS = WorkerSessionCreate.__required_keys__ | WorkerSessionCreate.__optional_keys__
_APPEND_OPTION_KEYS = TranscriptAppendOptions.__required_keys__ | TranscriptAppendOptions.__optional_keys__


def _key_set(keys: Any, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> None:
    """A keyword payload's key set, checked before the store."""
    if set(keys) - allowed or required - set(keys):
        raise AuthorityRefusal('invalid_params')


def _record(public: Mapping[str, Any]) -> WorkerExecutionRecord:
    return WorkerExecutionRecord(
        execution_id=WorkerExecutionId(public['execution_id']), session_id=SessionId(public['session_id']),
        kind=WorkerKind(public['kind']), owner_epoch=AuthorityEpoch(public['owner_epoch']),
        generation=ExecutionGeneration(public['generation']), status=WorkerStatus(public['status']),
        last_sequence=public['last_sequence'],
    )


def _messages(messages: Sequence[Any]) -> list[Any]:
    return [dict(message) if isinstance(message, Mapping) else message for message in messages]


def _annotations(result: Mapping[str, Any]) -> tuple[RowAnnotation, ...]:
    return tuple(RowAnnotation.from_wire(annotation) for annotation in result['annotations'])


class SQLiteWorkerExecutionRepository:
    """See ``persistence.workers.executions.WorkerExecutionRepository`` and its six sibling protocols."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def _persist(self, operation: str, payload: Any, *, epoch: Any, execution_id: Any, session_id: Any,
                 generation: Any, sequence: Any) -> Any:
        with refused():
            return mutate_worker_execution(self._store.holder(), epoch=epoch, execution_id=execution_id,
                                           session_id=session_id, generation=generation, sequence=sequence,
                                           operation=operation, payload=payload)

    def _value(self, operation: str, payload: Any, epoch: Any, execution_id: Any, session_id: Any, generation: Any,
               sequence: Any) -> Any:
        return self._persist(operation, payload, epoch=epoch, execution_id=execution_id, session_id=session_id,
                             generation=generation, sequence=sequence)['value']

    def _ledger(self, operation: str, payload: Any, epoch: Any, execution_id: Any, session_id: Any, generation: Any,
                sequence: Any, worker_pid: Any, worker_birth: Any, owner_started_at: int | None = None,
                probe_refused: str | None = None) -> Any:
        with refused():
            return _gateway.persist_worker_delegation(
                self._store.holder(), epoch=epoch, execution_id=execution_id, session_id=session_id,
                generation=generation, sequence=sequence, operation=operation, payload=payload, worker_pid=worker_pid,
                worker_birth=worker_birth, owner_started_at=owner_started_at, probe_refused=probe_refused)['value']

    # ── WorkerExecutionRepository ─────────────────────────────────────────────

    def register(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, kind: WorkerKind, adoption_secret: str, require_idle: bool = False) -> WorkerExecutionRecord:
        from ._executions import register_worker_execution
        with refused():
            row = register_worker_execution(self._store.holder(), epoch=epoch, execution_id=execution_id,
                                            session_id=session_id, generation=generation,
                                            kind=getattr(kind, 'value', kind), adoption_secret=adoption_secret,
                                            require_idle=require_idle)
        return _record(row)

    def adopt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        from ._executions import adopt_worker_execution
        with refused():
            row = adopt_worker_execution(self._store.holder(), epoch=epoch, execution_id=execution_id,
                                         session_id=session_id, generation=generation, adoption_secret=adoption_secret)
        return _record(row)

    def reserve_for_admission(self, *, epoch: AuthorityEpoch, admission_id: AdmissionId, principal_id: str, profile_id: str, pid: int, birth: float, secret: str,
                              owner: OwnerRule | None, worker_subject: str | None = None) -> AdmissionWorkerScope:
        with refused():
            scope = _gateway.reserve_admission_worker(self._store.holder(), epoch=epoch, admission_id=admission_id,
                                                      principal_id=principal_id, profile_id=profile_id, pid=pid,
                                                      birth=birth, secret=secret, rule=owner,
                                                      worker_subject=worker_subject)
        return AdmissionWorkerScope(
            profile_id=scope['profile_id'], session_id=SessionId(scope['session_id']),
            execution_id=WorkerExecutionId(scope['execution_id']), generation=ExecutionGeneration(scope['generation']),
            pid=scope['pid'], birth=scope['birth'], secret=scope['secret'], epoch=AuthorityEpoch(scope['epoch']))

    def lose_reservation(self, *, epoch: AuthorityEpoch, admission_id: AdmissionId, execution_id: WorkerExecutionId, generation: ExecutionGeneration) -> None:
        with refused():
            _gateway.lose_admission_worker(self._store.holder(), {'admission_id': admission_id}, {
                'epoch': epoch, 'execution_id': execution_id, 'generation': generation})

    def reserve_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, secret: str | WorkerDelegationPayload, profile_id: str, pid: int, birth: float, principal: str, max_concurrent_children: int, max_spawn_depth: int, refused: str | None = None) -> ChildReservation:
        from ..authority._runtime import refused as refusals
        payload = secret.payload if isinstance(secret, WorkerDelegationPayload) else {'secret': secret}
        with refusals():
            result = _gateway.reserve_child(
                self._store.holder(), epoch=epoch, execution_id=execution_id, session_id=session_id,
                generation=generation, sequence=sequence, payload=payload,
                identity={'profile_id': profile_id, 'pid': pid, 'birth': birth}, principal=principal,
                limits=(max_concurrent_children, max_spawn_depth), refused=refused)
        child = result.get('child')
        if child is None:
            return ChildReservation(refused=result['refused'])
        return ChildReservation(child=ChildExecution(
            session_id=SessionId(child['session_id']), execution_id=WorkerExecutionId(child['execution_id']),
            generation=ExecutionGeneration(child['generation'])))

    def persist_operation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        try:
            return self._persist(operation, payload, epoch=epoch, execution_id=execution_id, session_id=session_id,  # type: ignore[no-any-return]
                                 generation=generation, sequence=sequence)
        except RuntimeError as exc:
            # The publication body's precondition refusals are plain RuntimeErrors; its typed refusals subclass it.
            if operation != 'compression.publish' or type(exc) is not RuntimeError:
                raise
            raise CompressionPublishRefused(session_id, str(exc)) from exc

    def finish(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._persist('execution.finish', {}, epoch=epoch, execution_id=execution_id, session_id=session_id,
                      generation=generation, sequence=sequence)

    def record_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **usage: Unpack[WorkerMainUsage]) -> None:
        _key_set(usage, _MAIN_USAGE_KEYS)
        self._value('usage.main', dict(usage), epoch, execution_id, session_id, generation, sequence)

    def record_auxiliary_usage(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, task: str, **usage: Unpack[WorkerAuxiliaryUsage]) -> None:
        _key_set(usage, _AUXILIARY_USAGE_KEYS)
        self._value('usage.auxiliary', dict(usage, task=task), epoch, execution_id, session_id, generation, sequence)

    def acquire_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('turn.acquire', payload, epoch, execution_id, session_id, generation, sequence))

    def renew_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('turn.renew', payload, epoch, execution_id, session_id, generation, sequence))

    def cleanup_turn_lease(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        self._persist('turn.cleanup', {'target': target_session_id, 'holder': holder}, epoch=epoch,
                      execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)

    def verify_claim(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, adoption_secret: str) -> WorkerExecutionRecord:
        with refused():
            row = _gateway.verify_claim(self._store.holder(), {'execution_id': execution_id, 'session_id': session_id,
                                                                'generation': generation}, adoption_secret)
        return _record(_worker_public(row))

    def execution_session(self, *, execution_id: WorkerExecutionId) -> SessionId | None:
        with refused():
            session_id = _gateway.execution_session(self._store.holder(), execution_id)
        return None if session_id is None else SessionId(session_id)

    def has_live_worker(self) -> bool:
        with refused():
            return _gateway.has_live_worker(self._store.holder())

    # ── WorkerSessionVerbs ────────────────────────────────────────────────────

    def set_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, title: str | None, source: TitleSource) -> WorkerTitleResult:
        result = self._persist('session.title', {'title': title, 'source': getattr(source, 'value', source)}, epoch=epoch,
                               execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)
        return WorkerTitleResult(value=bool(result['value']), error=result.get('error'),
                                 title_candidate=result.get('title_candidate'))

    def set_session_title_source(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, source: TitleSource) -> bool:
        return bool(self._value('session.title_source', {'source': getattr(source, 'value', source)}, epoch,
                                execution_id, session_id, generation, sequence))

    def next_session_title(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, base_title: str) -> str:
        return str(self._value('session.next_title', {'base_title': base_title}, epoch, execution_id, session_id,
                               generation, sequence))

    def touch_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, activity: WorkerActivity) -> None:
        payload = {'ts': activity.ts, 'description': activity.description, 'provenance': activity.provenance}
        self._value('session.activity', payload, epoch, execution_id, session_id, generation, sequence)

    def clear_session_activity(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._value('session.activity_clear', {}, epoch, execution_id, session_id, generation, sequence)

    def set_session_billing_route(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, route: WorkerBillingRoute) -> None:
        payload = {'provider': route.provider, 'base_url': route.base_url, 'billing_mode': route.billing_mode}
        self._value('session.billing_route', payload, epoch, execution_id, session_id, generation, sequence)

    def set_latest_user_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, content: JsonValue, api_content: str) -> int:
        return int(self._value('session.api_content', {'content': content, 'api_content': api_content}, epoch,
                               execution_id, session_id, generation, sequence))

    def set_message_api_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, api_content: str) -> int:
        payload = {'row_id': row_id, 'content': content, 'api_content': api_content}
        return int(self._value('session.message_api_content', payload, epoch, execution_id, session_id, generation,
                               sequence))

    def set_user_message_content(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: MessageId, content: JsonValue, display_metadata: Optional[Mapping[str, JsonValue]] = None) -> int:
        payload: dict[str, Any] = {'row_id': row_id, 'content': content}
        if display_metadata is not None:
            payload['display_metadata'] = dict(display_metadata)
        return int(self._value('session.user_content', payload, epoch, execution_id, session_id, generation, sequence))

    def create_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, **fields: Unpack[WorkerSessionCreate]) -> SessionId:
        _key_set(fields, _CREATE_KEYS, frozenset({'source'}))
        return SessionId(self._value('session.create', dict(fields), epoch, execution_id, session_id, generation,
                                     sequence))

    def end_session(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, end_reason: str) -> None:
        self._value('session.end', {'end_reason': end_reason}, epoch, execution_id, session_id, generation, sequence)

    def session_lifecycle(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> SessionLifecycleStatus:
        return SessionLifecycleStatus(self._value('session.lifecycle', {}, epoch, execution_id, session_id, generation,
                                                  sequence))

    def set_session_system_prompt(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, system_prompt: str | None) -> None:
        self._value('session.prompt', {'system_prompt': system_prompt}, epoch, execution_id, session_id, generation,
                    sequence)

    def patch_session_sidecars(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, patch: Mapping[str, JsonValue]) -> None:
        payload = {'patch': dict(patch) if isinstance(patch, Mapping) else patch}
        self._value('session.sidecars', payload, epoch, execution_id, session_id, generation, sequence)

    def set_session_tool_pin(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, tool_pin: ToolPin | None) -> None:
        payload = {'tool_names': dict(tool_pin) if isinstance(tool_pin, Mapping) else tool_pin}
        self._value('session.tools', payload, epoch, execution_id, session_id, generation, sequence)

    def latest_conversation_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        return self._value('session.tail_role', {}, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def message_role(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, row_id: int) -> str | None:
        return self._value('session.row_role', {'row_id': row_id}, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def append_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], **options: Unpack[TranscriptAppendOptions]) -> WorkerAppendResult:
        _key_set(options, _APPEND_OPTION_KEYS)
        result = self._persist('transcript.append', {'messages': _messages(messages), **options}, epoch=epoch,
                               execution_id=execution_id, session_id=session_id, generation=generation, sequence=sequence)
        return WorkerAppendResult(count=result['count'], annotations=_annotations(result))

    # ── WorkerCompressionRepository ───────────────────────────────────────────

    def acquire_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('compression.lock.acquire', payload, epoch, execution_id, session_id, generation, sequence))

    def renew_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, lease: WorkerTurnLease) -> bool:
        payload = {'holder': lease.holder, 'ttl_seconds': lease.ttl_seconds}
        return bool(self._value('compression.lock.renew', payload, epoch, execution_id, session_id, generation, sequence))

    def compression_lock_holder(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> str | None:
        return self._value('compression.lock.holder', {}, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    def cleanup_compression_lock(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, holder: str) -> None:
        self._value('compression.cleanup', {'target': target_session_id, 'holder': holder}, epoch, execution_id,
                    session_id, generation, sequence)

    def record_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, cooldown: WorkerCooldown) -> None:
        payload = {'cooldown_until': cooldown.cooldown_until, 'error': cooldown.error}
        self._value('compression.cooldown.record', payload, epoch, execution_id, session_id, generation, sequence)

    def restore_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, snapshot: WorkerCooldownSnapshot) -> None:
        payload = {'snapshot': {'session_exists': snapshot.session_exists, 'cooldown_until': snapshot.cooldown_until,
                                'error': snapshot.error}}
        self._value('compression.cooldown.restore', payload, epoch, execution_id, session_id, generation, sequence)

    def clear_compression_cooldown(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> None:
        self._value('compression.cooldown.clear', {}, epoch, execution_id, session_id, generation, sequence)

    def set_compression_fallback_streak(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, streak: int) -> None:
        self._value('compression.fallback', {'value': streak}, epoch, execution_id, session_id, generation, sequence)

    def set_compression_ineffective_count(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, count: int) -> None:
        self._value('compression.ineffective', {'value': count}, epoch, execution_id, session_id, generation, sequence)

    def set_compression_recovery_deadline(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, deadline: float | None) -> None:
        self._value('compression.recovery', {'value': deadline}, epoch, execution_id, session_id, generation, sequence)

    def compression_watermark(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> int:
        return int(self._value('compression.watermark', {}, epoch, execution_id, session_id, generation, sequence))

    def reopen_orphaned_compression(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int) -> bool:
        return bool(self._value('compression.reopen', {}, epoch, execution_id, session_id, generation, sequence))

    def compression_boundary(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, peer: ConversationPeer) -> int | None:
        payload = {'session_key': peer.session_key, 'source': peer.source}
        return self._value('compression.boundary', payload, epoch, execution_id, session_id, generation, sequence)  # type: ignore[no-any-return]

    # ── WorkerCompressionReads ────────────────────────────────────────────────

    def compression_session_context(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> Mapping[str, JsonValue]:
        return self._persist('compression.context', {'target': target_session_id}, epoch=epoch, execution_id=execution_id,  # type: ignore[no-any-return]
                             session_id=session_id, generation=generation, sequence=sequence)['session']

    def compression_lineage_view(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId) -> WorkerLineageView:
        view = self._persist('compression.lineage', {'target': target_session_id}, epoch=epoch, execution_id=execution_id,
                             session_id=session_id, generation=generation, sequence=sequence)
        return WorkerLineageView(
            readable_ids=tuple(SessionId(s) for s in view['readable_ids']), lineage=tuple(SessionId(s) for s in view['lineage']),
            root=SessionId(view['root']), tip=SessionId(view['tip']), resume=SessionId(view['resume']),
            identity=(bool(view['identity'][0]), str(view['identity'][1])))

    def compression_history(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, target_session_id: SessionId, include_ancestors: bool, include_inactive: bool, repair_alternation: bool, include_row_ids: bool, include_compacted: bool) -> tuple[Mapping[str, JsonValue], ...]:
        payload = {'target': target_session_id, 'include_ancestors': include_ancestors, 'include_inactive': include_inactive,
                   'repair_alternation': repair_alternation, 'include_row_ids': include_row_ids,
                   'include_compacted': include_compacted}
        return tuple(self._persist('compression.history', payload, epoch=epoch, execution_id=execution_id,
                                   session_id=session_id, generation=generation, sequence=sequence)['messages'])

    # ── WorkerCompactionVerbs ─────────────────────────────────────────────────

    def compact_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, compaction: WorkerCompaction) -> WorkerArchiveResult:
        result = self._persist('compression.archive', compaction.payload(), epoch=epoch, execution_id=execution_id,
                               session_id=session_id, generation=generation, sequence=sequence)
        return WorkerArchiveResult(value=int(result['value']), annotations=_annotations(result))

    def publish_compression_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, publication: WorkerCompressionPublication) -> WorkerPublishResult:
        result = self.persist_operation(epoch=epoch, execution_id=execution_id, session_id=session_id,
                                        generation=generation, sequence=sequence, operation='compression.publish',
                                        payload=publication.payload())
        moved = result['worker_assignment']
        return WorkerPublishResult(parent=SessionId(moved['parent']), session_id=SessionId(moved['session_id']),
                                   annotations=_annotations(result))

    def append_compression_transcript(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, messages: Sequence[WorkerMessage], compression_lock_holder: str | None, turn_lease_holder: str | None, turn_lease_ttl_seconds: float) -> WorkerAppendResult:
        payload = {'messages': _messages(messages), 'compression_lock_holder': compression_lock_holder,
                   'turn_lease_holder': turn_lease_holder, 'turn_lease_ttl_seconds': turn_lease_ttl_seconds}
        result = self._persist('compression.append', payload, epoch=epoch, execution_id=execution_id,
                               session_id=session_id, generation=generation, sequence=sequence)
        return WorkerAppendResult(count=result['count'], annotations=_annotations(result))

    # ── WorkerReceiptReads ────────────────────────────────────────────────────

    def claim_origin(self, *, execution_id: WorkerExecutionId, session_id: SessionId) -> SessionId:
        with refused():
            return SessionId(worker_claim_origin(self._store.holder(), {'execution_id': execution_id,
                                                                        'session_id': session_id}))

    def publication_target(self, *, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, operation: str, payload: Mapping[str, JsonValue]) -> SessionId:
        params = {'execution_id': execution_id, 'session_id': session_id, 'generation': generation,
                  'sequence': sequence, 'operation': operation, 'payload': payload}
        with refused():
            return SessionId(worker_retry_target(self._store.holder(), params))

    # ── WorkerDelegationLedger ────────────────────────────────────────────────

    def dispatch_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, dispatch: WorkerDelegationDispatch | WorkerDelegationPayload, worker_pid: int, worker_birth: float, owner_started_at: int | None, probe_refused: Literal['permission_denied', 'worker_not_live'] | None = None) -> None:
        self._ledger('delegation.dispatch', delegation_payload(dispatch), epoch, execution_id, session_id, generation, sequence,
                     worker_pid, worker_birth, owner_started_at=owner_started_at, probe_refused=probe_refused)

    def complete_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, completion: WorkerDelegationCompletion | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger('delegation.complete', delegation_payload(completion), epoch,
                     execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def record_delegation_child(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, child: WorkerDelegationChild | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger('delegation.child', delegation_payload(child), epoch,
                     execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def read_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> WorkerDelegationView:
        return WorkerDelegationView(**self._ledger('delegation.read', delegation_payload(delegation_id), epoch,
                                                   execution_id, session_id, generation, sequence, worker_pid,
                                                   worker_birth))

    def _delivery(self, action: str, claim: WorkerDelegationClaim | WorkerDelegationPayload, *identity: Any) -> bool:
        return bool(self._ledger('delegation.' + action, delegation_payload(claim), *identity))

    def claim_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('claim', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def ack_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('ack', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def drop_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('drop', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def defer_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('defer', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def release_delegation_delivery(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, claim: WorkerDelegationClaim | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> bool:
        return self._delivery('release', claim, epoch, execution_id, session_id, generation, sequence, worker_pid, worker_birth)

    def withdraw_delegation(self, *, epoch: AuthorityEpoch, execution_id: WorkerExecutionId, session_id: SessionId, generation: ExecutionGeneration, sequence: int, delegation_id: str | WorkerDelegationPayload, worker_pid: int, worker_birth: float) -> None:
        self._ledger('delegation.unscheduled', delegation_payload(delegation_id), epoch, execution_id, session_id,
                     generation, sequence, worker_pid, worker_birth)

    def pending_delegation_event(self, *, execution_id: WorkerExecutionId, session_id: SessionId, delegation_id: str) -> Mapping[str, JsonValue] | None:
        with refused():
            return _gateway.pending_delegation_event(self._store.holder(), session_id=session_id,  # type: ignore[no-any-return]
                                                     execution_id=execution_id, delegation_id=delegation_id)


register_repository("worker_executions", SQLiteWorkerExecutionRepository, {"receipted_writes": True})
