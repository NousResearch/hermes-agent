import sqlite3
import threading
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.discord_edge_protocol import (
    DiscordEdgeAuthorityKind,
    DiscordEdgeErrorCode,
    DiscordEdgeIntent,
    DiscordEdgeOperation,
    DiscordEdgeProtocolError,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeReconciliationQuery,
    DiscordEdgeThreadReadback,
    DiscordPublicTarget,
    DiscordPublicTargetType,
    SignedDiscordEdgeEnvelope,
    make_request,
    sign_capability,
    verify_receipt,
    verify_request_capability,
    verify_request_capability_for_reconciliation,
)
from gateway.discord_edge_runtime import (
    DiscordEdgeBlockerCode,
    DiscordEdgeJournalState,
    DiscordEdgeRuntime,
    DiscordEdgeRuntimeError,
    DiscordEdgeRuntimeErrorCode,
    DiscordLivePublicTargetProof,
    DiscordMutationAccepted,
    DiscordMutationReadback,
    DurableDiscordEdgeJournal,
)

NOW_MS = 2_000_000_000_000
GUILD_ID = "100000000000000001"
CHANNEL_ID = "100000000000000002"
OTHER_CHANNEL_ID = "100000000000000003"
MESSAGE_ID = "100000000000000004"
THREAD_ID = "100000000000000005"
BOT_USER_ID = "100000000000000006"
OTHER_USER_ID = "100000000000000007"
OTHER_GUILD_ID = "100000000000000008"


class SimulatedProcessCrash(BaseException):
    pass


def _target(channel_id=CHANNEL_ID):
    return DiscordPublicTarget.from_mapping(
        {
            "target_type": "public_guild_channel",
            "guild_id": GUILD_ID,
            "channel_id": channel_id,
        }
    )


def _intent(
    operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
    *,
    content="Exact public response",
    idempotency_key="case-1:routeback:1",
    target=None,
    reply_to_message_id=None,
):
    if operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND:
        payload = {"content": content}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
    elif operation is DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT:
        payload = {"message_id": MESSAGE_ID, "content": content}
    else:
        payload = {
            "name": "Exact public thread",
            "auto_archive_minutes": 1_440,
        }
    return DiscordEdgeIntent(
        operation=operation,
        target=target or _target(),
        payload=payload,
        idempotency_key=idempotency_key,
    )


def _request(
    writer_key,
    intent=None,
    *,
    request_id=None,
    capability_id=None,
    now_ms=NOW_MS,
):
    intent = intent or _intent()
    authority = (
        DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK
        if intent.operation is DiscordEdgeOperation.PUBLIC_MESSAGE_SEND
        else DiscordEdgeAuthorityKind.CANONICAL_PLAN
    )
    capability = sign_capability(
        writer_key,
        intent,
        authority_kind=authority,
        authority_ref="routeauth:case-1:1",
        issued_at_unix_ms=now_ms,
        expires_at_unix_ms=now_ms + 60_000,
        capability_id=capability_id or str(uuid.uuid4()),
    )
    return make_request(
        intent,
        capability,
        request_id=request_id or str(uuid.uuid4()),
        now_unix_ms=now_ms,
    )


def _reconciliation_query(request):
    return DiscordEdgeReconciliationQuery(
        idempotency_key=request.intent.idempotency_key,
        operation=request.intent.operation,
        target=request.intent.target,
        request_sha256=request.intent.request_sha256,
        content_sha256=request.intent.content_sha256,
    )


class FakeTargetProver:
    def __init__(self):
        self.calls = []
        self.publicly_viewable = True
        self.bot_can_view = True
        self.bot_has_required_permission = True
        self.bot_has_readback_permission = True
        self.target_override = None
        self.raise_error = False

    def _prove(self, operation, target, now_unix_ms):
        self.calls.append((operation, target, now_unix_ms))
        if self.raise_error:
            raise RuntimeError("proof unavailable")
        return DiscordLivePublicTargetProof(
            operation=operation,
            target=self.target_override or target,
            bot_user_id=BOT_USER_ID,
            observed_at_unix_ms=now_unix_ms,
            publicly_viewable=self.publicly_viewable,
            bot_can_view=self.bot_can_view,
            bot_has_required_permission=self.bot_has_required_permission,
        )

    def prove_public_message_send(
        self,
        target,
        *,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert deadline_unix_ms > now_unix_ms
        return self._prove(
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target,
            now_unix_ms,
        )

    def prove_public_message_edit(
        self,
        target,
        *,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert deadline_unix_ms > now_unix_ms
        return self._prove(
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
            target,
            now_unix_ms,
        )

    def prove_public_thread_create(
        self,
        target,
        *,
        has_initial_message,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert isinstance(has_initial_message, bool)
        assert deadline_unix_ms > now_unix_ms
        return self._prove(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target,
            now_unix_ms,
        )

    def prove_public_readback(
        self,
        operation,
        target,
        *,
        require_message_history,
        deadline_unix_ms,
        now_unix_ms,
    ):
        assert isinstance(operation, DiscordEdgeOperation)
        assert isinstance(require_message_history, bool)
        assert deadline_unix_ms > now_unix_ms
        original = self.bot_has_required_permission
        self.bot_has_required_permission = self.bot_has_readback_permission
        try:
            return self._prove(operation, target, now_unix_ms)
        finally:
            self.bot_has_required_permission = original


class FakeDiscordTransport:
    def __init__(self):
        self.mutation_calls = []
        self.read_calls = []
        self.last_operation = None
        self.last_target = None
        self.last_content = None
        self.last_reply_to_message_id = None
        self.crash_after_mutation_call = False
        self.crash_on_readback = False
        self.uncertain_after_mutation_call = False
        self.accepted_target_override = None
        self.accepted_bot_override = None
        self.readback_content_override = None
        self.readback_author_override = None
        self.readback_drop_reply = False
        self.thread_name_override = None
        self.thread_archive_override = None
        self.thread_parent_override = None
        self.thread_guild_override = None
        self.expected_deadline_unix_ms = NOW_MS + 15_000

    def _accepted(self, operation, target, content, object_id):
        self.mutation_calls.append((operation, target, content, object_id))
        self.last_operation = operation
        self.last_target = target
        self.last_content = content
        if self.crash_after_mutation_call:
            raise SimulatedProcessCrash()
        if self.uncertain_after_mutation_call:
            raise RuntimeError("mutation outcome unavailable")
        return DiscordMutationAccepted(
            operation=operation,
            target=self.accepted_target_override or target,
            discord_object_id=object_id,
            bot_user_id=self.accepted_bot_override or BOT_USER_ID,
        )

    def send_public_message(
        self,
        target,
        *,
        content,
        reply_to_message_id,
        deadline_unix_ms,
    ):
        assert deadline_unix_ms == self.expected_deadline_unix_ms
        self.last_reply_to_message_id = reply_to_message_id
        return self._accepted(
            DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
            target,
            content,
            MESSAGE_ID,
        )

    def edit_public_message(
        self,
        target,
        *,
        message_id,
        content,
        deadline_unix_ms,
    ):
        assert deadline_unix_ms == self.expected_deadline_unix_ms
        assert message_id == MESSAGE_ID
        return self._accepted(
            DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
            target,
            content,
            message_id,
        )

    def create_public_thread(
        self,
        target,
        *,
        name,
        initial_message,
        auto_archive_minutes,
        deadline_unix_ms,
    ):
        assert deadline_unix_ms == self.expected_deadline_unix_ms
        assert name == "Exact public thread"
        assert auto_archive_minutes == 1_440
        return self._accepted(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            target,
            initial_message or "",
            THREAD_ID,
        )

    def _readback(self, target, object_id):
        if self.crash_on_readback:
            raise SimulatedProcessCrash()
        self.read_calls.append((self.last_operation, target, object_id))
        content = (
            self.last_content
            if self.readback_content_override is None
            else self.readback_content_override
        )
        author = self.readback_author_override or BOT_USER_ID
        thread = None
        if self.last_operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE:
            thread_target = DiscordPublicTarget(
                target_type=DiscordPublicTargetType.PUBLIC_GUILD_THREAD,
                guild_id=self.thread_guild_override or target.guild_id,
                channel_id=object_id,
                parent_channel_id=self.thread_parent_override or target.channel_id,
            )
            thread = DiscordEdgeThreadReadback(
                target=thread_target,
                name=self.thread_name_override or "Exact public thread",
                auto_archive_minutes=(
                    self.thread_archive_override
                    if self.thread_archive_override is not None
                    else 1_440
                ),
            )
        return DiscordMutationReadback(
            operation=self.last_operation,
            target=target,
            discord_object_id=object_id,
            author_user_id=author,
            content=content,
            reply_to_message_id=(
                None if self.readback_drop_reply else self.last_reply_to_message_id
            ),
            thread=thread,
        )

    def read_public_message(
        self,
        target,
        *,
        operation,
        message_id,
        expected_reply_to_message_id,
    ):
        assert operation is self.last_operation
        assert expected_reply_to_message_id == self.last_reply_to_message_id
        return self._readback(target, message_id)

    def read_created_public_thread(self, target, *, thread_id, expected_content):
        assert expected_content == self.last_content
        return self._readback(target, thread_id)


class BarrierDiscordTransport(FakeDiscordTransport):
    """Hold one mutation at an exact in-process race boundary."""

    def __init__(self, *, crash_after_release=False):
        super().__init__()
        self.entered = threading.Barrier(2)
        self.release = threading.Barrier(2)
        self.crash_after_release = crash_after_release

    def _accepted(self, operation, target, content, object_id):
        self.mutation_calls.append((operation, target, content, object_id))
        self.last_operation = operation
        self.last_target = target
        self.last_content = content
        self.entered.wait(timeout=5)
        self.release.wait(timeout=5)
        if self.crash_after_release:
            raise SimulatedProcessCrash()
        return DiscordMutationAccepted(
            operation=operation,
            target=target,
            discord_object_id=object_id,
            bot_user_id=BOT_USER_ID,
        )


def _runtime(
    tmp_path,
    writer_key,
    edge_key,
    *,
    prover=None,
    transport=None,
    journal=None,
    clock_ms=None,
):
    prover = prover or FakeTargetProver()
    transport = transport or FakeDiscordTransport()
    journal = journal or DurableDiscordEdgeJournal.bootstrap(
        tmp_path / "discord-edge.sqlite3"
    )
    runtime = DiscordEdgeRuntime(
        writer_public_key=writer_key.public_key(),
        edge_private_key=edge_key,
        journal=journal,
        target_prover=prover,
        transport=transport,
        clock_ms=clock_ms or (lambda: NOW_MS),
    )
    return runtime, prover, transport, journal


def _verify_uncertainty_receipt(request, envelope, writer_key, edge_key):
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    receipt = verify_receipt(
        envelope,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert receipt.outcome is DiscordEdgeReceiptOutcome.DISPATCH_UNCERTAIN
    assert receipt.adapter_accepted is None
    assert receipt.discord_object_id is None
    assert receipt.bot_user_id is None
    assert receipt.readback_verified is False
    assert receipt.readback_content_sha256 is None
    assert receipt.readback_thread is None
    return receipt


@pytest.fixture
def writer_key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def edge_key():
    return Ed25519PrivateKey.generate()


def test_runtime_accepts_only_parsed_requests_and_dm_cannot_reach_edge(
    tmp_path,
    writer_key,
    edge_key,
):
    runtime, prover, transport, journal = _runtime(tmp_path, writer_key, edge_key)

    with pytest.raises(TypeError, match="parsed DiscordEdgeRequest"):
        runtime.execute(
            {
                "target": {
                    "target_type": "dm",
                    "channel_id": CHANNEL_ID,
                }
            }
        )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        DiscordPublicTarget.from_mapping(
            {
                "target_type": "private_thread",
                "guild_id": GUILD_ID,
                "channel_id": CHANNEL_ID,
            }
        )
    assert exc.value.code is DiscordEdgeErrorCode.FORBIDDEN_TARGET
    assert prover.calls == []
    assert transport.mutation_calls == []
    assert journal.get("case-1:routeback:1") is None


def test_live_permission_revocation_blocks_before_mutation_with_signed_receipt(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    prover = FakeTargetProver()
    prover.bot_has_required_permission = False
    runtime, _, transport, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        prover=prover,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.BLOCKED
    assert result.blocker_code is DiscordEdgeBlockerCode.PERMISSION_REVOKED
    assert result.receipt is not None
    assert transport.mutation_calls == []
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    receipt = verify_receipt(
        result.receipt,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert receipt.outcome is DiscordEdgeReceiptOutcome.BLOCKED_BEFORE_DISPATCH
    assert receipt.blocker_code == "permission_revoked"
    assert receipt.adapter_accepted is False
    assert journal.get(request.intent.idempotency_key).state is (
        DiscordEdgeJournalState.BLOCKED
    )


@pytest.mark.parametrize(
    "operation",
    [
        DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        DiscordEdgeOperation.PUBLIC_MESSAGE_EDIT,
        DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
    ],
)
def test_each_fixed_operation_requires_exact_bot_authored_readback(
    tmp_path,
    writer_key,
    edge_key,
    operation,
):
    request = _request(
        writer_key,
        _intent(
            operation,
            idempotency_key=f"case-1:{operation.value}:1",
        ),
    )
    runtime, prover, transport, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.VERIFIED
    assert result.blocker_code is None
    assert len(prover.calls) == 1
    assert prover.calls[0][0] is operation
    assert len(transport.mutation_calls) == 1
    assert transport.mutation_calls[0][0] is operation
    assert len(transport.read_calls) == 1
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    receipt = verify_receipt(
        result.receipt,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert receipt.outcome is DiscordEdgeReceiptOutcome.VERIFIED
    assert receipt.bot_user_id == BOT_USER_ID
    assert receipt.target == request.intent.target
    assert receipt.request_sha256 == request.intent.request_sha256
    assert receipt.content_sha256 == request.intent.content_sha256
    assert receipt.readback_content_sha256 == request.intent.content_sha256
    if operation is DiscordEdgeOperation.PUBLIC_THREAD_CREATE:
        assert receipt.readback_thread is not None
        assert receipt.readback_thread.name == request.intent.payload["name"]
        assert receipt.readback_thread.auto_archive_minutes == 1_440
        assert receipt.readback_thread.target.guild_id == GUILD_ID
        assert receipt.readback_thread.target.parent_channel_id == CHANNEL_ID
        assert receipt.readback_thread.target.channel_id == THREAD_ID
    else:
        assert receipt.readback_thread is None


def test_reply_reference_is_bound_through_verified_readback(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(
        writer_key,
        _intent(reply_to_message_id=OTHER_USER_ID),
    )
    runtime, _prover, transport, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.VERIFIED
    assert result.blocker_code is None
    assert transport.last_reply_to_message_id == OTHER_USER_ID


def test_runtime_checks_slow_live_proof_against_post_proof_clock(
    tmp_path,
    writer_key,
    edge_key,
):
    class SlowProofProver(FakeTargetProver):
        def _prove(self, operation, target, now_unix_ms):
            del now_unix_ms
            return DiscordLivePublicTargetProof(
                operation=operation,
                target=target,
                bot_user_id=BOT_USER_ID,
                observed_at_unix_ms=NOW_MS + 2_000,
                publicly_viewable=True,
                bot_can_view=True,
                bot_has_required_permission=True,
            )

    ticks = iter(
        [
            NOW_MS,
            NOW_MS + 2_000,
            NOW_MS + 2_000,
            NOW_MS + 2_001,
            NOW_MS + 2_002,
        ]
    )
    runtime, _prover, _transport, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        prover=SlowProofProver(),
        clock_ms=lambda: next(ticks),
    )

    result = runtime.execute(_request(writer_key))

    assert result.state is DiscordEdgeJournalState.VERIFIED
    assert result.blocker_code is None


def test_missing_reply_reference_cannot_produce_verified_receipt(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(
        writer_key,
        _intent(reply_to_message_id=OTHER_USER_ID),
    )
    transport = FakeDiscordTransport()
    transport.readback_drop_reply = True
    runtime, _prover, _, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.DISPATCHING
    assert result.blocker_code is DiscordEdgeBlockerCode.READBACK_REPLY_MISMATCH


def test_exact_duplicate_returns_identical_persisted_receipt_without_resend(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    runtime, prover, transport, journal = _runtime(tmp_path, writer_key, edge_key)

    first = runtime.execute(request)
    second = runtime.execute(request)

    assert first.state is second.state is DiscordEdgeJournalState.VERIFIED
    assert second.replayed is True
    assert second.receipt.to_message() == first.receipt.to_message()
    assert len(prover.calls) == 1
    assert len(transport.mutation_calls) == 1
    assert len(transport.read_calls) == 1
    assert journal.get(request.intent.idempotency_key).receipt.to_message() == (
        first.receipt.to_message()
    )


def test_conflicting_request_cannot_reuse_idempotency_key(
    tmp_path,
    writer_key,
    edge_key,
):
    original = _request(writer_key)
    runtime, _prover, transport, journal = _runtime(tmp_path, writer_key, edge_key)
    runtime.execute(original)
    changed = _request(
        writer_key,
        _intent(content="Different exact content"),
    )

    with pytest.raises(DiscordEdgeRuntimeError) as exc:
        runtime.execute(changed)

    assert exc.value.code is DiscordEdgeRuntimeErrorCode.IDEMPOTENCY_CONFLICT
    assert len(transport.mutation_calls) == 1
    record = journal.get(original.intent.idempotency_key)
    assert record.request_sha256 == original.intent.request_sha256
    assert record.state is DiscordEdgeJournalState.VERIFIED


def test_prepared_exact_intent_rebind_is_monotonic_and_old_request_cannot_dispatch(
    tmp_path,
    writer_key,
    edge_key,
):
    intent = _intent()
    original = _request(writer_key, intent, now_ms=NOW_MS)
    fresh_now = NOW_MS + 1_000
    replacement = _request(writer_key, intent, now_ms=fresh_now)
    transport = FakeDiscordTransport()
    transport.expected_deadline_unix_ms = fresh_now + 15_000
    runtime, prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
        clock_ms=lambda: fresh_now,
    )
    original_capability = verify_request_capability(
        original,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    journal.prepare(original, original_capability, now_unix_ms=NOW_MS)
    replacement_capability = verify_request_capability(
        replacement,
        writer_key.public_key(),
        now_unix_ms=fresh_now,
    )

    rebound = journal.prepare(
        replacement,
        replacement_capability,
        now_unix_ms=fresh_now,
    )

    assert rebound.created is False
    assert rebound.record.state is DiscordEdgeJournalState.PREPARED
    assert rebound.record.request_envelope is not None
    assert rebound.record.request_envelope.to_message() == replacement.to_message()
    with pytest.raises(DiscordEdgeRuntimeError) as stale_exc:
        runtime.execute(original)
    assert stale_exc.value.code is DiscordEdgeRuntimeErrorCode.IDEMPOTENCY_CONFLICT
    assert prover.calls == []
    assert transport.mutation_calls == []

    completed = runtime.execute(replacement)

    assert completed.state is DiscordEdgeJournalState.VERIFIED
    assert len(transport.mutation_calls) == 1


def test_expired_prepared_request_is_recoverable_by_fresh_exact_intent(
    tmp_path,
    writer_key,
    edge_key,
):
    intent = _intent(idempotency_key="case-1:routeback:expired-prepared")
    expired = _request(writer_key, intent, now_ms=NOW_MS)
    fresh_now = NOW_MS + 120_000
    replacement = _request(writer_key, intent, now_ms=fresh_now)
    transport = FakeDiscordTransport()
    transport.expected_deadline_unix_ms = fresh_now + 15_000
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
        clock_ms=lambda: fresh_now,
    )
    expired_capability = verify_request_capability(
        expired,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    journal.prepare(expired, expired_capability, now_unix_ms=NOW_MS)

    result = runtime.execute(replacement)

    assert result.state is DiscordEdgeJournalState.VERIFIED
    assert len(transport.mutation_calls) == 1
    stored = journal.get(intent.idempotency_key)
    assert stored is not None
    assert stored.request_envelope is not None
    assert stored.request_envelope.to_message() == replacement.to_message()


def test_prepared_rebind_rejects_any_intent_change_before_proof_or_dispatch(
    tmp_path,
    writer_key,
    edge_key,
):
    original = _request(writer_key)
    changed = _request(
        writer_key,
        _intent(content="Changed prepared content"),
        now_ms=NOW_MS + 1_000,
    )
    runtime, prover, transport, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        clock_ms=lambda: NOW_MS + 1_000,
    )
    original_capability = verify_request_capability(
        original,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    journal.prepare(original, original_capability, now_unix_ms=NOW_MS)

    with pytest.raises(DiscordEdgeRuntimeError) as exc:
        runtime.execute(changed)

    assert exc.value.code is DiscordEdgeRuntimeErrorCode.IDEMPOTENCY_CONFLICT
    assert prover.calls == []
    assert transport.mutation_calls == []
    stored = journal.get(original.intent.idempotency_key)
    assert stored is not None
    assert stored.state is DiscordEdgeJournalState.PREPARED
    assert stored.request_envelope is not None
    assert stored.request_envelope.to_message() == original.to_message()


def test_reconciliation_during_active_dispatch_stays_pending_without_receipt(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = BarrierDiscordTransport()
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )
    outcomes = []

    def execute() -> None:
        try:
            outcomes.append(runtime.execute(request))
        except BaseException as exc:  # pragma: no cover - asserted below
            outcomes.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    transport.entered.wait(timeout=5)
    try:
        with pytest.raises(DiscordEdgeRuntimeError) as exc:
            runtime.reconcile(_reconciliation_query(request))
        assert (
            exc.value.code
            is DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE
        )
        active = journal.get(request.intent.idempotency_key)
        assert active is not None
        assert active.state is DiscordEdgeJournalState.DISPATCHING
        assert active.receipt is None
        assert journal.receipt_history(request.intent.idempotency_key) == ()
        assert len(transport.mutation_calls) == 1
    finally:
        transport.release.wait(timeout=5)
        worker.join(timeout=5)

    assert worker.is_alive() is False
    assert len(outcomes) == 1
    assert not isinstance(outcomes[0], BaseException)
    assert outcomes[0].state is DiscordEdgeJournalState.VERIFIED
    assert len(transport.mutation_calls) == 1
    reconciled = runtime.reconcile(_reconciliation_query(request))
    assert reconciled.execution.state is DiscordEdgeJournalState.VERIFIED
    assert len(transport.mutation_calls) == 1


def test_restart_after_barrier_dispatch_crash_persists_uncertainty_without_resend(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = BarrierDiscordTransport(crash_after_release=True)
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )
    failures = []

    def execute() -> None:
        try:
            runtime.execute(request)
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=execute)
    worker.start()
    transport.entered.wait(timeout=5)
    with pytest.raises(DiscordEdgeRuntimeError) as active_exc:
        runtime.reconcile(_reconciliation_query(request))
    assert (
        active_exc.value.code
        is DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE
    )
    assert journal.receipt_history(request.intent.idempotency_key) == ()
    transport.release.wait(timeout=5)
    worker.join(timeout=5)

    assert worker.is_alive() is False
    assert len(failures) == 1
    assert isinstance(failures[0], SimulatedProcessCrash)
    assert len(transport.mutation_calls) == 1
    restarted_journal = DurableDiscordEdgeJournal(journal.path)
    restarted_runtime, prover, clean_transport, _ = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        journal=restarted_journal,
        clock_ms=lambda: NOW_MS + 120_000,
    )

    reconciled = restarted_runtime.reconcile(_reconciliation_query(request))

    assert reconciled.execution.state is DiscordEdgeJournalState.DISPATCHING
    assert (
        reconciled.execution.blocker_code
        is DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN
    )
    assert reconciled.execution.receipt is not None
    assert prover.calls == []
    assert clean_transport.mutation_calls == []
    assert len(transport.mutation_calls) == 1


def test_process_crash_after_dispatching_commit_is_durable_and_never_resent(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    crashing_transport = FakeDiscordTransport()
    crashing_transport.crash_after_mutation_call = True
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=crashing_transport,
    )

    with pytest.raises(SimulatedProcessCrash):
        runtime.execute(request)

    record = journal.get(request.intent.idempotency_key)
    assert record.state is DiscordEdgeJournalState.DISPATCHING
    assert record.receipt is None
    assert len(crashing_transport.mutation_calls) == 1

    restarted_journal = DurableDiscordEdgeJournal(journal.path)
    restarted_runtime, restarted_prover, restarted_transport, _ = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        journal=restarted_journal,
        clock_ms=lambda: NOW_MS + 120_000,
    )
    replay = restarted_runtime.execute(request)

    assert replay.state is DiscordEdgeJournalState.DISPATCHING
    assert replay.blocker_code is DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN
    assert replay.receipt is not None
    assert replay.replayed is True
    assert restarted_prover.calls == []
    assert restarted_transport.mutation_calls == []
    receipt = _verify_uncertainty_receipt(
        request,
        replay.receipt,
        writer_key,
        edge_key,
    )
    assert receipt.blocker_code == "dispatch_outcome_uncertain"
    assert receipt.outcome is not DiscordEdgeReceiptOutcome.FAILED_BEFORE_DISPATCH
    assert restarted_journal.get(request.intent.idempotency_key).receipt is not None
    repeated = restarted_runtime.execute(request)
    assert repeated.receipt.to_message() == replay.receipt.to_message()
    assert restarted_transport.mutation_calls == []


def test_crash_after_acceptance_reconciles_staged_object_without_resend(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.crash_on_readback = True
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    with pytest.raises(SimulatedProcessCrash):
        runtime.execute(request)

    staged = journal.get(request.intent.idempotency_key)
    assert staged.state is DiscordEdgeJournalState.DISPATCHING
    assert staged.receipt is not None
    capability = verify_request_capability_for_reconciliation(
        request,
        writer_key.public_key(),
    )
    accepted_receipt = verify_receipt(
        staged.receipt,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert accepted_receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
    assert accepted_receipt.discord_object_id == MESSAGE_ID
    assert accepted_receipt.bot_user_id == BOT_USER_ID

    transport.crash_on_readback = False
    restarted_journal = DurableDiscordEdgeJournal(journal.path)
    restarted_runtime, _prover, _, _ = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        journal=restarted_journal,
        transport=transport,
        clock_ms=lambda: NOW_MS + 120_000,
    )
    replay = restarted_runtime.execute(request)

    assert replay.state is DiscordEdgeJournalState.VERIFIED
    assert replay.replayed is True
    assert len(transport.mutation_calls) == 1
    assert len(transport.read_calls) == 1
    assert len(restarted_journal.receipt_history(request.intent.idempotency_key)) == 2
    repeated = restarted_runtime.execute(request)
    assert repeated.receipt.to_message() == replay.receipt.to_message()
    assert len(transport.mutation_calls) == 1


def test_query_recovers_original_request_after_acceptance_crash_and_expiry(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.crash_on_readback = True
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    with pytest.raises(SimulatedProcessCrash):
        runtime.execute(request)

    transport.crash_on_readback = False
    restarted_journal = DurableDiscordEdgeJournal(journal.path)
    restarted_runtime, _prover, _, _ = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        journal=restarted_journal,
        transport=transport,
        clock_ms=lambda: NOW_MS + 120_000,
    )
    reconciled = restarted_runtime.reconcile(_reconciliation_query(request))

    assert reconciled.request.to_message() == request.to_message()
    assert reconciled.execution.state is DiscordEdgeJournalState.VERIFIED
    assert reconciled.execution.replayed is True
    assert len(transport.mutation_calls) == 1
    assert len(transport.read_calls) == 1


def test_query_turns_pre_acceptance_crash_into_signed_uncertainty_without_dispatch(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    crashing_transport = FakeDiscordTransport()
    crashing_transport.crash_after_mutation_call = True
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=crashing_transport,
    )
    with pytest.raises(SimulatedProcessCrash):
        runtime.execute(request)

    restarted_journal = DurableDiscordEdgeJournal(journal.path)
    restarted_runtime, prover, clean_transport, _ = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        journal=restarted_journal,
        clock_ms=lambda: NOW_MS + 120_000,
    )
    reconciled = restarted_runtime.reconcile(_reconciliation_query(request))

    assert reconciled.request.to_message() == request.to_message()
    assert reconciled.execution.state is DiscordEdgeJournalState.DISPATCHING
    assert (
        reconciled.execution.blocker_code
        is DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN
    )
    assert reconciled.execution.receipt is not None
    assert reconciled.execution.replayed is True
    assert prover.calls == []
    assert clean_transport.mutation_calls == []
    assert clean_transport.read_calls == []


def test_query_rejects_prepared_or_mismatched_binding_without_any_dispatch(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    runtime, prover, transport, journal = _runtime(tmp_path, writer_key, edge_key)
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    journal.prepare(request, capability, now_unix_ms=NOW_MS)

    with pytest.raises(DiscordEdgeRuntimeError) as prepared_exc:
        runtime.reconcile(_reconciliation_query(request))
    assert (
        prepared_exc.value.code
        is DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE
    )

    other_target = _target(OTHER_CHANNEL_ID)
    mismatched = DiscordEdgeReconciliationQuery(
        idempotency_key=request.intent.idempotency_key,
        operation=request.intent.operation,
        target=other_target,
        request_sha256=request.intent.request_sha256,
        content_sha256=request.intent.content_sha256,
    )
    with pytest.raises(DiscordEdgeRuntimeError) as mismatch_exc:
        runtime.reconcile(mismatched)
    assert (
        mismatch_exc.value.code
        is DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE
    )
    assert prover.calls == []
    assert transport.mutation_calls == []
    assert transport.read_calls == []


def test_transport_uncertainty_is_persisted_and_never_resent(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.uncertain_after_mutation_call = True
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    first = runtime.execute(request)
    second = runtime.execute(request)

    assert first.state is DiscordEdgeJournalState.DISPATCHING
    assert first.blocker_code is DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN
    assert first.receipt is not None
    assert second.replayed is True
    assert second.receipt.to_message() == first.receipt.to_message()
    assert len(transport.mutation_calls) == 1
    record = journal.get(request.intent.idempotency_key)
    assert record.state is DiscordEdgeJournalState.DISPATCHING
    assert record.blocker_code is DiscordEdgeBlockerCode.DISPATCH_OUTCOME_UNCERTAIN
    receipt = _verify_uncertainty_receipt(
        request,
        first.receipt,
        writer_key,
        edge_key,
    )
    assert receipt.blocker_code == "dispatch_outcome_uncertain"


def test_signed_dispatch_uncertainty_rejects_tampering(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.uncertain_after_mutation_call = True
    runtime, _prover, _, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )
    result = runtime.execute(request)
    tampered = result.receipt.to_message()
    tampered["payload"]["adapter_accepted"] = False
    envelope = SignedDiscordEdgeEnvelope.from_mapping(
        tampered,
        code=DiscordEdgeErrorCode.INVALID_RECEIPT,
        label="uncertainty receipt",
    )

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        verify_receipt(
            envelope,
            edge_key.public_key(),
            expected_request=request,
            now_unix_ms=NOW_MS,
        )

    assert exc.value.code is DiscordEdgeErrorCode.SIGNATURE_INVALID


def test_invalid_post_dispatch_binding_gets_signed_uncertainty_without_resend(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.accepted_target_override = _target(OTHER_CHANNEL_ID)
    runtime, _prover, _, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    first = runtime.execute(request)
    second = runtime.execute(request)

    assert first.state is DiscordEdgeJournalState.DISPATCHING
    assert first.blocker_code is DiscordEdgeBlockerCode.DISPATCH_BINDING_MISMATCH
    assert second.receipt.to_message() == first.receipt.to_message()
    assert len(transport.mutation_calls) == 1
    receipt = _verify_uncertainty_receipt(
        request,
        first.receipt,
        writer_key,
        edge_key,
    )
    assert receipt.blocker_code == "dispatch_binding_mismatch"
    assert receipt.outcome is not DiscordEdgeReceiptOutcome.FAILED_BEFORE_DISPATCH


def test_forged_writer_capability_is_rejected_before_journal_or_proof(
    tmp_path,
    writer_key,
    edge_key,
):
    rogue_writer_key = Ed25519PrivateKey.generate()
    request = _request(rogue_writer_key)
    runtime, prover, transport, journal = _runtime(tmp_path, writer_key, edge_key)

    with pytest.raises(DiscordEdgeProtocolError) as exc:
        runtime.execute(request)

    assert exc.value.code is DiscordEdgeErrorCode.SIGNATURE_INVALID
    assert journal.get(request.intent.idempotency_key) is None
    assert prover.calls == []
    assert transport.mutation_calls == []


@pytest.mark.parametrize(
    ("content_override", "author_override", "blocker"),
    [
        (
            "Discord returned different content",
            None,
            DiscordEdgeBlockerCode.READBACK_CONTENT_MISMATCH,
        ),
        (None, OTHER_USER_ID, DiscordEdgeBlockerCode.READBACK_AUTHOR_MISMATCH),
    ],
)
def test_non_exact_readback_never_produces_verified_receipt_or_resend(
    tmp_path,
    writer_key,
    edge_key,
    content_override,
    author_override,
    blocker,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.readback_content_override = content_override
    transport.readback_author_override = author_override
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    first = runtime.execute(request)
    second = runtime.execute(request)

    assert first.state is DiscordEdgeJournalState.DISPATCHING
    assert first.blocker_code is blocker
    assert first.receipt is not None
    assert second.receipt.to_message() == first.receipt.to_message()
    assert second.replayed is True
    assert len(transport.mutation_calls) == 1
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    receipt = verify_receipt(
        first.receipt,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
    assert receipt.blocker_code == blocker.value
    assert receipt.readback_verified is False
    assert journal.get(request.intent.idempotency_key).state is (
        DiscordEdgeJournalState.DISPATCHING
    )


def test_readback_only_reconciliation_upgrades_without_resend_and_keeps_history(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    transport = FakeDiscordTransport()
    transport.readback_content_override = "Wrong content"
    runtime, prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )
    first = runtime.execute(request)
    first_message = first.receipt.to_message()
    assert first.state is DiscordEdgeJournalState.DISPATCHING
    assert len(journal.receipt_history(request.intent.idempotency_key)) == 2

    transport.readback_content_override = None
    transport.readback_author_override = OTHER_USER_ID
    prover.bot_has_required_permission = False
    prover.bot_has_readback_permission = False
    permission_blocked = runtime.reconcile_accepted_unverified(request)
    assert permission_blocked.receipt.to_message() == first_message
    assert len(transport.read_calls) == 1

    prover.bot_has_readback_permission = True
    wrong_author = runtime.reconcile_accepted_unverified(request)
    assert wrong_author.receipt.to_message() == first_message
    assert len(transport.read_calls) == 2

    transport.readback_author_override = None
    upgraded = runtime.reconcile_accepted_unverified(request)

    assert upgraded.state is DiscordEdgeJournalState.VERIFIED
    assert upgraded.replayed is True
    assert upgraded.receipt.to_message() != first_message
    assert len(transport.mutation_calls) == 1
    assert len(transport.read_calls) == 3
    history = journal.receipt_history(request.intent.idempotency_key)
    assert len(history) == 3
    assert history[1].to_message() == first_message
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    staged_receipt = verify_receipt(
        history[0],
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    historical_receipt = verify_receipt(
        history[1],
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert staged_receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
    assert staged_receipt.blocker_code == "readback_pending"
    verified = verify_receipt(
        history[2],
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert historical_receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
    assert verified.outcome is DiscordEdgeReceiptOutcome.VERIFIED
    repeated = runtime.reconcile_accepted_unverified(request)
    assert repeated.receipt.to_message() == upgraded.receipt.to_message()
    assert len(transport.mutation_calls) == 1


def test_thread_reconciliation_requires_exact_external_thread_evidence(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(
        writer_key,
        _intent(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            idempotency_key="case-1:thread:reconcile",
        ),
    )
    transport = FakeDiscordTransport()
    transport.thread_name_override = "Wrong thread name"
    runtime, _prover, _, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )
    first = runtime.execute(request)
    assert first.state is DiscordEdgeJournalState.DISPATCHING

    transport.thread_name_override = None
    transport.thread_archive_override = 60
    wrong_archive = runtime.reconcile_accepted_unverified(request)
    assert wrong_archive.state is DiscordEdgeJournalState.DISPATCHING

    transport.thread_archive_override = None
    transport.thread_parent_override = OTHER_CHANNEL_ID
    wrong_parent = runtime.reconcile_accepted_unverified(request)
    assert wrong_parent.state is DiscordEdgeJournalState.DISPATCHING

    transport.thread_parent_override = None
    upgraded = runtime.reconcile_accepted_unverified(request)

    assert upgraded.state is DiscordEdgeJournalState.VERIFIED
    assert len(transport.mutation_calls) == 1
    assert len(journal.receipt_history(request.intent.idempotency_key)) == 3
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    receipt = verify_receipt(
        upgraded.receipt,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert receipt.outcome is DiscordEdgeReceiptOutcome.VERIFIED
    assert receipt.readback_thread.name == request.intent.payload["name"]
    assert receipt.readback_thread.auto_archive_minutes == 1_440
    assert receipt.readback_thread.target.parent_channel_id == CHANNEL_ID


def test_live_proof_must_bind_the_exact_public_target(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    prover = FakeTargetProver()
    prover.target_override = _target(OTHER_CHANNEL_ID)
    runtime, _, transport, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        prover=prover,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.BLOCKED
    assert result.blocker_code is DiscordEdgeBlockerCode.TARGET_PROOF_MISMATCH
    assert transport.mutation_calls == []


def test_live_private_visibility_proof_blocks_even_for_structurally_public_target(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    prover = FakeTargetProver()
    prover.publicly_viewable = False
    runtime, _, transport, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        prover=prover,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.BLOCKED
    assert result.blocker_code is DiscordEdgeBlockerCode.TARGET_NOT_PUBLIC
    assert transport.mutation_calls == []


@pytest.mark.parametrize(
    ("attribute", "value", "blocker"),
    [
        (
            "thread_name_override",
            "Different public thread",
            DiscordEdgeBlockerCode.READBACK_THREAD_NAME_MISMATCH,
        ),
        (
            "thread_archive_override",
            60,
            DiscordEdgeBlockerCode.READBACK_THREAD_ARCHIVE_MISMATCH,
        ),
        (
            "thread_parent_override",
            OTHER_CHANNEL_ID,
            DiscordEdgeBlockerCode.READBACK_THREAD_TARGET_MISMATCH,
        ),
        (
            "thread_guild_override",
            OTHER_GUILD_ID,
            DiscordEdgeBlockerCode.READBACK_THREAD_TARGET_MISMATCH,
        ),
    ],
)
def test_thread_verification_requires_external_name_archive_parent_and_guild(
    tmp_path,
    writer_key,
    edge_key,
    attribute,
    value,
    blocker,
):
    request = _request(
        writer_key,
        _intent(
            DiscordEdgeOperation.PUBLIC_THREAD_CREATE,
            idempotency_key=f"case-1:thread-mismatch:{attribute}",
        ),
    )
    transport = FakeDiscordTransport()
    setattr(transport, attribute, value)
    runtime, _prover, _, _journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        transport=transport,
    )

    result = runtime.execute(request)

    assert result.state is DiscordEdgeJournalState.DISPATCHING
    assert result.blocker_code is blocker
    assert len(transport.mutation_calls) == 1
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=NOW_MS,
    )
    receipt = verify_receipt(
        result.receipt,
        edge_key.public_key(),
        expected_request=request,
        expected_capability=capability,
        now_unix_ms=NOW_MS,
    )
    assert receipt.outcome is DiscordEdgeReceiptOutcome.ACCEPTED_UNVERIFIED
    assert receipt.outcome is not DiscordEdgeReceiptOutcome.VERIFIED
    assert receipt.blocker_code == blocker.value


def test_journal_rejects_non_normalized_path(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="normalized"):
        DurableDiscordEdgeJournal(nested / ".." / "edge.sqlite3")


def _downgrade_exact_journal_fixture_to_v1(path):
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        assert str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() == "wal"
        marker_id = str(
            conn.execute(
                "SELECT marker_id FROM discord_edge_journal_meta_v1 WHERE singleton = 1"
            ).fetchone()[0]
        )
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            CREATE TABLE discord_edge_idempotency_legacy_v1 (
                idempotency_key TEXT PRIMARY KEY,
                request_envelope_sha256 TEXT NOT NULL,
                request_id TEXT NOT NULL,
                capability_id TEXT NOT NULL,
                request_sha256 TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                state TEXT NOT NULL CHECK (
                    state IN ('prepared', 'dispatching', 'verified', 'blocked')
                ),
                receipt_json TEXT,
                blocker_code TEXT,
                created_at_unix_ms INTEGER NOT NULL CHECK (created_at_unix_ms > 0),
                updated_at_unix_ms INTEGER NOT NULL CHECK (updated_at_unix_ms > 0),
                CHECK (length(request_envelope_sha256) = 64),
                CHECK (length(request_sha256) = 64),
                CHECK (length(content_sha256) = 64),
                CHECK (
                    (state = 'prepared' AND receipt_json IS NULL
                        AND blocker_code IS NULL)
                    OR state = 'dispatching'
                    OR (state = 'verified' AND receipt_json IS NOT NULL
                        AND blocker_code IS NULL)
                    OR (state = 'blocked' AND receipt_json IS NOT NULL
                        AND blocker_code IS NOT NULL)
                )
            )
            """
        )
        conn.execute(
            """
            INSERT INTO discord_edge_idempotency_legacy_v1 (
                idempotency_key, request_envelope_sha256, request_id,
                capability_id, request_sha256, content_sha256, state,
                receipt_json, blocker_code, created_at_unix_ms,
                updated_at_unix_ms
            )
            SELECT idempotency_key, request_envelope_sha256, request_id,
                   capability_id, request_sha256, content_sha256, state,
                   receipt_json, blocker_code, created_at_unix_ms,
                   updated_at_unix_ms
              FROM discord_edge_idempotency_v1
            """
        )
        conn.execute("DROP TABLE discord_edge_idempotency_v1")
        conn.execute(
            "ALTER TABLE discord_edge_idempotency_legacy_v1 "
            "RENAME TO discord_edge_idempotency_v1"
        )
        conn.execute("DROP TABLE discord_edge_journal_meta_v1")
        conn.execute(
            """
            CREATE TABLE discord_edge_journal_meta_v1 (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                marker_id TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL CHECK (schema_version = 1)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO discord_edge_journal_meta_v1 (
                singleton, marker_id, schema_version
            ) VALUES (1, ?, 1)
            """,
            (marker_id,),
        )
        conn.execute("PRAGMA user_version=1")
        conn.execute("COMMIT")
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert checkpoint is not None and int(checkpoint[0]) == 0
    finally:
        conn.close()


def test_journal_persists_canonical_original_request_across_reopen(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    runtime, _prover, _transport, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
    )
    runtime.execute(request)

    record = journal.get(request.intent.idempotency_key)
    assert record is not None
    assert record.request_envelope is not None
    assert record.request_envelope.to_message() == request.to_message()

    reopened = DurableDiscordEdgeJournal(journal.path)
    reopened_record = reopened.get(request.intent.idempotency_key)
    assert reopened_record is not None
    assert reopened_record.request_envelope is not None
    assert reopened_record.request_envelope.to_message() == request.to_message()


def test_v1_migration_keeps_legacy_rows_unavailable_to_query_reconciliation(
    tmp_path,
    writer_key,
    edge_key,
):
    request = _request(writer_key)
    runtime, _prover, _transport, journal = _runtime(
        tmp_path,
        writer_key,
        edge_key,
    )
    runtime.execute(request)
    _downgrade_exact_journal_fixture_to_v1(journal.path)

    migrated = DurableDiscordEdgeJournal(journal.path)
    record = migrated.get(request.intent.idempotency_key)
    assert record is not None
    assert record.request_envelope is None
    with sqlite3.connect(journal.path) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 2
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(discord_edge_idempotency_v1)"
            )
        }
    assert "request_envelope_json" in columns

    restarted_runtime, prover, transport, _ = _runtime(
        tmp_path,
        writer_key,
        edge_key,
        journal=migrated,
        clock_ms=lambda: NOW_MS + 120_000,
    )
    with pytest.raises(DiscordEdgeRuntimeError) as exc:
        restarted_runtime.reconcile(_reconciliation_query(request))
    assert exc.value.code is DiscordEdgeRuntimeErrorCode.RECONCILIATION_NOT_AVAILABLE
    assert prover.calls == []
    assert transport.mutation_calls == []
    assert transport.read_calls == []


def test_journal_securely_creates_private_service_owned_database(tmp_path):
    journal = DurableDiscordEdgeJournal.bootstrap(tmp_path / "edge.sqlite3")
    database_stat = journal.path.stat()
    marker_stat = journal.marker_path.stat()

    assert database_stat.st_uid == tmp_path.stat().st_uid
    assert database_stat.st_mode & 0o777 == 0o600
    assert database_stat.st_nlink == 1
    assert marker_stat.st_uid == database_stat.st_uid
    assert marker_stat.st_mode & 0o777 == 0o600
    assert marker_stat.st_nlink == 1
    reopened = DurableDiscordEdgeJournal(journal.path)
    assert reopened.get("missing") is None

    with pytest.raises(FileExistsError):
        DurableDiscordEdgeJournal(journal.path, bootstrap=True)


def test_normal_journal_open_never_bootstraps_a_missing_database(tmp_path):
    path = tmp_path / "edge.sqlite3"

    with pytest.raises(DiscordEdgeRuntimeError) as exc:
        DurableDiscordEdgeJournal(path)

    assert exc.value.code is DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED
    assert not path.exists()
    assert not type(path)(f"{path}.initialized").exists()


def test_deleted_journal_fails_closed_without_recreation(tmp_path):
    journal = DurableDiscordEdgeJournal.bootstrap(tmp_path / "edge.sqlite3")
    journal.path.unlink()

    with pytest.raises(DiscordEdgeRuntimeError) as active_exc:
        journal.get("case-1:lost")
    assert active_exc.value.code is DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED

    with pytest.raises(DiscordEdgeRuntimeError) as restart_exc:
        DurableDiscordEdgeJournal(journal.path)
    assert restart_exc.value.code is DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED
    assert not journal.path.exists()


def test_empty_replacement_cannot_satisfy_durable_journal_marker(tmp_path):
    journal = DurableDiscordEdgeJournal.bootstrap(tmp_path / "edge.sqlite3")
    journal.path.unlink()
    journal.path.write_bytes(b"")
    journal.path.chmod(0o600)

    with pytest.raises(DiscordEdgeRuntimeError) as exc:
        DurableDiscordEdgeJournal(journal.path)

    assert exc.value.code is DiscordEdgeRuntimeErrorCode.JOURNAL_NOT_INITIALIZED
    assert journal.marker_path.exists()


def test_journal_rejects_unprotected_parent_mode(tmp_path):
    parent = tmp_path / "permissive"
    parent.mkdir(mode=0o700)
    parent.chmod(0o770)
    try:
        with pytest.raises(PermissionError, match="parent"):
            DurableDiscordEdgeJournal(parent / "edge.sqlite3")
    finally:
        parent.chmod(0o700)


def test_journal_rejects_existing_database_with_open_permissions(tmp_path):
    path = tmp_path / "edge.sqlite3"
    path.write_bytes(b"")
    path.chmod(0o640)

    with pytest.raises(PermissionError, match="journal database"):
        DurableDiscordEdgeJournal(path)


def test_journal_rejects_existing_database_with_owner_execute_bit(tmp_path):
    path = tmp_path / "edge.sqlite3"
    path.write_bytes(b"")
    path.chmod(0o700)

    with pytest.raises(PermissionError, match="exact mode 0600"):
        DurableDiscordEdgeJournal(path)


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_journal_rejects_insecure_companion_permissions(tmp_path, suffix):
    path = tmp_path / "edge.sqlite3"
    path.write_bytes(b"")
    path.chmod(0o600)
    companion = type(path)(f"{path}{suffix}")
    companion.write_bytes(b"")
    companion.chmod(0o644)

    with pytest.raises(PermissionError, match="journal companion"):
        DurableDiscordEdgeJournal(path)
