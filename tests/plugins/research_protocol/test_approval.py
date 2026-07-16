"""TDD contract tests for exact workflow approvals and atomic claims."""

import asyncio
from datetime import UTC, datetime, timedelta
import json
import threading

import pytest

from plugins.research_protocol.approval import (
    ApprovalRecord,
    ApprovalService,
    ApprovalScope,
    ApprovalVerdict,
    WorkflowApprovalDecision,
    make_approval_record,
    request_workflow_approval,
)
from plugins.research_protocol.models import (
    BudgetLimits,
    Capability,
    ExternalRight,
)
from plugins.research_protocol.storage.postgres import (
    ApprovalStoreTimeoutError,
    PostgresApprovalStore,
)


NOW = datetime.now(UTC)
PLAN_SHA = "a" * 64
INPUT_A = "b" * 64
INPUT_B = "c" * 64


def make_scope(**changes):
    values = {
        "capability": Capability.RESEARCH_COLLECT,
        "run_id": "run-001",
        "plan_sha256": PLAN_SHA,
        "input_hashes": [INPUT_B, INPUT_A],
        "budgets": {
            "max_duration_seconds": 60,
            "max_executions": 2,
            "max_records": 10,
            "max_bytes": 1000,
            "max_external_calls": 0,
        },
        "expires_at": NOW + timedelta(minutes=10),
        "max_executions": 2,
        "external_rights": [
            ExternalRight.DATABASE_READ,
            ExternalRight.NETWORK_READ,
        ],
    }
    values.update(changes)
    return ApprovalScope.model_validate(values)


def test_scope_is_canonical_and_order_independent():
    left = make_scope()
    right = make_scope(
        input_hashes=[INPUT_A, INPUT_B],
        external_rights=[
            ExternalRight.NETWORK_READ,
            ExternalRight.DATABASE_READ,
        ],
    )

    assert left.input_hashes == (INPUT_A, INPUT_B)
    assert left.external_rights == (
        ExternalRight.DATABASE_READ,
        ExternalRight.NETWORK_READ,
    )
    assert left.scope_sha256() == right.scope_sha256()
    assert left.canonical_json() == right.canonical_json()
    with pytest.raises(AttributeError):
        left.input_hashes.append(INPUT_A)
    with pytest.raises(AttributeError):
        left.external_rights.append(ExternalRight.NETWORK_READ)


def test_approval_record_rejects_any_scope_binding_mismatch():
    record = make_approval_record(make_scope())
    valid = record.model_dump()
    mutations = (
        {"scope_json": "{}"},
        {"scope_json": json.dumps(json.loads(record.scope_json), indent=2)},
        {"scope_sha256": "d" * 64},
        {"plan_sha256": "e" * 64},
        {"expires_at": record.expires_at + timedelta(seconds=1)},
        {"max_executions": record.max_executions - 1},
        {"consumed_count": record.max_executions + 1},
    )

    for mutation in mutations:
        with pytest.raises(ValueError):
            ApprovalRecord.model_validate(valid | mutation)


class RecordingApprovalStore:
    def __init__(self):
        self.records = []

    async def create_approval(self, record):
        self.records.append(record)


def test_approval_service_does_not_block_the_async_event_loop(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_native_approval(scope, summary):
        entered.set()
        release.wait(timeout=0.5)
        finished.set()
        return WorkflowApprovalDecision.DENY

    monkeypatch.setattr(
        "plugins.research_protocol.approval.request_workflow_approval",
        blocking_native_approval,
    )
    store = RecordingApprovalStore()
    service = ApprovalService(store=store)

    async def exercise():
        task = asyncio.create_task(service.request(make_scope(), "bounded summary"))
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        await asyncio.sleep(0)
        assert finished.is_set() is False
        release.set()
        assert await task is None
        assert len(store.records) == 1
        assert store.records[0].verdict is ApprovalVerdict.DENIED
        assert store.records[0].scope_json == make_scope().canonical_json()

    asyncio.run(exercise())


def test_approval_service_persists_expired_denial_with_exact_scope(monkeypatch):
    monkeypatch.setattr(
        "plugins.research_protocol.approval.request_workflow_approval",
        lambda _scope, _summary: WorkflowApprovalDecision.DENY,
    )
    expired_scope = make_scope(expires_at=NOW - timedelta(seconds=1))
    store = RecordingApprovalStore()
    service = ApprovalService(store=store)

    assert asyncio.run(service.request(expired_scope, "bounded summary")) is None

    assert len(store.records) == 1
    denied = store.records[0]
    assert denied.verdict is ApprovalVerdict.DENIED
    assert denied.scope_json == expired_scope.canonical_json()
    assert denied.scope_sha256 == expired_scope.scope_sha256()
    assert denied.created_at > denied.expires_at


def test_workflow_approval_always_calls_native_api_even_when_yolo_or_off(
    monkeypatch,
    tmp_path,
):
    import tools.approval as native_approval

    calls = []

    def fake_prompt(message, description, **kwargs):
        calls.append((message, description, kwargs))
        return "once"

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "approvals:\n  mode: off\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(native_approval, "_YOLO_MODE_FROZEN", True)
    monkeypatch.setattr(native_approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(native_approval, "prompt_dangerous_approval", fake_prompt)

    decision = request_workflow_approval(make_scope(), "exact plan summary")

    assert decision == WorkflowApprovalDecision.ACCEPT
    assert len(calls) == 1
    assert calls[0][0] == "exact plan summary"
    assert calls[0][2]["allow_permanent"] is False


def test_workflow_approval_fails_closed_without_explicit_strict_native_contract(
    monkeypatch,
):
    calls = []

    def permissive_legacy_consent(
        message,
        description,
        *,
        timeout_seconds=None,
        surface="mcp-elicitation",
    ):
        calls.append((message, description, timeout_seconds, surface))
        return "accept"

    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        permissive_legacy_consent,
    )

    assert (
        request_workflow_approval(make_scope(), "summary")
        is WorkflowApprovalDecision.DENY
    )
    assert calls == []


def test_workflow_approval_requests_explicit_strict_one_shot(monkeypatch):
    calls = []

    def strict_consent(
        message,
        description,
        *,
        timeout_seconds=None,
        surface="mcp-elicitation",
        strict_one_shot=False,
    ):
        calls.append((message, description, timeout_seconds, surface, strict_one_shot))
        return "accept"

    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        strict_consent,
    )

    assert (
        request_workflow_approval(make_scope(), "summary")
        is WorkflowApprovalDecision.ACCEPT
    )
    assert len(calls) == 1
    assert calls[0][0] == "summary"
    assert calls[0][4] is True


def test_workflow_approval_rejects_expired_scope_before_native_prompt(monkeypatch):
    calls = []

    def recording_prompt(*_args, **_kwargs):
        calls.append(True)
        return "accept"

    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        recording_prompt,
    )
    expired = make_scope(expires_at=datetime.now(UTC) - timedelta(seconds=1))

    assert (
        request_workflow_approval(expired, "summary") is WorkflowApprovalDecision.DENY
    )
    assert calls == []


@pytest.mark.parametrize("native_result", ["decline", "cancel"])
def test_workflow_approval_denies_non_accept_results(monkeypatch, native_result):
    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        lambda *_args, **_kwargs: native_result,
    )

    assert (
        request_workflow_approval(make_scope(), "summary")
        is WorkflowApprovalDecision.DENY
    )


def test_workflow_approval_denies_native_exception(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("native approval unavailable")

    monkeypatch.setattr("tools.approval.request_elicitation_consent", explode)

    assert (
        request_workflow_approval(make_scope(), "summary")
        is WorkflowApprovalDecision.DENY
    )


class FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class FakeConnection:
    def __init__(self, row):
        self.row = row
        self.calls = []
        self.executions = []

    def transaction(self):
        return FakeTransaction()

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return self.row

    async def execute(self, query, *args):
        self.executions.append((query, args))


class FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        return False


class FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return FakeAcquire(self.connection)


def test_postgres_create_uses_fixed_store_function_and_writer_timeout():
    scope = make_scope()
    record = make_approval_record(scope)
    connection = FakeConnection(None)
    store = PostgresApprovalStore(FakePool(connection), timeout_seconds=2.5)

    asyncio.run(store.create_approval(record))

    assert len(connection.executions) == 2
    timeout_sql, timeout_args = connection.executions[0]
    assert "set_config" in timeout_sql
    assert timeout_args == ("2500ms",)
    query, args = connection.executions[1]
    assert "research_protocol.store_approval" in query
    assert "INSERT" not in query
    assert args == (
        record.approval_id,
        record.scope_sha256,
        record.plan_sha256,
        record.scope_json,
        record.verdict.value,
        record.surface,
        record.created_at,
        record.expires_at,
        record.max_executions,
    )


def test_postgres_create_accepts_fresh_denied_record():
    denied = make_approval_record(make_scope(), verdict=ApprovalVerdict.DENIED)
    connection = FakeConnection(None)
    store = PostgresApprovalStore(FakePool(connection))

    asyncio.run(store.create_approval(denied))

    _query, args = connection.executions[1]
    assert args[4] == "denied"


def test_postgres_writer_deadline_includes_pool_acquisition():
    class SlowAcquire:
        async def __aenter__(self):
            await asyncio.sleep(1)

        async def __aexit__(self, *_args):
            return False

    class SlowPool:
        @staticmethod
        def acquire():
            return SlowAcquire()

    store = PostgresApprovalStore(SlowPool(), timeout_seconds=0.01)

    with pytest.raises(ApprovalStoreTimeoutError, match="timed out"):
        asyncio.run(store.create_approval(make_approval_record(make_scope())))


def test_postgres_claim_is_atomic_and_binds_exact_scope_and_hashes():
    scope = make_scope()
    row = {
        "approval_id": "approval-001",
        "scope_sha256": scope.scope_sha256(),
        "plan_sha256": PLAN_SHA,
        "consumed_count": 1,
        "max_executions": 2,
    }
    connection = FakeConnection(row)
    store = PostgresApprovalStore(FakePool(connection))

    claimed = asyncio.run(
        store.claim_approval(
            approval_id="approval-001",
            scope=scope,
            plan_sha256=PLAN_SHA,
        )
    )

    assert claimed is True
    query, args = connection.calls[0]
    assert "research_protocol.claim_approval" in query
    assert "UPDATE" not in query
    assert args[0] == "approval-001"
    assert scope.scope_sha256() in args
    assert PLAN_SHA in args


def test_postgres_claim_returns_false_for_no_matching_atomic_row():
    connection = FakeConnection(None)
    store = PostgresApprovalStore(FakePool(connection))

    claimed = asyncio.run(
        store.claim_approval(
            approval_id="missing",
            scope=make_scope(),
            plan_sha256=PLAN_SHA,
        )
    )

    assert claimed is False
