"""Contracts for the closed planner tool surface."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import threading

from plugins.research_protocol import register
from plugins.research_protocol.approval import (
    ApprovalRecord,
    ApprovalScope,
    ApprovalVerdict,
)
from plugins.research_protocol.models import (
    BudgetLimits,
    Capability,
    CapabilityGrant,
    ExternalRight,
    OutputRequirement,
    PlanV1,
)
from plugins.research_protocol.planner_tools import (
    PlannerToolHandlers,
    _scope_matches_plan,
    configure_planner_runtime,
    plan_context_read,
)
from plugins.research_protocol.storage.artifacts import ArtifactReceipt, ArtifactStore

EXPECTED_TOOLS = {
    "plan_context_read",
    "plan_artifact_write",
    "plan_approval_request",
}
PLAN_SHA_INPUT = "a" * 64


def _plan_payload() -> dict:
    return {
        "schema_version": "plan.v1",
        "producer_version": "test",
        "run_id": "run-001",
        "created_at": datetime(2026, 7, 14, 2, 0, tzinfo=UTC),
        "objective": "SECRET_OBJECTIVE_MUST_NOT_REACH_CONSENT",
        "constraints": ("offline",),
        "inputs": (),
        "outputs": (
            OutputRequirement(
                artifact_id="manifest-001",
                artifact_type="manifest",
            ),
        ),
        "budgets": BudgetLimits(
            max_duration_seconds=60,
            max_executions=1,
            max_records=10,
            max_bytes=1000,
            max_external_calls=0,
        ),
        "capabilities": (
            CapabilityGrant(
                capability=Capability.RESEARCH_COLLECT,
                input_hashes=(PLAN_SHA_INPUT,),
                external_rights=(),
            ),
        ),
    }


def _approval_scope(plan_sha256: str, **changes) -> ApprovalScope:
    values = {
        "capability": Capability.RESEARCH_COLLECT,
        "run_id": "run-001",
        "plan_sha256": plan_sha256,
        "input_hashes": [PLAN_SHA_INPUT],
        "budgets": BudgetLimits(
            max_duration_seconds=60,
            max_executions=1,
            max_records=10,
            max_bytes=1000,
            max_external_calls=0,
        ),
        "expires_at": datetime(2099, 1, 1, tzinfo=UTC),
        "max_executions": 1,
        "external_rights": [],
    }
    values.update(changes)
    return ApprovalScope.model_validate(values)


class RecordingContext:
    def __init__(self):
        self.calls = []

    def register_tool(self, **kwargs):
        self.calls.append(kwargs)


def test_register_exposes_exactly_three_planner_tools():
    context = RecordingContext()

    register(context)

    assert {call["name"] for call in context.calls} == EXPECTED_TOOLS
    assert {call["toolset"] for call in context.calls} == {"planner"}
    assert all(call["check_fn"] is not None for call in context.calls)
    assert all(call["is_async"] is True for call in context.calls)


def test_planner_tool_schemas_have_closed_top_level_inputs_without_free_authority():
    context = RecordingContext()
    register(context)

    for call in context.calls:
        parameters = call["schema"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert not ({"path", "sql", "dsn"} & set(parameters.get("properties", {})))

    schemas = {call["name"]: call["schema"] for call in context.calls}
    assert schemas["plan_context_read"]["parameters"]["properties"]["query_id"]["enum"]
    assert schemas["plan_artifact_write"]["parameters"]["properties"]["artifact_type"][
        "enum"
    ]
    assert "summary" not in schemas["plan_approval_request"]["parameters"]["properties"]

    def assert_objects_are_closed(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            for value in node.values():
                assert_objects_are_closed(value)
        elif isinstance(node, list):
            for value in node:
                assert_objects_are_closed(value)

    for schema in schemas.values():
        assert_objects_are_closed(schema["parameters"])


class FakeContextReader:
    def __init__(self):
        self.calls = []

    async def read(self, query_id, parameters):
        self.calls.append((query_id, parameters))
        return [{"approval_id": parameters["approval_id"], "verdict": "approved"}]


class FailingContextReader:
    async def read(self, query_id, parameters):
        del query_id, parameters
        raise RuntimeError("postgresql://user:secret@example.invalid/database")


class FailingArtifactStore:
    def persist(self, artifact_type, artifact_id, payload):
        del artifact_type, artifact_id, payload
        raise OSError("cannot write /operator/private/artifacts")


def test_artifact_write_does_not_block_the_event_loop():
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingArtifactStore(ArtifactStore):
        def __init__(self):
            pass

        def persist(self, artifact_type, artifact_id, payload):
            del payload
            entered.set()
            release.wait(timeout=0.5)
            finished.set()
            return ArtifactReceipt(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                schema_version="plan.v1",
                path_relative="plans/plan-001.json",
                sha256="a" * 64,
                byte_length=1,
                created_at="2026-07-14T02:00:00+00:00",
            )

    handlers = PlannerToolHandlers(
        artifact_store=BlockingArtifactStore(),
        context_reader=None,
        approval_service=None,
    )

    async def scenario():
        task = asyncio.create_task(
            handlers.artifact_write({
                "artifact_type": "plan",
                "artifact_id": "plan-001",
                "payload": _plan_payload(),
            })
        )
        while not entered.is_set():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0)
        assert not finished.is_set(), "synchronous artifact I/O blocked the event loop"
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert result["ok"] is True


class RecordingApprovalService:
    def __init__(self):
        self.calls = []

    async def request(self, scope, summary):
        self.calls.append((scope, summary))
        return ApprovalRecord(
            approval_id="approval-record-000001",
            scope_sha256=scope.scope_sha256(),
            plan_sha256=scope.plan_sha256,
            scope_json=scope.canonical_json(),
            verdict=ApprovalVerdict.APPROVED,
            surface="test",
            created_at=datetime(2026, 7, 14, 2, 0, tzinfo=UTC),
            expires_at=scope.expires_at,
            max_executions=scope.max_executions,
            consumed_count=0,
        )


def test_approval_plan_read_does_not_block_the_event_loop():
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    plan_bytes = (
        PlanV1.model_validate(_plan_payload()).model_dump_json().encode("utf-8")
    )

    class BlockingArtifactStore(ArtifactStore):
        def __init__(self):
            pass

        def read_verified(
            self,
            artifact_type,
            artifact_id,
            *,
            expected_sha256=None,
        ):
            del artifact_type, artifact_id, expected_sha256
            entered.set()
            release.wait(timeout=0.5)
            finished.set()
            return plan_bytes

    expected_sha256 = "c" * 64
    handlers = PlannerToolHandlers(
        artifact_store=BlockingArtifactStore(),
        context_reader=None,
        approval_service=RecordingApprovalService(),
    )

    async def scenario():
        task = asyncio.create_task(
            handlers.approval_request({
                "artifact_id": "plan-001",
                "expected_sha256": expected_sha256,
                "scope": _approval_scope(expected_sha256),
            })
        )
        while not entered.is_set():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0)
        assert not finished.is_set(), "synchronous plan read blocked the event loop"
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert result["ok"] is True
    assert result["approved"] is True


def _handlers(tmp_path, *, approvals=None, reader=None):
    return PlannerToolHandlers(
        artifact_store=ArtifactStore(tmp_path),
        context_reader=reader or FakeContextReader(),
        approval_service=approvals or RecordingApprovalService(),
    )


def test_context_read_delegates_only_query_id_and_validated_parameters(tmp_path):
    reader = FakeContextReader()
    handlers = _handlers(tmp_path, reader=reader)

    result = asyncio.run(
        handlers.context_read({
            "query_id": "approval_status.v1",
            "parameters": {"approval_id": "approval-001"},
        })
    )

    assert result == {
        "ok": True,
        "query_id": "approval_status.v1",
        "rows": [{"approval_id": "approval-001", "verdict": "approved"}],
        "row_count": 1,
    }
    assert reader.calls == [("approval_status.v1", {"approval_id": "approval-001"})]


def test_context_read_redacts_dependency_error_details(tmp_path):
    handlers = _handlers(tmp_path, reader=FailingContextReader())

    result = asyncio.run(
        handlers.context_read({
            "query_id": "approval_status.v1",
            "parameters": {"approval_id": "approval-001"},
        })
    )

    assert result == {"ok": False, "error": "planner context read failed"}
    assert "secret" not in repr(result)


def test_registered_handler_accepts_registry_keyword_arguments(tmp_path):
    reader = FakeContextReader()
    configure_planner_runtime(_handlers(tmp_path, reader=reader))
    try:
        result = asyncio.run(
            plan_context_read(
                query_id="approval_status.v1",
                parameters={"approval_id": "approval-001"},
            )
        )
    finally:
        configure_planner_runtime(None)

    assert result["ok"] is True
    assert reader.calls == [("approval_status.v1", {"approval_id": "approval-001"})]


def test_artifact_write_persists_without_accepting_a_path(tmp_path):
    result = asyncio.run(
        _handlers(tmp_path).artifact_write({
            "artifact_type": "plan",
            "artifact_id": "plan-001",
            "payload": _plan_payload(),
        })
    )

    assert result["ok"] is True
    assert result["receipt"]["path_relative"] == "plans/plan-001.json"
    assert set(result["receipt"]) == {
        "artifact_id",
        "artifact_type",
        "schema_version",
        "path_relative",
        "sha256",
        "byte_length",
        "created_at",
    }


def test_artifact_write_accepts_json_deserialized_payload(tmp_path):
    json_payload = json.loads(PlanV1.model_validate(_plan_payload()).model_dump_json())

    result = asyncio.run(
        _handlers(tmp_path).artifact_write({
            "artifact_type": "plan",
            "artifact_id": "plan-json-boundary",
            "payload": json_payload,
        })
    )

    assert result["ok"] is True
    assert result["receipt"]["path_relative"] == "plans/plan-json-boundary.json"
    persisted = json.loads(
        (tmp_path / result["receipt"]["path_relative"]).read_text(encoding="utf-8")
    )
    assert persisted == json_payload


def test_artifact_write_redacts_filesystem_error_details():
    handlers = PlannerToolHandlers(
        artifact_store=FailingArtifactStore(),
        context_reader=FakeContextReader(),
        approval_service=RecordingApprovalService(),
    )

    result = asyncio.run(
        handlers.artifact_write({
            "artifact_type": "plan",
            "artifact_id": "plan-redaction",
            "payload": _plan_payload(),
        })
    )

    assert result == {"ok": False, "error": "artifact write failed"}
    assert "/operator/private" not in repr(result)


def test_scope_matching_is_order_independent_for_exact_hashes_and_rights():
    plan_sha256 = "c" * 64
    payload = _plan_payload()
    payload["capabilities"] = (
        CapabilityGrant(
            capability=Capability.RESEARCH_COLLECT,
            input_hashes=("b" * 64, "a" * 64),
            external_rights=(
                ExternalRight.NETWORK_READ,
                ExternalRight.DATABASE_READ,
            ),
        ),
    )
    plan = PlanV1.model_validate(payload)
    scope = _approval_scope(
        plan_sha256,
        input_hashes=["a" * 64, "b" * 64],
        external_rights=[
            ExternalRight.DATABASE_READ,
            ExternalRight.NETWORK_READ,
        ],
    )

    assert _scope_matches_plan(scope, plan, plan_sha256) is True


def test_approval_request_reloads_exact_plan_and_builds_fixed_redacted_summary(
    tmp_path,
):
    store = ArtifactStore(tmp_path)
    receipt = store.persist("plan", "plan-001", _plan_payload())
    approvals = RecordingApprovalService()
    handlers = PlannerToolHandlers(
        artifact_store=store,
        context_reader=FakeContextReader(),
        approval_service=approvals,
    )

    result = asyncio.run(
        handlers.approval_request({
            "artifact_id": "plan-001",
            "expected_sha256": receipt.sha256,
            "scope": _approval_scope(receipt.sha256),
        })
    )

    assert result["ok"] is True
    assert result["approval"]["approval_id"] == "approval-record-000001"
    assert len(approvals.calls) == 1
    scope, summary = approvals.calls[0]
    assert receipt.sha256 in summary
    assert f"scope_sha256={scope.scope_sha256()}" in summary
    assert f"scope_json={scope.canonical_json()}" in summary
    assert "SECRET_OBJECTIVE" not in summary


def test_approval_request_rejects_scope_drift_before_native_consent(tmp_path):
    store = ArtifactStore(tmp_path)
    receipt = store.persist("plan", "plan-001", _plan_payload())
    approvals = RecordingApprovalService()
    handlers = PlannerToolHandlers(
        artifact_store=store,
        context_reader=FakeContextReader(),
        approval_service=approvals,
    )

    result = asyncio.run(
        handlers.approval_request({
            "artifact_id": "plan-001",
            "expected_sha256": receipt.sha256,
            "scope": _approval_scope(receipt.sha256, run_id="other-run"),
        })
    )

    assert result == {"ok": False, "error": "approval scope does not match plan"}
    assert approvals.calls == []
