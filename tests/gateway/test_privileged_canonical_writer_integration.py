"""Cross-boundary security contracts for the privileged Canonical writer.

These tests exercise the public gateway/tool call sites, not merely the writer
protocol in isolation.  They intentionally keep the model-facing tool schemas
unchanged while proving that database authority and route-back receipts cross
only the typed writer boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.canonical_brain_routeback_context as routeback_context
import gateway.canonical_writer_boundary as writer_boundary
from gateway.canonical_writer_client import (
    CanonicalWriterClient,
    CanonicalWriterClientError,
    ErrorCode,
    ExactServerMainPidAuthorizer,
    ServerPeerCredentials,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.canonical_writer_db import QueryResult
from gateway.canonical_writer_postgres_backend import (
    PRODUCTION_CATALOG_SHA256,
    PRODUCTION_STATEMENT_CATALOG,
)
from gateway.discord_edge_writer_authority import (
    derive_routeback_edge_idempotency_key,
)
from gateway.canonical_writer_bootstrap import build_service
from gateway.canonical_writer_service import DispatchContext, PeerCredentials
import tools.canonical_brain_tool as canonical_tool


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CREDENTIAL_CALL_SITES = (
    REPO_ROOT / "tools" / "canonical_brain_tool.py",
    REPO_ROOT / "gateway" / "canonical_brain_routeback_context.py",
    REPO_ROOT / "gateway" / "canonical_brain_audit.py",
    REPO_ROOT / "scripts" / "canonical_brain_event_projector.py",
)

DISCORD_GUILD_ID = "1282725267068157972"
DISCORD_CHANNEL_ID = "1504852408227069993"
EDGE_REQUEST_ID = "11111111-1111-4111-8111-111111111111"


def _enable_boundary(monkeypatch, call):
    monkeypatch.setattr(writer_boundary, "writer_boundary_configured", lambda *_: True)
    monkeypatch.setattr(writer_boundary, "in_writer_service", lambda: False)
    monkeypatch.setattr(writer_boundary, "canonical_writer_call", call)


def _public_target(_target_ref):
    return {
        "channel_id": DISCORD_CHANNEL_ID,
        "channel_type": "public_channel",
        "target_type": "public_guild_channel",
        "guild_id": DISCORD_GUILD_ID,
        "target_kind": "exact_public_directory_target",
        "target_member_key": None,
        "target_member_id": None,
        "target_mention": None,
    }


def _routeback_kwargs():
    return {
        "case_id": "case:writer-boundary",
        "target_ref": {"channel_id": DISCORD_CHANNEL_ID},
        "message": "A verified public route-back",
        "message_summary": "public route-back",
        "source_refs": {
            "platform": "discord",
            "thread_id": "requester-thread-1",
            "message_id": "request-message-1",
        },
        "idempotency_key": "routeback:writer-boundary:1",
    }


def _prepare_routeback_execution(monkeypatch):
    monkeypatch.setattr(canonical_tool, "_resolve_route_back_public_target", _public_target)
    monkeypatch.setattr(
        canonical_tool,
        "_discord_expected_content_sha256",
        lambda _message: "a" * 64,
    )
    monkeypatch.setattr(canonical_tool, "_discord_edge_preconnect", object)

    class _NoEdgeRecord(RuntimeError):
        code = "discord_edge_reconciliation_not_available"
        dispatch_uncertain = False

    monkeypatch.setattr(
        canonical_tool,
        "_discord_edge_reconcile",
        lambda _client, _intent: (_ for _ in ()).throw(
            _NoEdgeRecord("no exact durable edge record")
        ),
    )


def _signed_edge_request():
    return {
        "protocol": "discord-edge.v1",
        "request_id": EDGE_REQUEST_ID,
        "sequence": 1,
        "deadline_unix_ms": 4_000_000_000_000,
        "operation": "public.message.send",
        "target": {
            "target_type": "public_guild_channel",
            "guild_id": DISCORD_GUILD_ID,
            "channel_id": DISCORD_CHANNEL_ID,
        },
        "payload": {"content": "A verified public route-back"},
        "idempotency_key": derive_routeback_edge_idempotency_key(
            case_id="case:writer-boundary",
            canonical_idempotency_key="routeback:writer-boundary:1",
        ),
        "capability": {
            "key_id": "1" * 64,
            "payload": {
                "protocol": "discord-edge-capability.v1",
                "capability_id": "22222222-2222-4222-8222-222222222222",
            },
            "signature": "A" * 86,
        },
    }


def _signed_edge_receipt(*, outcome="blocked_before_dispatch"):
    return {
        "key_id": "2" * 64,
        "payload": {
            "protocol": "discord-edge-receipt.v1",
            "receipt_id": "33333333-3333-4333-8333-333333333333",
            "edge_request_id": EDGE_REQUEST_ID,
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": DISCORD_CHANNEL_ID,
            },
            "idempotency_key": derive_routeback_edge_idempotency_key(
                case_id="case:writer-boundary",
                canonical_idempotency_key="routeback:writer-boundary:1",
            ),
            "content_sha256": "a" * 64,
            "outcome": outcome,
            "blocker_code": "discord_permission_denied",
        },
        "signature": "B" * 86,
    }


def test_model_append_runs_authoritative_validation_before_writer(monkeypatch):
    order: list[str] = []
    calls: list[tuple[str, dict, str | None]] = []
    original_validator = canonical_tool._validate_append_request

    def validating(**kwargs):
        order.append("validator")
        return original_validator(**kwargs)

    def call(operation, payload, *, idempotency_key=None):
        order.append("writer")
        calls.append((str(operation), dict(payload), idempotency_key))
        return {
            "request_id": "writer-request-1",
            "status": "inserted",
            "success": True,
            "inserted": True,
            "readback_verified": True,
        }

    _enable_boundary(monkeypatch, call)
    monkeypatch.setattr(canonical_tool, "_validate_append_request", validating)
    monkeypatch.setattr(
        canonical_tool,
        "_load_helper",
        lambda: pytest.fail("gateway append attempted direct database access"),
    )

    rejected = json.loads(canonical_tool.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:writer-boundary",
        summary="must be rejected before IPC",
        source_refs={"platform": "discord", "message_id": "message-1"},
        safety={"contains_secret": True},
    ))

    assert "safety flags block append" in rejected["error"]
    assert order == ["validator"]
    assert calls == []

    accepted = json.loads(canonical_tool.canonical_event_append_tool(
        event_type="case.note",
        case_id="case:writer-boundary",
        summary="validated model decision",
        source_refs={"platform": "discord", "message_id": "message-2"},
        payload={"decision": {"reason": "model-owned semantics"}},
        idempotency_key="case:writer-boundary:note:1",
    ))

    assert accepted["success"] is True
    assert order == ["validator", "validator", "writer"]
    assert calls == [(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        {
            "event_type": "case.note",
            "case_id": "case:writer-boundary",
            "summary": "validated model decision",
            "source_refs": {"platform": "discord", "message_id": "message-2"},
            "actors": {},
            "payload": {"decision": {"reason": "model-owned semantics"}},
            "safety": {},
            "idempotency_key": "case:writer-boundary:note:1",
        },
        "case:writer-boundary:note:1",
    )]


def test_verified_writer_alias_event_materializes_local_alias_projection(monkeypatch):
    learned = []

    def call(operation, payload, *, idempotency_key=None):
        assert operation == CanonicalWriterOperation.EVENT_APPEND_MODEL.value
        assert idempotency_key == "alias:event:1"
        return {
            "success": True,
            "event_id": "11111111-1111-4111-8111-111111111111",
            "inserted": True,
            "deduped": False,
        }

    _enable_boundary(monkeypatch, call)
    monkeypatch.setattr(
        "gateway.support_ops_team_registry.learn_team_member_alias",
        lambda alias, member_key: learned.append((alias, member_key)) or {
            "alias": alias,
            "member_key": member_key,
        },
    )

    result = json.loads(canonical_tool.canonical_event_append_tool(
        event_type="person.alias.learned",
        case_id="case:alias",
        summary="Requester clarified the teammate alias",
        source_refs={"platform": "discord", "message_id": "message-alias-1"},
        payload={"alias": "Niki", "member_key": "nikolay"},
        idempotency_key="alias:event:1",
    ))

    assert learned == [("Niki", "nikolay")]
    assert result["success"] is True
    assert result["alias"]["member_key"] == "nikolay"


def test_gateway_and_projector_have_no_legacy_helper_or_secret_path():
    """Guard the complete public runtime surface against credential regression."""

    forbidden = (
        "cloud_sql_synthetic_write_gate",
        "spec_from_file_location",
        "get_secret_value",
        "CLOUD_SQL_HELPER",
        "_CLOUD_SQL_HELPER",
    )
    violations: list[str] = []
    for path in LEGACY_CREDENTIAL_CALL_SITES:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{token}")

    assert violations == [], "legacy Canonical credential path remains: " + ", ".join(
        violations
    )


@pytest.mark.parametrize(
    "claim",
    [
        {"success": False, "inserted": False, "error_code": "writer_unavailable"},
        {
            "success": True,
            "inserted": False,
            "deduped": True,
            "readback_verified": True,
        },
    ],
)
def test_routeback_never_sends_without_new_durable_claim(monkeypatch, claim):
    operations: list[str] = []
    edge_calls: list[tuple[object, dict]] = []

    def call(operation, payload, *, idempotency_key=None):
        operations.append(str(operation))
        assert operation == CanonicalWriterOperation.ROUTEBACK_CLAIM.value
        assert payload["discord_edge_intent"] == {
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": DISCORD_CHANNEL_ID,
            },
            "payload": {"content": "A verified public route-back"},
            "idempotency_key": derive_routeback_edge_idempotency_key(
                case_id="case:writer-boundary",
                canonical_idempotency_key="routeback:writer-boundary:1",
            ),
        }
        return dict(claim)

    _enable_boundary(monkeypatch, call)
    _prepare_routeback_execution(monkeypatch)
    monkeypatch.setattr(
        canonical_tool,
        "_discord_edge_execute",
        lambda client, request: edge_calls.append((client, request)),
    )

    result = json.loads(canonical_tool.route_back_execute_tool(**_routeback_kwargs()))

    assert edge_calls == []
    assert operations == [CanonicalWriterOperation.ROUTEBACK_CLAIM.value]
    assert result["status"] in {
        "ROUTE_BACK_EXECUTE_INTENT_FAILED",
        "ROUTE_BACK_EXECUTE_OUTCOME_UNCERTAIN_PENDING_RECONCILIATION",
    }


def test_child_owned_client_and_fake_writer_identity_cannot_forge_claim(monkeypatch):
    class _NeverAuthorized:
        def authorize(self, _peer):
            return False

    client = CanonicalWriterClient(
        "/tmp/canonical-writer-must-not-connect.sock",
        server_authorizer=_NeverAuthorized(),
    )
    monkeypatch.setattr(client, "_owner_pid", os.getpid() + 1)

    with pytest.raises(CanonicalWriterClientError) as exc_info:
        client.call(
            CanonicalWriterOperation.ROUTEBACK_CLAIM,
            {},
            runtime={},
        )

    assert exc_info.value.code == ErrorCode.UNAUTHORIZED_PEER
    assert client.fileno == -1

    class _WriterMainPid:
        @staticmethod
        def main_pid(unit_name):
            assert unit_name == "muncho-canonical-writer.service"
            return 9001

    authorizer = ExactServerMainPidAuthorizer(
        server_unit="muncho-canonical-writer.service",
        expected_server_uid=2002,
        main_pid_provider=_WriterMainPid(),
    )
    assert authorizer.authorize(ServerPeerCredentials(9001, 2002, 2002)) is True
    assert authorizer.authorize(ServerPeerCredentials(9002, 2002, 2002)) is False
    assert authorizer.authorize(ServerPeerCredentials(9001, 2001, 2002)) is False


def test_nonverified_edge_receipt_uses_typed_writer_finalize_blocked(monkeypatch):
    calls: list[tuple[str, dict, str | None]] = []
    edge_request = _signed_edge_request()
    edge_receipt = _signed_edge_receipt()

    def call(operation, payload, *, idempotency_key=None):
        calls.append((str(operation), dict(payload), idempotency_key))
        if operation == CanonicalWriterOperation.ROUTEBACK_CLAIM.value:
            return {
                "success": True,
                "inserted": True,
                "readback_verified": True,
                "discord_edge_request": edge_request,
            }
        if operation == CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value:
            return {
                "success": True,
                "inserted": True,
                "outcome": "blocked",
            }
        pytest.fail(f"unexpected writer operation: {operation}")

    _enable_boundary(monkeypatch, call)
    _prepare_routeback_execution(monkeypatch)
    monkeypatch.setattr(
        canonical_tool,
        "_discord_edge_execute",
        lambda client, request: {
            "state": "blocked",
            "blocker": "discord_permission_denied",
            "replayed": False,
            "receipt": edge_receipt,
        },
    )

    result = json.loads(canonical_tool.route_back_execute_tool(**_routeback_kwargs()))

    assert result["status"] == "ROUTE_BACK_EXECUTE_BLOCKED"
    assert [item[0] for item in calls] == [
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED.value,
    ]
    assert calls[0][1]["execution_binding"] == calls[1][1]["execution_binding"]
    assert calls[1][1]["discord_edge_request"] == edge_request
    assert calls[1][1]["discord_edge_receipt"] == edge_receipt
    assert "blocker_reason" not in calls[1][1]


def test_public_query_active_plan_and_routeback_context_contracts(monkeypatch):
    calls: list[tuple[str, dict]] = []

    query_result = {
        "success": True,
        "status": "CANONICAL_BRAIN_QUERY_PASS",
        "query": {
            "case_id": "case:writer-boundary",
            "thread_id": None,
            "limit": 25,
            "view": "summary",
        },
        "event_count": 1,
        "window_event_count": 1,
        "support_event_count": 0,
        "support_incomplete": False,
        "support": {
            "complete": True,
            "reasons": [],
            "missing_verification_event_ids": [],
        },
        "truncated": False,
        "candidate_cases_truncated": False,
        "case_count": 1,
        "cases": [{"case_id": "case:writer-boundary"}],
    }

    def call(operation, payload, *, idempotency_key=None):
        assert idempotency_key is None
        calls.append((str(operation), dict(payload)))
        if operation == CanonicalWriterOperation.CASE_QUERY.value:
            return dict(query_result)
        if operation == CanonicalWriterOperation.PLAN_ACTIVE_MATCH.value:
            return {"matches": True, "active": True, "plan_revision": 1}
        if operation == CanonicalWriterOperation.ROUTEBACK_CONTEXT.value:
            return {
                "thread_id": "owner-thread",
                "cases": [
                    {
                        "case_id": f"case:linked-{index}",
                        "source_thread_id": f"requester-{index}",
                    }
                    for index in range(1, 5)
                ],
                "truncated": True,
            }
        pytest.fail(f"unexpected operation: {operation}")

    _enable_boundary(monkeypatch, call)

    query = json.loads(canonical_tool.canonical_brain_query_tool(
        case_id="case:writer-boundary",
        limit=25,
    ))
    active = canonical_tool.canonical_active_plan_matches(
        case_id="case:writer-boundary",
        plan_id="plan:writer-boundary",
    )
    context = routeback_context.lookup_routeback_context_for_thread("owner-thread")

    assert query == query_result
    assert active is True
    assert context == routeback_context.RouteBackContextLookup(
        cases=(
            routeback_context.RouteBackCaseContext("case:linked-1", "requester-1"),
            routeback_context.RouteBackCaseContext("case:linked-2", "requester-2"),
            routeback_context.RouteBackCaseContext("case:linked-3", "requester-3"),
        ),
        truncated=True,
    )
    assert calls == [
        (
            CanonicalWriterOperation.CASE_QUERY.value,
            {
                "case_id": "case:writer-boundary",
                "thread_id": "",
                "limit": 25,
                "view": "summary",
            },
        ),
        (
            CanonicalWriterOperation.PLAN_ACTIVE_MATCH.value,
            {
                "case_id": "case:writer-boundary",
                "plan_id": "plan:writer-boundary",
            },
        ),
        (
            CanonicalWriterOperation.ROUTEBACK_CONTEXT.value,
            {"thread_id": "owner-thread"},
        ),
    ]


def test_raw_privileged_query_is_folded_into_public_contract(monkeypatch):
    event_id = "11111111-1111-4111-8111-111111111111"

    def call(operation, payload, *, idempotency_key=None):
        assert operation == CanonicalWriterOperation.CASE_QUERY.value
        assert idempotency_key is None
        return {
            "events": [
                {
                    "event_id": event_id,
                    "schema_version": "canonical_event.v1",
                    "event_type": "case.note",
                    "occurred_at": "2026-07-12T10:00:00+00:00",
                    "case_id": "case:writer-boundary",
                    "source": {},
                    "actor": {},
                    "subject": {},
                    "evidence": [],
                    "decision": {},
                    "status": {"summary": "Scoped durable note"},
                    "next_action": {},
                    "safety": {},
                    "payload": {"summary": "Scoped durable note"},
                }
            ],
            "support_events": [],
            "view": "summary",
            "has_more": False,
            "candidate_cases_truncated": False,
            "support_incomplete_reasons": [],
            "missing_verification_event_ids": [],
        }

    _enable_boundary(monkeypatch, call)

    result = json.loads(canonical_tool.canonical_brain_query_tool(
        case_id="case:writer-boundary",
        limit=25,
    ))

    assert result["success"] is True
    assert result["status"] == "CANONICAL_BRAIN_QUERY_PASS"
    assert result["event_count"] == 1
    assert result["support"]["complete"] is True
    assert result["case_count"] == 1
    assert result["cases"][0]["case_id"] == "case:writer-boundary"
    assert result["cases"][0]["summary"] == "Scoped durable note"


def test_bootstrap_dispatcher_preserves_flat_public_writer_contract(monkeypatch):
    """Exercise the dispatcher actually assembled by the production bootstrap."""

    class _Database:
        statement_names = PRODUCTION_STATEMENT_CATALOG.names
        statement_catalog_sha256 = PRODUCTION_CATALOG_SHA256

        def __init__(self, **_kwargs):
            self.attested = False

        def startup_attest(self):
            self.attested = True

        def query_fixed(self, statement_name, parameters):
            assert statement_name == "op_plan_active_match"
            assert parameters["request"] == {
                "case_id": "case:writer-boundary",
                "plan_id": "plan:writer-boundary",
            }
            return QueryResult(
                ("response",),
                ((json.dumps({
                    "ok": True,
                    "result": {"matches": True, "active": True},
                }),),),
                "SELECT 1",
            )

    config = SimpleNamespace(
        writer_uid=2002,
        writer_gid=2002,
        socket_gid=2001,
        gateway_uid=2001,
        owner_discord_user_ids=frozenset({"owner-1"}),
        gateway_unit="hermes-cloud-gateway.service",
        socket_path=Path("/run/muncho-canonical-writer/writer.sock"),
        connection_timeout_seconds=2.0,
        max_connections=1,
        database=object(),
        privileges=object(),
        discord_edge_authority=SimpleNamespace(enabled=False),
    )
    monkeypatch.setattr(os, "getuid", lambda: 2002)
    monkeypatch.setattr(os, "getgid", lambda: 2002)

    bootstrap = build_service(config, _database_factory=_Database)
    response = bootstrap.server.dispatcher.dispatch(
        CanonicalWriterOperation.PLAN_ACTIVE_MATCH,
        {
            "case_id": "case:writer-boundary",
            "plan_id": "plan:writer-boundary",
        },
        DispatchContext(
            request_id="11111111-1111-4111-8111-111111111111",
            sequence=1,
            deadline_unix_ms=1,
            idempotency_key=None,
            peer=PeerCredentials(pid=1234, uid=2001, gid=2001),
            runtime={"platform": "discord", "thread_id": "thread-1"},
        ),
    )

    assert bootstrap.database.attested is True
    assert response.status == "ok"
    assert response.result == {"matches": True, "active": True}
