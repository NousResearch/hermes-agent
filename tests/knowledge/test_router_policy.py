"""Tests for the standard knowledge routing layer.

These tests intentionally target pure routing and adapter orchestration rather
than SiYuan/Hindsight network details. The router should be deterministic,
backend-agnostic, and safe to dry-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from knowledge.adapters.base import ExistingKnowledge, KnowledgeAdapter, WriteResult
from knowledge.adapters.siyuan import run_sql
from knowledge.policy import RoutePolicy
from knowledge.router import KnowledgeRouter, route_knowledge_write
from knowledge.types import (
    DuplicatePolicy,
    KnowledgeDestination,
    KnowledgeWriteRequest,
    RouteAction,
)


@dataclass
class FakeStaticAdapter(KnowledgeAdapter):
    existing: list[ExistingKnowledge] = field(default_factory=list)
    searches: list[KnowledgeWriteRequest] = field(default_factory=list)
    writes: list[KnowledgeWriteRequest] = field(default_factory=list)
    updates: list[tuple[str, KnowledgeWriteRequest]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake-static"

    def search_existing(self, request: KnowledgeWriteRequest) -> list[ExistingKnowledge]:
        self.searches.append(request)
        return list(self.existing)

    def write(self, request: KnowledgeWriteRequest) -> WriteResult:
        self.writes.append(request)
        return WriteResult(success=True, backend=self.name, action="create", id="new-doc")

    def update(self, existing: ExistingKnowledge, request: KnowledgeWriteRequest) -> WriteResult:
        self.updates.append((existing.id, request))
        return WriteResult(success=True, backend=self.name, action="update", id=existing.id)


@dataclass
class FakeDynamicAdapter(KnowledgeAdapter):
    writes: list[KnowledgeWriteRequest] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "fake-dynamic"

    def write(self, request: KnowledgeWriteRequest) -> WriteResult:
        self.writes.append(request)
        return WriteResult(success=True, backend=self.name, action="retain", id="memory-1")


def test_route_static_reference_goes_to_static_knowledge():
    decision = route_knowledge_write(
        KnowledgeWriteRequest(content_type="runbook", title="Backup", content="# Backup")
    )

    assert decision.destination is KnowledgeDestination.STATIC_KNOWLEDGE
    assert decision.action is RouteAction.WRITE_DOCUMENT
    assert decision.reason == "static_reference"
    assert decision.requires_title is True


def test_route_dynamic_memory_goes_to_dynamic_backend():
    decision = route_knowledge_write(
        KnowledgeWriteRequest(content_type="user_preference", content="User prefers concise Chinese.")
    )

    assert decision.destination is KnowledgeDestination.DYNAMIC_MEMORY
    assert decision.action is RouteAction.RETAIN_MEMORY
    assert decision.requires_title is False


def test_route_ephemeral_noise_skips_by_default():
    decision = route_knowledge_write(
        KnowledgeWriteRequest(content_type="task_log", content="pytest passed")
    )

    assert decision.destination is KnowledgeDestination.NONE
    assert decision.action is RouteAction.SKIP
    assert decision.reason == "ephemeral_noise"


def test_unknown_content_type_is_review_only():
    decision = route_knowledge_write(
        KnowledgeWriteRequest(content_type="misc", content="unknown")
    )

    assert decision.destination is KnowledgeDestination.REVIEW
    assert decision.action is RouteAction.DRY_RUN_ONLY
    assert decision.reason == "unknown_content_type"


def test_dry_run_does_not_call_adapters():
    static = FakeStaticAdapter()
    dynamic = FakeDynamicAdapter()
    router = KnowledgeRouter(static_adapter=static, dynamic_adapter=dynamic)

    result = router.write(
        KnowledgeWriteRequest(
            content_type="runbook",
            title="Backup",
            content="# Backup",
            dry_run=True,
        )
    )

    assert result.success is True
    assert result.written is False
    assert result.dry_run is True
    assert static.searches == []
    assert static.writes == []
    assert dynamic.writes == []


def test_static_write_updates_existing_when_duplicate_policy_requests_it():
    existing = ExistingKnowledge(id="doc-1", title="Backup", path="/Backup", score=1.0)
    static = FakeStaticAdapter(existing=[existing])
    router = KnowledgeRouter(static_adapter=static, dynamic_adapter=FakeDynamicAdapter())

    result = router.write(
        KnowledgeWriteRequest(
            content_type="runbook",
            title="Backup",
            content="# Updated",
            duplicate_policy=DuplicatePolicy.UPDATE_EXISTING,
        )
    )

    assert result.success is True
    assert result.written is True
    assert result.action == "update"
    assert result.id == "doc-1"
    assert static.searches
    assert static.writes == []
    assert static.updates == [("doc-1", static.updates[0][1])]


def test_static_write_creates_when_no_existing_match():
    static = FakeStaticAdapter(existing=[])
    router = KnowledgeRouter(static_adapter=static, dynamic_adapter=FakeDynamicAdapter())

    result = router.write(
        KnowledgeWriteRequest(content_type="architecture", title="System", content="# System")
    )

    assert result.success is True
    assert result.action == "create"
    assert static.searches
    assert len(static.writes) == 1


def test_dynamic_write_retains_with_router_tags():
    dynamic = FakeDynamicAdapter()
    router = KnowledgeRouter(static_adapter=FakeStaticAdapter(), dynamic_adapter=dynamic)

    result = router.write(
        KnowledgeWriteRequest(
            content_type="lesson",
            content="Prefer official APIs.",
            context="test context",
            tags=("lesson",),
        )
    )

    assert result.success is True
    assert result.action == "retain"
    assert len(dynamic.writes) == 1
    request = dynamic.writes[0]
    assert request.context == "test context"
    assert request.tags == ("lesson", "knowledge-router", "hindsight")


def test_static_write_requires_title():
    router = KnowledgeRouter(static_adapter=FakeStaticAdapter(), dynamic_adapter=FakeDynamicAdapter())

    result = router.write(KnowledgeWriteRequest(content_type="runbook", content="# Missing title"))

    assert result.success is False
    assert "title" in (result.error or "")


def test_content_type_policy_is_configurable():
    policy = RoutePolicy(static_types=frozenset({"playbook"}))

    decision = route_knowledge_write(
        KnowledgeWriteRequest(content_type="playbook", title="Ops", content="# Ops"),
        policy=policy,
    )

    assert decision.destination is KnowledgeDestination.STATIC_KNOWLEDGE


def test_siyuan_sql_allows_simple_select_by_default():
    class Client:
        def __init__(self):
            self.payload = None

        def post(self, path, payload):
            self.payload = (path, payload)
            return [{"id": "1"}]

    client = Client()

    result = run_sql(client, "select * from blocks")

    assert result == {"success": True, "rows": [{"id": "1"}]}
    assert client.payload == ("/api/query/sql", {"stmt": "select * from blocks limit 20"})


def test_siyuan_sql_blocks_non_select_without_explicit_unsafe():
    class Client:
        def post(self, path, payload):  # pragma: no cover - must not be called
            raise AssertionError("unexpected SQL call")

    with pytest.raises(ValueError, match="allow_unsafe"):
        run_sql(Client(), "delete from blocks")
