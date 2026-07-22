"""Tests for ProviderHealthMonitor and ProviderHealthSnapshot."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agent.contracts import ContractIdentity
from agent.providers import registry as provider_registry
from agent.providers.fake import FakeConfig, FakeProviderAdapter
from agent.provider_adapter import ProviderHealth
from agent.provider_health_monitor import (
    ProviderHealthMonitor,
    ProviderHealthSnapshot,
    ProviderHealthStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_provider_registry():
    original_get = provider_registry.get_provider
    yield
    provider_registry.get_provider = original_get


def _patch_fake_health(health: ProviderHealth | None = None) -> FakeConfig:
    config = FakeConfig(configured_health=health)
    original_get = provider_registry.get_provider

    def patched_get(name):
        if name == "fake":
            return FakeProviderAdapter(provider_name="fake", config=config)
        return original_get(name)

    provider_registry.get_provider = patched_get
    return config


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_contract_identity(self) -> None:
        snap = ProviderHealthSnapshot(
            provider="fake",
            status=ProviderHealthStatus.AVAILABLE,
            is_available=True,
            consecutive_failures=0,
            last_error=None,
            last_checked_at_utc="2026-06-27T00:00:00Z",
        )
        ident = snap.contract_identity()
        assert isinstance(ident, ContractIdentity)
        assert ident.contract_name == "provider_health_snapshot"
        assert ident.contract_version == 1
        assert ident.schema_version == 1

    def test_to_dict_serializes_contract_fields(self) -> None:
        snap = ProviderHealthSnapshot(
            provider="fake",
            status=ProviderHealthStatus.QUOTA_BLOCKED,
            is_available=False,
            consecutive_failures=3,
            last_error="quota_exceeded",
            last_checked_at_utc="2026-06-27T00:00:00Z",
        )
        d = snap.to_dict()
        assert d["contract_name"] == "provider_health_snapshot"
        assert d["contract_version"] == 1
        assert d["schema_version"] == 1
        assert d["status"] == "quota_blocked"
        assert d["provider"] == "fake"

    def test_is_frozen(self) -> None:
        snap = ProviderHealthSnapshot(
            provider="fake",
            status=ProviderHealthStatus.AVAILABLE,
            is_available=True,
            consecutive_failures=0,
            last_error=None,
            last_checked_at_utc="",
        )
        with pytest.raises(FrozenInstanceError):
            snap.provider = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------


class TestStatusEnum:
    def test_statuses_are_closed(self) -> None:
        # Spot check the canonical set is present.
        names = {s.name for s in ProviderHealthStatus}
        for required in (
            "AVAILABLE",
            "QUOTA_BLOCKED",
            "RATE_LIMITED",
            "CONTEXT_LIMIT",
            "AUTH_BLOCKED",
            "TRANSIENT",
            "UNKNOWN",
        ):
            assert required in names


# ---------------------------------------------------------------------------
# Monitor: contract + identity
# ---------------------------------------------------------------------------


class TestMonitorContract:
    def test_component_identity(self) -> None:
        m = ProviderHealthMonitor()
        ident = m.component_identity()
        assert ident.component_name == "provider_health_monitor"
        assert ident.component_version == 1

    def test_component_contract(self) -> None:
        m = ProviderHealthMonitor()
        contract = m.component_contract()
        assert contract.identity.component_name == "provider_health_monitor"
        compat = contract.supported_contracts.get("provider_health_snapshot")
        assert compat is not None
        assert compat.min_contract_version == 1
        assert compat.max_contract_version == 1


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


class TestStatusMapping:
    def test_available_when_no_failures(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=True,
                consecutive_failures=0,
                last_error=None,
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.AVAILABLE
        assert snap.is_available is True

    def test_quota_blocked_when_error_indicates_quota(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=5,
                last_error="quota_exceeded: monthly credits exhausted",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.QUOTA_BLOCKED
        assert snap.is_available is False

    def test_rate_limited_when_error_indicates_rate(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=2,
                last_error="rate_limit_exceeded: 429",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.RATE_LIMITED

    def test_context_limit_when_error_indicates_context(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=1,
                last_error="context_length_exceeded: 128k tokens",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.CONTEXT_LIMIT

    def test_auth_blocked_when_error_indicates_auth(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=1,
                last_error="401 unauthorized: invalid_api_key",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.AUTH_BLOCKED

    def test_transient_when_error_indicates_transient(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=1,
                last_error="503 service unavailable",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.TRANSIENT

    def test_unknown_when_no_error_and_not_available(self) -> None:
        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=0,
                last_error=None,
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        snap = ProviderHealthMonitor().check("fake")
        assert snap.status is ProviderHealthStatus.UNKNOWN
        assert snap.is_available is False


# ---------------------------------------------------------------------------
# Unknown provider / no provider
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_provider_returns_unknown(self) -> None:
        snap = ProviderHealthMonitor().check("not_registered")
        assert snap.status is ProviderHealthStatus.UNKNOWN
        assert snap.is_available is False
        assert "unknown_provider" in (snap.last_error or "")

    def test_no_provider_returns_unknown(self) -> None:
        snap = ProviderHealthMonitor().check(None)
        assert snap.status is ProviderHealthStatus.UNKNOWN
        assert snap.provider == "(none)"
        assert snap.is_available is False


# ---------------------------------------------------------------------------
# LLMExecutor integration: monitor replaces inline probe
# ---------------------------------------------------------------------------


class TestExecutorIntegration:
    def test_llm_executor_uses_monitor_by_default(self) -> None:
        from agent.llm_executor import LLMExecutor

        executor = LLMExecutor(engine=object())
        assert executor.health_monitor is not None
        ident = executor.health_monitor.component_identity()
        assert ident.component_name == "provider_health_monitor"

    def test_llm_executor_accepts_custom_monitor(self) -> None:
        from agent.llm_executor import LLMExecutor

        custom = ProviderHealthMonitor()
        executor = LLMExecutor(engine=object(), health_monitor=custom)
        assert executor.health_monitor is custom

    def test_llm_executor_blocks_on_unavailable_provider(self) -> None:
        """When monitor reports is_available=False, executor blocks.

        Crucially: NO automatic fallback to any other provider.
        """
        from agent.execution_dispatcher import (
            ExecutionContext,
            ExecutionDispatchRequest,
            ExecutionTrace,
        )
        from agent.execution_router import ExecutionConstraints, ExecutionDecision
        from agent.llm_executor import LLMExecutor

        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=3,
                last_error="quota_exceeded",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )

        executor = LLMExecutor(engine=object())  # engine never invoked.
        decision = ExecutionDecision(
            decision_id="exec:h-001",
            decided_at_utc="2026-06-27T17:00:00Z",
            execution_mode="llm",
            provider="fake",
            requires_worker=False,
            requires_human_approval=False,
            safety_level="safe",
            estimated_cost_usd_micros=0,
            estimated_latency_ms=0,
            rationale="",
            input_hash="",
            decision_hash="",
            fallback_chain=(),
            execution_constraints=ExecutionConstraints(),
            provider_selection=None,
            intent_type_ref="",
            routing_strategy_ref="",
            orchestrator_decision_id="",
        )
        ctx = ExecutionContext.new(
            context_id="c",
            conversation_id="conv",
            request_id="req",
            created_at_utc="2026-06-27T17:00:00Z",
        )
        trace = ExecutionTrace(
            trace_id="t",
            conversation_id=ctx.identity.conversation_id,
            request_id=ctx.identity.request_id,
            decision_id=decision.decision_id,
        )
        request = ExecutionDispatchRequest(trace=trace, decision=decision, context=ctx)
        from agent.execution_dispatcher import DispatchStatus

        result = executor.execute(request)
        assert result.status is DispatchStatus.BLOCKED
        assert result.error == "provider_unavailable"
        # Critical: provider stays "fake", no automatic fallback.
        assert result.provider == "fake"
        # Health output is the structured snapshot dict.
        assert result.output["health"]["status"] == "quota_blocked"

    def test_llm_executor_passes_through_when_available(self) -> None:
        from agent.execution_dispatcher import (
            DispatchStatus,
            ExecutionContext,
            ExecutionDispatchRequest,
            ExecutionTrace,
        )
        from agent.execution_router import ExecutionConstraints, ExecutionDecision
        from agent.llm_executor import LLMExecutor
        from agent.llm_execution_engine import LLMExecutionEngine

        _patch_fake_health()  # default: available, no failures
        executor = LLMExecutor(engine=LLMExecutionEngine())
        decision = ExecutionDecision(
            decision_id="exec:h-002",
            decided_at_utc="2026-06-27T17:00:00Z",
            execution_mode="llm",
            provider="fake",
            requires_worker=False,
            requires_human_approval=False,
            safety_level="safe",
            estimated_cost_usd_micros=0,
            estimated_latency_ms=0,
            rationale="",
            input_hash="",
            decision_hash="",
            fallback_chain=(),
            execution_constraints=ExecutionConstraints(),
            provider_selection=None,
            intent_type_ref="",
            routing_strategy_ref="",
            orchestrator_decision_id="",
        )
        ctx = ExecutionContext.new(
            context_id="c",
            conversation_id="conv",
            request_id="req",
            created_at_utc="2026-06-27T17:00:00Z",
        )
        ctx.record_runtime("user_input", "hi")
        trace = ExecutionTrace(
            trace_id="t",
            conversation_id=ctx.identity.conversation_id,
            request_id=ctx.identity.request_id,
            decision_id=decision.decision_id,
        )
        request = ExecutionDispatchRequest(trace=trace, decision=decision, context=ctx)
        result = executor.execute(request)
        assert result.status is DispatchStatus.OK

    def test_llm_executor_does_not_call_engine_when_blocked(self) -> None:
        """Regression: monitor block must short-circuit before engine.process()."""
        from agent.execution_dispatcher import (
            ExecutionContext,
            ExecutionDispatchRequest,
            ExecutionTrace,
        )
        from agent.execution_router import ExecutionConstraints, ExecutionDecision
        from agent.llm_executor import LLMExecutor

        class _SpyEngine:
            def __init__(self) -> None:
                self.calls = 0

            def process(self, *args, **kwargs):  # pragma: no cover
                self.calls += 1
                raise AssertionError("engine.process must not be called")

        _patch_fake_health(
            ProviderHealth(
                is_available=False,
                consecutive_failures=5,
                last_error="quota_exceeded",
                last_checked_at_utc="2026-06-27T00:00:00Z",
            )
        )
        engine = _SpyEngine()
        executor = LLMExecutor(engine=engine)
        decision = ExecutionDecision(
            decision_id="exec:h-003",
            decided_at_utc="2026-06-27T17:00:00Z",
            execution_mode="llm",
            provider="fake",
            requires_worker=False,
            requires_human_approval=False,
            safety_level="safe",
            estimated_cost_usd_micros=0,
            estimated_latency_ms=0,
            rationale="",
            input_hash="",
            decision_hash="",
            fallback_chain=(),
            execution_constraints=ExecutionConstraints(),
            provider_selection=None,
            intent_type_ref="",
            routing_strategy_ref="",
            orchestrator_decision_id="",
        )
        ctx = ExecutionContext.new(
            context_id="c",
            conversation_id="conv",
            request_id="req",
            created_at_utc="2026-06-27T17:00:00Z",
        )
        trace = ExecutionTrace(
            trace_id="t",
            conversation_id=ctx.identity.conversation_id,
            request_id=ctx.identity.request_id,
            decision_id=decision.decision_id,
        )
        request = ExecutionDispatchRequest(trace=trace, decision=decision, context=ctx)
        from agent.execution_dispatcher import DispatchStatus

        result = executor.execute(request)
        assert result.status is DispatchStatus.BLOCKED
        assert engine.calls == 0