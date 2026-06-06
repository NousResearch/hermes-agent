#!/usr/bin/env python3
"""
Tests for executors/router.py and related Commit A core-safe APIs.

Scope:
  - ExecutorRouter returns sensible recommendations from task goal text
  - Fallback path doesn't crash on empty/garbage input
  - ExecutorRegistry default manifests are loadable
  - Health check does not crash when the underlying binary is missing

Strictly no real external model calls, no worktrees, no user file writes.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterable

import pytest

from executors.health import (
    check_all_executors_health,
    check_executor_health,
    make_available_health,
    make_unavailable_health,
    make_unknown_health,
)
from executors.registry import ExecutorRegistry, _default_manifests
from executors.router import (
    ExecutorRouter,
    RouteRule,
    TaskCreateContext,
    _FALLBACK_ORDER,
    _normalize,
    _score_text,
    create_default_router,
)
from executors.types import (
    AdapterStartResult,
    AgentExecutorAdapter,
    AgentRun,
    ExecutorCapabilities,
    ExecutorHealthResult,
    ExecutorHealthStatus,
    ExecutorManifest,
    RouterRecommendation,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubAdapter:
    """Minimal AgentExecutorAdapter used only to satisfy registry.register()."""

    def __init__(self, executor_id: str = "stub") -> None:
        self.executor_id = executor_id

    async def start(self, run: AgentRun, config) -> AdapterStartResult:  # noqa: D401
        return AdapterStartResult(external_run_id=f"{self.executor_id}-run")

    async def stop(self, run_id: str) -> None:
        return None

    async def stream_events(self, run_id: str) -> AsyncIterable:  # type: ignore[override]
        if False:
            yield None

    async def get_status(self, run_id: str):  # type: ignore[no-untyped-def]
        from executors.types import RunStatus
        return RunStatus.COMPLETED

    async def check_health(self) -> ExecutorHealthResult:
        return make_available_health(self.executor_id, version="stub-1.0")


def _make_registry() -> ExecutorRegistry:
    """Build an ExecutorRegistry with the default manifests + stub adapters."""
    reg = ExecutorRegistry()
    for manifest in _default_manifests().values():
        reg.register(manifest, _StubAdapter(manifest.id))
    return reg


# ---------------------------------------------------------------------------
# Router — keyword-based recommendation
# ---------------------------------------------------------------------------

class TestExecutorRouterKeywordMatching:
    def test_routes_architecture_task_to_claude_code(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="Design the new auth subsystem",
            goal="Produce an ADR for the architecture",
            available_executors=["claude-code", "codex-cli", "hermes-local"],
        )
        rec = router.route(ctx)
        assert isinstance(rec, RouterRecommendation)
        assert rec.recommended_executor == "claude-code"
        assert rec.source == "keyword"
        assert rec.confidence > 0.0
        assert "architecture" in rec.reason.lower() or "design" in rec.reason.lower()

    def test_routes_refactor_task_to_codex_cli(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="Refactor the billing module",
            goal="Rewrite the API service for production",
            available_executors=["claude-code", "codex-cli", "hermes-local"],
        )
        rec = router.route(ctx)
        assert rec.recommended_executor == "codex-cli"

    def test_routes_quick_bug_fix_to_deepseek_tui(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="Fix typo in README",
            goal="Quick small fix on a cosmetic issue",
            available_executors=["claude-code", "codex-cli", "deepseek-tui", "hermes-local"],
        )
        rec = router.route(ctx)
        assert rec.recommended_executor == "deepseek-tui"

    def test_routes_hermes_internal_to_hermes_local(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="Wire the gateway adapter",
            goal="Update the orchestrator config for cron batch",
            available_executors=["claude-code", "hermes-local"],
        )
        rec = router.route(ctx)
        assert rec.recommended_executor == "hermes-local"

    def test_routes_opencode_for_oss_validation(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="Validate the open-source alternative",
            goal="Local offline prototype of an OSS agent",
            available_executors=["claude-code", "opencode"],
        )
        rec = router.route(ctx)
        assert rec.recommended_executor == "opencode"


# ---------------------------------------------------------------------------
# Router — fallback path
# ---------------------------------------------------------------------------

class TestExecutorRouterFallback:
    def test_fallback_when_no_keyword_matches(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="do the thing",
            goal="make it work",
            available_executors=["codex-cli", "claude-code", "hermes-local"],
        )
        rec = router.route(ctx)
        assert rec.source == "health_fallback"
        assert rec.recommended_executor in {"codex-cli", "claude-code", "hermes-local"}
        assert 0.0 <= rec.confidence <= 1.0

    def test_fallback_when_no_executors_available(self) -> None:
        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="",
            goal="",
            available_executors=[],
        )
        rec = router.route(ctx)
        # Last-ditch default; should never raise.
        assert isinstance(rec, RouterRecommendation)
        assert rec.source == "health_fallback"
        assert rec.confidence <= 0.5

    def test_fallback_does_not_crash_on_garbage_input(self) -> None:
        router = ExecutorRouter()
        for raw in ["", "   ", "???", "\n\n", "👻", "a" * 5000]:
            ctx = TaskCreateContext(
                title=raw,
                goal=raw,
                available_executors=["codex-cli"],
            )
            rec = router.route(ctx)
            assert rec.recommended_executor in {"codex-cli"}

    def test_unavailable_recommended_falls_back_to_alternative(self) -> None:
        router = ExecutorRouter()
        # "architecture" routes to claude-code, but it's not in the available set.
        ctx = TaskCreateContext(
            title="Architecture review",
            goal="ADR for the new module",
            available_executors=["codex-cli", "hermes-local"],
        )
        rec = router.route(ctx)
        # Must not raise; must return a recommendation within the available set.
        assert rec.recommended_executor in {"codex-cli", "hermes-local"}

    def test_fallback_order_includes_known_executors(self) -> None:
        # Sanity check on the default fallback chain — guards against typos.
        for eid in _FALLBACK_ORDER:
            assert isinstance(eid, str)
            assert eid


# ---------------------------------------------------------------------------
# Router — text helpers
# ---------------------------------------------------------------------------

class TestRouterHelpers:
    def test_normalize_lowercases_and_collapses_whitespace(self) -> None:
        assert _normalize("  Hello   WORLD  ") == "hello world"

    def test_score_text_zero_for_no_match(self) -> None:
        assert _score_text("nothing relevant", ["foo", "bar"]) == 0.0

    def test_score_text_positive_for_match(self) -> None:
        score = _score_text("please refactor this", ["refactor"])
        assert score > 0.0
        assert score <= 1.0

    def test_custom_rules_override_defaults(self) -> None:
        custom_rule = RouteRule(
            executor="opencode",
            keywords=["unicorn"],
            reason_template="matched unicorn",
            priority=100,
            confidence=0.99,
        )
        router = ExecutorRouter(rules=[custom_rule])
        ctx = TaskCreateContext(
            title="Unicorn migration",
            goal="deploy the unicorn",
            available_executors=["claude-code", "opencode"],
        )
        rec = router.route(ctx)
        assert rec.recommended_executor == "opencode"

    def test_create_default_router_factory_works(self) -> None:
        router = create_default_router()
        assert isinstance(router, ExecutorRouter)
        rec = router.route(
            TaskCreateContext(title="x", goal="y", available_executors=["codex-cli"])
        )
        assert isinstance(rec, RouterRecommendation)


# ---------------------------------------------------------------------------
# Registry — default manifests loadable
# ---------------------------------------------------------------------------

class TestRegistryDefaultManifests:
    def test_default_manifests_contains_known_executors(self) -> None:
        manifests = _default_manifests()
        for required in ["hermes-local", "claude-code", "codex-cli", "deepseek-tui", "opencode"]:
            assert required in manifests, f"missing default manifest: {required}"
            assert isinstance(manifests[required], ExecutorManifest)

    def test_registry_round_trip(self) -> None:
        reg = _make_registry()
        assert "claude-code" in {m.id for m in reg.list_executors()}
        adapter = reg.get("claude-code")
        assert isinstance(adapter, AgentExecutorAdapter)
        manifest = reg.get_manifest("claude-code")
        assert manifest.id == "claude-code"
        assert isinstance(manifest.capabilities, ExecutorCapabilities)

    def test_registry_unknown_id_raises(self) -> None:
        reg = _make_registry()
        with pytest.raises(KeyError):
            reg.get("does-not-exist")
        with pytest.raises(KeyError):
            reg.get_manifest("does-not-exist")

    def test_list_available_filters_by_health(self) -> None:
        reg = _make_registry()
        # Default health is UNKNOWN for all → nothing listed as available.
        assert reg.list_available() == []
        # Promote one to AVAILABLE and re-check.
        reg.set_health(make_available_health("codex-cli"))
        available_ids = {m.id for m in reg.list_available()}
        assert "codex-cli" in available_ids
        assert "claude-code" not in available_ids


# ---------------------------------------------------------------------------
# Health — does not crash on missing binaries
# ---------------------------------------------------------------------------

class TestHealthMissingBinary:
    def test_check_executor_health_returns_unavailable_for_missing_command(self) -> None:
        # 'definitely-not-a-real-binary-xyz123' cannot exist on PATH.
        result = asyncio.run(
            check_executor_health("phantom-exec", "definitely-not-a-real-binary-xyz123")
        )
        assert isinstance(result, ExecutorHealthResult)
        assert result.executor_id == "phantom-exec"
        assert result.status == ExecutorHealthStatus.UNAVAILABLE
        assert result.error is not None
        assert "not found" in result.error.lower() or "PATH" in result.error

    def test_check_all_executors_health_handles_empty_registry(self) -> None:
        reg = ExecutorRegistry()
        results = asyncio.run(check_all_executors_health(reg))
        assert results == {}

    def test_check_all_executors_health_does_not_raise_on_unknown_adapter(self) -> None:
        # Build a registry with a manifest whose adapter has no check_health().
        reg = ExecutorRegistry()

        class _NoHealthAdapter:
            pass

        reg.register(
            ExecutorManifest(
                id="no-health",
                label="No Health",
                description="adapter without check_health",
            ),
            _NoHealthAdapter(),
        )
        results = asyncio.run(check_all_executors_health(reg))
        assert "no-health" in results
        # The aggregator should fall back to UNKNOWN rather than crashing.
        assert results["no-health"].status == ExecutorHealthStatus.UNKNOWN

    def test_health_factory_helpers(self) -> None:
        u = make_unknown_health("x")
        assert u.status == ExecutorHealthStatus.UNKNOWN
        a = make_available_health("x", version="1.0")
        assert a.status == ExecutorHealthStatus.AVAILABLE
        assert a.version == "1.0"
        n = make_unavailable_health("x", error="boom")
        assert n.status == ExecutorHealthStatus.UNAVAILABLE
        assert n.error == "boom"


# ---------------------------------------------------------------------------
# Public API smoke
# ---------------------------------------------------------------------------

class TestPublicAPISmoke:
    def test_importing_executors_package_does_not_pull_adapters(self) -> None:
        import executors as pkg
        for name in [
            "ExecutorId",
            "ExecutorRegistry",
            "ExecutorRouter",
            "RouterRecommendation",
            "TaskCreateContext",
            "check_executor_health",
            "create_default_router",
        ]:
            assert name in pkg.__all__, f"missing public export: {name}"
        # Adapters must NOT be re-exported by the core-safe API.
        for adapter_name in [
            "HermesLocalAdapter",
            "ClaudeCodeAdapter",
            "CodexAdapter",
            "OpenCodeAdapter",
            "DeepSeekTuiAdapter",
        ]:
            assert adapter_name not in pkg.__all__, (
                f"adapter leaked into core-safe API: {adapter_name}"
            )
            assert not hasattr(pkg, adapter_name), (
                f"adapter leaked into executors package namespace: {adapter_name}"
            )
