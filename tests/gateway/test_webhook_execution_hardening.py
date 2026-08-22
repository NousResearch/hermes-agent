"""Hardening regressions for webhook execution capability authority."""

from __future__ import annotations

import asyncio

import pytest

from gateway.platforms.webhook_executions import (
    WebhookExecutionRegistry,
    _AUTH_MAX_ATTEMPTS,
)


def _registry(tmp_path, *, ttl: int = 60) -> WebhookExecutionRegistry:
    return WebhookExecutionRegistry(
        tmp_path / "executions.json",
        ttl_seconds=ttl,
        max_records=16,
    )


def _accept(registry: WebhookExecutionRegistry):
    return registry.accept(
        profile="worker",
        route="build",
        provider="github",
        delivery_id="provider-delivery-123",
        session_key="webhook:build:provider-delivery-123",
    )


def test_public_projection_omits_session_and_provider_delivery_identity(tmp_path):
    registry = _registry(tmp_path)
    accepted = _accept(registry)

    public = registry.public(accepted["execution_id"])

    assert "token_hash" not in public
    assert "session_key" not in public
    assert "delivery_id" not in public
    assert public["execution_id"] == accepted["execution_id"]
    assert public["state"] == "accepted"


def test_scoped_capability_authority_has_shared_attempt_ceiling(tmp_path):
    registry = _registry(tmp_path)
    accepted = _accept(registry)
    execution_id = accepted["execution_id"]
    token = accepted["access_token"]

    for _ in range(_AUTH_MAX_ATTEMPTS):
        assert registry.authorize_scoped(
            execution_id, token, profile="worker", route="build"
        ) is True
    assert registry.authorize_scoped(
        execution_id, token, profile="worker", route="build"
    ) is False


def test_stale_unbound_active_record_becomes_interrupted(tmp_path):
    registry = _registry(tmp_path, ttl=10)
    accepted = _accept(registry)
    execution_id = accepted["execution_id"]
    record = registry._records[execution_id]
    record["created_at"] = 1.0

    changed = registry.prune(now=20.0)

    assert changed is True
    public = registry.public(execution_id)
    assert public["state"] == "interrupted"
    assert public["finished_at"] == 20.0


@pytest.mark.asyncio
async def test_bind_is_fail_soft_when_ledger_persist_fails(tmp_path, monkeypatch):
    registry = _registry(tmp_path)
    accepted = _accept(registry)
    execution_id = accepted["execution_id"]

    blocker = asyncio.Event()

    async def work():
        await blocker.wait()

    task = asyncio.create_task(work())

    def fail_persist():
        raise OSError("disk full")

    monkeypatch.setattr(registry, "_persist", fail_persist)
    try:
        assert registry.bind(execution_id, task) is True
        assert registry.is_bound(execution_id) is True
    finally:
        blocker.set()
        await task
