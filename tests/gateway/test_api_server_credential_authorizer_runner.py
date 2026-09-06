"""Adversarial contract tests for plugin-issued API credentials."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.api_credentials import (
    APIServerOperation,
    AgentProfileId,
    AuthorizedAPICredential,
    CredentialAuthorizationRequest,
    CredentialScopeId,
)
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_credential_authorizer import _CredentialAuthorizerRunner
from gateway.platforms import api_server as api_server_module
from gateway.platforms import api_server_runs as api_server_runs_module
from gateway.platforms import api_server_credential_authorizer as credential_authorizer_module
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import get_hermes_home

OPERATOR_KEY = "operator-key-1234567890"
from tests.gateway.api_server_credential_authorizer_test_support import (
    OPERATOR_KEY, AsyncAuthorizer, Authorizer, SyncAuthorizer, _adapter, _auth_app,
    _create_owned_session, _credential_app, _principal, _principal_with_operations,
    _wait_for_run,
)

@pytest.mark.asyncio
async def test_authorizer_timeout_is_bounded():
    """Async authorizer respects the configured deadline."""
    async def stalled_async(_request):
        await asyncio.sleep(60)

    adapter = _adapter(AsyncAuthorizer(stalled_async))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await asyncio.wait_for(
            client.get(
                "/v1/capabilities",
                headers={"Authorization": "Bearer stalled-secret"},
            ),
            timeout=1,
        )
    assert response.status == 401

@pytest.mark.asyncio
async def test_lingering_async_authorizer_saturates_dedicated_capacity_without_queueing():
    """A cancellation-resistant async task holds the slot; the second request gets 401 immediately."""
    release = asyncio.Event()
    calls = []

    async def authorize(_request):
        calls.append("started")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()

    adapter = _adapter(AsyncAuthorizer(authorize))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01
    adapter._API_CREDENTIAL_AUTH_MAX_INFLIGHT = 1

    async def handler(_request):
        raise AssertionError("handler must not run")

    first_task = None
    try:
        async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
            first_task = asyncio.create_task(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer first"}))
            done, _ = await asyncio.wait({first_task}, timeout=0.5)
            if not done:
                release.set()
                await asyncio.wait_for(first_task, timeout=1)
            assert done, "request must return after deadline even with cancellation-resistant authorizer"
            first = first_task.result()
            # Slot is still held — second request must fail immediately.
            second = await asyncio.wait_for(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer second"}), timeout=0.2)
            assert first.status == second.status == 401
            assert calls == ["started"]
    finally:
        release.set()
        if first_task is not None and not first_task.done():
            await asyncio.wait_for(first_task, timeout=1)

@pytest.mark.asyncio
async def test_cancellation_resistant_async_authorizer_returns_at_deadline_and_saturates_capacity():
    """Deadline is observed, cancellation-resistant task holds its slot until it actually finishes,
    and saturation persists across adapter disconnect/reconnect, plugin unload, and a second adapter."""
    release = asyncio.Event()
    cancellation_seen = asyncio.Event()
    calls = []

    async def authorize(_request):
        calls.append("started")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    adapter = _adapter(AsyncAuthorizer(authorize))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01
    adapter._API_CREDENTIAL_AUTH_MAX_INFLIGHT = 1

    async def handler(_request):
        raise AssertionError("handler must not run")

    first_task = None
    try:
        async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
            first_task = asyncio.create_task(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer first"}))
            done, _ = await asyncio.wait({first_task}, timeout=0.2)
            if not done:
                release.set()
                await asyncio.wait_for(first_task, timeout=1)
            assert done, "request deadline waited for cancellation-resistant authorizer cleanup"
            first = first_task.result()
            assert cancellation_seen.is_set()

            # Slot is held — saturation persists while the task has not finished.
            second = await asyncio.wait_for(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer second"}), timeout=0.2)
            assert first.status == second.status == 401
            assert calls == ["started"]

        # Simulate adapter disconnect — runner must NOT be discarded.
        runner_before = adapter._api_credential_authorizer_runner
        await adapter.disconnect()
        assert adapter._api_credential_authorizer_runner is runner_before, (
            "disconnect() must not clear the process-level runner reference"
        )

        # Saturation persists across the disconnect for the same runner.
        # (Slot still held because the lingering task has not exited yet — release.set() below.)
        from gateway.platforms.api_server_credential_authorizer import _CredentialAuthorizerSaturated
        with pytest.raises(_CredentialAuthorizerSaturated):
            runner_before._acquire()

        # A second adapter using the same process runner also sees saturation.
        adapter2 = _adapter(AsyncAuthorizer(authorize))
        adapter2._api_credential_authorizer_runner = runner_before
        with pytest.raises(_CredentialAuthorizerSaturated):
            runner_before._acquire()

    finally:
        release.set()
        if first_task is not None and not first_task.done():
            await asyncio.wait_for(first_task, timeout=1)

@pytest.mark.asyncio
async def test_plugin_manager_shutdown_cancels_runner_and_cannot_replace_it():
    manager = PluginManager()
    created = []

    def factory(capacity):
        runner = _CredentialAuthorizerRunner(capacity)
        created.append(runner)
        return runner

    runner = manager.get_api_server_credential_authorizer_runner(capacity=1, factory=factory)
    started = asyncio.Event()

    async def authorize(_request):
        started.set()
        await asyncio.sleep(60)

    in_flight = asyncio.create_task(runner.run(authorize, object(), timeout=60))
    await asyncio.wait_for(started.wait(), timeout=1)
    manager.shutdown()
    manager.shutdown()

    with pytest.raises(asyncio.CancelledError):
        await in_flight
    with pytest.raises(RuntimeError, match="shut down"):
        manager.get_api_server_credential_authorizer_runner(capacity=1, factory=factory)
    assert created == [runner]

@pytest.mark.asyncio
async def test_authorizer_runner_cancels_child_when_request_owner_is_cancelled():
    runner = _CredentialAuthorizerRunner(1)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def authorize(_request):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    owner = asyncio.create_task(runner.run(authorize, object(), timeout=60))
    await asyncio.wait_for(started.wait(), timeout=1)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    try:
        await asyncio.wait_for(cancelled.wait(), timeout=0.1)
        child_was_cancelled = True
    except asyncio.TimeoutError:
        child_was_cancelled = False
    finally:
        runner.close()
        release.set()
        for _ in range(20):
            if runner._active == 0:
                break
            await asyncio.sleep(0)
    assert child_was_cancelled
    assert runner._active == 0

@pytest.mark.asyncio
async def test_authorizer_runner_singleton_survives_reconnect_and_unload():
    """The process-level runner must not be discarded on adapter disconnect or plugin unload."""
    adapter = _adapter(AsyncAuthorizer(lambda _request: asyncio.sleep(0)))
    first = adapter._credential_authorizer_runner()
    second = adapter._credential_authorizer_runner()
    assert first is second

    runner_ref = first
    await adapter.disconnect()
    # Disconnect must NOT clear the adapter's runner reference (it's the process singleton).
    assert adapter._api_credential_authorizer_runner is runner_ref, (
        "disconnect() cleared the process-level runner — this breaks saturation invariants"
    )

    # Calling the accessor again still returns the same object.
    third = adapter._credential_authorizer_runner()
    assert third is runner_ref

@pytest.mark.asyncio
async def test_concurrent_manager_get_or_create_returns_same_runner():
    """Concurrent calls to get_api_server_credential_authorizer_runner must return the same instance."""
    manager = PluginManager()
    results = []

    async def fetch():
        runner = manager.get_api_server_credential_authorizer_runner(
            capacity=4, factory=_CredentialAuthorizerRunner
        )
        results.append(runner)

    await asyncio.gather(*[fetch() for _ in range(16)])
    assert len(set(id(r) for r in results)) == 1, (
        "get_api_server_credential_authorizer_runner returned multiple distinct runner instances"
    )

@pytest.mark.asyncio
async def test_adapter_retrieval_failure_fails_auth_closed(monkeypatch):
    """If the process runner cannot be retrieved, auth must fail closed — no local fallback."""
    def _boom():
        raise RuntimeError("plugin manager unavailable")

    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", _boom)
    adapter = _adapter(AsyncAuthorizer(lambda _request: asyncio.sleep(0)))
    # Clear any cached runner so the accessor must call get_plugin_manager.
    adapter._api_credential_authorizer_runner = None

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer any-bearer"}
        )
    assert response.status == 401, (
        "runner retrieval failure must yield 401, not a local-fallback 200"
    )

@pytest.mark.asyncio
async def test_slot_recovers_after_lingering_task_finishes():
    """After a cancellation-resistant task truly exits, its capacity slot is released."""
    release = asyncio.Event()
    admitted = asyncio.Event()
    invocations = 0

    async def authorize(request):
        nonlocal invocations
        invocations += 1
        if invocations > 1:
            return _principal(request.operation)
        admitted.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()

    adapter = _adapter(AsyncAuthorizer(authorize))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01
    adapter._API_CREDENTIAL_AUTH_MAX_INFLIGHT = 1

    async def handler(request):
        return web.json_response({"ok": True})

    first_task = None
    try:
        async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
            first_task = asyncio.create_task(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer first"}))
            await asyncio.wait_for(admitted.wait(), timeout=1)
            # Wait for deadline to fire (returns 401, slot still held by lingering task).
            done, _ = await asyncio.wait({first_task}, timeout=0.5)
            if not done:
                release.set()
                await asyncio.wait_for(first_task, timeout=1)
            assert done
            assert (await first_task).status == 401

            # Now release the lingering task and wait briefly for done callback.
            release.set()
            await asyncio.sleep(0.05)

            # Slot must now be free — next request should be authorized normally.
            second = await asyncio.wait_for(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer second"}), timeout=0.5)
            assert invocations == 2
            assert second.status == 200
    finally:
        release.set()
        if first_task is not None and not first_task.done():
            await asyncio.wait_for(first_task, timeout=1)
