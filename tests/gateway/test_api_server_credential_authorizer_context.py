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
@pytest.mark.parametrize("failure", ["bind", "cleanup"])
async def test_run_agent_restores_executor_auth_context_after_setup_or_cleanup_failure(
    failure, monkeypatch
):
    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    leaked = credential_authorizer_module._CredentialAuthContext(
        _principal(APIServerOperation.RUNS_CREATE), "owner-key"
    )
    token = credential_authorizer_module._api_request_auth_context.set(leaked)
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    if failure == "bind":
        monkeypatch.setattr(
            adapter, "_bind_api_server_session",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
        )
    else:
        monkeypatch.setattr(
            "gateway.session_context.clear_session_vars",
            lambda _tokens: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
        )
    try:
        with pytest.raises(RuntimeError, match=failure):
            await adapter._run_agent("hello", [])
        observed = await loop.run_in_executor(
            None, credential_authorizer_module._api_request_auth_context.get
        )
    finally:
        credential_authorizer_module._api_request_auth_context.reset(token)
        executor.shutdown(wait=False, cancel_futures=True)

    assert observed is None

@pytest.mark.asyncio
@pytest.mark.parametrize("failed_cleanup", ["ownership", "declared_binding"])
async def test_run_agent_clears_all_session_context_after_each_cleanup_failure(
    failed_cleanup, monkeypatch
):
    from gateway import session_context

    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    agent = MagicMock(session_id="rotated-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    calls = []

    def clear_ownership(_agent):
        calls.append("ownership")
        if failed_cleanup == "ownership":
            raise RuntimeError("ownership cleanup failed")

    def bind_declared(_session_id, _session_key):
        calls.append("declared_binding")
        if failed_cleanup == "declared_binding":
            raise RuntimeError("declared binding cleanup failed")

    monkeypatch.setattr(api_server_module, "_clear_turn_process_ownership", clear_ownership)
    monkeypatch.setattr(adapter, "_bind_declared_conversation", bind_declared)

    def observe_session_context():
        return {
            var.name: session_context.get_session_env(var.name, "missing")
            for var in session_context._SESSION_VARS
        }

    try:
        with pytest.raises(RuntimeError, match=failed_cleanup.replace("_", " ")):
            await adapter._run_agent(
                "hello", [], session_id="bound-session",
                gateway_session_key="declared-key", bind_declared_conversation=True,
            )
        observed = await loop.run_in_executor(None, observe_session_context)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert sorted(calls) == ["declared_binding", "ownership"]
    assert set(observed.values()) == {""}

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_cleanup",
    [
        "ownership", "declared_binding", "notify", "room_policy",
        "session_vars", "approval_session", "auth_context",
    ],
)
async def test_runs_executor_clears_every_context_after_each_cleanup_failure(
    failed_cleanup, monkeypatch
):
    from gateway import hosted_room_execution_policy, session_context
    from tools import approval, approval_context

    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    leaked = credential_authorizer_module._CredentialAuthContext(
        _principal(APIServerOperation.RUNS_CREATE), "owner-key"
    )
    policy = hosted_room_execution_policy.execution_policy_mapping(
        target_profile="default", config={"agent": {}, "approvals": {}}
    )
    run = api_server_runs_module._RunLaunch(
        owner=adapter,
        run_id="run_cleanup_failure",
        queue=asyncio.Queue(),
        session_id="session_cleanup_failure",
        gateway_session_key="declared-key",
        declared_selected=True,
        user_message="hello",
        conversation_history=[],
        agent_kwargs={
            "room_dispatch": {"room_id": "room-one"},
            "room_execution_policy": policy,
        },
        request_profile=None,
        request_auth_context=leaked,
        browser_control_principal="browser-principal",
        browser_control_transport_family="cloud",
    )
    agent = MagicMock(session_id="rotated-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    calls = []

    def cleanup(name, real=lambda *_args: None):
        def wrapped(*args):
            calls.append(name)
            if failed_cleanup == name:
                raise RuntimeError(f"{name} cleanup failed")
            return real(*args)
        return wrapped

    monkeypatch.setattr(
        api_server_module, "_clear_turn_process_ownership", cleanup("ownership")
    )
    monkeypatch.setattr(
        adapter, "_bind_declared_conversation", cleanup("declared_binding")
    )
    real_unregister = approval.unregister_gateway_notify
    monkeypatch.setattr(
        approval, "unregister_gateway_notify",
        cleanup("notify", real_unregister),
    )
    monkeypatch.setattr(
        hosted_room_execution_policy, "reset_room_execution_policy",
        cleanup("room_policy", hosted_room_execution_policy.reset_room_execution_policy),
    )
    monkeypatch.setattr(
        session_context, "clear_session_vars",
        cleanup("session_vars", session_context.clear_session_vars),
    )
    monkeypatch.setattr(
        approval_context, "reset_current_session_key",
        cleanup("approval_session", approval_context.reset_current_session_key),
    )

    real_auth_context = credential_authorizer_module._api_request_auth_context

    class AuthContext:
        def get(self):
            return real_auth_context.get()

        def set(self, value):
            return real_auth_context.set(value)

        def reset(self, token):
            calls.append("auth_context")
            if failed_cleanup == "auth_context":
                raise RuntimeError("auth_context cleanup failed")
            return real_auth_context.reset(token)

    monkeypatch.setattr(credential_authorizer_module, "_api_request_auth_context", AuthContext())

    def observe_contexts():
        return {
            "session": {var.name: var.get() for var in session_context._SESSION_VARS},
            "async_delivery": session_context._SESSION_ASYNC_DELIVERY.get(),
            "approval": approval_context._approval_session_key.get(),
            "room": hosted_room_execution_policy.current_room_execution_policy(),
            "auth": real_auth_context.get(),
        }

    try:
        with pytest.raises(RuntimeError, match=f"{failed_cleanup} cleanup failed"):
            await loop.run_in_executor(
                None,
                lambda: api_server_runs_module._run_agent_sync(
                    adapter, run, agent, lambda _event: None,
                    _api_server=api_server_module,
                ),
            )
        observed = await loop.run_in_executor(None, observe_contexts)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        real_unregister(run.approval_session_key)

    assert sorted(calls) == sorted([
        "ownership", "declared_binding", "notify", "room_policy",
        "session_vars", "approval_session", "auth_context",
    ])
    assert set(observed["session"].values()) == {""}
    assert observed["async_delivery"] is session_context._UNSET
    assert observed["approval"] == ""
    assert observed["room"] is None
    assert observed["auth"] is None

@pytest.mark.asyncio
async def test_run_agent_preclear_failure_clears_session_and_browser_context_on_reused_worker(
    monkeypatch,
):
    from gateway import session_context

    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    agent = MagicMock(session_id="bound-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    monkeypatch.setattr(
        session_context,
        "clear_session_vars",
        lambda _tokens: (_ for _ in ()).throw(RuntimeError("pre-clear failed")),
    )

    def observe():
        return (
            {var.name: var.get() for var in session_context._SESSION_VARS},
            session_context._SESSION_ASYNC_DELIVERY.get(),
        )

    try:
        with pytest.raises(RuntimeError, match="pre-clear failed"):
            await adapter._run_agent(
                "hello", [], session_id="bound-session",
                gateway_session_key="bound-key",
            )
        values, async_delivery = await loop.run_in_executor(None, observe)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert set(values.values()) == {""}
    assert async_delivery is session_context._UNSET

@pytest.mark.asyncio
async def test_run_agent_auth_reset_failure_sets_none_on_reused_worker(monkeypatch):
    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    leaked = credential_authorizer_module._CredentialAuthContext(
        _principal(APIServerOperation.RUNS_CREATE), "owner-key"
    )
    token = credential_authorizer_module._api_request_auth_context.set(leaked)
    real_var = credential_authorizer_module._api_request_auth_context
    agent = MagicMock(session_id="bound-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    class ResetFailure:
        def get(self):
            return real_var.get()

        def set(self, value):
            return real_var.set(value)

        def reset(self, _token):
            raise RuntimeError("auth reset failed")

    monkeypatch.setattr(credential_authorizer_module, "_api_request_auth_context", ResetFailure())
    try:
        with pytest.raises(RuntimeError, match="auth reset failed"):
            await adapter._run_agent("hello", [], session_id="bound-session")
        observed = await loop.run_in_executor(None, real_var.get)
    finally:
        real_var.reset(token)
        executor.shutdown(wait=False, cancel_futures=True)

    assert observed is None

def test_expected_profile_key_reset_failure_forces_safe_baseline(monkeypatch):
    adapter = _adapter(None)
    real_var = api_server_module._api_request_profile

    class ResetFailure:
        def get(self):
            return real_var.get()

        def set(self, value):
            return real_var.set(value)

        def reset(self, _token):
            raise RuntimeError("profile reset failed")

    monkeypatch.setattr(api_server_module, "_api_request_profile", ResetFailure())
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "key")

    try:
        with pytest.raises(RuntimeError, match="profile reset failed"):
            adapter._expected_api_key_for_profile("worker")
        assert real_var.get() is None
    finally:
        real_var.set(None)

@pytest.mark.asyncio
async def test_agent_request_reservation_reset_failure_forces_safe_baseline(monkeypatch):
    adapter = _adapter(None)
    real_var = api_server_module._api_agent_request_reservation

    class ResetFailure:
        def get(self):
            return real_var.get()

        def set(self, value):
            return real_var.set(value)

        def reset(self, _token):
            raise RuntimeError("reservation reset failed")

    monkeypatch.setattr(api_server_module, "_api_agent_request_reservation", ResetFailure())
    monkeypatch.setattr(adapter, "_check_auth", lambda _request: None)

    @api_server_module._admit_api_agent_request
    async def handler(_adapter, _request):
        return web.Response(status=204)

    with pytest.raises(RuntimeError, match="reservation reset failed"):
        await handler(adapter, MagicMock())
    assert real_var.get() is None
    assert adapter._pending_agent_requests == 0

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_name",
    ["profile", "auth", "browser_principal", "browser_family"],
)
async def test_scoped_request_resets_each_context_independently(failed_name, monkeypatch):
    variables = {
        "profile": api_server_module._api_request_profile,
        "auth": credential_authorizer_module._api_request_auth_context,
        "browser_principal": api_server_module._api_request_browser_control_principal,
        "browser_family": api_server_module._api_request_browser_control_transport_family,
    }
    baselines = {name: variable.get() for name, variable in variables.items()}

    class ResetFailure:
        def __init__(self, real, name):
            self.real = real
            self.name = name

        def get(self):
            return self.real.get()

        def set(self, value):
            return self.real.set(value)

        def reset(self, token):
            if self.name == failed_name:
                raise RuntimeError(f"{self.name} reset failed")
            return self.real.reset(token)

    monkeypatch.setattr(
        api_server_module, "_api_request_profile", ResetFailure(variables["profile"], "profile")
    )
    monkeypatch.setattr(
        credential_authorizer_module, "_api_request_auth_context", ResetFailure(variables["auth"], "auth")
    )
    monkeypatch.setattr(
        api_server_module,
        "_api_request_browser_control_principal",
        ResetFailure(variables["browser_principal"], "browser_principal"),
    )
    monkeypatch.setattr(
        api_server_module,
        "_api_request_browser_control_transport_family",
        ResetFailure(variables["browser_family"], "browser_family"),
    )
    adapter = _adapter(None)

    async def handler(_request):
        return web.Response(status=204)

    with pytest.raises(RuntimeError, match=f"{failed_name} reset failed"):
        await adapter._run_scoped_request(
            MagicMock(), handler, "default",
            credential_authorizer_module._CredentialAuthContext(
                _principal(APIServerOperation.RUNS_CREATE), "owner-key"
            ),
        )

    assert variables[failed_name].get() is None
    assert all(
        variable.get() == baselines[name]
        for name, variable in variables.items()
        if name != failed_name
    )
