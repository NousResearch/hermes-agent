"""Behavior contracts for the fail-closed local LLM admission hook."""

import builtins
from types import SimpleNamespace

import pytest

from hermes_cli import plugins
from hermes_cli.middleware import (
    LocalModelAdmissionBlockedError,
    _is_local_llm_route,
    require_local_model_admission,
    run_llm_admission_middleware,
)


def _scope():
    return {
        "task_id": "task-1",
        "kanban_task_id": "task-1",
        "task_run_id": "7",
        "kanban_board": "board",
        "kanban_db": "/tmp/board.sqlite3",
        "kanban_workspace": "/tmp/workspace",
        "kanban_claim_lock": "claim-1",
        "capability_id": "capability-1",
        "task_contract_sha256": "a" * 64,
        "api_request_id": "request-1",
        "context_length": "4096",
        "ollama_num_ctx": "4096",
        "session_id": "session-1",
        "model": "fixture-model",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1/",
        "streaming": False,
        "has_tools": False,
        "api_call_count": 1,
        "existing_response_format": False,
    }


def _manager(callbacks):
    return SimpleNamespace(_middleware={"llm_admission": callbacks})


def _scope_echo(kwargs, scope):
    # Controller payloads use their canonical string representation.
    return {key: str(kwargs[key]).lower() if isinstance(kwargs[key], bool) else str(kwargs[key])
            for key in scope}


def test_local_route_without_controller_is_denied(monkeypatch):
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: False)

    decision = run_llm_admission_middleware(**_scope())

    assert not decision.allowed
    assert decision.code == "controller_missing"


def test_local_capability_cannot_fall_through_to_remote_fallback():
    decision = run_llm_admission_middleware(
        task_id="task-1", kanban_task_id="task-1", capability_id="capability-1",
        provider="openai", base_url="https://api.openai.com/v1",
    )
    assert not decision.allowed
    assert decision.code == "local_capability_route_changed"


def test_unscoped_remote_route_remains_outside_local_admission():
    decision = run_llm_admission_middleware(
        provider="openai", base_url="https://api.openai.com/v1",
    )
    assert decision.allowed
    assert decision.code == "not_local_route"


def test_controller_discovery_import_error_is_denied(monkeypatch):
    real_import = builtins.__import__

    def fail_plugin_import(name, *args, **kwargs):
        if name == "hermes_cli.plugins":
            raise RuntimeError("plugin loader unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_plugin_import)

    decision = run_llm_admission_middleware(**_scope())

    assert not decision.allowed
    assert decision.code == "controller_unavailable"


def test_controller_exception_is_a_terminal_denial(monkeypatch):
    def broken_controller(**_kwargs):
        raise RuntimeError("controller unavailable")

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([broken_controller]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)

    decision = run_llm_admission_middleware(**_scope())

    assert not decision.allowed
    assert decision.code == "controller_error"


def test_admission_requires_exact_controller_scope_echo(monkeypatch):
    scope = _scope()

    def allow(**kwargs):
        return {
            "decision": "allow",
            "policy_id": "test-policy",
            "reason": "test evidence",
            "evidence_id": "evidence-1",
            "lease_ids": ["lease-1"],
            "scope": _scope_echo(kwargs, scope),
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([allow]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)

    decision = run_llm_admission_middleware(**scope)

    assert decision.allowed
    assert decision.lease_ids == ("lease-1",)


def test_admission_scope_mismatch_is_denied(monkeypatch):
    scope = _scope()

    def allow(**kwargs):
        echoed = {key: kwargs[key] for key in scope}
        echoed["task_run_id"] = "6"
        return {
            "decision": "allow", "policy_id": "test-policy", "reason": "test evidence",
            "evidence_id": "evidence-1", "lease_ids": ["lease-1"], "scope": echoed,
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([allow]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)

    decision = run_llm_admission_middleware(**scope)

    assert not decision.allowed
    assert decision.code == "admission_scope_mismatch"


def test_registry_schema_is_returned_only_for_nonstreaming_tool_free_first_request(monkeypatch):
    scope = _scope()
    response_format = {
        "type": "json_schema",
        "json_schema": {"schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        }},
    }

    def allow(**kwargs):
        return {
            "decision": "allow", "policy_id": "test-policy", "reason": "test evidence",
            "evidence_id": "evidence-1", "lease_ids": ["lease-1"],
                "scope": _scope_echo(kwargs, scope), "response_format": response_format,
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([allow]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)
    allowed = run_llm_admission_middleware(
        **scope, request_format_phase="prepare",
    )
    assert allowed.allowed
    assert allowed.response_format == response_format

    streamed = run_llm_admission_middleware(
        **{**scope, "streaming": True}, request_format_phase="prepare",
    )
    assert not streamed.allowed
    assert streamed.code == "request_format_incompatible"


def test_provider_boundary_refuses_when_request_schema_was_not_applied(monkeypatch):
    scope = _scope()
    response_format = {
        "type": "json_schema",
        "json_schema": {"schema": {"type": "object"}},
    }

    def allow(**kwargs):
        return {
            "decision": "allow", "policy_id": "test-policy", "reason": "test evidence",
            "evidence_id": "evidence-1", "lease_ids": ["lease-1"],
                "scope": _scope_echo(kwargs, scope), "response_format": response_format,
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([allow]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)
    decision = run_llm_admission_middleware(
        **scope, request_format_phase="verify", supplied_response_format=None,
    )
    assert not decision.allowed
    assert decision.code == "request_format_not_applied"


def test_provider_boundary_accepts_the_exact_applied_schema(monkeypatch):
    scope = {**_scope(), "existing_response_format": True}
    response_format = {
        "type": "json_schema",
        "json_schema": {"schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        }},
    }

    def allow(**kwargs):
        return {
            "decision": "allow", "policy_id": "test-policy", "reason": "test evidence",
            "evidence_id": "evidence-1", "lease_ids": ["lease-1"],
            "scope": _scope_echo(kwargs, scope), "response_format": response_format,
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([allow]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)
    decision = run_llm_admission_middleware(
        **scope, request_format_phase="verify", supplied_response_format=response_format,
    )

    assert decision.allowed
    assert decision.response_format == response_format


def test_local_provider_call_stops_before_provider_io_without_task_identity(monkeypatch):
    provider_calls = []
    monkeypatch.setattr(
        "agent.chat_completion_helpers.direct_api_call",
        lambda *_args, **_kwargs: provider_calls.append("called"),
    )
    agent = SimpleNamespace(provider="ollama", base_url="http://127.0.0.1:11434/v1/")

    from agent.chat_completion_helpers import interruptible_api_call

    with pytest.raises(LocalModelAdmissionBlockedError, match="missing_admission_identity"):
        interruptible_api_call(agent, {"model": "fixture-model"})

    assert provider_calls == []


def test_interruptible_provider_adapter_validates_and_forwards_admitted_schema(monkeypatch):
    import agent.chat_completion_helpers as helpers

    response_format = {
        "type": "json_schema",
        "json_schema": {"schema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        }},
    }
    fields = (
        "task_id", "kanban_task_id", "task_run_id", "kanban_board", "kanban_db",
        "kanban_workspace", "kanban_claim_lock", "capability_id", "task_contract_sha256",
        "api_request_id", "context_length", "ollama_num_ctx", "session_id", "model",
        "provider", "base_url", "streaming", "has_tools", "api_call_count",
        "existing_response_format",
    )

    def allow(**kwargs):
        return {
            "decision": "allow", "policy_id": "test-policy", "reason": "test evidence",
            "evidence_id": "evidence-1", "lease_ids": ["lease-1"],
            "scope": _scope_echo(kwargs, fields), "response_format": response_format,
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([allow]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda _agent: False)
    provider_requests = []

    class FakeNonStreamRequest:
        def __init__(self, _agent, api_kwargs):
            provider_requests.append(api_kwargs)

        def run(self):
            return "fixture-response"

    import agent.chat_completion_nonstream as nonstream
    monkeypatch.setattr(nonstream, "_NonStreamRequest", FakeNonStreamRequest)
    for key, value in {
        "HERMES_KANBAN_TASK": "task-1", "HERMES_KANBAN_RUN_ID": "7",
        "HERMES_KANBAN_BOARD": "board", "HERMES_KANBAN_DB": "/tmp/board.sqlite3",
        "HERMES_KANBAN_WORKSPACE": "/tmp/workspace", "HERMES_KANBAN_CLAIM_LOCK": "claim-1",
    }.items():
        monkeypatch.setenv(key, value)
    from agent.delegation_context import set_owned_local_gpu_task_context
    set_owned_local_gpu_task_context({"capability_id": "capability-1", "task_contract_sha256": "a" * 64})
    agent = SimpleNamespace(
        provider="ollama", base_url="http://127.0.0.1:11434/v1/", model="fixture-model",
        session_id="session-1", api_mode="chat_completions", _api_call_count=1,
        context_compressor=SimpleNamespace(context_length=4096), _ollama_num_ctx=4096,
    )

    result = helpers.interruptible_api_call(
        agent, {"model": "fixture-model", "response_format": response_format},
    )

    assert result == "fixture-response"
    assert provider_requests == [{"model": "fixture-model", "response_format": response_format}]
    assert agent._local_gpu_admission_lease_ids == ("lease-1",)


def test_resource_balancer_block_reason_stops_provider_adapter_before_io(monkeypatch):
    import agent.chat_completion_helpers as helpers

    provider_calls = []
    monkeypatch.setattr(
        helpers, "direct_api_call",
        lambda *_args, **_kwargs: provider_calls.append("called"),
    )
    for key, value in {
        "HERMES_KANBAN_TASK": "task-1",
        "HERMES_KANBAN_RUN_ID": "7",
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_DB": "/tmp/kanban.sqlite3",
        "HERMES_KANBAN_WORKSPACE": "/tmp/workspace",
        "HERMES_KANBAN_CLAIM_LOCK": "claim-1",
    }.items():
        monkeypatch.setenv(key, value)
    from agent.delegation_context import set_owned_local_gpu_task_context
    set_owned_local_gpu_task_context({"capability_id": "capability-1", "task_contract_sha256": "a" * 64})

    def blocked(**_kwargs):
        return {
            "decision": "block",
            "policy_id": "resource_balancer_blocked",
            "reason": "Resource Balancer blocked admission: pressure/recovery",
        }

    monkeypatch.setattr(plugins, "_delivery_manager", lambda: _manager([blocked]))
    monkeypatch.setattr(plugins, "has_middleware", lambda _kind: True)
    agent = SimpleNamespace(
        provider="ollama", base_url="http://127.0.0.1:11434/v1/",
        model="fixture-model", session_id="session-1", api_mode="chat_completions",
        context_compressor=SimpleNamespace(context_length=4096), _ollama_num_ctx=4096,
    )

    with pytest.raises(LocalModelAdmissionBlockedError, match="pressure/recovery"):
        helpers.interruptible_api_call(agent, {"model": "fixture-model"})

    assert provider_calls == []


def test_local_streaming_call_stops_before_provider_io_without_task_identity(monkeypatch):
    provider_calls = []
    import agent.chat_completion_helpers as helpers

    monkeypatch.setattr(
        helpers,
        "_StreamingCall",
        lambda *_args, **_kwargs: provider_calls.append("constructed"),
    )
    agent = SimpleNamespace(provider="ollama", base_url="http://127.0.0.1:11434/v1/")

    with pytest.raises(LocalModelAdmissionBlockedError, match="missing_admission_identity"):
        helpers.interruptible_streaming_api_call(agent, {"model": "fixture-model"})

    assert provider_calls == []


@pytest.mark.parametrize(
    "provider,base_url,expected",
    [
        ("ollama", "https://remote.example/v1", True),
        ("openai", "http://192.168.1.10:8000/v1", True),
        ("openai", "https://api.openai.com/v1", False),
    ],
)
def test_local_route_classifier(provider, base_url, expected):
    assert _is_local_llm_route(provider, base_url) is expected


def test_official_provider_through_loopback_proxy_is_local_route(monkeypatch):
    from hermes_cli import local_model_routes

    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:8080")

    assert local_model_routes.is_local_llm_route(
        "openai", "https://api.openai.com/v1",
    ) is True


def test_official_provider_bypassing_proxy_stays_remote(monkeypatch):
    from hermes_cli import local_model_routes

    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("NO_PROXY", "api.openai.com")
    monkeypatch.setenv("no_proxy", "api.openai.com")

    assert local_model_routes.is_local_llm_route(
        "openai", "https://api.openai.com/v1",
    ) is False


def test_custom_fqdn_resolving_to_private_address_requires_local_admission(monkeypatch):
    from hermes_cli import local_model_routes

    monkeypatch.setattr(local_model_routes, "_is_known_official_provider_host", lambda *_args: False)
    monkeypatch.setattr(local_model_routes, "_host_resolves_to_local_address", lambda _host: True)

    assert _is_local_llm_route("openai", "https://gpu.region.example/v1") is True


def test_unresolvable_custom_fqdn_fails_closed_as_local_route(monkeypatch):
    from hermes_cli import local_model_routes

    monkeypatch.setattr(local_model_routes, "_is_known_official_provider_host", lambda *_args: False)
    monkeypatch.setattr(local_model_routes, "_host_resolves_to_local_address", lambda _host: True)

    assert _is_local_llm_route("custom", "https://unknown.example/v1") is True


@pytest.mark.parametrize(
    "resolved,expected",
    [
        ([(2, 1, 6, "", ("10.4.5.6", 0))], True),
        ([(2, 1, 6, "", ("8.8.8.8", 0))], False),
        (OSError("resolver unavailable"), True),
    ],
)
def test_custom_hostname_resolution_is_private_or_fail_closed(monkeypatch, resolved, expected):
    from hermes_cli import local_model_routes

    monkeypatch.setattr(local_model_routes, "_is_known_official_provider_host", lambda *_args: False)
    if isinstance(resolved, BaseException):
        def fail_lookup(*_args, **_kwargs):
            raise resolved
        monkeypatch.setattr(local_model_routes.socket, "getaddrinfo", fail_lookup)
    else:
        monkeypatch.setattr(local_model_routes.socket, "getaddrinfo", lambda *_args, **_kwargs: resolved)

    assert local_model_routes._host_resolves_to_local_address("gpu.region.example") is expected


def test_custom_hostname_resolution_timeout_fails_closed(monkeypatch):
    from hermes_cli import local_model_routes

    monkeypatch.setattr(local_model_routes, "_LOCAL_ROUTE_DNS_TIMEOUT_SECONDS", 0.001)
    original_thread = local_model_routes.threading.Thread

    class StalledThread:
        def __init__(self, *, target, **kwargs):
            self._target = target
            self._thread = original_thread(target=target, **kwargs)

        def start(self):
            # Do not start resolver work: this deterministically exercises the wait timeout.
            return None

    monkeypatch.setattr(local_model_routes.threading, "Thread", StalledThread)

    assert local_model_routes._host_resolves_to_local_address("slow.example") is True
