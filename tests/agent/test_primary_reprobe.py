"""Tests for optional primary re-probes while provider fallback is active."""
from types import SimpleNamespace
from unittest.mock import Mock


def _agent(**overrides):
    values = {
        "_provider_fallback_active": True,
        "_fallback_activated": True,
        "_primary_runtime": {"model": "example-model", "provider": "custom"},
        "_fallback_activated_at_call": 10,
        "_last_primary_reprobe_at": 0.0,
        "model": "example-model",
        "provider": "custom",
        "_emit_diagnostic_status": Mock(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_primary_probe_skips_unsupported_api_mode(monkeypatch):
    import httpx
    from agent.primary_reprobe import _primary_probe

    client = Mock()
    monkeypatch.setattr(httpx, "Client", client)
    assert not _primary_probe(_agent(), {
        "base_url": "https://example.invalid/v1",
        "model": "example-model",
        "api_key": "unit-test",
        "api_mode": "responses",
    })
    client.assert_not_called()


def test_primary_probe_skips_missing_or_dynamic_credentials(monkeypatch):
    import httpx
    from agent.primary_reprobe import _primary_probe

    client = Mock()
    monkeypatch.setattr(httpx, "Client", client)
    route = {"base_url": "https://example.invalid/v1", "model": "example-model", "api_mode": "chat_completions"}
    assert not _primary_probe(_agent(), route)
    assert not _primary_probe(_agent(), {**route, "api_key": lambda: "dynamic"})
    client.assert_not_called()


def test_primary_probe_uses_static_auth_and_route_tls(monkeypatch):
    import httpx
    from agent.primary_reprobe import _primary_probe

    response = SimpleNamespace(status_code=200)
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.post.return_value = response
    monkeypatch.setattr(httpx, "Client", Mock(return_value=client))
    verify = object()
    resolve_verify = Mock(return_value=verify)
    monkeypatch.setattr("agent.ssl_verify.resolve_httpx_verify", resolve_verify)

    route = {
        "base_url": "https://example.invalid/v1",
        "model": "example-model",
        "api_mode": "chat_completions",
        "api_key": "unit-test",
        "client_kwargs": {"ssl_verify": False, "ssl_ca_cert": None},
    }
    assert _primary_probe(_agent(), route)
    resolve_verify.assert_called_once_with(
        ca_bundle=None, ssl_verify=False, base_url="https://example.invalid/v1"
    )
    client.post.assert_called_once()
    args, kwargs = client.post.call_args
    assert args == ("https://example.invalid/v1/chat/completions",)
    assert kwargs["headers"]["Authorization"] == "Bearer unit-test"
    assert kwargs["json"]["messages"] == [{"role": "user", "content": "ping"}]


def test_primary_probe_does_not_restore_manual_once_model(monkeypatch):
    import agent.primary_reprobe as reprobe

    probe = Mock(return_value=True)
    monkeypatch.setattr(reprobe, "_primary_probe", probe)
    agent = _agent(_provider_fallback_active=False, _fallback_activated=True)
    assert not reprobe.maybe_reprobe_primary(agent, 15)
    probe.assert_not_called()


def test_primary_probe_runs_on_interval_and_restores(monkeypatch):
    import agent.primary_reprobe as reprobe

    monkeypatch.setattr(reprobe, "_reprobe_cfg", lambda: (5, 300.0))
    probe = Mock(return_value=True)
    monkeypatch.setattr(reprobe, "_primary_probe", probe)
    restore = Mock(return_value=True)
    monkeypatch.setattr("agent.agent_runtime_helpers.restore_primary_runtime", restore)
    agent = _agent()

    assert not reprobe.maybe_reprobe_primary(agent, 14)
    probe.assert_not_called()
    assert reprobe.maybe_reprobe_primary(agent, 15)
    probe.assert_called_once_with(agent, agent._primary_runtime)
    restore.assert_called_once_with(agent)
    assert agent._fallback_activated_at_call is None
    agent._emit_diagnostic_status.assert_called_once()


def test_failed_probe_leaves_fallback_active(monkeypatch):
    import agent.primary_reprobe as reprobe

    monkeypatch.setattr(reprobe, "_reprobe_cfg", lambda: (5, 300.0))
    monkeypatch.setattr(reprobe, "_primary_probe", Mock(return_value=False))
    restore = Mock(return_value=True)
    monkeypatch.setattr("agent.agent_runtime_helpers.restore_primary_runtime", restore)
    agent = _agent()

    assert not reprobe.maybe_reprobe_primary(agent, 14)
    assert not reprobe.maybe_reprobe_primary(agent, 15)
    restore.assert_not_called()


def test_preflight_reprobe_refunds_iteration_and_restarts_the_gate(monkeypatch):
    import sys
    from agent.turn_preflight_gate import run_preflight_gate

    # Keep this unit test outside conversation_loop's process-bootstrap import.
    monkeypatch.setitem(
        sys.modules,
        "agent.conversation_loop",
        SimpleNamespace(_ollama_context_limit_error=lambda *_args: None),
    )
    monkeypatch.setattr("agent.primary_reprobe.maybe_reprobe_primary", lambda _a, _n: True)
    budget = Mock()
    agent = SimpleNamespace(iteration_budget=budget, _api_call_count=3)
    verdict = run_preflight_gate(
        agent,
        request_pressure_tokens=0,
        _moa_prepared_request=None,
        pending_moa_prepared_request=None,
        messages=[],
        system_message="",
        user_message="",
        active_system_prompt="",
        conversation_history=[],
        api_call_count=3,
        compression_attempts=0,
        max_compression_attempts=3,
        effective_task_id=None,
        final_response=None,
        failed=False,
        _turn_exit_reason=None,
        _compression_timeout_exhausted=False,
        _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False,
        _last_preflight_pressure=None,
    )
    assert verdict.action == "continue"
    assert verdict.api_call_count == 2
    assert agent._api_call_count == 2
    budget.refund.assert_called_once_with()


def test_default_config_keeps_reprobe_opt_in():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    fallback = DEFAULT_CONFIG["fallback"]
    assert fallback["reprobe_per_call"] == 0
    assert fallback["reprobe_cooldown_sec"] == 300
