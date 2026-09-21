"""Runtime-path doctor contracts: one isolated request, attribution, and redaction."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from hermes_cli import doctor_runtime


class _FakeAgent:
    def __init__(self):
        self.prompt_calls = 0
        self.request_calls = 0
        self.close_calls = 0
        self._last_api_first_chunk_at = None

    def _build_system_prompt(self, instruction):
        self.prompt_calls += 1
        return f"system: {instruction}"

    def _build_api_kwargs(self, messages):
        return {"messages": messages}

    def _interruptible_streaming_api_call(self, kwargs):
        self.request_calls += 1
        doctor_runtime.time = lambda: 10.0
        self._last_api_first_chunk_at = 10.025
        return SimpleNamespace(choices=[])

    def close(self):
        self.close_calls += 1


def _install_runtime_seams(monkeypatch, *, plugin_error=None):
    config = {"model": {"provider": "custom:work", "default": "model-x"}}
    runtime = {
        "provider": "custom",
        "requested_provider": "custom:work",
        "model": "model-x",
        "api_mode": "chat_completions",
        "base_url": "https://user:secret@example.test/v1?token=hidden#fragment",
        "api_key": "top-secret",
    }
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested, target_model: runtime,
    )

    def _plugins():
        if plugin_error:
            raise plugin_error

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", _plugins)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **kwargs: None,
    )
    return runtime


def test_runtime_probe_uses_one_request_and_emits_redacted_stable_report(monkeypatch):
    from hermes_cli.subcommands.doctor import build_doctor_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_doctor_parser(subparsers, cmd_doctor=lambda args: None)
    assert parser.parse_args(["doctor"]).runtime is False
    parsed = parser.parse_args(["doctor", "--runtime", "--json"])
    assert parsed.runtime is True and parsed.json is True

    _install_runtime_seams(monkeypatch)
    agent = _FakeAgent()
    monkeypatch.setattr(doctor_runtime, "_build_agent", lambda runtime, model: agent)
    monkeypatch.setattr(doctor_runtime, "time", lambda: 10.0)

    payload = doctor_runtime.run_runtime_diagnostic().to_dict()

    assert agent.prompt_calls == 1 and agent.request_calls == 1
    assert agent.close_calls == 1
    assert payload["schema_version"] == 1
    assert payload["resolved_runtime"] == {
        "provider": "custom",
        "requested_provider": "custom:work",
        "model": "model-x",
        "api_mode": "chat_completions",
        "base_url": "https://example.test/v1",
    }
    serialized = repr(payload)
    assert "top-secret" not in serialized and "secret" not in serialized and "hidden" not in serialized
    assert payload["timings"]["provider_ttfb_ms"] == 25.0


def test_runtime_phase_failure_isolated_and_persistence_disabled(monkeypatch):
    _install_runtime_seams(monkeypatch, plugin_error=RuntimeError("contains secret material"))
    agent = _FakeAgent()
    real_build_agent = doctor_runtime._build_agent
    monkeypatch.setattr(doctor_runtime, "_build_agent", lambda runtime, model: agent)
    monkeypatch.setattr(doctor_runtime, "time", lambda: 10.0)

    payload = doctor_runtime.run_runtime_diagnostic().to_dict()

    assert payload["failed_phase"] == "plugin_hook_initialization"
    assert payload["error_class"] == "RuntimeError"
    assert "contains secret material" not in repr(payload)
    assert agent.request_calls == 1
    assert agent.close_calls == 1

    failed_prompt_agent = _FakeAgent()
    failed_prompt_agent._build_system_prompt = lambda _instruction: (_ for _ in ()).throw(
        ValueError("prompt details")
    )
    monkeypatch.setattr(doctor_runtime, "_build_agent", lambda runtime, model: failed_prompt_agent)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    failed_prompt = doctor_runtime.run_runtime_diagnostic().to_dict()
    assert failed_prompt["failed_phase"] == "prompt_construction"
    assert failed_prompt_agent.request_calls == 0
    assert failed_prompt_agent.close_calls == 1

    seen = {}

    class _ConstructedAgent:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", _ConstructedAgent)
    built = real_build_agent(
        {
            "provider": "custom",
            "requested_provider": "custom:work",
            "api_mode": "chat_completions",
            "base_url": "https://example.test/v1",
            "api_key": "key",
        },
        "model-x",
    )
    assert seen["session_db"] is None
    assert seen["max_iterations"] == 1
    assert built._persist_disabled is True
    assert built._session_db is None
    assert built._end_session_on_close is False
