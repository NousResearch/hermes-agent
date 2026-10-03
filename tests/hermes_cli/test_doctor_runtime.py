"""Runtime-path doctor contracts: one isolated request, attribution, and redaction."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from hermes_cli import doctor_runtime


class _FakeAgent:
    def __init__(self):
        self.prompt_calls = 0
        self.request_calls = 0
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


def _install_runtime_seams(monkeypatch, *, plugin_error=None):
    config = {
        "model": {"provider": "custom:work", "default": "model-x"},
    }
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

    report = doctor_runtime.run_runtime_diagnostic()
    payload = report.to_dict()

    assert report.status == "ok"
    assert agent.prompt_calls == 1 and agent.request_calls == 1
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
    assert payload["timings"]["hermes_first_chunk_ms"] >= 25.0


def test_runtime_phase_failure_is_isolated_and_agent_build_disables_persistence(monkeypatch):
    _install_runtime_seams(monkeypatch, plugin_error=RuntimeError("contains secret material"))
    agent = _FakeAgent()
    real_build_agent = doctor_runtime._build_agent
    monkeypatch.setattr(doctor_runtime, "_build_agent", lambda runtime, model: agent)
    monkeypatch.setattr(doctor_runtime, "time", lambda: 10.0)

    report = doctor_runtime.run_runtime_diagnostic()
    payload = report.to_dict()

    assert report.status == "fail"
    assert report.failed_phase == "plugin_hook_initialization"
    assert payload["error_class"] == "RuntimeError"
    assert "contains secret material" not in repr(payload)
    assert agent.request_calls == 1
    assert {phase.name for phase in report.phases} >= {
        "plugin_hook_initialization",
        "mcp_initialization",
        "agent_tool_context_initialization",
        "prompt_construction",
        "provider_first_chunk",
    }

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
    assert seen["platform"] == "doctor"
    assert seen["max_iterations"] == 1
    assert seen["skip_background_review"] is True
    assert built._persist_disabled is True
    assert built._session_db is None
    assert built._end_session_on_close is False
    assert built.suppress_status_output is True