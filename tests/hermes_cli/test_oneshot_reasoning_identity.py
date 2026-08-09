"""Contract tests for reasoning identity in non-interactive one-shot runs."""

import json

import pytest

from hermes_cli import oneshot


def _stub_runtime(monkeypatch, tmp_path, *, configured_effort=None):
    import hermes_cli.config as config_mod
    import run_agent
    from hermes_cli import mcp_startup, runtime_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    cfg = {"model": {"default": "test-model", "provider": "openai-api"}}
    if configured_effort is not None:
        cfg["agent"] = {"reasoning_effort": configured_effort}
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(
        mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test",
            "base_url": "http://provider.invalid/v1",
            "provider": "openai-api",
            "requested_provider": "openai-api",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    observed = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt):
            return {"final_response": "ok", "completed": True, "failed": False}

        def close(self):
            pass

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    return observed


@pytest.mark.parametrize(
    ("reasoning", "expected_config", "expected_effort"),
    [
        ("low", {"enabled": True, "effort": "low"}, "low"),
        ("xhigh", {"enabled": True, "effort": "xhigh"}, "xhigh"),
        ("none", {"enabled": False}, "none"),
    ],
)
def test_explicit_reasoning_is_passed_and_attested(
    monkeypatch, tmp_path, reasoning, expected_config, expected_effort
):
    observed = _stub_runtime(monkeypatch, tmp_path)
    response, result = oneshot._run_agent(
        "identity", model="test-model", provider="openai-api", reasoning=reasoning
    )
    assert response == "ok"
    assert observed["reasoning_config"] == expected_config
    assert result["reasoning_effort"] == expected_effort


def test_absent_cli_override_preserves_configured_reasoning(monkeypatch, tmp_path):
    observed = _stub_runtime(monkeypatch, tmp_path, configured_effort="high")
    _response, result = oneshot._run_agent(
        "identity", model="test-model", provider="openai-api", reasoning=None
    )
    assert observed["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert result["reasoning_effort"] == "high"


def test_invalid_reasoning_fails_before_agent_or_provider(monkeypatch, capsys):
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("agent/provider reached")
        ),
    )
    assert oneshot.run_oneshot("identity", reasoning="turbo") == 2
    assert "invalid --reasoning" in capsys.readouterr().err


def test_failure_usage_keeps_resolved_reasoning_identity(monkeypatch, tmp_path):
    import run_agent

    usage_path = tmp_path / "usage.json"
    _stub_runtime(monkeypatch, tmp_path)

    class FailingAgent:
        def __init__(self, **_kwargs):
            raise RuntimeError("provider failed")

    monkeypatch.setattr(run_agent, "AIAgent", FailingAgent)
    assert oneshot.run_oneshot(
        "identity",
        model="test-model",
        provider="openai-api",
        reasoning="high",
        usage_file=str(usage_path),
    ) == 1
    report = json.loads(usage_path.read_text())
    assert report["failed"] is True
    assert report["reasoning_effort"] == "high"


def test_main_oneshot_wrapper_forwards_reasoning(monkeypatch):
    from hermes_cli import main

    observed = {}
    monkeypatch.setattr(
        oneshot,
        "run_oneshot",
        lambda *_args, **kwargs: observed.update(kwargs) or 0,
    )
    monkeypatch.setattr(main, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(
        main, "_exit_after_oneshot", lambda rc: observed.update(exit_code=rc)
    )
    main._run_and_exit_oneshot("identity", reasoning="low")
    assert observed["reasoning"] == "low"
    assert observed["exit_code"] == 0
