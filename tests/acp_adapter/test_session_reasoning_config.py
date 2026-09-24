"""ACP sessions honor the configured reasoning setting (#85153).

``SessionManager._make_agent`` builds its ``AIAgent`` from config like every other surface but never
passed ``reasoning_config``, so ``agent.reasoning_effort: none`` was ignored and the transport applied its
default effort — a 400 on non-reasoning models such as ``gpt-4o-mini``. Real config-file → ``load_config``
→ ``resolve_reasoning_config`` chain on the per-test ``HERMES_HOME``; only the agent constructor and
provider resolution are stubbed.
"""

import os
from pathlib import Path

import pytest
import yaml

from acp_adapter.session import SessionManager


class _CapturingAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model") or "stub-model"


@pytest.fixture
def acp_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kwargs: {"provider": requested or "openai-api", "api_mode": "codex_responses",
                                           "base_url": "https://example.invalid/v1", "api_key": "test-key"},
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kwargs: None)

    def _write_config(cfg: dict) -> None:
        (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    return _write_config


def test_acp_agent_receives_configured_reasoning(acp_env):
    acp_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"}, "agent": {"reasoning_effort": "none"}})
    sm = SessionManager(db=None)
    sm._get_db = lambda: None
    agent = sm._make_agent(session_id="s1", cwd=".")
    assert agent.kwargs["reasoning_config"] == {"enabled": False}

    # Per-model overrides key off the session's model, not ``model.default``.
    acp_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"},
             "agent": {"reasoning_effort": "none", "reasoning_overrides": {"gpt-5.6": "high"}}})
    agent = sm._make_agent(session_id="s2", cwd=".", model="gpt-5.6")
    assert agent.kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}


def test_acp_reasoning_override_beats_config(acp_env):
    """``hermes acp --reasoning LEVEL``: the process-scoped override replaces the config-resolved
    effort for every agent the manager builds, on any model — headless clients (task runners,
    editors) pass args after the subcommand and cannot type /reasoning mid-session."""
    acp_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"},
             "agent": {"reasoning_effort": "none", "reasoning_overrides": {"gpt-5.6": "high"}}})
    sm = SessionManager(db=None, reasoning_config_override={"enabled": True, "effort": "max"})
    sm._get_db = lambda: None
    # Both the default model and a differently-overridden model get the launch flag value.
    assert sm._make_agent(session_id="s1", cwd=".").kwargs["reasoning_config"] == {
        "enabled": True, "effort": "max"}
    assert sm._make_agent(session_id="s2", cwd=".", model="gpt-5.6").kwargs["reasoning_config"] == {
        "enabled": True, "effort": "max"}
    # No override → the config resolution from #85153 is untouched.
    sm_plain = SessionManager(db=None)
    sm_plain._get_db = lambda: None
    assert sm_plain._make_agent(session_id="s3", cwd=".", model="gpt-5.6").kwargs["reasoning_config"] == {
        "enabled": True, "effort": "high"}
