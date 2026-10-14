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
import hermes_yaml as yaml

from acp_adapter.session import SessionManager, SessionState


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


def test_acp_agent_receives_custom_provider_request_body(acp_env, monkeypatch):
    """#103738: the resolver's ``request_overrides`` (a custom entry's ``extra_body``) reach the ACP agent, as
    on the CLI; an explicit base_url naming a different endpoint does not inherit them."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kwargs: {"provider": "custom", "api_mode": "chat_completions",
                                           "base_url": "https://proxy.example/v1", "api_key": "k",
                                           "request_overrides": {"extra_body": {"user": "proxy-user"}}},
    )
    acp_env({"model": {"default": "some-model", "provider": "custom:my-proxy"}})
    sm = SessionManager(db=None)
    sm._get_db = lambda: None
    assert sm._make_agent(session_id="s1", cwd=".").kwargs["request_overrides"] == {"extra_body": {"user": "proxy-user"}}
    same = sm._make_agent(session_id="s2", cwd=".", base_url="https://proxy.example/v1")
    assert same.kwargs["request_overrides"] == {"extra_body": {"user": "proxy-user"}}
    other = sm._make_agent(session_id="s3", cwd=".", base_url="https://elsewhere.example/v1")
    assert "request_overrides" not in other.kwargs


def test_acp_agent_rebuild_honours_an_acp_set_reasoning_effort(acp_env):
    """An effort set through ACP ``thought_level`` (``session/set_config_option``) must reach the
    rebuilt agent verbatim — an explicit effort outranks config.yaml's ``agent.reasoning_effort``.
    Without the carry, a model switch silently reverted the editor's choice."""
    acp_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"}, "agent": {"reasoning_effort": "none"}})
    sm = SessionManager(db=None)
    sm._get_db = lambda: None
    agent = sm._make_agent(session_id="s1", cwd=".", reasoning_config={"enabled": True, "effort": "high"})
    assert agent.kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}  # type: ignore[attr-defined]
    # No explicit effort: the config-derived default still applies (here: disabled).
    agent2 = sm._make_agent(session_id="s2", cwd=".")
    assert agent2.kwargs["reasoning_config"] == {"enabled": False}  # type: ignore[attr-defined]


def test_acp_fork_carries_an_acp_set_reasoning_effort(acp_env, monkeypatch):
    """A fork is a rebuild: an effort set through ACP ``thought_level`` must reach the forked
    agent and state, not silently fall back to config.yaml's effort."""
    import threading

    made: dict = {}
    acp_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"}, "agent": {"reasoning_effort": "none"}})
    sm = SessionManager(db=None)
    sm._get_db = lambda: None

    def _fake_make_agent(**kwargs):
        made.update(kwargs)
        made.setdefault("calls", 0)
        made["calls"] += 1
        return _CapturingAgent(**kwargs)

    monkeypatch.setattr(sm, "_make_agent", _fake_make_agent)
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)

    original = SessionState(
        session_id="s1", agent=_CapturingAgent(), cwd=".", model="gpt-4o-mini",
        history=[{"role": "user", "content": "hi"}], cancel_event=threading.Event())
    original.reasoning_effort = ("high", {"enabled": True, "effort": "high"})
    monkeypatch.setattr(sm, "get_session", lambda _sid: original)

    forked = sm.fork_session("s1", cwd=".")
    assert forked is not None
    assert forked.reasoning_effort == ("high", {"enabled": True, "effort": "high"})
    # The fork's agent was built with the carried effort, not the config.yaml default.
    assert made["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert made["calls"] == 1
