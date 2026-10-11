"""ACP sessions honor the configured tool_preview_length (#135554).

``SessionManager._make_agent`` never mapped ``display.tool_preview_length`` onto
``agent.display.set_tool_preview_max_len`` (the gateway does it per turn in
``run_turn._run_agent_display_settings``), and ``build_tool_title`` hardcoded ``max_len=80``,
so ACP tool-call titles truncated at 80 regardless of config. Real config-file →
``load_config`` → ``resolve_display_setting`` chain on the per-test ``HERMES_HOME``; only the
agent constructor and provider resolution are stubbed.
"""

import os
from pathlib import Path

import hermes_yaml as yaml

import run_agent  # noqa: F401  collected before the home-io guard: run_agent's module-level recovery
# probe stats the checkout's git dir, which lives under the real home when this suite runs in a
# linked worktree (the guard then refuses the stat and every run_agent-importing test errors).

from acp_adapter.session import SessionManager


class _CapturingAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model") or "stub-model"


def _make_session_manager(monkeypatch, tmp_path, cfg):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kwargs: {"provider": requested or "openai-api", "api_mode": "codex_responses",
                                           "base_url": "https://example.invalid/v1", "api_key": "test-key"},
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kwargs: None)
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    sm = SessionManager(db=None)
    sm._get_db = lambda: None
    return sm


def test_acp_session_applies_configured_tool_preview_length(tmp_path, monkeypatch):
    from agent.display import get_tool_preview_max_len, set_tool_preview_max_len

    set_tool_preview_max_len(0)
    try:
        sm = _make_session_manager(monkeypatch, tmp_path, {"display": {"tool_preview_length": 120}})
        sm._make_agent(session_id="s1", cwd=".")
        assert get_tool_preview_max_len() == 120

        # Per-platform override (display.platforms.acp) wins over the top-level value, as in the gateway.
        sm = _make_session_manager(
            monkeypatch, tmp_path,
            {"display": {"tool_preview_length": 120, "platforms": {"acp": {"tool_preview_length": 40}}}},
        )
        sm._make_agent(session_id="s2", cwd=".")
        assert get_tool_preview_max_len() == 40

        # Unset falls back to the global default 0 (no limit), matching the gateway's display default.
        sm = _make_session_manager(monkeypatch, tmp_path, {})
        sm._make_agent(session_id="s3", cwd=".")
        assert get_tool_preview_max_len() == 0
    finally:
        set_tool_preview_max_len(0)
