"""Production-builder skills-index ordering control for #137307.

Only external availability and runtime inputs are controlled; prompt renderers
stay real.
"""
from datetime import UTC, datetime
from os.path import commonprefix
from pathlib import Path
import socket

import pytest

from agent.system_prompt import _skills_prompt, build_system_prompt_parts


@pytest.fixture
def agent_factory(monkeypatch, tmp_path):
    import hermes_time
    import tools.registry as registry_module
    from run_agent import AIAgent
    from tools import env_probe
    from tools.skills_sync import sync_skills

    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "plain"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(hermes_time, "now", lambda: datetime(2026, 1, 15, 12, tzinfo=UTC))
    monkeypatch.setattr(registry_module, "_check_fn_cached", lambda _fn: True)
    monkeypatch.setattr(env_probe, "get_environment_probe_line", lambda **_kw: "")

    def refuse_network(*_args, **_kwargs):
        raise OSError("network disabled for prompt-builder control")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "create_connection", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    sync_skills(quiet=True)
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    agents = []

    def make(**overrides):
        options = dict(
            model="anthropic/claude-sonnet-4.5", api_key="inspect-only",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            quiet_mode=True, save_trajectories=False, platform="cli",
            skip_context_files=True, skip_background_review=True,
            session_id="20260115_120000_prompt_control", enabled_toolsets=["hermes-cli"],
        )
        options.update(overrides)
        agent = AIAgent(**options)
        agents.append(agent)
        return agent

    yield make
    for agent in reversed(agents):
        agent.close()


def _joined(parts):
    return "\n\n".join(parts[tier] for tier in ("stable", "context", "volatile") if parts[tier])


def test_shared_skills_index_precedes_memory_guidance_divergence(agent_factory, record_property):
    """A memory exposure change must not discard an identical skills index behind it."""
    toolsets = ["terminal", "file", "web", "skills", "session_search", "memory"]
    full = agent_factory(enabled_toolsets=toolsets)
    restricted = agent_factory(enabled_toolsets=toolsets, disabled_toolsets=["memory"])
    assert "memory" in full.valid_tool_names
    assert "memory" not in restricted.valid_tool_names
    assert full.valid_tool_names == restricted.valid_tool_names | {"memory"}
    index = _skills_prompt(full).strip()
    assert index and "<available_skills>" in index
    assert _skills_prompt(restricted).strip() == index
    full_parts = build_system_prompt_parts(full)
    restricted_parts = build_system_prompt_parts(restricted)
    assert full_parts["stable"] == restricted_parts["stable"]
    assert full_parts["context"] == restricted_parts["context"]
    full_prompt, restricted_prompt = _joined(full_parts), _joined(restricted_parts)
    shared = commonprefix([full_prompt, restricted_prompt])
    record_property("shared_prefix_chars", len(shared))
    record_property("skills_index_chars", len(index))
    assert index in shared
    assert full_parts["volatile"].startswith(index)
    assert restricted_parts["volatile"].startswith(index)
    assert "Memory is the narrow exception" in full_parts["volatile"]
    assert "Memory is the narrow exception" not in restricted_parts["volatile"]
