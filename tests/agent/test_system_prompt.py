"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(
        cwd=None, skip_soul=False, context_length=None,
        allow_install_tree_fallback=False,
    ):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _init_code_repo(path):
    """A git repo that actually holds code — the coding posture requires a source
    file (or manifest), not a bare ``.git`` (a prose/notes repo stays general)."""
    import subprocess

    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "main.py").write_text("print('hi')\n")


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)


class TestTelegramRichMessagesHint:
    """Verify that TELEGRAM_RICH_MESSAGES_HINT is conditionally included."""

    def test_base_hint_without_rich_messages(self, monkeypatch):
        """When rich_messages is False (default), only the base hint is used."""
        agent = _make_agent(platform="telegram")
        # Mock config to return rich_messages: false (default)
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "platforms": {"telegram": {"extra": {"rich_messages": False}}}
            }
            stable = _stable_prompt(agent)
        # Base hint should be present
        assert "Standard Markdown is automatically converted" in stable
        # Rich-messages extension should NOT be present
        assert "lean into it" not in stable
        assert "task lists" not in stable

    def test_rich_hint_with_rich_messages_enabled(self, monkeypatch):
        """When rich_messages is True, the rich-messages extension is appended."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "platforms": {"telegram": {"extra": {"rich_messages": True}}}
            }
            stable = _stable_prompt(agent)
        # Base hint should be present
        assert "Standard Markdown is automatically converted" in stable
        # Rich-messages extension should be present
        assert "lean into it" in stable
        assert "task lists" in stable
        assert "math/formulas" in stable

    def test_base_hint_without_config(self, monkeypatch):
        """When config has no telegram section, only base hint is used."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {}
            stable = _stable_prompt(agent)
        assert "Standard Markdown is automatically converted" in stable
        assert "lean into it" not in stable


# ── skill-graph gateway injection tests ──────────────────────────────────

def _make_skill_graph_agent(**overrides):
    """Agent with skill-graph mode enabled and skill_graph_search tool."""
    return _make_agent(
        valid_tool_names=["skill_graph_search"],
        _skill_graph_mode=True,
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        **overrides,
    )


def test_gateway_extras_from_routing_extensions(tmp_path, monkeypatch):
    """Gateway skills declared in routing-extensions.md are injected."""
    # Write routing-extensions.md with a gateway skill
    ext_file = tmp_path / "routing-extensions.md"
    ext_file.write_text("""## Pre-installed Gateways (Extensions)

| Gateway Skill | Purpose |
|--------------|---------|
| `project-directories` | Project code directory map |
| `troupe-lookup` | Troupe roster queries |
""")

    # Mock config to return our temp file
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "skills": {
                "config": {
                    "skill-graph": {
                        "extensions_file": str(ext_file),
                    }
                }
            }
        },
    )

    agent = _make_skill_graph_agent()

    captured = []
    def fake_context_files(**kw):
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        parts = build_system_prompt_parts(agent)

    stable = parts.get("stable", "")
    assert "Available Skills" in stable
    assert "skill-graph" in stable
    assert "project-directories" in stable
    assert "troupe-lookup" in stable


def test_gateway_extras_missing_file_no_crash(tmp_path, monkeypatch):
    """Missing extensions_file should not crash — just inject skill-graph only."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "skills": {
                "config": {
                    "skill-graph": {
                        "extensions_file": "/nonexistent/path/routing-extensions.md",
                    }
                }
            }
        },
    )

    agent = _make_skill_graph_agent()

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        parts = build_system_prompt_parts(agent)

    stable = parts.get("stable", "")
    assert "Available Skills" in stable
    assert "skill-graph" in stable
    # Gateway extras not injected (file missing)
    assert "project-directories" not in stable


def test_gateway_extras_no_extensions_file_config(tmp_path, monkeypatch):
    """No extensions_file in config — only skill-graph in Available Skills."""
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"skills": {"config": {"skill-graph": {}}}},
    )

    agent = _make_skill_graph_agent()

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        parts = build_system_prompt_parts(agent)

    stable = parts.get("stable", "")
    assert "Available Skills" in stable
    assert "skill-graph" in stable


def test_gateway_extras_not_injected_without_skill_graph_mode(tmp_path, monkeypatch):
    """Without skill-graph mode, Available Skills section should not appear."""
    agent = _make_agent(
        valid_tool_names=["skills_list", "skill_view"],
    )

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        parts = build_system_prompt_parts(agent)

    stable = parts.get("stable", "")
    # "Available Skills\n  skill-graph —" is the injection pattern;
    # plain "Available Skills" might appear in other contexts
    assert "Available Skills\n  skill-graph" not in stable
