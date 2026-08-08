"""Tests for the ``hermes agents`` CLI command.

Uses a temporary HERMES_HOME with fake profile dirs/configs so the real
user config is never touched. Exercises registry loading, default
creation, table output, JSON output, init, and orchestrate handoff.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Ensure the repo root is on sys.path so `hermes_cli` / `hermes_constants`
# import correctly regardless of pytest invocation cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli import agents as agents_mod


@pytest.fixture
def temp_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir and patch get_hermes_home accordingly."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Patch the get_hermes_home used inside hermes_cli.agents so it
    # reflects the temp home regardless of any cached imports.
    monkeypatch.setattr(
        agents_mod, "get_hermes_home", lambda: Path(os.environ["HERMES_HOME"])
    )
    return home


def _write_fake_profile(home: Path, name: str, model: str, provider: str) -> None:
    """Create a fake profile directory with a config.yaml."""
    if name == "default":
        profile_dir = home
    else:
        profile_dir = home / "profiles" / name
        profile_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"model": {"default": model, "provider": provider}}
    (profile_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _write_registry(home: Path, registry: dict) -> None:
    (home / "agents.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")


# ---------------------------------------------------------------------------
# load_agents
# ---------------------------------------------------------------------------

class TestLoadAgents:
    def test_missing_registry_uses_discovered_profiles(self, temp_hermes_home, monkeypatch):
        _write_fake_profile(temp_hermes_home, "default", "gpt-5.5", "openai-codex")
        # Force discovery to see our fake profiles. We patch
        # _discover_profiles directly so the test is hermetic.
        monkeypatch.setattr(
            agents_mod,
            "_discover_profiles",
            lambda: [("default", "gpt-5.5", "openai-codex")],
        )
        agents = agents_mod.load_agents()
        assert len(agents) >= 1
        assert any(a.profile == "default" for a in agents)
        assert any(a.model == "gpt-5.5" for a in agents)

    def test_explicit_registry_is_loaded(self, temp_hermes_home, monkeypatch):
        _write_fake_profile(temp_hermes_home, "default", "gpt-5.5", "openai-codex")
        _write_registry(temp_hermes_home, {
            "version": 1,
            "agents": {
                "lead": {
                    "profile": "default",
                    "role": "orchestrator",
                    "description": "Team lead",
                    "can_edit_files": False,
                    "skills": ["kanban-orchestrator"],
                },
            },
        })
        monkeypatch.setattr(
            agents_mod,
            "_discover_profiles",
            lambda: [("default", "gpt-5.5", "openai-codex")],
        )
        agents = agents_mod.load_agents()
        lead = next(a for a in agents if a.name == "lead")
        assert lead.profile == "default"
        assert lead.role == "orchestrator"
        assert lead.model == "gpt-5.5"
        assert lead.skills == ["kanban-orchestrator"]

    def test_missing_profile_marked(self, temp_hermes_home, monkeypatch):
        _write_fake_profile(temp_hermes_home, "default", "gpt-5.5", "openai-codex")
        _write_registry(temp_hermes_home, {
            "version": 1,
            "agents": {
                "ghost": {
                    "profile": "nonexistent_profile",
                    "role": "agent",
                },
            },
        })
        # Discovery only finds 'default'; 'nonexistent_profile' won't exist.
        from hermes_cli.profiles import list_profiles
        monkeypatch.setattr(
            agents_mod,
            "_discover_profiles",
            lambda: [(p.name, p.model or "", p.provider or "") for p in list_profiles()],
        )
        agents = agents_mod.load_agents()
        ghost = next(a for a in agents if a.name == "ghost")
        assert ghost.status == "missing_profile"

    def test_registry_agent_without_model_inherits_discovered(self, temp_hermes_home, monkeypatch):
        _write_fake_profile(temp_hermes_home, "default", "gpt-5.5", "openai-codex")
        _write_registry(temp_hermes_home, {
            "version": 1,
            "agents": {
                "lead": {"profile": "default", "role": "orchestrator"},
            },
        })
        monkeypatch.setattr(
            agents_mod,
            "_discover_profiles",
            lambda: [("default", "gpt-5.5", "openai-codex")],
        )
        agents = agents_mod.load_agents()
        lead = next(a for a in agents if a.name == "lead")
        assert lead.model == "gpt-5.5"
        assert lead.provider == "openai-codex"


# ---------------------------------------------------------------------------
# render_table / to_json
# ---------------------------------------------------------------------------

class TestRender:
    def test_table_contains_expected_columns_and_model(self, temp_hermes_home, monkeypatch):
        monkeypatch.setattr(
            agents_mod,
            "_discover_profiles",
            lambda: [("coding", "glm-5.2:cloud", "custom")],
        )
        agents = agents_mod.load_agents()
        out = agents_mod.render_table(agents)
        assert "Agent" in out
        assert "Model" in out
        assert "Provider" in out
        assert "Role" in out
        assert "Status" in out
        assert "glm-5.2:cloud" in out

    def test_json_is_valid(self, temp_hermes_home, monkeypatch):
        monkeypatch.setattr(
            agents_mod,
            "_discover_profiles",
            lambda: [("coding", "glm-5.2:cloud", "custom")],
        )
        agents = agents_mod.load_agents()
        raw = agents_mod.to_json(agents)
        data = json.loads(raw)
        assert isinstance(data, list)
        assert any(entry["profile"] == "coding" for entry in data)
        assert any(entry["model"] == "glm-5.2:cloud" for entry in data)


# ---------------------------------------------------------------------------
# init_registry
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_creates_file(self, temp_hermes_home):
        path = agents_mod.init_registry(force=False)
        assert path == temp_hermes_home / "agents.yaml"
        assert path.is_file()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "agents" in data
        assert "orch" in data["agents"]

    def test_init_refuses_overwrite_without_force(self, temp_hermes_home):
        agents_mod.init_registry(force=False)
        with pytest.raises(FileExistsError):
            agents_mod.init_registry(force=False)

    def test_init_force_overwrites(self, temp_hermes_home):
        agents_mod.init_registry(force=False)
        # Overwrite manually with different content to prove --force replaces.
        path = temp_hermes_home / "agents.yaml"
        path.write_text("version: 99\nagents: {}\n", encoding="utf-8")
        agents_mod.init_registry(force=True)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "orch" in data["agents"]


# ---------------------------------------------------------------------------
# orchestrate
# ---------------------------------------------------------------------------

class TestOrchestrate:
    def test_orchestrate_builds_handoff_body(self, temp_hermes_home):
        result = agents_mod.orchestrate("Test task", create_card=False)
        assert result["task"] == "Test task"
        assert "hermes kanban create" in result["command"]
        assert "orch" in result["command"]
        assert "Test task" in result["body"]

    def test_orchestrate_empty_task_raises(self, temp_hermes_home):
        with pytest.raises(ValueError):
            agents_mod.orchestrate("   ", create_card=False)

    def test_orchestrate_no_create_has_no_card_id(self, temp_hermes_home):
        result = agents_mod.orchestrate("Some task", create_card=False)
        assert result["card_id"] is None

    def test_render_summary_contains_command(self, temp_hermes_home):
        result = agents_mod.orchestrate("Do something", create_card=False)
        summary = agents_mod.render_orchestrate_summary(result)
        assert "hermes kanban create" in summary

    def test_orchestrate_quoting_safe_with_quotes_and_dollar(self, temp_hermes_home):
        """The printed command must survive shell metacharacters: quotes, $, backticks."""
        tricky = 'tarefa com "aspas" e $HOME e `cmd`'
        result = agents_mod.orchestrate(tricky, create_card=False)
        cmd = result["command"]
        # The command must be safe to copy/paste into a shell: shlex.quote
        # wraps the args in single quotes, so $ and `` are literal.
        # Verify by parsing the command back with shlex.
        import shlex as _shlex
        parts = _shlex.split(cmd)
        assert parts[0] == "hermes"
        assert parts[1] == "kanban"
        assert parts[2] == "create"
        # The title arg must reconstruct to the exact title (not a mangled one).
        assert "orchestrate: " + tricky in parts[3]
        # $HOME must appear literally (not expanded), and aspas must be intact.
        assert "$HOME" in parts[3]
        assert '"aspas"' in parts[3]
        # The --body arg must contain the original task text too.
        assert any("$HOME" in p for p in parts)

    def test_render_summary_no_duplicate_command_when_card_created(self, temp_hermes_home):
        """When card_id exists, summary must NOT print the kanban create command."""
        result = agents_mod.orchestrate("Some task", create_card=False)
        # Simulate a card having been created.
        result["card_id"] = "t_fake123"
        summary = agents_mod.render_orchestrate_summary(result)
        assert "Created Kanban card: t_fake123" in summary
        # Must NOT contain the create command (would invite duplicates).
        assert "hermes kanban create" not in summary
        assert "hermes kanban list" in summary  # points to board instead

    def test_render_summary_shows_command_when_no_card(self, temp_hermes_home):
        """When card_id is None, summary MUST print the manual create command."""
        result = agents_mod.orchestrate("Some task", create_card=False)
        summary = agents_mod.render_orchestrate_summary(result)
        assert "hermes kanban create" in summary
        assert "Kanban card was not created" in summary

    def test_render_summary_suggests_gateway_when_dispatcher_down(self, temp_hermes_home):
        """When card exists but dispatcher is down, suggest `hermes gateway start`."""
        result = agents_mod.orchestrate("Some task", create_card=False)
        result["card_id"] = "t_fake456"
        result["dispatcher_running"] = False
        result["dispatcher_message"] = "No gateway is running"
        summary = agents_mod.render_orchestrate_summary(result)
        assert "hermes gateway start" in summary
        assert "hermes kanban list" in summary
        # Still must not suggest creating a duplicate card.
        assert "hermes kanban create" not in summary


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class TestDashboard:
    """Tests for the ``hermes agents dashboard`` subcommand and rendering."""

    def _make_agents(self) -> list:
        """Build a small set of fake agents for rendering tests."""
        from hermes_cli.agents import AgentSpec
        return [
            AgentSpec(
                name="orch", profile="orch", role="orchestrator",
                description="Team lead", can_edit_files=False,
                escalates_to="user", skills=["kanban-orchestrator"],
                model="gpt-5.5", provider="openai-codex", status="ready",
            ),
            AgentSpec(
                name="coding", profile="coding", role="implementer",
                description="Writes code", can_edit_files=True,
                escalates_to="orch", skills=["test-driven-development"],
                model="glm-5.2:cloud", provider="custom", status="ready",
            ),
            AgentSpec(
                name="default", profile="default", role="fallback",
                description="Legacy base", can_edit_files=False,
                escalates_to="orch", skills=[],
                model="glm-5.2", provider="ollama-cloud", status="ready",
            ),
        ]

    def test_render_dashboard_str_contains_agents(self):
        from hermes_cli.agents_dashboard import render_dashboard_str
        agents = self._make_agents()
        out = render_dashboard_str(agents)
        assert "orch" in out
        assert "coding" in out
        assert "default" in out
        assert "gpt-5.5" in out
        assert "glm-5.2:cloud" in out

    def test_render_dashboard_str_contains_actions(self):
        from hermes_cli.agents_dashboard import render_dashboard_str
        agents = self._make_agents()
        out = render_dashboard_str(agents)
        assert "hermes agents orchestrate" in out
        assert "hermes kanban list" in out

    def test_render_dashboard_str_marks_fallback(self):
        from hermes_cli.agents_dashboard import render_dashboard_str
        agents = self._make_agents()
        out = render_dashboard_str(agents)
        # The default profile should be marked as fallback.
        assert "fallback" in out

    def test_render_agent_detail_contains_all_fields(self):
        from hermes_cli.agents_dashboard import render_agent_detail_str
        agents = self._make_agents()
        coding = agents[1]
        detail = render_agent_detail_str(coding)
        assert "coding" in detail
        assert "implementer" in detail
        assert "glm-5.2:cloud" in detail
        assert "custom" in detail
        assert "test-driven-development" in detail
        assert "orch" in detail  # escalates_to

    def test_render_agent_detail_marks_fallback(self):
        from hermes_cli.agents_dashboard import render_agent_detail_str
        agents = self._make_agents()
        default_agent = agents[2]
        detail = render_agent_detail_str(default_agent)
        assert "fallback" in detail

    def test_run_dashboard_non_interactive_prints_and_exits(self, temp_hermes_home, monkeypatch):
        """In a non-TTY, run_dashboard should print the table and return 0."""
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        # Force non-interactive
        monkeypatch.setattr(dash_mod, "_is_interactive", lambda: False)
        result = dash_mod.run_dashboard(agents=agents)
        assert result == 0

    def test_run_dashboard_no_agents_prints_message(self, temp_hermes_home, monkeypatch):
        """When there are no agents, run_dashboard prints a helpful message."""
        from hermes_cli import agents_dashboard as dash_mod
        result = dash_mod.run_dashboard(agents=[])
        assert result == 0

    def test_dashboard_no_detail_flag_renders_non_interactive(self, temp_hermes_home, monkeypatch):
        """``--no-detail`` should force non-interactive rendering."""
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        out = dash_mod.render_dashboard_str(agents)
        assert "orch" in out
        assert "coding" in out

    def test_plain_table_fallback_has_columns(self, monkeypatch):
        """The plain-text fallback table should have the expected columns."""
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        # Force plain rendering by making rich import fail.
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name.startswith("rich"):
                raise ImportError("no rich")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        out = dash_mod._render_plain_table(agents)
        assert "Agent" in out
        assert "Model" in out
        assert "Provider" in out
        assert "Role" in out
        assert "Status" in out
        assert "orch" in out

    def test_esc_quits_main_prompt(self, temp_hermes_home, monkeypatch):
        """Pressing Esc (\\x1b) at the main prompt should exit with 0."""
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        # Force interactive mode so the loop runs.
        monkeypatch.setattr(dash_mod, "_is_interactive", lambda: True)
        # Simulate Esc at the first prompt.
        monkeypatch.setattr("builtins.input", lambda prompt="": "\x1b")
        result = dash_mod.run_dashboard(agents=agents)
        assert result == 0

    def test_q_quits_main_prompt(self, temp_hermes_home, monkeypatch):
        """Pressing q at the main prompt should exit with 0."""
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        monkeypatch.setattr(dash_mod, "_is_interactive", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "q")
        result = dash_mod.run_dashboard(agents=agents)
        assert result == 0

    def test_esc_in_detail_goes_back_not_quit(self, temp_hermes_home, monkeypatch):
        """Esc at the detail prompt should go back, not quit.

        The sequence: select agent 1, then Esc at the detail prompt
        (goes back), then q at the main prompt (quits).
        """
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        monkeypatch.setattr(dash_mod, "_is_interactive", lambda: True)
        # First input: "1" (select agent 1)
        # Second input: "\x1b" (Esc at detail → goes back)
        # Third input: "q" (quit at main prompt)
        inputs = iter(["1", "\x1b", "q"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        result = dash_mod.run_dashboard(agents=agents)
        assert result == 0

    def test_non_interactive_actions_hint_mentions_tty(self, temp_hermes_home, monkeypatch):
        """The non-interactive actions panel should not say 'Press a number'."""
        from hermes_cli import agents_dashboard as dash_mod
        agents = self._make_agents()
        monkeypatch.setattr(dash_mod, "_is_interactive", lambda: False)
        out = dash_mod.render_dashboard_str(agents)
        assert "Press a number" not in out
        assert "TTY" in out