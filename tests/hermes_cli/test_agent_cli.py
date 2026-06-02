"""Tests for the ``hermes agent`` CLI subcommand surface.

Covers ``list``, ``show``, ``add``, and ``remove`` operations on the
``agents:`` and ``routes:`` sections of config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import agent as agent_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior config state."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Reset hermes_constants cache so get_hermes_home() re-reads.
    try:
        import hermes_constants
        hermes_constants._cached_hermes_home = None  # type: ignore[attr-defined]
    except Exception:
        pass
    return home


@pytest.fixture
def sample_cfg() -> Dict[str, Any]:
    return {
        "default_agent": "main",
        "agents": {
            "main": {},
            "coder": {
                "model": "anthropic/claude-opus-4-6",
                "home_dir": "~/.hermes/profiles/coder",
                "enabled_toolsets": ["filesystem", "terminal"],
            },
            "research": {
                "model": "anthropic/claude-sonnet-4-6",
            },
        },
        "routes": [
            {"match": {"platform": "telegram", "chat_id": "-1001234"}, "agent": "coder"},
            {"match": {"platform": "slack", "guild_id": "T0ABC"}, "agent": "coder"},
            {"match": {"platform": "matrix"}, "agent": "research"},
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace from keyword args."""
    return argparse.Namespace(**kwargs)


# ---------------------------------------------------------------------------
# cmd_agent_list
# ---------------------------------------------------------------------------

class TestAgentList:
    def test_empty_config_shows_message(self, fresh_home, monkeypatch, capsys):
        monkeypatch.setattr(agent_mod, "load_config", lambda: {})
        rv = agent_mod.cmd_agent_list(_args())
        assert rv == 0
        out = capsys.readouterr().out
        assert "No agents configured" in out

    def test_lists_all_agents(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_list(_args())
        assert rv == 0
        out = capsys.readouterr().out
        assert "main" in out
        assert "coder" in out
        assert "research" in out
        assert "anthropic/claude-opus-4-6" in out
        assert "anthropic/claude-sonnet-4-6" in out

    def test_default_marker(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_list(_args())
        out = capsys.readouterr().out
        # main is the default, so it should have the "*" marker
        lines = out.splitlines()
        main_line = [l for l in lines if l.strip().startswith("*") and "main" in l]
        assert main_line, "default agent should be marked with *"

    def test_route_counts(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_list(_args())
        out = capsys.readouterr().out
        # coder has 2 routes, research has 1, main has 0
        lines = [l for l in out.splitlines() if "coder" in l or "research" in l]
        coder_line = [l for l in lines if "coder" in l][0]
        research_line = [l for l in lines if "research" in l][0]
        assert "2" in coder_line
        assert "1" in research_line


# ---------------------------------------------------------------------------
# cmd_agent_show
# ---------------------------------------------------------------------------

class TestAgentShow:
    def test_existing_agent(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_show(_args(agent_id="coder"))
        assert rv == 0
        out = capsys.readouterr().out
        assert "Agent: coder" in out
        assert "anthropic/claude-opus-4-6" in out
        assert "memories" in out
        assert "skills" in out
        assert "Sessions" in out

    def test_shows_routes(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_show(_args(agent_id="coder"))
        out = capsys.readouterr().out
        assert "Routes (2)" in out
        assert "platform=telegram" in out
        assert "platform=slack" in out

    def test_missing_agent(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_show(_args(agent_id="nonexistent"))
        assert rv == 1
        out = capsys.readouterr().out
        assert "not found" in out

    def test_uses_default_home_dir(self, fresh_home, monkeypatch, capsys, sample_cfg):
        """Agent without home_dir should show the default HERMES_HOME path."""
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg)
        rv = agent_mod.cmd_agent_show(_args(agent_id="main"))
        out = capsys.readouterr().out
        # main has no home_dir, so it should show the default
        assert str(fresh_home) in out


# ---------------------------------------------------------------------------
# cmd_agent_add
# ---------------------------------------------------------------------------

class TestAgentAdd:
    def test_add_basic(self, fresh_home, monkeypatch, capsys):
        saved = {}

        def fake_load():
            return saved.copy() if saved else {}

        def fake_save(cfg):
            nonlocal saved
            saved = cfg

        monkeypatch.setattr(agent_mod, "load_config", fake_load)
        monkeypatch.setattr(agent_mod, "save_config", fake_save)

        rv = agent_mod.cmd_agent_add(_args(
            agent_id="reviewer",
            model="gpt-4",
            provider="openai",
            home_dir=None,
            enabled_toolsets=None,
            from_profile=None,
        ))
        assert rv == 0
        out = capsys.readouterr().out
        assert "added" in out
        assert saved["agents"]["reviewer"]["model"] == "gpt-4"
        assert saved["agents"]["reviewer"]["provider"] == "openai"

    def test_add_with_toolsets(self, fresh_home, monkeypatch, capsys):
        saved = {}

        def fake_load():
            return saved.copy() if saved else {}

        def fake_save(cfg):
            nonlocal saved
            saved = cfg

        monkeypatch.setattr(agent_mod, "load_config", fake_load)
        monkeypatch.setattr(agent_mod, "save_config", fake_save)

        rv = agent_mod.cmd_agent_add(_args(
            agent_id="ops",
            model=None,
            provider=None,
            home_dir=None,
            enabled_toolsets="terminal,k8s",
            from_profile=None,
        ))
        assert rv == 0
        assert saved["agents"]["ops"]["enabled_toolsets"] == ["terminal", "k8s"]

    def test_rejects_invalid_id(self, fresh_home, monkeypatch, capsys):
        monkeypatch.setattr(agent_mod, "load_config", lambda: {})
        rv = agent_mod.cmd_agent_add(_args(
            agent_id="bad id!",
            model=None,
            provider=None,
            home_dir=None,
            enabled_toolsets=None,
            from_profile=None,
        ))
        assert rv == 1
        out = capsys.readouterr().out
        assert "Invalid agent ID" in out

    def test_rejects_duplicate(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg.copy())
        rv = agent_mod.cmd_agent_add(_args(
            agent_id="coder",
            model=None,
            provider=None,
            home_dir=None,
            enabled_toolsets=None,
            from_profile=None,
        ))
        assert rv == 1
        out = capsys.readouterr().out
        assert "already exists" in out

    def test_clone_from_profile(self, fresh_home, monkeypatch, capsys):
        """Clone from an existing profile directory."""
        # Create a source profile
        src = fresh_home / "profiles" / "main"
        src.mkdir(parents=True)
        (src / "SOUL.md").write_text("# Main SOUL")

        saved = {"agents": {}, "routes": [], "default_agent": "main"}

        def fake_load():
            return saved.copy()

        def fake_save(cfg):
            nonlocal saved
            saved = cfg

        # Patch the module-level load_config that _load_config() calls
        import hermes_cli.config
        monkeypatch.setattr(hermes_cli.config, "load_config", fake_load)
        monkeypatch.setattr(agent_mod, "save_config", fake_save)

        rv = agent_mod.cmd_agent_add(_args(
            agent_id="cloned",
            model=None,
            provider=None,
            home_dir=None,
            enabled_toolsets=None,
            from_profile="main",
        ))
        assert rv == 0
        out = capsys.readouterr().out
        assert "Cloned" in out
        dst = fresh_home / "profiles" / "cloned"
        assert dst.exists()
        assert (dst / "SOUL.md").exists()
        assert saved["agents"]["cloned"]["home_dir"] == str(dst)

    def test_clone_missing_source(self, fresh_home, monkeypatch, capsys):
        monkeypatch.setattr(agent_mod, "load_config", lambda: {})
        rv = agent_mod.cmd_agent_add(_args(
            agent_id="cloned",
            model=None,
            provider=None,
            home_dir=None,
            enabled_toolsets=None,
            from_profile="nonexistent",
        ))
        assert rv == 1
        out = capsys.readouterr().out
        assert "not found" in out


# ---------------------------------------------------------------------------
# cmd_agent_remove
# ---------------------------------------------------------------------------

class TestAgentRemove:
    def test_remove_existing(self, fresh_home, monkeypatch, capsys, sample_cfg):
        saved = sample_cfg.copy()

        def fake_load():
            return saved.copy()

        def fake_save(cfg):
            nonlocal saved
            saved = cfg

        monkeypatch.setattr(agent_mod, "load_config", fake_load)
        monkeypatch.setattr(agent_mod, "save_config", fake_save)

        rv = agent_mod.cmd_agent_remove(_args(agent_id="research", yes=True))
        assert rv == 0
        out = capsys.readouterr().out
        assert "removed" in out
        assert "research" not in saved["agents"]
        # Its route should also be cleaned up
        assert not any(r.get("agent") == "research" for r in saved["routes"])

    def test_cannot_remove_main(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg.copy())
        rv = agent_mod.cmd_agent_remove(_args(agent_id="main", yes=True))
        assert rv == 1
        out = capsys.readouterr().out
        assert "Cannot remove" in out

    def test_warns_about_orphaned_routes(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg.copy())
        # Without --yes, should warn and refuse
        rv = agent_mod.cmd_agent_remove(_args(agent_id="coder", yes=False))
        assert rv == 1
        out = capsys.readouterr().out
        assert "Warning" in out
        assert "route(s) reference" in out
        assert "Use --yes" in out

    def test_missing_agent(self, fresh_home, monkeypatch, capsys, sample_cfg):
        monkeypatch.setattr(agent_mod, "load_config", lambda: sample_cfg.copy())
        rv = agent_mod.cmd_agent_remove(_args(agent_id="ghost", yes=True))
        assert rv == 1
        out = capsys.readouterr().out
        assert "not found" in out

    def test_removal_cleans_routes(self, fresh_home, monkeypatch, capsys, sample_cfg):
        saved = sample_cfg.copy()

        def fake_load():
            return saved.copy()

        def fake_save(cfg):
            nonlocal saved
            saved = cfg

        monkeypatch.setattr(agent_mod, "load_config", fake_load)
        monkeypatch.setattr(agent_mod, "save_config", fake_save)

        # coder has 2 routes; after removal they should be gone
        assert sum(1 for r in saved["routes"] if r.get("agent") == "coder") == 2
        rv = agent_mod.cmd_agent_remove(_args(agent_id="coder", yes=True))
        assert rv == 0
        assert not any(r.get("agent") == "coder" for r in saved["routes"])


# ---------------------------------------------------------------------------
# _ensure_agent_section
# ---------------------------------------------------------------------------

class TestEnsureAgentSection:
    def test_creates_missing_keys(self):
        cfg = {}
        result = agent_mod._ensure_agent_section(cfg)
        assert "agents" in result
        assert "routes" in result
        assert result["default_agent"] == "main"

    def test_preserves_existing_values(self):
        cfg = {"agents": {"x": {}}, "routes": [{"agent": "x"}], "default_agent": "x"}
        result = agent_mod._ensure_agent_section(cfg)
        assert result["default_agent"] == "x"
        assert "x" in result["agents"]


# ---------------------------------------------------------------------------
# _summarize_soul
# ---------------------------------------------------------------------------

class TestSummarizeSoul:
    def test_missing_file(self, tmp_path):
        assert agent_mod._summarize_soul(tmp_path / "nope.md") == "(no SOUL.md)"

    def test_extracts_content(self, tmp_path):
        p = tmp_path / "SOUL.md"
        p.write_text("# Title\n\nYou are a helpful assistant.\n\nMore text here.")
        summary = agent_mod._summarize_soul(p)
        assert "helpful assistant" in summary

    def test_skips_comments_and_empty(self, tmp_path):
        p = tmp_path / "SOUL.md"
        p.write_text("# Header\n  \n  \nReal content here.")
        summary = agent_mod._summarize_soul(p, max_lines=1)
        assert "Real content" in summary
        assert "#" not in summary
