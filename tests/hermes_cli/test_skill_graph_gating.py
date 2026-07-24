"""Tests for skill-graph pre_tool_call gating hook."""
import pytest
import sys
from unittest.mock import patch, MagicMock


# ── Helper: recreate the gating closure logic ────────────────────────────

def _make_gating_hook(mock_config_skill_graph_mode: bool):
    """Build a pre_tool_call hook with the same logic as the skill-graph plugin."""
    from hermes_cli.config import load_config

    _gated_tools = frozenset({"find", "read_file"})
    _graph_searched_turn: dict[str, bool] = {}

    # Resolve from config at registration time (startup constant)
    _graph_mode = False
    try:
        _graph_mode = load_config().get("agent", {}).get("skill_graph_mode", False)
    except Exception:
        pass

    def _on_pre_tool_call(tool_name: str, args: dict | None = None, **kw):
        nonlocal _graph_searched_turn, _graph_mode
        turn_id = kw.get("turn_id", "")
        if not turn_id:
            return None

        if turn_id not in _graph_searched_turn:
            _graph_searched_turn.clear()
            _graph_searched_turn[turn_id] = False

        if tool_name == "skill_graph_search":
            _graph_searched_turn[turn_id] = True
            return None

        if (
            _graph_mode
            and tool_name in _gated_tools
            and not _graph_searched_turn.get(turn_id, False)
        ):
            return {"action": "block", "message": "blocked"}

        return None

    return _on_pre_tool_call, _graph_searched_turn


# ── Tests ─────────────────────────────────────────────────────────────────

class TestSkillGraphGating:
    """Test pre_tool_call gating when agent.skill_graph_mode is enabled."""

    def test_blocks_gated_tools_before_search(self):
        """read_file is blocked until skill_graph_search is called."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        result = hook("read_file", turn_id="t1")
        assert result is not None
        assert result["action"] == "block"

    def test_allows_gated_tools_after_search(self):
        """After skill_graph_search, gated tools are allowed."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        hook("skill_graph_search", turn_id="t1")
        result = hook("read_file", turn_id="t1")
        assert result is None

    def test_allows_non_gated_tools_before_search(self):
        """Tools not in the gated set are always allowed."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        result = hook("terminal", turn_id="t1")
        assert result is None

    def test_session_search_not_gated(self):
        """session_search is not in the gated set — always allowed."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        result = hook("session_search", turn_id="t1")
        assert result is None

    def test_skill_graph_search_itself_always_allowed(self):
        """skill_graph_search is never blocked."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        result = hook("skill_graph_search", turn_id="t1")
        assert result is None

    def test_turn_boundary_resets_flag(self):
        """New turn_id resets the searched flag."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        hook("skill_graph_search", turn_id="t1")
        assert hook("read_file", turn_id="t1") is None

        # New turn → should be blocked again
        result = hook("read_file", turn_id="t2")
        assert result is not None
        assert result["action"] == "block"

    def test_no_turn_id_returns_none(self):
        """Without turn_id, hook is a no-op."""
        mock_cfg = {"agent": {"skill_graph_mode": True}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(True)

        result = hook("read_file")
        assert result is None


class TestSkillGraphGatingDisabled:
    """Test pre_tool_call when agent.skill_graph_mode is disabled or absent."""

    def test_no_blocking_when_mode_false(self):
        """When skill_graph_mode is False, nothing is blocked."""
        mock_cfg = {"agent": {"skill_graph_mode": False}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(False)

        assert hook("read_file", turn_id="t1") is None
        assert hook("session_search", turn_id="t1") is None

    def test_no_blocking_when_mode_missing(self):
        """When skill_graph_mode is not in config, nothing is blocked."""
        mock_cfg = {"agent": {}}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(False)

        assert hook("read_file", turn_id="t1") is None

    def test_no_blocking_when_agent_section_missing(self):
        """When agent section is absent, nothing is blocked."""
        mock_cfg = {}
        with patch("hermes_cli.config.load_config", return_value=mock_cfg):
            hook, _ = _make_gating_hook(False)

        assert hook("read_file", turn_id="t1") is None

    def test_no_blocking_when_config_fails(self):
        """When config loading raises, nothing is blocked (graceful degradation)."""
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            hook, _ = _make_gating_hook(False)

        assert hook("read_file", turn_id="t1") is None


class TestSkillGraphProfileIsolation:
    """Test skill-graph respects HERMES_HOME for hermetic profile isolation."""

    @staticmethod
    def _import_skill_graph():
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "skill_graph",
            Path(__file__).parent.parent.parent / "plugins" / "skill-graph" / "__init__.py",
            submodule_search_locations=[
                str(Path(__file__).parent.parent.parent / "plugins" / "skill-graph")
            ],
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_db_path_uses_hermes_home(self, tmp_path, monkeypatch):
        """_db_path() resolves under the current HERMES_HOME, not global ~/.hermes."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_BUNDLED_PLUGINS", raising=False)

        mod = self._import_skill_graph()
        db = mod._db_path()
        assert str(tmp_path) in str(db)
        assert "personal" in str(db)

    def test_db_path_with_bundled_plugins(self, tmp_path, monkeypatch):
        """_db_path() goes to root when HERMES_BUNDLED_PLUGINS is set."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", "/some/path")

        mod = self._import_skill_graph()
        db = mod._db_path()
        assert str(tmp_path) in str(db)
        assert "personal" not in str(db)

    def test_find_skills_dirs_includes_hermes_home(self, tmp_path, monkeypatch):
        """_find_all_skills_dirs() includes the profile's own skills/ dir."""
        profile_skills = tmp_path / "skills" / "test-skill"
        profile_skills.mkdir(parents=True)
        (profile_skills / "SKILL.md").write_text("---\nname: test-skill\n---\n# Test\n")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        mod = self._import_skill_graph()
        dirs = mod._find_all_skills_dirs()
        dir_strings = [str(d) for d in dirs]
        assert any(str(tmp_path) in s for s in dir_strings), \
            f"HERMES_HOME skills/ not scanned: {dir_strings}"

    def test_graph_db_does_not_leak_across_profiles(self, tmp_path, monkeypatch):
        """Skills indexed under one HERMES_HOME don't appear under another."""
        home_a = tmp_path / "profile_a"
        skill_a = home_a / "skills" / "skill-a"
        skill_a.mkdir(parents=True)
        (skill_a / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Only in profile A\n---\n# Skill A\n"
        )

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        mod = self._import_skill_graph()
        conn_a = mod._get_conn()
        mod._init_db(conn_a)
        mod._full_rebuild(conn_a)
        conn_a.commit()
        conn_a.close()

        home_b = tmp_path / "profile_b"
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        conn_b = mod._get_conn()
        mod._init_db(conn_b)
        row = conn_b.execute(
            "SELECT COUNT(*) FROM skill_nodes WHERE name = 'skill-a'"
        ).fetchone()
        conn_b.close()
        assert row[0] == 0, "skill-a leaked from profile A to profile B"
