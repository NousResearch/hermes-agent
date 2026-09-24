"""Integration tests for the bundled skill-graph plugin's gating and profile scope."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_PATH = Path(__file__).parents[2] / "plugins" / "skill-graph" / "__init__.py"


def _import_skill_graph():
    spec = importlib.util.spec_from_file_location("test_skill_graph_plugin", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Context:
    def __init__(self) -> None:
        self.tools = {}
        self.commands = {}
        self.hooks = {}
        self.prompt_sections = {}

    def register_tool(self, *, name, **kwargs):
        self.tools[name] = kwargs

    def register_command(self, *, name, **kwargs):
        self.commands[name] = kwargs

    def register_hook(self, name, callback):
        self.hooks[name] = callback

    def register_system_prompt_section(self, section_id, content, **kwargs):
        self.prompt_sections[section_id] = (content, kwargs)


def _registered_plugin(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True):
    module = _import_skill_graph()
    monkeypatch.setattr(module, "_skill_graph_mode_enabled", lambda: enabled)
    context = _Context()
    module.register(context)
    return module, context


class TestSkillGraphGating:
    def test_blocks_only_search_files_before_discovery(self, monkeypatch):
        _, context = _registered_plugin(monkeypatch)
        hook = context.hooks["pre_tool_call"]

        blocked = hook("search_files", turn_id="t1")
        assert blocked and blocked["action"] == "block"
        assert hook("read_file", turn_id="t1") is None
        assert hook("session_search", turn_id="t1") is None
        assert hook("terminal", turn_id="t1") is None

    @pytest.mark.parametrize("discovery_tool", ["skill_graph_search", "skill_load"])
    def test_search_or_preinjected_skill_load_unlocks_current_turn(
        self, monkeypatch, discovery_tool
    ):
        _, context = _registered_plugin(monkeypatch)
        hook = context.hooks["pre_tool_call"]

        assert hook(discovery_tool, turn_id="t1") is None
        assert hook("search_files", turn_id="t1") is None
        blocked = hook("search_files", turn_id="t2")
        assert blocked and blocked["action"] == "block"

    def test_disabled_mode_never_blocks(self, monkeypatch):
        _, context = _registered_plugin(monkeypatch, enabled=False)
        assert context.hooks["pre_tool_call"]("search_files", turn_id="t1") is None

    def test_registration_exposes_cache_safe_prompt_section(self, monkeypatch):
        _, context = _registered_plugin(monkeypatch)

        render, options = context.prompt_sections["skill-graph.discovery"]
        assert callable(render)
        assert options == {"position": "after_memory", "max_chars": 8000}

        search_schema = context.tools["skill_graph_search"]["schema"]["parameters"]
        assert "query" not in search_schema.get("required", [])

    def test_search_handler_returns_successful_results(self, monkeypatch):
        module = _import_skill_graph()

        class _CountCursor:
            @staticmethod
            def fetchone():
                return (1,)

        class _Conn:
            @staticmethod
            def execute(_sql, _params=()):
                return _CountCursor()

        result = [{
            "name": "example",
            "category": "test",
            "description": "Example skill",
            "tags": [],
            "file_path": "/skills/example/SKILL.md",
            "relevance": "direct",
            "relationship_chain": [],
            "score": 0.9,
        }]
        monkeypatch.setattr(module, "_ensure_graph", lambda: _Conn())
        monkeypatch.setattr(module, "_search_graph", lambda _query, _conn, limit: result[:limit])

        payload = json.loads(module._handle_skill_graph_search({"query": "example"}))

        assert payload["success"] is True
        assert payload["result_count"] == 1
        assert payload["results"][0]["name"] == "example"

    def test_prompt_requires_mode_and_both_graph_tools(self, monkeypatch):
        module = _import_skill_graph()
        monkeypatch.setattr(module, "_find_skill_path", lambda _name: None)
        monkeypatch.setattr(module, "_gateway_extension_skills", lambda: [])
        assert module._render_skill_graph_prompt({
            "skill_graph_mode": True,
            "valid_tool_names": ["skill_graph_search"],
        }) == ""
        rendered = module._render_skill_graph_prompt(
            {
                "skill_graph_mode": True,
                "valid_tool_names": ["skill_graph_search", "skill_load"],
            }
        )
        assert "## Skill Discovery Protocol" in rendered
        assert "Available Skills\n- skill-graph" in rendered

        assert module._render_skill_graph_prompt(
            {
                "skill_graph_mode": False,
                "valid_tool_names": ["skill_graph_search", "skill_load"],
            }
        ) == ""

    def test_flat_index_is_suppressed_only_when_both_graph_tools_exist(self, monkeypatch):
        from agent import system_prompt

        monkeypatch.setattr(
            system_prompt._pb,
            "build_skills_system_prompt",
            lambda **_kwargs: "native flat index",
        )
        monkeypatch.setattr("model_tools.get_toolset_for_tool", lambda _name: None)
        agent = SimpleNamespace(
            _skill_graph_mode=True,
            valid_tool_names={"skills_list", "skill_graph_search", "skill_load"},
            platform="cli",
        )
        assert system_prompt._skills_prompt(agent) == ""

        agent.valid_tool_names.remove("skill_load")
        assert system_prompt._skills_prompt(agent) == "native flat index"

    def test_skill_manage_batch_refreshes_every_affected_skill(self, monkeypatch):
        module, context = _registered_plugin(monkeypatch)
        updated = []

        class _Conn:
            def __init__(self):
                self.deleted = []
                self.commits = 0

            def execute(self, _sql, params):
                self.deleted.append(params[0])

            def commit(self):
                self.commits += 1

        conn = _Conn()
        monkeypatch.setattr(module, "_ensure_graph", lambda: conn)
        monkeypatch.setattr(
            module,
            "_update_single_skill",
            lambda _conn, name: updated.append(name) or True,
        )
        context.hooks["post_tool_call"](
            tool_name="skill_manage",
            args={"operations": [
                {"action": "patch", "name": "one"},
                {"action": "write_file", "name": "two"},
                {"action": "delete", "name": "three"},
            ]},
            result=json.dumps({"success": True}),
        )

        assert updated == ["one", "two"]
        assert conn.deleted == ["three"]
        assert conn.commits == 1


class TestSkillGraphProfileIsolation:
    def test_db_path_uses_active_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_BUNDLED_PLUGINS", raising=False)
        module = _import_skill_graph()

        assert module._db_path() == tmp_path / "personal" / "skill-graph.db"

    def test_bundled_db_path_uses_profile_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", "/bundled")
        module = _import_skill_graph()

        assert module._db_path() == tmp_path / "skill-graph.db"

    def test_scan_does_not_include_default_profile_when_another_profile_is_active(
        self, tmp_path, monkeypatch
    ):
        active_home = tmp_path / "profiles" / "work"
        active_skill = active_home / "skills" / "active"
        active_skill.mkdir(parents=True)
        (active_skill / "SKILL.md").write_text("---\nname: active\n---\n", encoding="utf-8")
        default_skills = tmp_path / "skills"
        default_skills.mkdir()

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(active_home))
        module = _import_skill_graph()
        monkeypatch.setattr(module, "get_bundled_skills_dir", lambda _fallback: tmp_path / "bundled")
        monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", lambda: [])

        dirs = module._find_all_skills_dirs()
        assert active_home / "skills" in dirs
        assert default_skills not in dirs

    def test_runtime_source_dir_is_ephemeral_and_rebuilds(self, tmp_path, monkeypatch):
        module = _import_skill_graph()
        source = tmp_path / "extra"
        source.mkdir()
        monkeypatch.setattr(module, "_ensure_graph", lambda: SimpleNamespace())
        monkeypatch.setattr(module, "_full_rebuild", lambda _conn: 7)

        result = json.loads(module._handle_skill_graph_config(
            {"action": "add_dir", "path": str(source), "persist": False}
        ))
        assert result["success"] is True
        assert result["skills_indexed"] == 7
        assert source.resolve() in module._RUNTIME_SOURCE_DIRS

        removed = json.loads(module._handle_skill_graph_config(
            {"action": "remove_dir", "path": str(source), "persist": False}
        ))
        assert removed["success"] is True
        assert source.resolve() not in module._RUNTIME_SOURCE_DIRS

    def test_graph_db_does_not_leak_across_profiles(self, tmp_path, monkeypatch):
        home_a = tmp_path / "profile-a"
        skill_a = home_a / "skills" / "skill-a"
        skill_a.mkdir(parents=True)
        (skill_a / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: Only in profile A\n---\n# Skill A\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(home_a))
        module = _import_skill_graph()
        conn_a = module._get_conn()
        module._init_db(conn_a)
        module._full_rebuild(conn_a)
        conn_a.close()

        home_b = tmp_path / "profile-b"
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        conn_b = module._get_conn()
        module._init_db(conn_b)
        count = conn_b.execute(
            "SELECT COUNT(*) FROM skill_nodes WHERE name = 'skill-a'"
        ).fetchone()[0]
        conn_b.close()
        assert count == 0


def test_normalizes_non_string_yaml_tag_metadata(tmp_path):
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: scalar-metadata\nmetadata:\n  hermes:\n"
        "    tags: [hardware, 0402]\n---\n# Scalar metadata\n",
        encoding="utf-8",
    )

    assert _import_skill_graph()._parse_skill_md(skill_md)["tags"] == ["hardware", "258"]
