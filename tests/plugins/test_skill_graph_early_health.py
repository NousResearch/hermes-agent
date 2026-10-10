"""Focused BOM and privacy-safe fallback tests for the bundled skill graph."""
from __future__ import annotations

import importlib.util
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_PATH = Path(__file__).parents[2] / "plugins" / "skill-graph" / "__init__.py"


def _import_skill_graph():
    spec = importlib.util.spec_from_file_location("test_skill_graph_early_health_plugin", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_skill_metadata_and_load_accept_bom(tmp_path, monkeypatch, encoding):
    module = _import_skill_graph()
    skill_md = tmp_path / "skill-bom" / "SKILL.md"
    skill_md.parent.mkdir()
    skill_md.write_text("---\nname: skill-bom\ndescription: BOM skill\n---\n# Body\n", encoding=encoding)
    monkeypatch.setattr(module, "_find_skill_path", lambda _name: skill_md)
    monkeypatch.setattr(module, "_ensure_graph", lambda: SimpleNamespace(
        execute=lambda *_args: None, commit=lambda: None,
    ))

    assert module._parse_skill_md(skill_md)["name"] == "skill-bom"
    payload = json.loads(module._handle_skill_load({"name": "skill-bom"}))
    assert payload["success"] is True
    assert payload["content"].startswith("---\n")
    assert payload["description"] == "BOM skill"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_gateway_extensions_accept_bom(tmp_path, monkeypatch, encoding):
    module = _import_skill_graph()
    path = tmp_path / "extensions.md"
    path.write_text("## Pre-installed Gateways\n| `example` | Example skill |\n", encoding=encoding)
    monkeypatch.setattr(module, "_skill_graph_config", lambda: {"extensions_file": str(path)})
    assert module._gateway_extension_skills() == [("example", "Example skill")]


def test_profile_config_failure_falls_back_without_logging_secret(monkeypatch, caplog):
    module = _import_skill_graph()

    def fail():
        raise ValueError("secret-config-value")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module._skill_graph_config() == {}
    assert "secret-config-value" not in caplog.text
    assert "stack:" in caplog.text
    assert "fail:" in caplog.text


def test_bad_db_override_uses_default_without_logging_secret(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_BUNDLED_PLUGINS", raising=False)
    monkeypatch.setattr(module, "_skill_graph_config", lambda: {"db_path": "secret-config-value"})

    def fail(_value):
        raise ValueError("secret-config-value")

    monkeypatch.setattr(module, "_resolve_config_path", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module._db_path() == tmp_path / "personal" / "skill-graph.db"
    assert "secret-config-value" not in caplog.text
    assert "fail:" in caplog.text


def test_source_dir_failure_keeps_prior_directories(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()
    good = tmp_path / "good"
    good.mkdir()
    monkeypatch.setattr(module, "_skill_graph_config", lambda: {
        "source_dirs": [str(good), "secret-config-value"],
    })

    def resolve(value):
        if value == "secret-config-value":
            raise ValueError("secret-config-value")
        return Path(value)

    monkeypatch.setattr(module, "_resolve_config_path", resolve)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module._read_source_dirs_from_config() == [good]
    assert "secret-config-value" not in caplog.text
    assert "resolve:" in caplog.text


def test_external_skill_dir_failure_keeps_profile_dirs(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setattr(module, "_read_source_dirs_from_config", lambda: [])
    monkeypatch.setattr(module, "get_bundled_skills_dir", lambda _fallback: tmp_path / "bundled")

    def fail():
        raise RuntimeError("secret-config-value")

    monkeypatch.setattr("agent.skill_utils.get_external_skills_dirs", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module._find_all_skills_dirs() == [skills]
    assert "secret-config-value" not in caplog.text
    assert "fail:" in caplog.text


def test_skill_parse_failure_returns_metadata_without_logging_secret(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()

    def fail(_path, **_kwargs):
        raise OSError("secret-config-value")

    monkeypatch.setattr(Path, "read_text", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        info = module._parse_skill_md(tmp_path / "skill-bad" / "SKILL.md")
    assert info["name"] == "skill-bad"
    assert info["description"] == ""
    assert "secret-config-value" not in caplog.text
    assert "fail:" in caplog.text


@pytest.mark.parametrize("failed_sql, message", [
    ("INSERT INTO skill_term_stats", "search stat update failed"),
    ("SELECT term, load_count", "search stat boost failed"),
])
def test_search_retains_results_when_optional_stats_fail(failed_sql, message, caplog):
    module = _import_skill_graph()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    module._init_db(db)
    module._migrate_db(db)
    db.execute("INSERT INTO skill_nodes (name, tags) VALUES (?, ?)", ("orchid", '["orchid"]'))
    db.execute("INSERT INTO skill_fts (name, tags) VALUES (?, ?)", ("orchid", "orchid"))
    db.execute(
        "INSERT INTO skill_terms (term, skill_name) VALUES (?, ?)", ("orchid", "orchid"),
    )
    db.commit()

    class FailingStatsConnection:
        commits = 0

        def execute(self, sql, params=()):
            if failed_sql in sql:
                raise sqlite3.OperationalError("secret-config-value")
            return db.execute(sql, params)

        def commit(self):
            self.commits += 1
            db.commit()

    conn = FailingStatsConnection()
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        results = module._search_graph("orchid", conn)
    assert results[0]["name"] == "orchid"
    assert results[0]["relevance"] == "tag_match"
    assert conn.commits == 1
    assert message in caplog.text
    assert "secret-config-value" not in caplog.text
    assert "execute:" in caplog.text
    db.close()


def test_search_updates_stats_when_available():
    module = _import_skill_graph()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    module._init_db(db)
    module._migrate_db(db)
    db.execute("INSERT INTO skill_nodes (name, tags) VALUES (?, ?)", ("orchid", '["orchid"]'))
    db.execute("INSERT INTO skill_fts (name, tags) VALUES (?, ?)", ("orchid", "orchid"))
    db.execute(
        "INSERT INTO skill_terms (term, skill_name) VALUES (?, ?)", ("orchid", "orchid"),
    )
    db.commit()
    assert module._search_graph("orchid", db)[0]["name"] == "orchid"
    assert db.execute(
        "SELECT search_count FROM skill_term_stats WHERE skill_name = ? AND term = ?",
        ("orchid", "orchid"),
    ).fetchone()[0] == 1
    db.close()


@pytest.mark.parametrize("boundary, expected", [
    ("show", "Config failed: could not display graph configuration"),
    ("slash", "Config add failed: could not change source directory"),
    ("tool", {"success": False, "error": "Could not update graph configuration"}),
])
def test_config_boundaries_redact_failure_and_log_location(monkeypatch, caplog, boundary, expected):
    module = _import_skill_graph()

    def fail(*_args, **_kwargs):
        raise RuntimeError("secret-config-value")

    if boundary == "show":
        monkeypatch.setattr(module, "_ensure_graph", fail)
        call = module._show_graph_config
    elif boundary == "slash":
        monkeypatch.setattr(module, "_change_source_dir", fail)
        call = lambda: module._handle_source_dir_config("add", "secret-source-path")
    else:
        monkeypatch.setattr(module, "_change_source_dir", fail)
        call = lambda: json.loads(module._handle_skill_graph_config({
            "action": "add_dir", "path": "secret-source-path",
        }))

    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert call() == expected
    assert len(caplog.records) == 1
    assert "fail:" in caplog.text
    assert "secret-config-value" not in caplog.text
    assert "secret-source-path" not in caplog.text


def test_config_display_and_source_dir_success_contract(tmp_path, monkeypatch):
    module = _import_skill_graph()
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("# Example", encoding="utf-8")
    db_path = tmp_path / "graph.db"
    db_path.write_bytes(b"db")
    monkeypatch.setattr(module, "_ensure_graph", lambda: SimpleNamespace(
        execute=lambda *_args: SimpleNamespace(fetchone=lambda: (2,)),
    ))
    monkeypatch.setattr(module, "_db_path", lambda: db_path)
    monkeypatch.setattr(module, "_find_all_skills_dirs", lambda: [source])
    monkeypatch.setattr(module, "_read_source_dirs_from_config", lambda: [source])
    monkeypatch.setattr(module, "_change_source_dir", lambda *_args, **_kw: (source, 2, "already present"))

    display = module._show_graph_config()
    assert "Skill Graph configuration" in display
    assert "Skills:      2" in display
    assert f"    {source}  (1 SKILL.md)" in display
    assert f"Source dirs (config): [{source!r}]" in display
    assert module._handle_source_dir_config("add", str(source)) == (
        f"✅ added {source} (already present)\n   Graph rebuilt: 2 skills indexed."
    )
    assert json.loads(module._handle_skill_graph_config({
        "action": "add_dir", "path": str(source), "persist": False,
    })) == {"success": True, "action": "add_dir", "path": str(source),
           "skills_indexed": 2, "persisted": False, "note": "already present"}
    assert json.loads(module._handle_skill_graph_config({"action": "list_dirs"}))["source_dirs"] == [
        str(source),
    ]


def test_invalid_source_path_does_not_leak_in_errors_or_logs(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()
    private_path = tmp_path / "secret-source-path"
    monkeypatch.setattr(module, "_resolve_config_path", lambda _path: private_path)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        slash = module._handle_source_dir_config("add", str(private_path))
        tool = json.loads(module._handle_skill_graph_config({
            "action": "add_dir", "path": str(private_path),
        }))
    assert slash == "Config add failed: could not change source directory"
    assert tool == {"success": False, "error": "Could not update graph configuration"}
    assert len(caplog.records) == 2
    assert "secret-source-path" not in caplog.text + slash + json.dumps(tool)


@pytest.mark.parametrize("formatter", ["_format_edges", "_format_terms"])
def test_graph_formatters_log_query_failure_without_secret(monkeypatch, caplog, formatter):
    module = _import_skill_graph()

    def fail(_sql, _params):
        raise sqlite3.OperationalError("secret-config-value")

    monkeypatch.setattr(module, "_ensure_graph", lambda: SimpleNamespace(execute=fail))
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert getattr(module, formatter)("orchid") == ""
    assert len(caplog.records) == 1
    assert "fail:" in caplog.text
    assert "secret-config-value" not in caplog.text


def test_edge_and_term_formatting_with_real_db(monkeypatch, caplog):
    module = _import_skill_graph()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    module._init_db(db)
    module._migrate_db(db)
    db.execute("INSERT INTO skill_edges VALUES (?, ?, ?, ?)",
               ("orchid", "fern", "depends_on", '{"reason": "for shade"}'))
    db.execute("INSERT INTO skill_edges VALUES (?, ?, ?, ?)",
               ("fern", "orchid", "similar_to", "unparsed edge detail"))
    db.execute("INSERT INTO skill_terms (term, skill_name, source) VALUES (?, ?, ?)",
               ("garden", "orchid", "tag"))
    db.execute("INSERT INTO skill_terms (term, skill_name, source) VALUES (?, ?, ?)",
               ("orchid", "fern", "description"))
    # Bind this in-memory graph to both renderers without touching the profile DB.
    monkeypatch.setattr(module, "_ensure_graph", lambda: db)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        edges = module._format_edges("orchid")
        terms = module._format_terms("orchid")
    assert "orchid ──(depends_on)──> fern" in edges
    assert "for shade" in edges
    assert "fern ──(similar_to)──> orchid" in edges
    assert "unparsed edge detail" in edges
    assert "invalid edge properties" in caplog.text
    assert "orchid ──(tag)──> garden" in terms
    assert "fern" in terms and "──(description)──> orchid" in terms
    assert "s=0/l=0/ok=0/b=-0.000" in terms
    assert module._format_edges("missing") == "No relations defined for: missing"
    assert module._format_terms("missing") == ""
    db.close()


def test_slash_discovery_routes_aliases_and_preserves_outputs(tmp_path, monkeypatch):
    module = _import_skill_graph()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    module._init_db(db)
    module._migrate_db(db)
    db.execute(
        "INSERT INTO skill_nodes (name, category, description) VALUES (?, ?, ?)",
        ("orchid", "garden", "A flowering skill"),
    )
    db.commit()
    monkeypatch.setattr(module, "_ensure_graph", lambda: db)
    monkeypatch.setattr(module, "_find_all_skills_dirs", lambda: [])
    monkeypatch.setattr(module, "_db_path", lambda: tmp_path / "graph.db")
    calls = []

    def search(query, conn, limit):
        calls.append((query, conn, limit))
        return [{"name": "orchid", "description": "A flowering skill", "score": 0.75,
                 "relevance": "direct", "relationship_chain": ["term -> orchid"]}]

    monkeypatch.setattr(module, "_search_graph", search)
    assert module._handle_slash_command("status") == module._handle_slash_command("stats")
    assert "Skills:  1" in module._handle_slash_command("status")
    assert "orchid" in module._handle_slash_command("list")
    assert "orchid" in module._handle_slash_command("search garden")
    assert "orchid" in module._handle_slash_command("score garden")
    assert module._handle_slash_command("score garden") == module._handle_slash_command("explain garden")
    assert calls == [("garden", db, 15), ("garden", db, 8),
                     ("garden", db, 8), ("garden", db, 8)]
    assert "Subcommands:" in module._handle_slash_command("search")
    assert module._handle_slash_command("score") == "Usage: /skill-graph score <query>"
    db.close()


@pytest.mark.parametrize("subcmd, expected", [
    ("status", "Status check failed: could not read graph status"),
    ("list", "List failed: could not read graph results"),
    ("search garden", "Search failed: could not search skills"),
    ("score garden", "Score breakdown failed: could not read graph results"),
])
def test_slash_discovery_failure_redacts_private_path(monkeypatch, caplog, subcmd, expected):
    module = _import_skill_graph()

    def fail():
        raise sqlite3.OperationalError("/private/secret-source-path/graph.db")

    monkeypatch.setattr(module, "_ensure_graph", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module._handle_slash_command(subcmd) == expected
    assert len(caplog.records) == 1
    assert "fail:" in caplog.text
    assert "secret-source-path" not in caplog.text


def test_slash_remaining_commands_preserve_success_and_missing(tmp_path, monkeypatch):
    module = _import_skill_graph()
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    module._init_db(db)
    module._migrate_db(db)
    db.execute(
        "INSERT INTO skill_nodes (name, category, description, file_path) VALUES (?, ?, ?, ?)",
        ("orchid", "garden", "A flowering skill", "orchid/SKILL.md"),
    )
    db.commit()
    md = tmp_path / "orchid" / "SKILL.md"
    md.parent.mkdir()
    md.write_text("---\nname: orchid\ndescription: A flowering skill\n---\n# Orchid", encoding="utf-8")
    monkeypatch.setattr(module, "_find_skill_path", lambda name: md if name == "orchid" else None)
    monkeypatch.setattr(module, "_ensure_graph", lambda: db)
    monkeypatch.setattr(module, "_full_rebuild", lambda _conn: 1)

    assert module._handle_slash_command("rebuild") == "Skill graph rebuilt: 1 skills indexed."
    assert "# Orchid" in module._handle_slash_command("show orchid")
    assert "Category:    garden" in module._handle_slash_command("info orchid")
    assert module._handle_slash_command("terms orchid") == ""
    assert "Loaded skill: orchid" in module._handle_slash_command("orchid")
    assert module._handle_slash_command("show") == "Usage: /skill-graph show <skill-name>"
    assert module._handle_slash_command("info") == "Usage: /skill-graph info <skill-name>"
    assert module._handle_slash_command("terms") == "Usage: /skill-graph terms <skill-name>"
    assert module._handle_slash_command("show missing") == "Not found: missing"
    assert module._handle_slash_command("info missing") == "Not found: missing  (try /sg list)"
    assert "Subcommands:" in module._handle_slash_command("missing")
    db.close()


@pytest.mark.parametrize("subcmd, patched, expected", [
    ("rebuild", "_full_rebuild", "Rebuild failed: could not rebuild skill graph"),
    ("show orchid", "_handle_skill_load", "Show failed: could not load skill"),
    ("info orchid", "_ensure_graph", "Info failed: could not read skill metadata"),
    ("terms orchid", "_format_terms", "Terms failed: could not read skill terms"),
    ("orchid", "_ensure_graph", None),
])
def test_slash_remaining_failures_redact_and_log(monkeypatch, caplog, subcmd, patched, expected):
    module = _import_skill_graph()

    def fail(*_args):
        raise sqlite3.OperationalError("/private/secret-source-path/graph.db")

    monkeypatch.setattr(module, "_ensure_graph", lambda: SimpleNamespace())
    monkeypatch.setattr(module, patched, fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        result = module._handle_slash_command(subcmd)
    if expected is None:
        assert "Subcommands:" in result
    else:
        assert result == expected
    assert len(caplog.records) == 1
    assert "fail:" in caplog.text
    assert "secret-source-path" not in caplog.text + result


def test_skill_load_optional_stats_failure_preserves_content(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()
    md = tmp_path / "orchid" / "SKILL.md"
    md.parent.mkdir()
    md.write_text("---\nname: orchid\ndescription: A flowering skill\n---\n# Orchid", encoding="utf-8")
    monkeypatch.setattr(module, "_find_skill_path", lambda _name: md)

    def fail(*_args):
        raise sqlite3.OperationalError("secret-config-value")

    monkeypatch.setattr(module, "_ensure_graph", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        result = json.loads(module._handle_skill_load({"name": "orchid"}))
    assert result["success"] is True
    assert "# Orchid" in result["content"]
    assert len(caplog.records) == 1
    assert "load stats update failed" in caplog.text
    assert "secret-config-value" not in caplog.text


def test_skill_load_read_failure_returns_redacted_json(tmp_path, monkeypatch, caplog):
    module = _import_skill_graph()
    md = tmp_path / "orchid" / "SKILL.md"
    monkeypatch.setattr(module, "_find_skill_path", lambda _name: md)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert json.loads(module._handle_skill_load({"name": "orchid"})) == {
            "success": False, "error": "Could not load skill",
        }
    assert "orchid" not in caplog.text
    assert "read_text:" in caplog.text


def test_mode_read_failure_is_logged_without_config_value(monkeypatch, caplog):
    module = _import_skill_graph()

    def fail():
        raise ValueError("secret-config-value")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module._skill_graph_mode_enabled() is False
    assert "fail:" in caplog.text
    assert "secret-config-value" not in caplog.text


def test_post_tool_success_stat_failure_resets_tracking(monkeypatch, caplog):
    module = _import_skill_graph()
    hooks = {}
    commands = {}

    class Context:
        def register_tool(self, **_kwargs):
            pass

        def register_command(self, name, handler, **_kwargs):
            commands[name] = handler

        def register_system_prompt_section(self, *_args, **_kwargs):
            pass

        def register_hook(self, name, handler):
            hooks[name] = handler

    monkeypatch.setattr(module, "_skill_graph_mode_enabled", lambda: False)
    module.register(Context())
    assert commands["sg"] is commands["skill-graph"] is module._handle_slash_command
    calls = []

    def fail():
        calls.append(1)
        raise sqlite3.OperationalError("secret-config-value")

    monkeypatch.setattr(module, "_get_conn", fail)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        hooks["post_tool_call"](tool_name="skill_load", args={"name": "orchid"})
        hooks["post_tool_call"](tool_name="skill_load", args={"name": "quality-gate"})
        hooks["post_tool_call"](tool_name="skill_load", args={"name": "quality-gate"})
    assert calls == [1]
    assert "success stat update failed" in caplog.text
    assert "secret-config-value" not in caplog.text
