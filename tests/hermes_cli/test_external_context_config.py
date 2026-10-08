"""External context paths survive the real CLI writer, loaders and dashboard schema."""

import pytest

from hermes_cli import config as cfg


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
    from hermes_cli import managed_scope
    managed_scope.invalidate_managed_cache()
    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return tmp_path


@pytest.mark.parametrize(("raw", "expected"), [
    ("~/.codex/AGENTS.md, ~/.claude/CLAUDE.md", ["~/.codex/AGENTS.md", "~/.claude/CLAUDE.md"]),
    ('["~/rules,team.md", "~/My Rules.md"]', ["~/rules,team.md", "~/My Rules.md"]),
    ('- "~/rules,team.md"\n- ~/other.md', ["~/rules,team.md", "~/other.md"]),
    ("off", ["off"]), ("123", ["123"]), ("null", ["null"]),
    ('["off", "123", "", "  "]', ["off", "123"]), ("[]", []),
])
def test_cli_round_trip_preserves_path_strings_and_siblings(home, raw, expected):
    cfg.set_config_value("context.engine", "custom")
    cfg.set_config_value("context.external_files", raw)
    assert cfg.read_raw_config()["context"] == {"engine": "custom", "external_files": expected}
    assert cfg.load_config_readonly()["context"]["external_files"] == expected


@pytest.mark.parametrize("raw", ['["unclosed', "[true, 123]", '{"path": "rules.md"}'])
def test_malformed_or_non_string_lists_leave_existing_config_unchanged(home, raw):
    cfg.set_config_value("context.external_files", "~/keep.md")
    before = (home / "config.yaml").read_bytes()
    with pytest.raises(SystemExit):
        cfg.set_config_value("context.external_files", raw)
    assert (home / "config.yaml").read_bytes() == before


def test_dashboard_marks_only_external_paths_as_line_list():
    from hermes_cli.web_server_config import CONFIG_SCHEMA
    assert CONFIG_SCHEMA["context.external_files"]["type"] == "list"
    assert CONFIG_SCHEMA["context.external_files"]["editor"] == "lines"
    assert CONFIG_SCHEMA["context.engine"]["type"] == "select"
    assert all(entry.get("editor") != "lines" for key, entry in CONFIG_SCHEMA.items()
               if key != "context.external_files" and entry["type"] == "list")
