"""Public prompt and /context must agree on configured instruction files."""
from pathlib import Path


def test_configured_file_matches_context_manifest(tmp_path, monkeypatch):
    from agent.prompt_builder import build_context_files_prompt
    from agent.context_file_sources import list_context_file_sources
    home = tmp_path / "profile"
    home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    external = tmp_path / "shared.md"
    external.write_text("Unique external instruction marker", encoding="utf-8")
    (home / "config.yaml").write_text(
        f"context:\n  engine: compressor\n  external_files:\n    - {external}\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    prompt = build_context_files_prompt(cwd=str(cwd), skip_soul=True)
    assert "Unique external instruction marker" in prompt
    matches = [entry for entry in list_context_file_sources(cwd=str(cwd), skip_soul=True)
               if Path(entry["path"]).resolve() == external.resolve()]
    assert len(matches) == 1
    assert matches[0]["loaded"] is True


def test_ignored_hermes_file_matches_context_manifest(tmp_path, monkeypatch):
    from agent.prompt_builder import build_context_files_prompt
    from agent.context_file_sources import list_context_file_sources
    home = tmp_path / "profile"
    home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    (home / "config.yaml").write_text("context:\n  ignore_hermes_md: true\n", encoding="utf-8")
    (cwd / "HERMES.md").write_text("Old hermes instruction marker", encoding="utf-8")
    (cwd / "AGENTS.md").write_text("Selected agents instruction marker", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    prompt = build_context_files_prompt(cwd=str(cwd), skip_soul=True)
    assert "Selected agents instruction marker" in prompt
    assert "Old hermes instruction marker" not in prompt
    sources = list_context_file_sources(cwd=str(cwd), skip_soul=True)
    agents = [entry for entry in sources if Path(entry["path"]).name == "AGENTS.md"]
    assert len(agents) == 1
    assert agents[0]["loaded"] is True
    assert all(not entry["loaded"] for entry in sources if Path(entry["path"]).name == "HERMES.md")


def test_external_manifest_order_statuses_and_no_truncation_warning(tmp_path, monkeypatch):
    from agent.context_file_sources import list_context_file_sources
    from agent.prompt_builder import build_context_files_prompt, drain_truncation_warnings
    from hermes_cli.config import atomic_config_replace

    launch = tmp_path / "launch"
    launch.mkdir()
    owner = tmp_path / "owner"
    owner.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    cursor = project / ".cursorrules"
    cursor.write_text("External cursor contract.")
    rule = project / ".cursor" / "rules" / "shared.mdc"
    rule.parent.mkdir(parents=True)
    rule.write_text("External rule contract.")
    alias = owner / "alias.md"
    alias.symlink_to(cursor)
    blocked = owner / "blocked.md"
    blocked.write_text("<!-- system_prompt -->")
    large = owner / "large.md"
    large.write_text("Large start.\n" + "x" * 4000 + "\nLarge end.")
    empty = owner / "empty.md"
    empty.write_text("")
    denied = owner / "denied.md"
    denied.write_text("Unreadable contract.")
    (project / "CLAUDE.md").write_text("Automatic contract.")
    (owner / "SOUL.md").write_text("Owner identity.")
    atomic_config_replace(owner / "config.yaml", {"context": {"external_files": [
        {"path": str(alias), "label": "Shared cursor"}, str(rule), str(cursor),
        str(blocked), str(large), str(empty), str(denied),
    ]}})
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("agent.prompt_builder._get_context_file_max_chars", lambda *_: 1000)
    read_text = Path.read_text

    def read_with_denial(path, *args, **kwargs):
        if path == denied:
            raise PermissionError("test denied")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_with_denial)
    drain_truncation_warnings()
    try:
        sources = list_context_file_sources(cwd=str(project), home_override=owner)
        assert drain_truncation_warnings() == []
        assert [entry["path"] for entry in sources] == [
            str(cursor), str(rule), str(blocked), str(large), str(empty), str(denied),
            str(project / "CLAUDE.md"), str(owner / "SOUL.md"),
        ]
        assert sources[0]["label"] == "Shared cursor"
        assert [entry["status"] for entry in sources] == [
            "loaded", "loaded", "blocked", "truncated", "empty", "unreadable", "loaded", "loaded",
        ]
        assert [entry["loaded"] for entry in sources] == [True, True, False, True, False, False, True, True]
        prompt = build_context_files_prompt(cwd=str(project), home_override=owner)
        assert prompt.count("External cursor contract.") == prompt.count("External rule contract.") == 1
        assert prompt.index("External cursor contract.") < prompt.index("External rule contract.")
        assert "[BLOCKED:" in prompt and "<!-- system_prompt -->" not in prompt
        assert "[...truncated" in prompt and "Large start." in prompt and "Large end." in prompt
        assert "Unreadable contract." not in prompt
        assert "Owner identity." in prompt and "Automatic contract." in prompt
    finally:
        drain_truncation_warnings()


def test_safe_worker_manifest_is_empty_like_prompt(tmp_path, monkeypatch):
    from agent.context_file_sources import list_context_file_sources
    from agent.prompt_builder import build_context_files_prompt

    (tmp_path / "AGENTS.md").write_text("Project contract.")
    monkeypatch.setattr("agent.safe_worker_policy.safe_worker_enabled", lambda: True)
    assert build_context_files_prompt(cwd=str(tmp_path), skip_soul=True) == ""
    assert list_context_file_sources(cwd=str(tmp_path), skip_soul=True) == []
