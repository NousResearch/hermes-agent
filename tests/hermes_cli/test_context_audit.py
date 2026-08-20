import json

from hermes_cli.context_audit import measure_context_budget, render_context_audit


def test_context_audit_reports_real_loaded_cwd_context_file(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    text = "x" * 21000
    (tmp_path / "AGENTS.md").write_text(text, encoding="utf-8")

    data = measure_context_budget(cwd=tmp_path, platform="cli")

    assert data["cwd"] == str(tmp_path.resolve())
    assert data["context_files"][0]["path"] == str((tmp_path / "AGENTS.md").resolve())
    assert data["context_files"][0]["over_default_limit"] is True

    rendered = render_context_audit(data)
    assert "Context audit" in rendered
    assert "AGENTS.md" in rendered
    assert "OVER-CAP" in rendered


def test_context_audit_payload_is_json_serializable(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "AGENTS.md").write_text("small context", encoding="utf-8")

    data = measure_context_budget(cwd=tmp_path, platform="cli")

    encoded = json.dumps(data, ensure_ascii=False)
    assert "small context" not in encoded
    assert str((tmp_path / "AGENTS.md").resolve()) in encoded


def test_context_audit_uses_top_level_context_file_cap(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("context_file_max_chars: 100\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "AGENTS.md").write_text("x" * 101, encoding="utf-8")

    data = measure_context_budget(cwd=tmp_path, platform="cli")

    assert data["context_file_char_limit"] == 100
    assert data["context_files"][0]["over_default_limit"] is True


def test_context_audit_reports_only_real_priority_context_source(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "repo"
    child = repo / "child"
    child.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / ".hermes.md").write_text("root hermes context", encoding="utf-8")
    (child / "AGENTS.md").write_text("child agents context", encoding="utf-8")

    data = measure_context_budget(cwd=child, platform="cli")

    paths = [item["path"] for item in data["context_files"]]
    assert paths == [str((repo / ".hermes.md").resolve())]
