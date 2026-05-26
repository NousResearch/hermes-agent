from pathlib import Path

from hermes_cli.attach_light import detect_attach_intent, render_attach_light_status


def test_detect_korean_attach_forms():
    assert detect_attach_intent("너는 DASHBOARD 세션이야").project == "DASHBOARD"
    assert detect_attach_intent("CHART_STUDY 붙어").project == "CHART_STUDY"
    assert detect_attach_intent("너는 CHARTAGI야").project == "CHARTAGI"


def test_non_attach_text_is_ignored():
    assert detect_attach_intent("DASHBOARD 상태 전체 점검해봐") is None
    assert detect_attach_intent("무슨 세션?") is None


def test_render_attach_status_uses_git_without_reading_project_docs(tmp_path: Path, monkeypatch):
    repo = tmp_path / "DASHBOARD"
    repo.mkdir()
    monkeypatch.chdir(repo)

    import subprocess
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("# dashboard\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)

    result = render_attach_light_status("너는 DASHBOARD 세션이야", config={"attach_light": {"project_roots": [str(tmp_path)]}}, cwd=tmp_path)

    assert result is not None
    assert "[attach-light]" in result.response
    assert "DASHBOARD" in result.response
    assert f"path: {repo}" in result.response
    assert "state: clean" in result.response
    assert "docs/skills not loaded" in result.response
