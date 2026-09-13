from __future__ import annotations

from pathlib import Path

from agent import agent_os_install as install


def test_apply_rule_preserves_existing_content_and_is_idempotent(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir()
    home.mkdir()
    (repo / "AGENT_OS_CORE.md").write_text(
        "# Agent OS Core\n\nAGENT_OS_RULE_VERSION: TEST-V1\n\nrule body\n",
        encoding="utf-8",
    )
    target = home / ".codex" / "AGENTS.md"
    target.parent.mkdir(parents=True)
    target.write_text("existing user rule\n", encoding="utf-8")

    original_which = install.shutil.which
    install.shutil.which = lambda name: f"/fake/{name}" if name == "codex" else None
    try:
        result = install.apply_rule("codex", repo_root=repo, home=home)
    finally:
        install.shutil.which = original_which

    content = target.read_text(encoding="utf-8")
    assert result["rule_loaded"] is True
    assert result["existing_user_content_preserved"] is True
    assert result["idempotent_second_apply"] is True
    assert "existing user rule" in content
    assert install.MANAGED_BEGIN in content
    assert content.count(install.MANAGED_BEGIN) == 1


def test_audit_targets_classifies_local_states(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    (repo / "AGENTS.md").write_text("repo\n", encoding="utf-8")
    (repo / "SOUL.md").write_text("soul\n", encoding="utf-8")
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text("cfg\n", encoding="utf-8")
    (home / ".claude").mkdir()
    (home / ".cursor").mkdir()
    (home / ".cursor" / "cli-config.json").write_text("{}", encoding="utf-8")

    results = install.audit_targets(
        repo_root=repo,
        home=home,
        command_paths={
            "codex": "/fake/codex",
            "claude": None,
            "cursor": "/fake/cursor",
            "gemini": None,
            "hermes": "/fake/hermes",
        },
        process_snapshot=[{"name": "Hermes.exe", "command_line": "Hermes.exe"}],
    )
    by_target = {item["target"]: item for item in results}

    assert by_target["codex"]["status"] == "SAFE_TO_INSTALL"
    assert by_target["claude"]["status"] == "NOT_INSTALLED"
    assert by_target["cursor"]["status"] == "UNSUPPORTED"
    assert by_target["gemini"]["status"] == "NOT_INSTALLED"
    assert by_target["hermes"]["status"] == "REQUIRES_RESTART"


def test_readback_rule_reports_version(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text(
        f"{install.MANAGED_BEGIN}\nAGENT_OS_RULE_VERSION: TEST-V2\nbody\n{install.MANAGED_END}\n",
        encoding="utf-8",
    )

    readback = install.readback_rule(path)

    assert readback["rule_loaded"] is True
    assert readback["rule_version"] == "TEST-V2"
