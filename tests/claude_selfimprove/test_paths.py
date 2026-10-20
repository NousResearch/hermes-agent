from __future__ import annotations

from claude_selfimprove import paths


def test_claude_home_defaults_under_home(sandbox):
    assert paths.claude_home() == sandbox.home / ".claude"
    assert paths.claude_hermes_home() == sandbox.home / ".claude-hermes"


def test_claude_home_override_env_var(sandbox, monkeypatch):
    custom = sandbox.home / "elsewhere" / ".claude"
    monkeypatch.setenv("CLAUDE_SELFIMPROVE_CLAUDE_DIR", str(custom))
    assert paths.claude_home() == custom


def test_hermes_home_defaults_and_override(sandbox, monkeypatch):
    assert paths.hermes_home() == sandbox.hermes_home
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert paths.hermes_home() == sandbox.home / ".hermes"


def test_source_roots_only_returns_existing_dirs(sandbox):
    assert paths.source_roots() == []

    (sandbox.claude_home / "projects").mkdir(parents=True)
    roots = paths.source_roots()
    assert len(roots) == 1
    assert roots[0].name == "claude"
    assert roots[0].projects_dir == sandbox.claude_home / "projects"

    (sandbox.claude_hermes_home / "projects").mkdir(parents=True)
    roots = paths.source_roots()
    assert {r.name for r in roots} == {"claude", "claude-hermes"}


def test_state_paths_are_scoped_to_hermes_home(sandbox):
    assert paths.state_dir() == sandbox.hermes_home / "state" / "claude-selfimprove"
    assert paths.candidates_db_path().parent == paths.state_dir()
    assert paths.checkpoints_path().parent == paths.state_dir()
    assert paths.audit_log_path().parent == paths.state_dir()
    assert paths.backups_dir() == paths.state_dir() / "backups"


def test_self_improvement_queue_path_matches_existing_watcher_convention(sandbox):
    expected = (
        sandbox.hermes_home / "state" / "self-improvement-notify" / "queue.jsonl"
    )
    assert paths.self_improvement_queue_path() == expected


def test_ensure_state_dirs_creates_directories(sandbox):
    assert not paths.state_dir().exists()
    paths.ensure_state_dirs()
    assert paths.state_dir().is_dir()
    assert paths.backups_dir().is_dir()


def test_target_paths(sandbox):
    assert paths.claude_md_path() == sandbox.claude_home / "CLAUDE.md"
    assert paths.rules_dir() == sandbox.claude_home / "rules"
    assert paths.skills_dir() == sandbox.claude_home / "skills"
