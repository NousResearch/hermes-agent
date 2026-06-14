from __future__ import annotations

from hermes_cli import codex_cockpit as cc


def test_load_cockpit_config_defaults_to_home_code_allowlist():
    cfg = cc.load_cockpit_config({})

    assert cfg.enabled is True
    assert cfg.driver == "codex_app_server"
    assert cfg.default_model == "gpt-5.5"
    assert cfg.branch_prefix == "codex/"
    assert cfg.repo_allowlist
    assert cfg.context_helper["enabled"] is False
    assert cfg.context_helper["harvest_launches"] is True
    assert cfg.context_helper["auto_promote_memory"] is True
    assert cfg.context_helper["auto_promote_skills"] is True


def test_load_cockpit_config_normalizes_user_values(tmp_path):
    allowed = tmp_path / "repos"
    cfg = cc.load_cockpit_config(
        {
            "codex_cockpit": {
                "enabled": "off",
                "driver": "codex_exec",
                "default_model": "gpt-test",
                "default_worktree_root": str(tmp_path / "worktrees"),
                "branch_prefix": "ai",
                "repo_allowlist": [str(allowed)],
                "readout": {"include_git_status": False},
                "context_helper": {"enabled": True, "auto_promote_memory": True},
            }
        }
    )

    assert cfg.enabled is False
    assert cfg.driver == "codex_exec"
    assert cfg.default_model == "gpt-test"
    assert cfg.branch_prefix == "ai"
    assert cfg.repo_allowlist == (str(allowed.resolve()),)
    assert cfg.readout["include_git_status"] is False
    assert cfg.context_helper["enabled"] is True
    assert cfg.context_helper["auto_promote_memory"] is True


def test_prepare_launch_builds_safe_worktree_command(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    worktrees = tmp_path / "worktrees"
    monkeypatch.setattr(
        cc,
        "resolve_git_root",
        lambda path: (str(repo.resolve()), None),
    )

    plan, error = cc.prepare_launch(
        'launch repo "Fix auth bug"',
        {
            "codex_cockpit": {
                "repo_allowlist": [str(tmp_path)],
                "default_worktree_root": str(worktrees),
                "default_model": "gpt-test",
                "branch_prefix": "codex/",
            }
        },
        cwd=str(tmp_path),
        now=1_800_000_000,
    )

    assert error is None
    assert plan is not None
    assert plan.repo_root == str(repo.resolve())
    assert plan.branch.startswith("codex/fix-auth-bug-")
    assert str(worktrees.resolve()) in plan.worktree_path
    assert "git -C" in plan.command
    assert "worktree add" in plan.command
    assert "codex exec" in plan.command
    assert "--model gpt-test" in plan.command
    assert "--sandbox workspace-write" in plan.command


def test_prepare_launch_rejects_repo_outside_allowlist(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(
        cc,
        "resolve_git_root",
        lambda path: (str(repo.resolve()), None),
    )

    plan, error = cc.prepare_launch(
        "launch repo fix-things",
        {"codex_cockpit": {"repo_allowlist": [str(allowed)]}},
        cwd=str(tmp_path),
    )

    assert plan is None
    assert error is not None
    assert "outside `codex_cockpit.repo_allowlist`" in error


def test_render_last_includes_codex_ids():
    class Agent:
        _last_codex_thread_id = "thread-1"
        _last_codex_turn_id = "turn-2"

    rendered = cc.render_last(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "done"},
        ],
        active_agent=Agent(),
    )

    assert "thread-1" in rendered
    assert "turn-2" in rendered
    assert "done" in rendered


def test_render_checks_filters_validation_processes():
    rendered = cc.render_checks(
        [
            {"session_id": "p1", "command": "sleep 1", "status": "running"},
            {
                "session_id": "p2",
                "command": "npm run build",
                "status": "exited",
                "exit_code": 0,
                "output_preview": "built",
            },
        ]
    )

    assert "p2" in rendered
    assert "built" in rendered
    assert "sleep 1" not in rendered


def test_render_status_includes_recent_tool_events(monkeypatch):
    monkeypatch.setattr(cc, "_codex_binary_status", lambda: (True, "test"))
    monkeypatch.setattr(cc, "_codex_auth_line", lambda: "logged in (test)")
    monkeypatch.setattr(cc, "_pending_context_counts", lambda: (0, 0))

    rendered = cc.render_status(
        {"codex_cockpit": {"readout": {"include_git_status": False, "max_events": 2}}},
        transcript=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "read_file"}},
                    {"function": {"name": "terminal"}},
                ],
            },
            {"role": "tool", "tool_name": "write_file", "content": "ok"},
        ],
    )

    assert "- Recent tools: `write_file`, `terminal`" in rendered
    assert "read_file" not in rendered


def test_render_status_includes_pending_learning(monkeypatch):
    monkeypatch.setattr(cc, "_codex_binary_status", lambda: (True, "test"))
    monkeypatch.setattr(cc, "_codex_auth_line", lambda: "logged in (test)")
    monkeypatch.setattr(cc, "_pending_context_counts", lambda: (0, 0))
    monkeypatch.setattr(cc, "_pending_learning_count", lambda: 2)

    rendered = cc.render_status(
        {"codex_cockpit": {"readout": {"include_git_status": False}}},
        transcript=[],
    )

    assert "- Pending Codex learning: 2" in rendered


def test_render_learn_dispatches_status(monkeypatch):
    from hermes_cli import codex_learning

    monkeypatch.setattr(codex_learning, "render_learn_status", lambda _cfg: "learn status")

    assert cc.render_learn(("status",), {}) == "learn status"
