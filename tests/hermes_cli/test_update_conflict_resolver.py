from pathlib import Path
from types import SimpleNamespace

from hermes_cli import main as hermes_main
from hermes_cli.update_conflict_resolver import (
    UpdateConflictResolverConfig,
    _resolver_command,
    _resolver_env,
    load_update_conflict_resolver_config,
    run_patched_main_conflict_resolver,
)


def test_conflict_resolver_config_reads_update_section_and_env(monkeypatch):
    cfg = load_update_conflict_resolver_config(
        {
            "update": {
                "auto_resolve_conflicts": True,
                "auto_resolve_timeout": 1200,
                "auto_resolve_model": "from-config",
                "auto_resolve_provider": "openai-codex",
                "auto_resolve_reasoning_effort": "medium",
                "auto_resolve_push": False,
                "auto_resolve_max_turns": 123,
            }
        }
    )
    assert cfg == UpdateConflictResolverConfig(
        enabled=True,
        timeout_seconds=1200,
        model="from-config",
        provider="openai-codex",
        reasoning_effort="medium",
        push=False,
        max_turns=123,
    )

    monkeypatch.setenv("HERMES_UPDATE_AUTO_RESOLVE_MODEL", "gpt-5.5")
    monkeypatch.setenv("HERMES_UPDATE_AUTO_RESOLVE_REASONING_EFFORT", "high")
    cfg = load_update_conflict_resolver_config({"update": {"auto_resolve_conflicts": True}})
    assert cfg.model == "gpt-5.5"
    assert cfg.reasoning_effort == "high"


def test_resolver_command_uses_configured_model_provider_and_reasoning_env(monkeypatch):
    cfg = UpdateConflictResolverConfig(
        enabled=True,
        model="gpt-5.5",
        provider="openai-codex",
        reasoning_effort="high",
        max_turns=777,
    )
    cmd = _resolver_command("fix the merge", cfg)

    assert "--provider" in cmd
    assert cmd[cmd.index("--provider") + 1] == "openai-codex"
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "gpt-5.5"
    assert "--max-turns" in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "777"
    assert "systematic-debugging" not in cmd

    env = _resolver_env(cfg)
    assert env["HERMES_REASONING_EFFORT"] == "high"
    assert env["HERMES_UPDATE_RESOLVER"] == "1"


def test_run_conflict_resolver_uses_temp_worktree_pushes_and_fast_forwards(monkeypatch, tmp_path):
    calls = []
    worktree_path = tmp_path / "resolver-worktree"
    worktree_path.mkdir()

    monkeypatch.setattr("hermes_cli.update_conflict_resolver._worktree_root", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.update_conflict_resolver._make_branch_name", lambda: "resolve-branch")
    monkeypatch.setattr("hermes_cli.update_conflict_resolver.tempfile.mkdtemp", lambda **kw: str(worktree_path))
    monkeypatch.setattr("hermes_cli.update_conflict_resolver.shutil.rmtree", lambda *a, **kw: None)

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        cwd_value = kwargs.get("cwd")
        cwd = Path(cwd_value) if cwd_value is not None else Path()
        if cmd[1:] == ["merge", "--abort"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:4] == ["worktree", "add", "-b"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cwd == worktree_path and cmd[1:] == ["merge", "--no-edit", "upstream/main"]:
            return SimpleNamespace(stdout="", stderr="conflict\n", returncode=1)
        if cwd == worktree_path and cmd[1:] == ["diff", "--name-only", "--diff-filter=U"]:
            count = sum(
                1
                for seen_cmd, seen_kwargs in calls
                if Path(seen_kwargs.get("cwd")) == worktree_path
                and seen_cmd[1:] == ["diff", "--name-only", "--diff-filter=U"]
            )
            return SimpleNamespace(stdout=("hermes_cli/main.py\n" if count == 1 else ""), stderr="", returncode=0)
        if cmd[0].endswith("python") or cmd[0] == "python":
            assert kwargs["timeout"] == 1800
            assert Path(kwargs["cwd"]) == tmp_path / "repo"
            assert kwargs["env"]["HERMES_REASONING_EFFORT"] == "high"
            assert kwargs["env"]["HERMES_UPDATE_RESOLVER_WORKTREE"] == str(worktree_path)
            return SimpleNamespace(stdout="done\n", stderr="", returncode=0)
        if cwd == worktree_path and cmd[1:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cwd == worktree_path and cmd[1:] == ["merge-base", "--is-ancestor", "upstream/main", "HEAD"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cwd == worktree_path and cmd[1:] == ["push", "origin", "HEAD:patched-main"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:] == ["fetch", "origin", "--quiet"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:] == ["merge", "--ff-only", "origin/patched-main"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:3] == ["worktree", "remove"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd[1:3] == ["branch", "-D"]:
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        raise AssertionError(f"unexpected command: {cmd} cwd={cwd}")

    monkeypatch.setattr("hermes_cli.update_conflict_resolver.subprocess.run", fake_run)

    ok = run_patched_main_conflict_resolver(
        ["git"],
        tmp_path / "repo",
        merge_stderr="conflict",
        config={
            "update": {
                "auto_resolve_conflicts": True,
                "auto_resolve_model": "gpt-5.5",
                "auto_resolve_provider": "openai-codex",
                "auto_resolve_reasoning_effort": "high",
                "auto_resolve_push": True,
            }
        },
    )

    assert ok is True
    commands = [cmd for cmd, _ in calls]
    assert ["git", "worktree", "add", "-b", "resolve-branch", str(worktree_path), "patched-main"] in commands
    assert ["git", "push", "origin", "HEAD:patched-main"] in commands
    assert ["git", "merge", "--ff-only", "origin/patched-main"] in commands


def test_sync_patched_main_invokes_auto_resolver_after_merge_conflict(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "remote", "get-url", "upstream"]:
            return SimpleNamespace(stdout="https://github.com/NousResearch/hermes-agent.git\n", stderr="", returncode=0)
        if cmd in (["git", "fetch", "upstream", "--quiet"], ["git", "fetch", "origin", "--quiet"]):
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        if cmd == ["git", "rev-list", "--count", "upstream/main..origin/main"]:
            return SimpleNamespace(stdout="0\n", stderr="", returncode=0)
        if cmd == ["git", "rev-list", "--count", "origin/main..upstream/main"]:
            return SimpleNamespace(stdout="0\n", stderr="", returncode=0)
        if cmd == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="before\n", stderr="", returncode=0)
        if cmd == ["git", "merge", "--no-edit", "upstream/main"]:
            return SimpleNamespace(stdout="", stderr="CONFLICT\n", returncode=1)
        raise AssertionError(f"unexpected command: {cmd}")

    resolver_calls = []

    def fake_resolver(git_cmd, cwd, *, merge_stderr="", config=None, stream=None):
        resolver_calls.append((git_cmd, cwd, merge_stderr))
        return True

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "hermes_cli.update_conflict_resolver.run_patched_main_conflict_resolver",
        fake_resolver,
    )

    assert hermes_main._sync_patched_main_with_upstream(["git"], tmp_path) is True
    assert resolver_calls == [(["git"], tmp_path, "CONFLICT\n")]
