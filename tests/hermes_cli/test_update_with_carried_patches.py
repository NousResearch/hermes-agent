from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_cmd_update_with_carried_patches_falls_back_to_cmd_update(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *a, **kw: "https://github.com/example/fork.git")
    monkeypatch.setattr(hermes_main, "_is_fork", lambda *a, **kw: False)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(hermes_main, "_list_carried_patch_commits", lambda *a, **kw: [])

    called = {}
    monkeypatch.setattr(hermes_main, "cmd_update", lambda args: called.setdefault("args", args))

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "fetch"]:
            return _Result()
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return _Result(stdout="main\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    hermes_main.cmd_update_with_carried_patches(SimpleNamespace(on_conflict="stop", gateway=False))

    assert called["args"].gateway is False


def test_cmd_update_with_carried_patches_stop_mode_prints_suggestions(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_get_origin_url", lambda *a, **kw: "https://github.com/example/fork.git")
    monkeypatch.setattr(hermes_main, "_is_fork", lambda *a, **kw: False)
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", lambda *a, **kw: None)
    monkeypatch.setattr(
        hermes_main,
        "_list_carried_patch_commits",
        lambda *a, **kw: [{"sha": "abc123", "subject": "carry local tweak"}],
    )
    monkeypatch.setattr(hermes_main, "_get_conflicted_files", lambda *a, **kw: ["hermes_cli/main.py"])

    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[:2] == ["git", "fetch"]:
            return _Result()
        if cmd[:3] == ["git", "rev-parse", "--abbrev-ref"]:
            return _Result(stdout="main\n")
        if cmd[:2] == ["git", "branch"]:
            return _Result()
        if cmd[:3] == ["git", "reset", "--hard"]:
            return _Result()
        if cmd[:2] == ["git", "cherry-pick"]:
            return _Result(returncode=1, stderr="conflict\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        hermes_main.cmd_update_with_carried_patches(SimpleNamespace(on_conflict="stop", gateway=False))

    out = capsys.readouterr().out
    assert "Carried patch reapply hit a conflict" in out
    assert "--on-conflict=plan" in out
    assert "--on-conflict=llm" in out


def test_resume_carried_patch_conflict_plan_mode_uses_llm(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "_get_conflicted_files", lambda *a, **kw: ["hermes_cli/main.py"])
    monkeypatch.setattr(hermes_main, "_llm_conflict_plan", lambda *a, **kw: "1. Keep upstream import order.\n2. Reapply the local branch switch.")

    state = {
        "patches": [{"sha": "abc123", "subject": "carry local tweak"}],
        "current_index": 0,
        "backup_branch": "backup/update-carry-test",
    }

    resumed = hermes_main._resume_carried_patch_conflict(["git"], tmp_path, state, "plan")

    assert resumed is False
    out = capsys.readouterr().out
    assert "LLM conflict plan" in out
    assert "Keep upstream import order" in out


def test_resume_carried_patch_conflict_llm_mode_continues(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(hermes_main, "_get_conflicted_files", lambda *a, **kw: ["hermes_cli/main.py"])
    monkeypatch.setattr(hermes_main, "_llm_resolve_conflicts", lambda *a, **kw: (True, "merged cleanly"))

    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        if cmd[:2] == ["git", "add"]:
            return _Result()
        if cmd[:3] == ["git", "cherry-pick", "--continue"]:
            return _Result()
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    state = {
        "patches": [{"sha": "abc123", "subject": "carry local tweak"}],
        "current_index": 0,
        "backup_branch": "backup/update-carry-test",
    }

    resumed = hermes_main._resume_carried_patch_conflict(["git"], tmp_path, state, "llm")

    assert resumed is True
    assert state["current_index"] == 1
    saved = hermes_main._load_carried_patch_state(tmp_path)
    assert saved["current_index"] == 1
    assert ["git", "add", "--", "hermes_cli/main.py"] in recorded
    assert ["git", "cherry-pick", "--continue"] in recorded
