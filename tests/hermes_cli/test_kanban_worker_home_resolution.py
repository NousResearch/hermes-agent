"""Worker spawn must resolve its HERMES_HOME deliberately, never by inheritance.

The dispatcher is a detached process: its ambient ``HERMES_HOME`` is whatever
the process that started it exported. The observed failure (erp-runtime
exit-126 incident, t_05e2e685) was a card workspace home, later pruned, that
workers kept inheriting. These tests pin the spawn-time resolution introduced
with the fix: the assignee profile under a live configured root wins, a
genuine custom root keeps working, and an inherited dead workspace home is
never what a worker boots against.
"""

from __future__ import annotations

import subprocess


def _make_task(kb, *, assignee: str):
    return kb.Task(
        id="t_worker_home",
        title="worker home",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=9,
    )


def _capture_spawn(monkeypatch, tmp_path):
    """Patch Popen and the argv resolver; return the captured spawn env."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from tools.environments import local as local_env

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")
    # The PATH prepend resolves the running machine's shim via ``which``; from
    # an isolated fixture that is real-home I/O the guard (rightly) refuses.
    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)

    captured = {}

    class FakeProc:
        pid = 7777

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return captured


def _spawn(tmp_path, captured, monkeypatch, *, assignee: str):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    pid = kbd._default_spawn(_make_task(kb, assignee=assignee), str(workspace))
    assert pid == 7777
    return captured["env"]


def test_spawn_resolves_assignee_home_despite_inherited_workspace_home(monkeypatch, tmp_path):
    """A dead workspace home in the dispatcher's env must not poison the spawn."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "elias"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    poisoned = root / "kanban" / "boards" / "hermes" / "workspaces" / "t_dead" / "home" / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(poisoned))
    monkeypatch.setenv("HOME", str(tmp_path))

    captured = _capture_spawn(monkeypatch, tmp_path)
    env = _spawn(tmp_path, captured, monkeypatch, assignee="elias")

    assert env["HERMES_HOME"] == str(profile)
    assert env["HERMES_HOME"] != str(poisoned)
    # Home-derived values must follow the resolved home too.
    assert not str(env.get("TMPDIR", "")).startswith(str(poisoned))
    assert not str(env.get("HERMES_REAL_HOME", "")).startswith(str(poisoned))


def test_spawn_never_passes_through_inherited_workspace_home(monkeypatch, tmp_path):
    """Even with no resolvable profile, the inherited dead home is not passed through."""
    root = tmp_path / ".hermes"
    root.mkdir(parents=True)
    poisoned = root / "kanban" / "workspaces" / "t_dead" / "home" / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(poisoned))
    monkeypatch.setenv("HOME", str(tmp_path))

    captured = _capture_spawn(monkeypatch, tmp_path)
    env = _spawn(tmp_path, captured, monkeypatch, assignee="ghost")

    assert env["HERMES_HOME"] == str(root)  # platform default, not the dead home
    assert env["HERMES_HOME"] != str(poisoned)


def test_spawn_keeps_custom_root_profile_home(monkeypatch, tmp_path):
    """A genuine custom root (Docker-style) keeps resolving its profiles."""
    custom = tmp_path / "opt" / "data"
    profile = custom / "profiles" / "elias"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(custom))
    monkeypatch.setenv("HOME", str(tmp_path / "not-a-hermes-home"))

    captured = _capture_spawn(monkeypatch, tmp_path)
    env = _spawn(tmp_path, captured, monkeypatch, assignee="elias")

    assert env["HERMES_HOME"] == str(profile)


def test_spawn_default_assignee_folds_workspace_home_back_to_root(monkeypatch, tmp_path):
    """Assigning the `default` profile must not adopt an inherited workspace home."""
    root = tmp_path / ".hermes"
    root.mkdir(parents=True)
    poisoned = root / "kanban" / "boards" / "hermes" / "workspaces" / "t_dead" / "home" / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(poisoned))
    monkeypatch.setenv("HOME", str(tmp_path))

    captured = _capture_spawn(monkeypatch, tmp_path)
    env = _spawn(tmp_path, captured, monkeypatch, assignee="default")

    assert env["HERMES_HOME"] == str(root)
    assert env["HERMES_HOME"] != str(poisoned)
