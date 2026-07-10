from __future__ import annotations

import subprocess


def _make_task(kb, *, assignee: str = "w"):
    return kb.Task(
        id="t_gh_auth_env",
        title="github auth env",
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


def _capture_spawn_env(kb, monkeypatch, workspace: str) -> dict[str, str]:
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict[str, dict[str, str]] = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb), workspace)
    return captured["env"]


def test_default_spawn_sets_gh_config_dir_from_real_home(monkeypatch, tmp_path):
    """Workers should see host GitHub CLI auth even when profile HOME is isolated.

    In profile/worker sessions HOME can point at ``{HERMES_HOME}/home``. GitHub
    CLI then misses the host's ``~/.config/gh`` unless the dispatcher pins
    GH_CONFIG_DIR into the worker env.
    """
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile_home = profile / "home"
    profile_home.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")

    real_home = tmp_path / "host-home"
    gh_config_dir = real_home / ".config" / "gh"
    gh_config_dir.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
    monkeypatch.setenv("HOME", str(profile_home))
    monkeypatch.delenv("GH_CONFIG_DIR", raising=False)

    from hermes_cli import kanban_db as kb

    env = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert env["HERMES_HOME"] == str(profile)
    assert env["GH_CONFIG_DIR"] == str(gh_config_dir)


def test_default_spawn_preserves_explicit_gh_config_dir(monkeypatch, tmp_path):
    """An operator-supplied GH_CONFIG_DIR is the source of truth."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    (profile / "home").mkdir(parents=True)
    profile.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")

    explicit = tmp_path / "custom-gh-config"
    explicit.mkdir()
    real_home = tmp_path / "host-home"
    (real_home / ".config" / "gh").mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
    monkeypatch.setenv("HOME", str(profile / "home"))
    monkeypatch.setenv("GH_CONFIG_DIR", str(explicit))

    from hermes_cli import kanban_db as kb

    env = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert env["GH_CONFIG_DIR"] == str(explicit)
