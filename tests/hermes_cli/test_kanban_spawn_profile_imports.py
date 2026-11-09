"""The assignee child pins its actual runtime import root and profile."""
from hermes_cli import kanban_db as kb


def test_default_spawn_pins_repo_root_on_module_worker_pythonpath(tmp_path, monkeypatch):
    """The dispatcher's worker must carry the import context of the install that spawned it.

    The worker is always ``sys.executable -m hermes_cli.kanban_worker_client`` (it only submits
    the claim to the profile owner), and the worker env scrub strips Hermes-owned PYTHONPATH
    entries, so a bare module child died on import and the board auto-blocked (#122299,
    #122487, #122500). The spawned env must put the running install's root first on
    PYTHONPATH. Main's carve-out for a resolved ``hermes`` shim path does not apply here: the
    worker never runs the resolved CLI argv, so a ``$HERMES_BIN``/PATH shim can neither replace
    the module child nor drop the pin.
    """
    import os
    import sys
    from pathlib import Path

    from hermes_cli import kanban_db_dispatch as kbd

    root = str(Path(kbd.__file__).resolve().parents[1])
    home = tmp_path / ".hermes"
    home.mkdir()
    coder_home = home / "profiles" / "coder"
    coder_home.mkdir(parents=True)
    (coder_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("HERMES_BIN", raising=False)

    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            self.pid = 4242

    monkeypatch.setattr("subprocess.Popen", _FakePopen)

    task = kb.Task(
        id="t_import_root", title="x", body=None, assignee="coder", status="ready",
        priority=0, created_by=None, created_at=0, started_at=None, completed_at=None,
        workspace_kind="worktree", workspace_path=str(tmp_path / "ws"), claim_lock=None,
        claim_expires=None, tenant=None, branch_name=None,
    )

    for resolved in ([sys.executable, "-m", "hermes_cli.main"], ["/opt/hermes/bin/hermes"]):
        monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda resolved=resolved: resolved)
        kbd._default_spawn(task, str(tmp_path / "ws"))
        assert captured["cmd"][-3:] == [sys.executable, "-m", "hermes_cli.kanban_worker_client"]
        assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == root
