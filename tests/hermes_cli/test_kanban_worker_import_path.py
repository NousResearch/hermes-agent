"""Kanban workers must import this checkout under a bundled interpreter.

The dispatcher launches ``sys.executable -m hermes_cli.main`` after the shared
sanitizer strips Hermes-owned PYTHONPATH entries. The gateway launcher inserts
the checkout on the parent ``sys.path`` and pops PYTHONPATH, and PYTHONSAFEPATH
does not put the task workspace on ``sys.path``. A bundled interpreter then
exits with ``ModuleNotFoundError: No module named 'hermes_cli'`` before
``hermes_cli.main`` can bootstrap itself. The pin is applied to the sanitized
env only — it must not resurrect stripped profile secrets.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _capture_spawn(monkeypatch, tmp_path, *, assignee: str = "default") -> dict:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    captured: dict = {}

    class _Proc:
        pid = 5151

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return _Proc()

    monkeypatch.delenv("HERMES_BIN", raising=False)
    real_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(kbd, "_retag_legacy_worker_sessions", lambda _root: None)
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    task = kb.Task(
        id="t_import_path",
        title="import path",
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
    )
    try:
        kbd._default_spawn(task, str(workspace))
    finally:
        # subprocess.run() uses Popen; leave the real constructor for the
        # interpreter probe that follows in the same test.
        monkeypatch.setattr(subprocess, "Popen", real_popen)
    assert captured, "spawn did not reach Popen"
    return captured


def _find_main(interpreter: str, pythonpath: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Resolve ``hermes_cli.main`` with site disabled so an editable install cannot cheat."""
    return subprocess.run(
        [interpreter, "-S", "-c",
         "import importlib.util\n"
         "spec = importlib.util.find_spec('hermes_cli.main')\n"
         "print(spec.origin if spec else '', end='')\n"],
        cwd=cwd,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": pythonpath,
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_worker_env_resolves_hermes_cli_main_under_pythonsafepath(monkeypatch, tmp_path):
    """The selected interpreter resolves this tree only because the worker env pins it."""
    user_libs = tmp_path / "user-libs"
    user_libs.mkdir()
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(user_libs))
    monkeypatch.setenv("HERMES_HOME", str(home))

    captured = _capture_spawn(monkeypatch, tmp_path)
    repo = _repo_root()
    entries = [entry for entry in captured["env"].get("PYTHONPATH", "").split(os.pathsep) if entry]
    assert entries[0] == str(repo)
    assert str(user_libs) in entries
    assert entries.count(str(repo)) == 1

    cmd = captured["cmd"]
    assert cmd[1:3] == ["-m", "hermes_cli.main"]
    interpreter = cmd[0]
    empty = tmp_path / "empty-cwd"
    empty.mkdir()

    pinned = _find_main(interpreter, captured["env"]["PYTHONPATH"], empty)
    assert pinned.returncode == 0, pinned.stderr
    assert Path(pinned.stdout).resolve() == (repo / "hermes_cli" / "main.py").resolve()

    bare = _find_main(interpreter, str(user_libs), empty)
    assert bare.returncode != 0
    assert "No module named 'hermes_cli'" in (bare.stderr or "")


def test_routed_worker_pin_does_not_restore_dispatcher_secrets(monkeypatch, tmp_path):
    """Re-pinning the checkout must not undo the sanitizer for another profile."""
    launch = tmp_path / "fakehome" / ".hermes"
    served = launch / "profiles" / "b"
    served.mkdir(parents=True)
    (served / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    (launch / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    user_libs = tmp_path / "user-libs"
    user_libs.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("OPENAI_API_KEY", "dispatcher-launch-key")
    monkeypatch.setenv("PYTHONPATH", str(user_libs))

    captured = _capture_spawn(monkeypatch, tmp_path, assignee="b")
    env = captured["env"]
    assert "OPENAI_API_KEY" not in env
    assert "dispatcher-launch-key" not in env.get("PYTHONPATH", "")
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(_repo_root())
