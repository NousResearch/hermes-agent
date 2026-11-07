"""Tests: Kanban workers use a task workspace without booting Hermes from it.

Regression coverage for #34619 and #41312 (same root cause): ``_default_spawn``
launched the worker subprocess with ``cwd=workspace`` and set
``HERMES_KANBAN_WORKSPACE``, but did NOT set ``TERMINAL_CWD``. Because
``TERMINAL_CWD`` takes precedence over the process cwd in both
``tools/file_tools.py::_resolve_base_dir`` (relative ``write_file`` paths) and
``agent_init``'s context-file loader (``AGENTS.md`` discovery), workers inherited
the dispatching gateway's cwd — relative writes landed in the gateway user's
home (#41312) and the wrong profile's ``AGENTS.md`` was loaded (#34619).
Pinning ``TERMINAL_CWD`` to the workspace fixes both.  The process must keep
that cwd because ordinary worker code can read it, while Python must not put
the untrusted workspace first on ``sys.path`` during worker bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _make_task(kb, *, assignee: str = "w"):
    return kb.Task(
        id="t_cwd",
        title="cwd pin",
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
        current_run_id=1,
    )


def _capture_spawn_env(kb, monkeypatch, workspace: str) -> dict:
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kbd._default_spawn(_make_task(kb), workspace)
    return captured


def test_terminal_cwd_pinned_to_workspace(monkeypatch, tmp_path):
    """A real, absolute workspace dir is pinned as TERMINAL_CWD."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    # The subprocess cwd and TERMINAL_CWD must agree — both anchor the workspace.
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)


def test_worker_bootstrap_comes_from_active_install_when_workspace_is_hermes_checkout(
    monkeypatch, tmp_path
):
    """The worker imports the pinned install without changing cwd or child env.

    The checkout running this test has no PM environment committed under the
    isolated home.  Without Python safe-path mode its bootstrap wins over the
    install pinned on ``PYTHONPATH`` and exits with ``no dependency environment
    is committed`` before the worker can reach its owner.
    """
    root = tmp_path / ".hermes"
    root.mkdir()
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    workspace = Path(__file__).resolve().parents[2]
    active_root = tmp_path / "active-install"
    (active_root / "hermes_cli").mkdir(parents=True)
    (active_root / "hermes_bootstrap.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        kbd, "__file__", str(active_root / "hermes_cli" / "kanban_db_dispatch.py")
    )

    base_python = getattr(sys, "_base_executable", sys.executable)
    monkeypatch.setattr(sys, "executable", base_python)
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(Path(sys.base_prefix).resolve().parent))
    worker_argv = kbd._worker_argv
    probe = (
        "import hermes_bootstrap, json, os, sys; "
        "print(json.dumps({"
        "'bootstrap': hermes_bootstrap.__file__, 'cwd': os.getcwd(), "
        "'workspace': os.environ.get('HERMES_KANBAN_WORKSPACE'), "
        "'terminal_cwd': os.environ.get('TERMINAL_CWD'), "
        "'python_safe_path': os.environ.get('PYTHONSAFEPATH'), "
        "'safe_path': sys.flags.safe_path}))"
    )

    def probe_argv(task, profile_arg, hermes_home):
        command = worker_argv(task, profile_arg, hermes_home)
        return [*command[:-2], "-c", probe]

    monkeypatch.setattr(kbd, "_worker_argv", probe_argv)
    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", lambda task, command: command)

    spawned = []
    real_popen = subprocess.Popen

    def tracked_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", tracked_popen)
    task = _make_task(kb, assignee="default")

    kbd._default_spawn(task, str(workspace))
    assert len(spawned) == 1
    spawned[0].wait(timeout=30)
    output = (kb.worker_logs_dir() / f"{task.id}.log").read_text(
        encoding="utf-8", errors="replace"
    )

    assert spawned[0].returncode == 0, output
    payload = json.loads(output)
    assert Path(payload["bootstrap"]).resolve() == active_root / "hermes_bootstrap.py"
    assert payload["cwd"] == str(workspace)
    assert payload["workspace"] == str(workspace)
    assert payload["terminal_cwd"] == str(workspace)
    assert payload["python_safe_path"] is None
    assert payload["safe_path"] is True


