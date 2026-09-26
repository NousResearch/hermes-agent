"""Kanban worker spawn on the PM store Python (#122620).

The store interpreter carries no application packages: ``hermes_cli`` resolves
in the dispatcher's process only via the launcher prelude's ``sys.path`` entry,
which children do not inherit. A bare ``sys.executable -m hermes_cli.main``
child therefore died with ``ModuleNotFoundError`` before any Hermes code ran,
while the parent-side ``find_spec("hermes_cli")`` guard was always true (the
dispatcher itself is started through the prelude) — so the ``which("hermes")``
fallback was unreachable dead code.

Contract: when ``sys.executable`` IS the install's store interpreter, the
worker command is built through the sanctioned launcher prelude
(``hermes_cli._launchers.runtime_command``); on any other interpreter the
module form keeps winning (PATH-planted ``hermes`` must not shadow the
running install, #111569).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _fake_store_layout(tmp_path, monkeypatch) -> Path:
    """Point the install's PM store at a directory whose Python is this interpreter."""
    runtime = tmp_path / "pm-store"
    entry = runtime / "python-3.14.7+fake"
    (entry / "bin").mkdir(parents=True)
    exe = entry / "bin" / ("python.exe" if os.name == "nt" else "python3")
    exe.symlink_to(Path(sys.executable))
    (runtime / "facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "python-3.14.7+fake"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(runtime))
    return exe


def test_store_python_worker_uses_launcher_prelude(tmp_path, monkeypatch):
    """Store interpreter -> launcher prelude argv; venv -> module argv still wins."""
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import _launchers

    monkeypatch.delenv("HERMES_BIN", raising=False)

    exe = _fake_store_layout(tmp_path, monkeypatch)
    assert _launchers.running_on_store_python() is True

    argv = kbd._resolve_hermes_argv()
    # The prelude contract from hermes_cli/_launchers.py::runtime_command: the
    # store interpreter in isolated mode, repo root + bootstrap before the entry.
    assert argv[0] == str(exe)
    assert argv[1:3] == ["-I", "-c"]
    bootstrap = argv[3]
    assert "import hermes_bootstrap" in bootstrap
    assert "runpy.run_module('hermes_cli.main'" in bootstrap
    repo_root = str(Path(kbd.__file__).resolve().parents[1])
    assert f"sys.path.insert(0, {repo_root!r})" in bootstrap

    # Same environment without a store record (venv/dev interpreter): the
    # module form must still win over a PATH-planted hermes (#111569).
    (Path(os.environ["HERMES_RUNTIME_DIR"]) / "facts.json").unlink()
    assert _launchers.running_on_store_python() is False
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    assert kbd._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]


def test_store_python_worker_argv_keeps_task_tail(tmp_path, monkeypatch):
    """The prelude is a prefix; the profile/task arguments ride after it."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.delenv("HERMES_BIN", raising=False)
    _fake_store_layout(tmp_path, monkeypatch)

    task = kb.Task(
        id="t_store_spawn",
        title="spawn on store python",
        body=None,
        assignee="main",
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
        current_run_id=7,
    )
    monkeypatch.setattr(kbd, "_resolve_worker_cli_toolsets", lambda home: None)

    cmd = kbd._worker_argv(task, "main", None)
    prelude = kbd._resolve_hermes_argv()
    assert cmd[: len(prelude)] == prelude
    assert cmd[len(prelude) :] == [
        "-p", "main",
        "--cli",
        "--accept-hooks",
        "chat", "-q", "work kanban task t_store_spawn",
    ]
