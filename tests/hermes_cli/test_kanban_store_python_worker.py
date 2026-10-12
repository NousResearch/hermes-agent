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

import pytest

pytestmark = pytest.mark.platforms("posix", "windows")


def _store_python_rel() -> str:
    """The relative path of the store interpreter inside its package entry.

    Mirrors ``hermes_cli._launchers.resolve_store_python``: Windows stores the
    interpreter at ``<entry>/python.exe``, POSIX at ``<entry>/bin/python3`` —
    a fixture that hardcodes ``bin/`` resolves to None on win32 and the
    detector collapses to False there.
    """
    return "python.exe" if os.name == "nt" else "bin/python3"


def _fake_store_layout(tmp_path, monkeypatch) -> Path:
    """Point the install's PM store at a directory whose Python is this interpreter."""
    runtime = tmp_path / "pm-store"
    entry = runtime / "python-3.14.7+fake"
    exe = entry / _store_python_rel()
    exe.parent.mkdir(parents=True, exist_ok=True)
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


def test_corrupt_store_manifest_fails_closed_not_crashes(tmp_path, monkeypatch):
    """A manifest/store the resolver cannot parse must not crash every caller
    of the detector: ``store_root`` parses ``<repo_root>.parent/manifest.json``
    unguarded (a truncated file raises JSONDecodeError; an escaping ``store``
    path raises RuntimeError) and ``resolve_store_python`` catches neither.
    "Cannot prove we are on the store interpreter" keeps the module-form argv
    (pre-fix behavior), and ``running_on_store_python`` returns False instead
    of propagating.

    The unguarded parsing is driven through the REAL ``pm.environments.store_root``
    on a staged payload layout: ``running_on_store_python`` derives the repo root
    from ``__file__`` (not controllable from a test), so ``_launchers.store_root``
    — the module-level name ``resolve_store_python`` calls — is repointed at the
    real function bound to the staged root. The exceptions themselves come from
    production code, and removing the guard in ``running_on_store_python``
    repropagates them here (red).
    """
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import _launchers
    from pm import environments as _env
    import json as _json

    monkeypatch.delenv("HERMES_BIN", raising=False)

    payload_root = tmp_path / "payload"
    repo_root = payload_root / "repo"
    repo_root.mkdir(parents=True)
    manifest_path = payload_root / "manifest.json"

    monkeypatch.setattr(_launchers, "store_root", lambda _root: _env.store_root(repo_root))

    def _assert_module_form():
        assert _launchers.running_on_store_python() is False
        assert kbd._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]

    # Sanity: the identity check passes for this layout, so parsing proceeds
    # to the unguarded store resolution (a well-formed store does not raise).
    manifest_path.write_text(_json.dumps({"repo": "repo", "store": "store"}), encoding="utf-8")
    assert _env.store_root(repo_root) == (payload_root / "store").resolve()

    # Truncated manifest.json → JSONDecodeError out of store_root().
    manifest_path.write_text('{"repo": "hermes', encoding="utf-8")
    with pytest.raises(ValueError):
        _env.store_root(repo_root)
    _assert_module_form()

    # A store path that escapes its root → RuntimeError out of store_root().
    manifest_path.write_text(_json.dumps({"repo": "repo", "store": "../../escape"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _env.store_root(repo_root)
    _assert_module_form()
