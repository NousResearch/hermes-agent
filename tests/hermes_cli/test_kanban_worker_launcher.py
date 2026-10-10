"""Kanban worker launcher resolution from unrelated workspaces."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import _runtime_command as runtime_command

# ---------------------------------------------------------------------------
# Dispatcher spawn invocation — _resolve_hermes_argv()
#
# Workers spawned by the dispatcher must use a `hermes` invocation that does
# not depend on PATH being set up correctly. cron jobs, systemd User= services,
# launchd jobs, and other detached processes routinely run with a stripped
# $PATH that doesn't include the venv's bin/, so a bare `["hermes", ...]`
# spawn fails with FileNotFoundError and the task gets stuck. The resolver
# prefers the interpreter-bound module form (exactly this install; a PATH
# shim could be attacker-planted or belong to another install, #111569) and
# only falls back to the PATH shim when ``hermes_cli`` is not importable.
# Isolated store Python (``python -I``) is the exception: ``-m hermes_cli.main``
# cannot see the package. Use the published POSIX launcher when present, or its
# installation-bound bootstrap command when it is missing.
# ---------------------------------------------------------------------------


def test_resolve_hermes_argv_prefers_module_form_over_path_shim(monkeypatch):
    """A `hermes` on PATH must not shadow the running install (#111569):
    the module argv wins whenever ``hermes_cli`` is importable; only an
    explicit ``$HERMES_BIN`` overrides it."""
    import shutil
    import sys
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_isolated_store_python", lambda: False)
    assert kbd._resolve_hermes_argv() == [sys.executable, "-m", "hermes_cli.main"]

    monkeypatch.setenv("HERMES_BIN", "/opt/hermes/bin/hermes")
    assert kbd._resolve_hermes_argv() == ["/opt/hermes/bin/hermes"]


def test_resolve_hermes_argv_isolated_python_uses_install_launcher(monkeypatch):
    """Isolated store Python cannot ``-m hermes_cli.main`` (no default path).
    A published POSIX launcher must win over a planted PATH ``hermes``."""
    import shutil
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_isolated_store_python", lambda: True)
    monkeypatch.setattr(
        runtime_command, "_published_posix_launcher", lambda root: "/opt/hermes/.hermes/bin/hermes"
    )
    assert kbd._resolve_hermes_argv() == ["/opt/hermes/.hermes/bin/hermes"]


def test_resolve_hermes_argv_isolated_python_bootstraps_without_launcher(
    monkeypatch, tmp_path,
):
    """A missing published shim still starts this source tree from an unrelated cwd."""
    import shutil
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_isolated_store_python", lambda: True)
    monkeypatch.setattr(runtime_command, "_published_posix_launcher", lambda root: None)
    argv = kbd._resolve_hermes_argv()
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-I", "-c"]
    assert "hermes_cli.main" in argv[3]
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "HERMES_BIN"}}
    env["HERMES_HOME"] = str(tmp_path / "home")
    result = subprocess.run(argv + ["--version"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-500:]
    assert str(Path(kbd.__file__).resolve().parents[1]) in result.stdout


def test_resolve_hermes_argv_isolated_does_not_fall_back_to_path_when_launchers_import_fails(
    monkeypatch,
):
    """A broken heavy launcher import must not turn isolation into PATH trust."""
    import shutil
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_isolated_store_python", lambda: True)
    monkeypatch.setattr(runtime_command, "_published_posix_launcher", lambda root: None)
    monkeypatch.setitem(sys.modules, "hermes_cli._launchers", None)

    argv = kbd._resolve_hermes_argv()

    assert argv[0] == sys.executable
    assert argv[1:3] == ["-I", "-c"]
    assert "/tmp/planted/hermes" not in argv


def test_resolve_hermes_argv_isolated_fails_closed_when_bootstrap_is_unavailable(
    monkeypatch,
):
    """A bootstrap failure must surface instead of selecting a PATH executable."""
    import shutil
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    monkeypatch.setattr(kbd, "_isolated_store_python", lambda: True)
    monkeypatch.setattr(
        kbd, "_module_hermes_argv", lambda: (_ for _ in ()).throw(ImportError("bootstrap"))
    )

    with pytest.raises(ImportError, match="bootstrap"):
        kbd._resolve_hermes_argv()


def test_resolve_hermes_argv_isolated_preserves_explicit_override_and_batch_safety(
    monkeypatch,
):
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_isolated_store_python", lambda: True)
    monkeypatch.setattr(runtime_command, "_published_posix_launcher", lambda root: None)
    monkeypatch.setenv("HERMES_BIN", "/opt/operator/hermes")
    assert kbd._resolve_hermes_argv() == ["/opt/operator/hermes"]

    monkeypatch.setattr(kbd._kb, "_IS_WINDOWS", True)
    monkeypatch.setenv("HERMES_BIN", "C:\\operator\\hermes.cmd")
    argv = kbd._resolve_hermes_argv()
    assert argv[:3] == [sys.executable, "-I", "-c"]
    assert "hermes_cli.main" in argv[3]
    assert not any(arg.lower().endswith((".cmd", ".bat")) for arg in argv)


def test_resolve_hermes_argv_module_actually_runs():
    """The fallback module name must be importable + runnable.

    A unit test that pins the literal string is necessary but not
    sufficient — if `hermes_cli.main` ever loses `if __name__ == "__main__"`
    handling or its argparse setup, `python -m hermes_cli.main --version`
    would fail and so would every dispatcher spawn that hits the fallback.
    Run it as a real subprocess to catch that regression.
    """
    import subprocess
    from hermes_cli import kanban_db_dispatch as kbd
    import shutil
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_BIN", None)
        with mock.patch.object(shutil, "which", return_value=None):
            argv = kbd._resolve_hermes_argv()
    r = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (
        f"`{' '.join(argv)} --version` failed (rc={r.returncode}); "
        f"stderr={r.stderr[:200]!r}"
    )
