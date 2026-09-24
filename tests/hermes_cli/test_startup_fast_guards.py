"""Guards for hermes_cli._startup_fast — the pre-import version fast path.

Two invariants, each of which has been broken before:

1. IMPORT WEIGHT: _startup_fast must stay stdlib-only. The whole point of
   the module is to run before main.py's heavy import wall; one careless
   ``from hermes_cli.config import ...`` silently makes `hermes --version`
   slow again for everyone (the regression would be invisible — everything
   still works, just 40x slower).

2. OUTPUT PARITY / LIVENESS: the fast path must actually produce version
   output and exit 0 in a real subprocess, on and off Termux. This is the
   test that would have caught eb4040242, which changed the canonical
   version output to reference the PROJECT_ROOT module constant inside the
   fast function — a name that doesn't exist yet at the fast exit point —
   NameError-ing the Termux fast path in production for weeks.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Modules that must NEVER be imported by the fast path. Each one either
# pulls yaml/argparse/logging config or is itself a god-module.
_FORBIDDEN_MODULES = (
    "hermes_cli.config",
    "hermes_cli.main",
    "yaml",
    "argparse",
    "cli",
    "run_agent",
    "model_tools",
    "httpx",
    "openai",
)


@pytest.mark.linux_only
def test_cli_starts_from_a_deleted_cwd(tmp_path):
    """A child spawned into a directory that was removed since (a cron delivery from a reaped
    kanban workspace) must still reach argv parsing: a relative ``sys.path`` entry made
    ``ensure_project_root_on_path`` die in ``realpath`` → ``getcwd`` (#102941)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    gone = tmp_path / "scratch"
    gone.mkdir()
    # The child needs the checkout on sys.path by absolute name; the cwd is exactly what is gone.
    env = {**os.environ, "HERMES_HOME": str(home), "TERMUX_VERSION": "",
           "PYTHONPATH": os.pathsep.join(p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH")) if p)}
    env.pop("HERMES_DEV", None)
    fd = os.open(gone, os.O_RDONLY)
    try:
        gone.rmdir()
        # ``cwd=`` of a removed path is refused by Popen, so start in the dead dir via a
        # preexec fchdir onto its still-open handle — the shape a reaped workspace leaves behind.
        result = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", "--version"],
            capture_output=True, text=True, timeout=60, env=env,
            cwd=REPO_ROOT, preexec_fn=lambda: os.fchdir(fd))
    finally:
        os.close(fd)
    assert result.returncode == 0, result.stderr
    assert "Hermes Agent v" in result.stdout
    assert "FileNotFoundError" not in result.stderr


def test_startup_fast_import_weight():
    """Importing _startup_fast must not drag in any heavy module."""
    probe = (
        "import sys, json\n"
        "import hermes_cli._startup_fast\n"
        "print(json.dumps(sorted(sys.modules.keys())))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    loaded = set(json.loads(result.stdout))
    offenders = [m for m in _FORBIDDEN_MODULES if m in loaded]
    assert not offenders, (
        f"hermes_cli._startup_fast imported heavy modules: {offenders} — "
        "the fast path must stay stdlib-only (see module docstring)."
    )


def _run_version(env_overrides: dict) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    env.pop("HERMES_DEV", None)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        env=env,
    )




def test_fast_version_parity_on_termux(tmp_path):
    """The historical Termux path — the one eb4040242 broke."""
    home = tmp_path / ".hermes"
    home.mkdir()
    result = _run_version(
        {"HERMES_HOME": str(home), "TERMUX_VERSION": "0.118"}
    )
    assert result.returncode == 0, result.stderr
    assert "Hermes Agent v" in result.stdout
    assert "Traceback" not in result.stderr


def test_fast_version_reports_install_method_stamp(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".install_method").write_text("git\n", encoding="utf-8")
    result = _run_version({"HERMES_HOME": str(home), "TERMUX_VERSION": ""})
    assert result.returncode == 0, result.stderr
    assert "Install method: git" in result.stdout


def test_literal_tilde_hermes_home_expands_before_any_reader(tmp_path):
    """A literal ``~`` in HERMES_HOME (fish, or any quoted value) is expanded at process entry.

    Before the fix ``Path("~/.x")`` was relative, so the real CLI resolved it against cwd and
    scaffolded a full home under ``<cwd>/~/.x``. The negative assertion on cwd is the
    reporter's own acceptance criterion.
    """
    fake_home = tmp_path / "home"
    cwd = tmp_path / "project"
    fake_home.mkdir()
    cwd.mkdir()
    env = {**os.environ, "HOME": str(fake_home), "USERPROFILE": str(fake_home), "HERMES_HOME": "~/.x"}
    env.pop("HERMES_DEV", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "config", "path"],
        capture_output=True, text=True, timeout=120, cwd=cwd, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_home / ".x" / "config.yaml")
    assert not (cwd / "~").exists(), sorted(p.name for p in cwd.iterdir())

    # Raw-reader observable: the ~30 ``os.environ["HERMES_HOME"]`` readers in hermes_cli/ never
    # call the resolver, so the entry-point hunk in main.py (not hermes_constants) must have
    # rewritten the env var by the time the module import finishes.
    probe = subprocess.run(
        [sys.executable, "-c", "import hermes_cli.main, os; print(os.environ['HERMES_HOME'])"],
        capture_output=True, text=True, timeout=120, cwd=cwd, env=env,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == str(fake_home / ".x")


def test_normalize_hermes_home_env_rewrites_tilde_and_leaves_absolute_alone(tmp_path, monkeypatch):
    from hermes_cli import _startup_fast

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", "~/.x")
    _startup_fast.normalize_hermes_home_env()
    assert os.environ["HERMES_HOME"] == str(tmp_path / ".x")

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "abs"))
    _startup_fast.normalize_hermes_home_env()
    assert os.environ["HERMES_HOME"] == str(tmp_path / "abs")

    monkeypatch.delenv("HERMES_HOME")
    _startup_fast.normalize_hermes_home_env()
    assert "HERMES_HOME" not in os.environ
