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


def test_fast_version_parity_off_termux(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    result = _run_version({"HERMES_HOME": str(home), "TERMUX_VERSION": ""})
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for field in ("Hermes Agent v", "Install directory:", "Python:", "OpenAI SDK:"):
        assert field in out, f"fast --version output missing {field!r}:\n{out}"


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


def test_ensure_project_root_tolerates_unresolvable_sys_path_entries(monkeypatch):
    """Relative/cwd-derived sys.path entries must not crash when realpath fails.

    Cron bot-chat delivery can spawn into a deleted worker cwd; realpath then
    raises FileNotFoundError via getcwd() (#102941).
    """
    import hermes_cli._startup_fast as startup_fast

    real_realpath = os.path.realpath
    boom = FileNotFoundError(2, "Unable to get the current working directory")

    def fake_realpath(path):
        if path in (".", "relative-entry"):
            raise boom
        return real_realpath(path)

    monkeypatch.setattr(os.path, "realpath", fake_realpath)
    monkeypatch.setattr(
        sys,
        "path",
        [".", "relative-entry", startup_fast.project_root_str(), "/unrelated"],
        raising=False,
    )

    startup_fast.ensure_project_root_on_path()

    assert sys.path[0] == startup_fast.project_root_str()
    assert "." in sys.path
    assert "relative-entry" in sys.path
    assert "/unrelated" in sys.path
    # The absolute project-root duplicate was dropped once.
    assert sys.path.count(startup_fast.project_root_str()) == 1


def test_sys_path_entry_match_treats_oserror_as_non_match(monkeypatch):
    from hermes_cli._startup_fast import _sys_path_entry_matches_root

    assert _sys_path_entry_matches_root("", "/any") is False

    def boom(_path):
        raise FileNotFoundError(2, "Unable to get the current working directory")

    monkeypatch.setattr(os.path, "realpath", boom)
    assert _sys_path_entry_matches_root(".", "/root") is False
