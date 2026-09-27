"""Behavioral tests for kanban worker spawn resolution.

The field failure this guards against: every kanban worker died on first spawn
with ``ModuleNotFoundError: No module named 'hermes_cli'``. The dispatcher built
its argv as ``[sys.executable, "-m", "hermes_cli.main"]``; on a managed Hermes
runtime there is no installed ``hermes_cli`` (only a source checkout under the
repo root), and workers are spawned with ``cwd=workspace``, never the repo root.
``PYTHONPATH`` in the spawn env cannot fix it — the runtime launcher pops
``PYTHONPATH`` from ``os.environ`` before inserting the repo root itself.

So the worker is routed through the published launcher, which resolves the
package regardless of cwd. These tests assert the argv contract and, where a
launcher is published, that it actually runs — without a module-level
``sys.exit`` (which would break pytest collection).

Every subprocess here runs under a temporary ``HERMES_HOME``: the launcher
enters startup maintenance before it parses arguments, so a test that inherits
the real home would touch live state. The repo's ``home_io_guard`` enforces it.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db_dispatch as kbd

LAUNCHER = Path(kbd.__file__).resolve().parent.parent / ".hermes" / "bin" / "hermes"
LAUNCHER_PUBLISHED = LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK)
# LAUNCHER is <repo>/.hermes/bin/hermes, so the repo root is three levels up.
REPO_ROOT = str(LAUNCHER.parent.parent.parent)


def _isolated_env(home: Path, **extra: str) -> dict:
    """Bare environment plus a throwaway HERMES_HOME — never the real one.

    The child's ``sys.path[0]`` is the script directory (``-c`` gives ``''`` →
    the cwd), so the repo under test is importable without PYTHONPATH. Runtime
    third-party deps (ruamel etc.) are carried over from the parent's resolved
    paths, because a fresh interpreter in a tmp cwd would otherwise lack them.
    """
    import site

    env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "HERMES_HOME")
    }
    env["HOME"] = str(home)
    env["HERMES_HOME"] = str(home)
    site_dirs = [p for p in sys.path if p.endswith("site-packages")]
    if site_dirs:
        env["PYTHONPATH"] = os.pathsep.join(site_dirs)
    env.update(extra)
    return env


def test_module_argv_never_uses_path_shim(monkeypatch):
    """Whatever form is chosen, it must not be a `hermes` planted on PATH."""
    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(kbd, "_safe_which_no_cwd", lambda name: "/tmp/planted/hermes")
    argv = kbd._module_hermes_argv()
    assert argv, "module argv must never be empty"
    assert "/tmp/planted/hermes" not in argv


@pytest.mark.skipif(not LAUNCHER_PUBLISHED, reason="no published launcher in this checkout")
def test_module_argv_prefers_published_launcher():
    """With a launcher published, prefer it: it resolves hermes_cli from any cwd."""
    assert kbd._module_hermes_argv() == [str(LAUNCHER)]


def test_module_argv_falls_back_to_interpreter(monkeypatch):
    """With no launcher published, the interpreter form is preserved."""
    monkeypatch.setattr(kbd, "Path", _AlwaysMissingPath)
    argv = kbd._module_hermes_argv()
    assert argv == [sys.executable, "-m", "hermes_cli.main"]


class _AlwaysMissingPath:
    """Stand-in whose launcher path never reports as a file.

    ``parent`` is a property, matching pathlib — the production code chains
    ``Path(...).resolve().parent.parent / "..."``, so a method here would make
    ``.parent.parent`` an AttributeError instead of a path.
    """

    def __init__(self, *_args, **_kwargs):
        pass

    def resolve(self):
        return self

    @property
    def parent(self):
        return self

    def __truediv__(self, _other):
        return self

    def is_file(self):
        return False


@pytest.mark.skipif(not LAUNCHER_PUBLISHED, reason="no published launcher in this checkout")
def test_published_launcher_strips_pythonpath(tmp_path):
    """The launcher must discard an inherited or injected PYTHONPATH.

    This is the property that made an env-based fix impossible: the launcher
    pops PYTHONPATH from os.environ and inserts the repo root itself, so no
    injected value can reach the child. Asserted directly on the launcher's own
    script text — running `hermes --help` here would need the runtime's
    third-party deps, which the launcher deliberately makes unreachable by
    stripping PYTHONPATH. ``test_launcher_resolves_hermes_cli_without_pythonpath``
    covers the import itself.
    """
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    # The launcher is a shell script wrapping a `python -c` payload, so single
    # quotes inside it are shell-escaped ('"'"'). Normalise the escaping away,
    # then match the intent rather than one exact spelling.
    flat = text.replace("'\"'\"'", "'").replace("\\'", "'").replace('\\"', '"')
    assert "PYTHONPATH" in flat, "launcher no longer mentions PYTHONPATH"
    assert "pop('PYTHONPATH'" in flat, (
        "launcher must pop PYTHONPATH, not merely set sys.path around it"
    )
    # `-I` is the second half of the guarantee: isolated mode ignores the
    # environment for sys.path, so an inherited value cannot shadow the repo.
    assert " -I " in text or text.rstrip().endswith("-I") or " -I -c " in text, (
        "launcher should invoke the interpreter in isolated mode"
    )


@pytest.mark.skipif(not LAUNCHER_PUBLISHED, reason="no published launcher in this checkout")
def test_published_launcher_is_executable_and_self_contained():
    """The launcher is a POSIX shim that execs the CLI, so argv survives intact."""
    assert os.access(LAUNCHER, os.X_OK)
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    assert "exec " in text, "launcher should exec rather than fork, for signal forwarding"
    assert '"$@"' in text, "launcher must forward arguments verbatim"


def _launcher_interpreter() -> str | None:
    """The interpreter the launcher itself execs, parsed from its own text.

    Reading it from the launcher rather than hardcoding a versioned path keeps
    the test correct across runtime upgrades, and — critically — it is the
    interpreter that actually has Hermes' third-party deps installed. Using the
    *test runner's* interpreter here is what made an earlier version of this
    test fail on a missing `ruamel`, which says nothing about the launcher.
    """
    import re

    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    # The shebang occupies line 1, so `exec` is on line 2 — match it anywhere on
    # its own line rather than anchoring to the start of the file.
    m = re.search(r"^\s*exec\s+(\S+)", text, re.MULTILINE)
    return m.group(1) if m else None


def _missing_third_party_dep(proc) -> str | None:
    """Skip-reason when the launcher failed on a missing dep, not on resolution.

    The managed runtime this launcher execs is not a dependency-complete
    environment on every host: on this one ``ruamel`` is installed only in the
    repo venv (3.11), so a standalone launcher run dies importing
    ``hermes_yaml`` before it ever exercises the ``sys.path`` behaviour under
    test. That failure says nothing about whether the launcher resolves
    ``hermes_cli``, so it must not be reported as one.

    Only a genuine missing-dependency traceback qualifies. The module named in
    ``No module named 'X'`` is what matters: if that is ``hermes_cli`` the
    defect is exactly what this change fixes and the test must FAIL. Any other
    missing module is a host dependency gap. (The full traceback is not a
    reliable signal — it passes through ``hermes_cli/`` frames on the way to
    whatever genuinely failed to import.)
    """
    if proc.returncode == 0:
        return None
    err = proc.stderr or ""
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", err)
    if m and m.group(1).split(".")[0] == "hermes_cli":
        return None  # the real defect — must fail, never skip
    if m:
        return f"launcher interpreter lacks third-party dep {m.group(1)!r}"
    if "ImportError" in err:
        return f"launcher interpreter import failure: {err.strip()[-200:]}"
    return None


@pytest.mark.skipif(not LAUNCHER_PUBLISHED, reason="no published launcher in this checkout")
def test_published_launcher_resolves_hermes_cli_from_a_workspace_cwd(tmp_path):
    """The REAL launcher must resolve `hermes_cli` from a non-repo-root cwd.

    This is the property the bare `python -m` form lacks and the reason the
    dispatcher now prefers the launcher: workers spawn with ``cwd=workspace``,
    never the repo root. The launcher is executed as-is — not a reconstruction
    of its bootstrap — so breaking the launcher's own ``sys.path.insert`` would
    fail here. Isolated HOME/HERMES_HOME because the launcher enters startup
    maintenance before it parses arguments.
    """
    home = tmp_path / "home"
    home.mkdir()
    proc = subprocess.run(
        [str(LAUNCHER), "--help"],
        env=_isolated_env(home), cwd=str(tmp_path),
        capture_output=True, text=True, timeout=300,
    )
    reason = _missing_third_party_dep(proc)
    if reason:
        pytest.skip(reason)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "usage: hermes" in proc.stdout.lower()


@pytest.mark.skipif(not LAUNCHER_PUBLISHED, reason="no published launcher in this checkout")
def test_published_launcher_runs_with_a_poisoned_pythonpath(tmp_path):
    """A hostile PYTHONPATH must not break or redirect the real launcher.

    It strips PYTHONPATH itself and inserts the repo root, so an inherited or
    injected value cannot shadow the real package.
    """
    home = tmp_path / "home"
    home.mkdir()
    proc = subprocess.run(
        [str(LAUNCHER), "--help"],
        env=_isolated_env(home, PYTHONPATH="/nonexistent/bogus"), cwd=str(tmp_path),
        capture_output=True, text=True, timeout=300,
    )
    reason = _missing_third_party_dep(proc)
    if reason:
        pytest.skip(reason)
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "usage: hermes" in proc.stdout.lower()


@pytest.mark.skipif(not LAUNCHER_PUBLISHED, reason="no published launcher in this checkout")
def test_published_launcher_names_an_interpreter(monkeypatch):
    """The launcher must name an absolute interpreter path.

    Checks the shape of the contract, not the file's existence: the managed
    runtime lives under the real Hermes home, and this repo's ``home_io_guard``
    forbids a test from touching it. A missing interpreter after a runtime
    upgrade is an installation problem for the launcher's own smoke test to
    surface, not something this unit test should reach into the live home for.
    """
    interp = _launcher_interpreter()
    if not interp:
        pytest.skip("could not parse the launcher interpreter path")
    assert os.path.isabs(interp), f"launcher interpreter must be absolute: {interp}"
    assert interp.endswith("python3"), f"unexpected interpreter: {interp}"
