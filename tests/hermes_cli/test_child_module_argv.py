"""child_hermes_module_argv: the launcher-blind resolver fix (2026-09-25 fleet paralysis).

The bug (#111569 regression, emberevosrv 03:34 incident): resolvers picked
``sys.executable -m hermes_cli.main`` whenever ``find_spec('hermes_cli')`` was
not None IN THE PARENT. Under the launcher bootstrap (systemd ExecStart ->
``.hermes/bin/hermes`` -> ``python -I -c '...sys.path.insert(0, root)...'``)
the repo root lives only in the parent's in-process path and Hermes-owned
PYTHONPATH entries are stripped from every child — so every spawned ``-m``
child died with ModuleNotFoundError. The invariant these tests pin: module
argv may only be chosen when a FRESH CHILD resolves the module under the
child's own environment; otherwise the same install is reached through the
published launcher (``--run-module``) or the equivalent inline bootstrap.

Fixtures are pure argv / sys.path / fake-tree construction and real
subprocesses — no fake OS.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import _launchers
from hermes_cli._launchers import child_hermes_module_argv


@pytest.fixture(autouse=True)
def _fresh_probe_cache(monkeypatch):
    # The probe verdict is process-cached per (interpreter, module); keep tests
    # from inheriting each other's (or an earlier real probe's) verdicts.
    monkeypatch.setattr(_launchers, "_child_module_probe_cache", {})


def _fake_spec(root: Path, module: str = "fakepkg"):
    return types.SimpleNamespace(
        origin=str(root / module / "__init__.py"),
        submodule_search_locations=None,
    )


def _fake_install(tmp_path: Path, *, launcher: bool = False) -> Path:
    """A self-contained two-module install tree under ``tmp_path``.

    ``fakepkg.main`` echoes its argv so spawn tests can prove the CHILD booted
    THIS root and received the appended worker arguments. ``hermes_bootstrap``
    / ``hermes_constants`` are no-op stand-ins so the bootstrap text resolves
    entirely inside the fixture.
    """
    root = tmp_path / "repo"
    pkg = root / "fakepkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        "import sys\n"
        "print('BOOTED:', *sys.argv[1:])\n"
        "sys.exit(0)\n"
    )
    (root / "hermes_bootstrap.py").write_text("pass\n")
    (root / "hermes_constants.py").write_text(
        "def get_default_hermes_root():\n"
        f"    return {str(root)!r}\n"
    )
    if launcher:
        if os.name == "nt":
            pytest.skip("published-launcher branch is exercised via the POSIX shell shim")
        bin_dir = root / ".hermes" / "bin"
        bin_dir.mkdir(parents=True)
        shim = bin_dir / "hermes"
        # Same contract as the published launcher: consumes
        # ``--run-module <mod>`` (positional 1-2), re-injects THIS root, and
        # forwards every remaining argument to the module.
        shim.write_text(
            "#!/bin/sh\n"
            "ROOT=$(cd \"$(dirname \"$0\")/../..\" && pwd)\n"
            "shift 2\n"
            "exec " + repr(sys.executable) + " -I -c "
            "\"import os,sys,runpy;"
            "os.environ.pop('PYTHONHOME',None);os.environ.pop('PYTHONPATH',None);"
            "os.environ.pop('VIRTUAL_ENV',None);"
            "sys.path.insert(0,sys.argv[1]);"
            "import hermes_bootstrap;"
            "sys.argv=[sys.argv[0]]+sys.argv[2:];"
            "runpy.run_module('fakepkg.main',run_name='__main__',alter_sys=True)\" "
            "\"$ROOT\" \"$@\"\n"
        )
        shim.chmod(0o755)
    return root


# ---------------------------------------------------------------------------
# Branch selection (pure fixtures)
# ---------------------------------------------------------------------------


def test_module_argv_wins_when_child_resolves_it(monkeypatch, tmp_path):
    """Invariant #111569 preserved byte-for-byte: when the fresh child resolves
    the module (venv / Nix / editable installs), module argv wins over PATH."""
    monkeypatch.setattr(_launchers, "_module_resolves_in_fresh_child", lambda m: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "/tmp/attacker/hermes")
    assert child_hermes_module_argv("fakepkg") == [sys.executable, "-m", "fakepkg.main"]


def test_no_path_consulted_ever(monkeypatch, tmp_path):
    """The helper never shells out to PATH — attacker-shim precedence is the
    helper's whole reason to exist on both sides of the branch."""
    import shutil as shutil_mod

    def tripwire(name, *a, **k):
        raise AssertionError(f"PATH consulted: {name}")

    monkeypatch.setattr(shutil_mod, "which", tripwire)
    monkeypatch.setattr(_launchers, "_module_resolves_in_fresh_child", lambda m: True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(tmp_path))
    child_hermes_module_argv("fakepkg")


def test_importable_nowhere_returns_none(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    assert child_hermes_module_argv("fakepkg") is None


def test_find_spec_error_returns_none(monkeypatch):
    def boom(name, *a, **k):
        raise ImportError(name)

    monkeypatch.setattr(importlib.util, "find_spec", boom)
    assert child_hermes_module_argv("fakepkg") is None


def test_published_launcher_form_when_parent_path_is_private(monkeypatch, tmp_path):
    root = _fake_install(tmp_path, launcher=True)
    monkeypatch.setattr(_launchers, "_module_resolves_in_fresh_child", lambda m: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(root))
    argv = child_hermes_module_argv("fakepkg")
    assert argv == [str(root / ".hermes" / "bin" / "hermes"), "--run-module", "fakepkg.main"]


def test_inline_bootstrap_form_when_no_launcher_published(monkeypatch, tmp_path):
    root = _fake_install(tmp_path)
    monkeypatch.setattr(_launchers, "_module_resolves_in_fresh_child", lambda m: False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: _fake_spec(root))
    argv = child_hermes_module_argv("fakepkg")
    assert argv is not None
    assert argv[:3] == [sys.executable, "-I", "-c"]
    assert str(root) in argv[3]  # the bootstrap re-injects the parent's root


def test_exotic_layout_keeps_historical_module_form(monkeypatch, tmp_path):
    # A spec with neither origin nor search locations, and a spec whose package
    # directory is missing from the derived root: both keep the old argv.
    monkeypatch.setattr(_launchers, "_module_resolves_in_fresh_child", lambda m: False)
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: types.SimpleNamespace(origin=None, submodule_search_locations=None))
    assert child_hermes_module_argv("fakepkg") == [sys.executable, "-m", "fakepkg.main"]

    ghost_root = tmp_path / "ghost"
    ghost_root.mkdir()
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: types.SimpleNamespace(
            origin=str(ghost_root / "fakepkg" / "__init__.py"),
            submodule_search_locations=None))
    assert child_hermes_module_argv("fakepkg") == [sys.executable, "-m", "fakepkg.main"]


# ---------------------------------------------------------------------------
# The regression itself: parent-only sys.path injection must not lie.
# Real subprocesses, real interpreter — no fake OS anywhere in these two.
# ---------------------------------------------------------------------------


def test_parent_only_path_injection_resolves_to_child_bootable_form(tmp_path):
    """Exactly the 03:34 shape: a parent boots via ``python -I -c`` with the
    repo root injected into ITS sys.path only. The resolver must not hand back
    ``-m fakepkg.main``; whatever it returns must spawn a working child."""
    root = _fake_install(tmp_path)
    parent_code = (
        "import os,sys,json;"
        "os.environ.pop('PYTHONHOME',None);os.environ.pop('PYTHONPATH',None);"
        "os.environ.pop('VIRTUAL_ENV',None);"
        f"sys.path.insert(0,{str(root)!r});"
        "import hermes_bootstrap;"
        f"sys.path.insert(0,{str(Path(_launchers.__file__).resolve().parent.parent)!r});"
        "from hermes_cli._launchers import child_hermes_module_argv;"
        "print('ARGV_JSON:'+json.dumps(child_hermes_module_argv('fakepkg')))"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    r = subprocess.run([sys.executable, "-I", "-c", parent_code],
                       capture_output=True, text=True, timeout=120,
                       cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stderr[-400:]
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("ARGV_JSON:"))
    argv = json.loads(line[len("ARGV_JSON:"):])
    assert argv != [sys.executable, "-m", "fakepkg.main"], (
        "module argv chosen although a fresh child cannot resolve fakepkg — "
        "this is the exact lie that paralyzed the fleet on 2026-09-25")
    # And the argv it DID choose must boot a real child with appended worker args.
    child = subprocess.run(argv + ["--id", "t_test", "--board", "b"],
                           capture_output=True, text=True, timeout=120,
                           cwd=str(tmp_path), env=env)
    assert child.returncode == 0, child.stderr[-400:]
    assert "BOOTED: --id t_test --board b" in child.stdout


def test_module_argv_returned_in_this_install_actually_boots():
    """The no-change guarantee for real installs (the venv this test runs in):
    the argv chosen here must run ``--version`` as a fresh child from a foreign
    cwd, whichever branch wins."""
    argv = child_hermes_module_argv()
    assert argv is not None
    import tempfile
    r = subprocess.run(argv + ["--version"], capture_output=True, text=True,
                       timeout=180, cwd=tempfile.gettempdir())
    assert r.returncode == 0, f"{argv} failed: {r.stderr[-300:]!r}"


# ---------------------------------------------------------------------------
# Probe honesty: it models the CHILD env (temp cwd, sanitizer applied, -P),
# not the parent's.
# ---------------------------------------------------------------------------


def test_probe_uses_child_shaped_env(monkeypatch, tmp_path):
    root = _fake_install(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(root))
    seen = {}

    class Done:
        returncode = 1

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env")
        seen["cwd"] = kw.get("cwd")
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _launchers._module_resolves_in_fresh_child("fakepkg") is False
    assert "-P" in seen["cmd"], "probe must drop the cwd from sys.path"
    assert Path(seen["cwd"]) != Path.cwd(), "probe must not run from the repo"
    assert isinstance(seen["env"], dict), "probe must run under the child env, not ambient"


def test_probe_failure_to_spawn_keeps_module_form(monkeypatch):
    """If we cannot even run our own interpreter probe, preserve historical
    behavior (module argv) rather than failing the spawn path."""
    def boom(*a, **k):
        raise OSError("no fork on this box")

    monkeypatch.setattr(subprocess, "run", boom)
    assert _launchers._module_resolves_in_fresh_child("fakepkg") is True


def test_probe_verdict_is_cached_per_interpreter_and_module(monkeypatch):
    calls = []

    class Done:
        returncode = 0

    def counting_run(cmd, **kw):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(subprocess, "run", counting_run)
    assert _launchers._module_resolves_in_fresh_child("mod_a") is True
    assert _launchers._module_resolves_in_fresh_child("mod_a") is True
    assert _launchers._module_resolves_in_fresh_child("mod_b") is True
    assert len(calls) == 2, "cache key must be (interpreter, module), probed once each"


# ---------------------------------------------------------------------------
# Mirror call sites consume the shared helper (one helper, three sites).
# ---------------------------------------------------------------------------


def test_kanban_resolver_uses_helper(monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    marker = ["LAUNCHER", "FORM"]
    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr("hermes_cli._launchers.child_hermes_module_argv", lambda *a: list(marker))
    assert kbd._resolve_hermes_argv() == marker


def test_kanban_resolver_hermes_bin_still_wins(monkeypatch):
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr("hermes_cli._launchers.child_hermes_module_argv",
                        lambda *a: ["should", "not", "appear"])
    monkeypatch.setenv("HERMES_BIN", "/opt/hermes/bin/hermes")
    assert kbd._resolve_hermes_argv() == ["/opt/hermes/bin/hermes"]


def test_gateway_mirror_uses_helper(monkeypatch):
    from gateway.run import _resolve_hermes_bin

    marker = [sys.executable, "-I", "-c", "bootstrap-text"]
    monkeypatch.setattr("hermes_cli._launchers.child_hermes_module_argv", lambda *a: list(marker))
    assert _resolve_hermes_bin() == marker

    monkeypatch.setattr("hermes_cli._launchers.child_hermes_module_argv", lambda *a: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/hermes")
    assert _resolve_hermes_bin() == ["/usr/local/bin/hermes"]
