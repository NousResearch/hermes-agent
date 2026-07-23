"""Tests for hermes_cli._early_recovery — the dependency-light bootstrap
repair that runs BEFORE hermes_cli.main's third-party imports (#57828 / #58004).

Covers:
- entry-point lifecycle: a broken core import (dotenv) crashes the import of
  hermes_cli.main WITHOUT early recovery, and imports fine when recovery runs
  first (proving main.py invokes recovery before its third-party imports)
- recover_if_needed unit behavior: fast path, marker gating, update-argv skip,
  lock single-flight, no marker clearing, pinned repair specs
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import _early_recovery as er

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Entry-point lifecycle (subprocess, real imports)
# ---------------------------------------------------------------------------

def _make_broken_dotenv_shadow(tmp_path: Path) -> Path:
    """A sys.path dir shadowing ``dotenv`` with the #57828 failure state:
    distribution metadata intact, import files wiped/broken."""
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "dotenv.py").write_text(
        "raise ImportError('import files wiped mid-install (#57828)')\n",
        encoding="utf-8",
    )
    return shadow


def _run_lifecycle_subprocess(tmp_path: Path, *, repair: bool) -> subprocess.CompletedProcess:
    shadow = _make_broken_dotenv_shadow(tmp_path)
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    script = tmp_path / "lifecycle.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys

            shadow = {str(shadow)!r}
            sys.path.insert(0, shadow)

            # _early_recovery must be importable on the corrupted venv
            # (stdlib-only) — this import itself is part of the contract.
            import hermes_cli._early_recovery as er

            REPAIR = {repair!r}

            def recorder(*args, **kwargs):
                print("EARLY_RECOVERY_CALLED", flush=True)
                if REPAIR:
                    sys.path.remove(shadow)
                    sys.modules.pop("dotenv", None)

            er.recover_if_needed = recorder

            import hermes_cli.main  # noqa: F401
            print("MAIN_IMPORTED_OK", flush=True)
            """
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "HERMES_HOME": str(hermes_home),
    }
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )


def test_broken_dotenv_crashes_main_import_without_repair(tmp_path):
    """Negative control: the shadow really breaks importing hermes_cli.main,
    and recovery was invoked BEFORE the crash (i.e. before third-party
    imports) — so a real repair at that point can save the launch."""
    result = _run_lifecycle_subprocess(tmp_path, repair=False)
    assert result.returncode != 0
    assert "EARLY_RECOVERY_CALLED" in result.stdout
    assert "MAIN_IMPORTED_OK" not in result.stdout
    assert "wiped mid-install" in result.stderr


def test_early_recovery_runs_before_main_imports_and_saves_launch(tmp_path):
    """When recovery repairs the broken package, hermes_cli.main imports
    cleanly — proving the recovery hook fires before env_loader/dotenv."""
    result = _run_lifecycle_subprocess(tmp_path, repair=True)
    assert "EARLY_RECOVERY_CALLED" in result.stdout
    assert "MAIN_IMPORTED_OK" in result.stdout, result.stderr
    assert result.returncode == 0


def test_early_recovery_module_is_stdlib_only(tmp_path):
    """The module must import in a process where every non-stdlib import
    fails — that is the whole point of its existence."""
    script = tmp_path / "stdlib_only.py"
    script.write_text(
        textwrap.dedent(
            """
            import builtins
            import sys

            STDLIB = set(sys.stdlib_module_names) | {"hermes_cli"}
            real_import = builtins.__import__

            def guard(name, *args, **kwargs):
                top = name.split(".")[0]
                if top not in STDLIB:
                    raise ImportError(f"non-stdlib import blocked: {name}")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = guard
            import hermes_cli._early_recovery  # noqa: F401
            print("STDLIB_ONLY_OK")
            """
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        timeout=60,
    )
    assert "STDLIB_ONLY_OK" in result.stdout, result.stderr


# ---------------------------------------------------------------------------
# recover_if_needed unit behavior
# ---------------------------------------------------------------------------

def _project(tmp_path: Path, *, pyproject: bool = True) -> Path:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    if pyproject:
        (root / "pyproject.toml").write_text(
            '[project]\nname = "x"\ndependencies = [\n'
            '  "PyYAML==6.0.2",\n'
            '  "python-dotenv==1.2.2",\n'
            '  "PyJWT[crypto]==2.13.0",\n'
            "]\n",
            encoding="utf-8",
        )
    return root


def test_fast_path_no_marker_never_probes(tmp_path, monkeypatch):
    root = _project(tmp_path)
    probed = []
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: probed.append(1) or [])
    er.recover_if_needed(project_root=root, argv=[])
    assert probed == []


def test_update_argv_skips_recovery(tmp_path, monkeypatch):
    root = _project(tmp_path)
    (root / ".lazy-refresh-incomplete").write_text("x", encoding="utf-8")
    probed = []
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: probed.append(1) or [])
    er.recover_if_needed(project_root=root, argv=["update"])
    assert probed == []


def test_no_pyproject_skips_and_preserves_marker(tmp_path, monkeypatch):
    root = _project(tmp_path, pyproject=False)
    marker = root / ".update-incomplete"
    marker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: ["PyYAML"])
    installs = []
    monkeypatch.setattr(er, "_run_repair_install", lambda specs, r: installs.append(specs) or True)
    er.recover_if_needed(project_root=root, argv=[])
    assert installs == []
    assert marker.exists()


def test_marker_plus_broken_probe_repairs_with_pinned_specs(tmp_path, monkeypatch):
    root = _project(tmp_path)
    marker = root / ".lazy-refresh-incomplete"
    marker.write_text("x", encoding="utf-8")

    probe_results = iter([["PyYAML", "python-dotenv"], []])
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: next(probe_results))
    installs = []
    monkeypatch.setattr(
        er, "_run_repair_install", lambda specs, r: installs.append(specs) or True
    )

    er.recover_if_needed(project_root=root, argv=[])

    assert installs == [["PyYAML==6.0.2", "python-dotenv==1.2.2"]]
    # Marker lifecycle belongs to main.py's full recovery — never cleared here.
    assert marker.exists()
    # Lock released for the full recovery pass.
    assert not (root / ".update-incomplete.lock").exists()


def test_healthy_probe_skips_install(tmp_path, monkeypatch):
    root = _project(tmp_path)
    (root / ".update-incomplete").write_text("x", encoding="utf-8")
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: [])
    installs = []
    monkeypatch.setattr(er, "_run_repair_install", lambda specs, r: installs.append(specs) or True)
    er.recover_if_needed(project_root=root, argv=[])
    assert installs == []


def test_lock_held_skips_repair(tmp_path, monkeypatch):
    root = _project(tmp_path)
    (root / ".lazy-refresh-incomplete").write_text("x", encoding="utf-8")
    (root / ".update-incomplete.lock").write_text("123\n", encoding="utf-8")
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: ["PyYAML"])
    installs = []
    monkeypatch.setattr(er, "_run_repair_install", lambda specs, r: installs.append(specs) or True)
    er.recover_if_needed(project_root=root, argv=[])
    assert installs == []
    # Fresh (non-stale) lock is left for its owner.
    assert (root / ".update-incomplete.lock").exists()


def test_failed_repair_prints_manual_command_with_pins(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    (root / ".lazy-refresh-incomplete").write_text("x", encoding="utf-8")
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: ["PyJWT"])
    monkeypatch.setattr(er, "_run_repair_install", lambda specs, r: False)
    er.recover_if_needed(project_root=root, argv=[])
    err = capsys.readouterr().err
    assert "--force-reinstall" in err
    assert "PyJWT[crypto]==2.13.0" in err


def test_pinned_specs_falls_back_to_bare_names_without_pyproject(tmp_path):
    root = _project(tmp_path, pyproject=False)
    assert er._pinned_specs(["PyYAML", "unknown-pkg"], root) == ["PyYAML", "unknown-pkg"]


def test_pinned_specs_strips_env_markers_and_matches_extras(tmp_path):
    root = _project(tmp_path)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = [\n'
        '  "cryptography==46.0.7; python_version >= \'3.11\'",\n'
        '  "PyJWT[crypto]==2.13.0",\n'
        "]\n",
        encoding="utf-8",
    )
    assert er._pinned_specs(["cryptography", "PyJWT"], root) == [
        "cryptography==46.0.7",
        "PyJWT[crypto]==2.13.0",
    ]


def test_probe_tables_shared_with_main():
    """The full recovery layer in main.py must probe/repair the same set as
    the early layer — the tables have one canonical home."""
    import hermes_cli.main as m

    assert m._LAZY_REFRESH_IMPORT_PROBES == er.LAZY_REFRESH_IMPORT_PROBES
    assert m._LAZY_REFRESH_REPAIR_PACKAGES == er.LAZY_REFRESH_REPAIR_PACKAGES
