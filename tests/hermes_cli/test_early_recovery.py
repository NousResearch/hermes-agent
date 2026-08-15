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


def test_pid_liveness_recognizes_current_process():
    assert er._pid_is_running(os.getpid()) is True
    assert er._pid_is_running(0) is False


def test_marker_owner_liveness_uses_recorded_pid(tmp_path, monkeypatch):
    marker = tmp_path / ".update-incomplete"
    marker.write_text("started=1\npid=4321\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        er, "_pid_is_running", lambda pid: seen.append(pid) or True
    )

    assert er._marker_owner_is_live(marker) is True
    assert seen == [4321]


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


# ---------------------------------------------------------------------------
# _run_repair_install: uv-managed base interpreters (#83569)
# ---------------------------------------------------------------------------

def test_repair_install_prefers_uv_when_base_is_externally_managed(
    tmp_path, monkeypatch
):
    """uv-managed base Pythons carry EXTERNALLY-MANAGED: plain
    ``python -m pip`` aborts, so the repair must go through ``uv pip`` with
    VIRTUAL_ENV pointed at the project venv."""
    root = _project(tmp_path)
    monkeypatch.setattr(er, "_base_interpreter_is_externally_managed", lambda: True)
    monkeypatch.setattr(er, "_find_uv_binary", lambda: "/fake/uv")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(er.subprocess, "run", fake_run)

    assert er._run_repair_install(["cryptography==50.0.0"], root) is True

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == ["/fake/uv", "pip", "install"]
    assert "--force-reinstall" in cmd
    assert "cryptography==50.0.0" in cmd


def test_repair_install_uv_sets_virtual_env_to_project_venv(tmp_path, monkeypatch):
    root = _project(tmp_path)
    monkeypatch.setattr(er, "_base_interpreter_is_externally_managed", lambda: True)
    monkeypatch.setattr(er, "_find_uv_binary", lambda: "/fake/uv")

    seen_env = {}

    def fake_run(cmd, **kwargs):
        seen_env.update(kwargs.get("env") or {})

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(er.subprocess, "run", fake_run)

    assert er._run_repair_install(["PyYAML==6.0.2"], root) is True
    assert seen_env.get("VIRTUAL_ENV") == str(root / "venv")
    # A leaked PYTHONHOME/PYTHONPATH from the parent shell must not steer
    # uv's venv resolution.
    assert "PYTHONHOME" not in seen_env
    assert "PYTHONPATH" not in seen_env


def test_repair_install_falls_back_to_break_system_packages_without_uv(
    tmp_path, monkeypatch
):
    """No uv anywhere: still attempt the repair with pip's PEP 668 override
    instead of no-oping behind externally-managed-environment."""
    root = _project(tmp_path)
    monkeypatch.setattr(er, "_base_interpreter_is_externally_managed", lambda: True)
    monkeypatch.setattr(er, "_find_uv_binary", lambda: None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(er.subprocess, "run", fake_run)

    assert er._run_repair_install(["cryptography==50.0.0"], root) is True

    pip_calls = [c for c in calls if "pip" in c]
    assert pip_calls, calls
    assert any("--break-system-packages" in c for c in pip_calls)


def test_repair_install_uses_plain_pip_when_not_externally_managed(
    tmp_path, monkeypatch
):
    """Self-contained venvs (no PEP 668 marker) keep the original behaviour:
    ensurepip + plain pip, no uv lookup, no override flag."""
    root = _project(tmp_path)
    monkeypatch.setattr(
        er, "_base_interpreter_is_externally_managed", lambda: False
    )
    monkeypatch.setattr(
        er, "_find_uv_binary", lambda: pytest.fail("uv must not be consulted")
    )

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0
            stderr = ""
            stdout = ""

        return R()

    monkeypatch.setattr(er.subprocess, "run", fake_run)

    assert er._run_repair_install(["cryptography==50.0.0"], root) is True

    flat = [part for cmd in calls for part in cmd]
    assert "--break-system-packages" not in flat
    assert any("ensurepip" in part for part in flat)


def test_externally_managed_detection(tmp_path, monkeypatch):
    """The probe keys off the EXTERNALLY-MANAGED marker next to the stdlib."""
    import sysconfig

    real_get_path = sysconfig.get_path
    monkeypatch.setattr(
        sysconfig,
        "get_path",
        lambda key: str(tmp_path) if key == "stdlib" else real_get_path(key),
    )
    assert er._base_interpreter_is_externally_managed() is False
    (tmp_path / "EXTERNALLY-MANAGED").write_text("", encoding="utf-8")
    assert er._base_interpreter_is_externally_managed() is True


# ---------------------------------------------------------------------------
# Pending core install (.update-incomplete) — completed BEFORE native imports
# (#83569 review: a deferred update must not re-lock itself on the next launch)
# ---------------------------------------------------------------------------

def test_core_marker_triggers_install_before_any_native_import(
    tmp_path, monkeypatch
):
    """The reviewer's exact case (comment 5254279935): ``.update-incomplete``
    present, venv HEALTHY (import probes would pass).  The early pass must
    STILL run the core install — crucially while no native extension module
    is loaded in this process — because deferring to main()'s post-import
    recovery lets a recurring eager import remap the .pyd first."""
    root = _project(tmp_path)
    core_marker = root / ".update-incomplete"
    core_marker.write_text('{"attempts": 0}', encoding="utf-8")

    from hermes_cli import _install_repair as ir

    calls: list[dict] = []

    def fake_install(project_root):
        calls.append(
            {
                "root": project_root,
                "native_loaded_at_call": sorted(
                    m for m in sys.modules if m.startswith("cryptography")
                ),
            }
        )

    monkeypatch.setattr(ir, "run_core_install", fake_install)
    # Early recovery imports _install_repair lazily inside the helper; make
    # sure the lazy import resolves to the SAME monkeypatched module object.
    import hermes_cli._install_repair  # noqa: F401  (pre-import for patch)

    er.recover_if_needed(project_root=root, argv=[])

    assert len(calls) == 1, "core install must run when the marker exists"
    assert calls[0]["root"] == root
    assert calls[0]["native_loaded_at_call"] == [], (
        "install must run BEFORE any cryptography module is loaded "
        "(that is the whole point of the early pass)"
    )
    assert not core_marker.exists(), "marker cleared on success"
    # And the lazy import-probe repair path must NOT also fire:
    # (no probe repair attempted — cryptography is irrelevant to this branch)


def test_core_marker_marks_attempts_and_keeps_marker_on_install_failure(
    tmp_path, monkeypatch
):
    root = _project(tmp_path)
    core_marker = root / ".update-incomplete"
    core_marker.write_text('{"attempts": 0}', encoding="utf-8")

    from hermes_cli import _install_repair as ir

    def boom(_project_root):
        raise RuntimeError("simulated install failure")

    monkeypatch.setattr(ir, "run_core_install", boom)
    import hermes_cli._install_repair  # noqa: F401

    er.recover_if_needed(project_root=root, argv=[])

    assert core_marker.exists(), "failure keeps the marker for the next try"
    import json

    body = json.loads(core_marker.read_text(encoding="utf-8"))
    assert body["attempts"] == 1
    # Recovery lock released even on failure (next launch may retry).
    assert not (root / ".update-incomplete.lock").exists()


def test_core_marker_retry_ceiling_hands_off_to_late_recovery(
    tmp_path, monkeypatch
):
    """A persistently failing install must not reinstall-hammer every launch."""
    root = _project(tmp_path)
    core_marker = root / ".update-incomplete"
    core_marker.write_text(
        f'{{"attempts": {er._EARLY_CORE_INSTALL_MAX_ATTEMPTS}}}', encoding="utf-8"
    )

    from hermes_cli import _install_repair as ir

    monkeypatch.setattr(
        ir,
        "run_core_install",
        lambda _r: (_ for _ in ()).throw(
            AssertionError("install must NOT run past the attempts ceiling")
        ),
    )
    import hermes_cli._install_repair  # noqa: F401

    er.recover_if_needed(project_root=root, argv=[])

    assert core_marker.exists(), "marker retained for main.py's late recovery"
    # Counter not bumped further by the skipped attempt.


def test_lazy_marker_alone_does_not_trigger_core_install(tmp_path, monkeypatch):
    """Invariant guard: a lone ``.lazy-refresh-incomplete`` must NOT trigger
    the core-install branch (lazy repair has its own narrow probe path and
    must NEVER clear the core marker per #58004)."""
    root = _project(tmp_path)
    (root / ".lazy-refresh-incomplete").write_text("x", encoding="utf-8")

    from hermes_cli import _install_repair as ir

    monkeypatch.setattr(
        ir,
        "run_core_install",
        lambda _r: (_ for _ in ()).throw(
            AssertionError("core install must not run for the lazy marker")
        ),
    )
    import hermes_cli._install_repair  # noqa: F401

    # Healthy probes → early pass does nothing (preserves existing behavior).
    monkeypatch.setattr(er, "_probe_broken_packages", lambda: [])

    er.recover_if_needed(project_root=root, argv=[])


def test_core_marker_from_dead_updater_is_recovered_on_update_retry(
    tmp_path, monkeypatch
):
    """Retrying ``hermes update`` must consume a prior deferral marker.

    The self-lock preflight exits after writing this marker.  Desktop and CLI
    retries both keep ``update`` in argv, so an argv-only skip loops forever.
    """
    root = _project(tmp_path)
    core_marker = root / ".update-incomplete"
    core_marker.write_text("started=1\npid=1234\n", encoding="utf-8")

    from hermes_cli import _install_repair as ir

    calls = []
    monkeypatch.setattr(ir, "run_core_install", lambda project_root: calls.append(project_root))
    monkeypatch.setattr(er, "_marker_owner_is_live", lambda _marker: False, raising=False)
    monkeypatch.setattr(er, "_UPDATE_RETRY_RECOVERED", False)
    import hermes_cli._install_repair  # noqa: F401

    er.recover_if_needed(project_root=root, argv=["update"])

    assert calls == [root]
    assert not core_marker.exists()
    assert er._should_skip_external_secret_sources() is True


def test_core_marker_owned_by_live_updater_is_not_recovered(
    tmp_path, monkeypatch
):
    """A second launch must not reinstall into an active updater's venv."""
    root = _project(tmp_path)
    core_marker = root / ".update-incomplete"
    core_marker.write_text("started=1\npid=1234\n", encoding="utf-8")

    from hermes_cli import _install_repair as ir

    monkeypatch.setattr(
        ir,
        "run_core_install",
        lambda _r: (_ for _ in ()).throw(
            AssertionError("must not race a live updater")
        ),
    )
    monkeypatch.setattr(er, "_marker_owner_is_live", lambda _marker: True, raising=False)
    import hermes_cli._install_repair  # noqa: F401

    er.recover_if_needed(project_root=root, argv=[])

    assert core_marker.exists()


def test_bump_marker_attempts_handles_missing_and_corrupt_bodies(tmp_path):
    from hermes_cli import _install_repair as ir

    m = tmp_path / ".update-incomplete"
    m.write_text("", encoding="utf-8")
    assert ir.bump_marker_attempts(m) == 1

    m.write_text("not json", encoding="utf-8")
    assert ir.bump_marker_attempts(m) == 1

    m.write_text('{"attempts": 2}', encoding="utf-8")
    assert ir.bump_marker_attempts(m) == 3










