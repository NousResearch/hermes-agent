"""Tests for the post-update *import* guard in ``hermes update``.

``_validate_critical_files_syntax`` only parses files, so it cannot detect a
partially-updated tree: when one package is refreshed and a sibling is not,
every file still parses but importing them together raises ``ImportError``.

Reference incident: a Windows user reported
``ImportError: cannot import name 'TODO_INJECTION_HEADER' from
'tools.todo_tool'`` on every startup after an update. ``agent/`` carried the
new ``context_compressor.py`` (which imports that name at module level) while
``tools/`` still held the pre-update ``todo_tool.py``. The ZIP-update path
replaces top-level entries one at a time in ``os.listdir`` order, so an
interruption between ``agent/`` and ``tools/`` produces exactly that skew --
and the syntax guard reported the update as successful.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import update_cmd
import hermes_cli.update_cmd_deps as update_cmd_deps
from hermes_constants import partial_update_hint


def _write_skewed_tree(root: Path, *, skewed: bool) -> None:
    """Build a tiny two-package tree that mimics the real failure.

    ``consumer`` imports a name from ``provider`` at module level. When
    ``skewed`` is True the name is absent -- both files still parse.
    """
    (root / "provider").mkdir(parents=True, exist_ok=True)
    (root / "provider" / "__init__.py").write_text("")
    (root / "provider" / "thing.py").write_text(
        "OTHER = 1\n" if skewed else "SHARED_NAME = 'x'\nOTHER = 1\n"
    )
    (root / "consumer.py").write_text("from provider.thing import SHARED_NAME\n")


def test_syntax_guard_passes_but_import_guard_catches_skew(monkeypatch, tmp_path):
    """The regression: a skewed tree parses cleanly but cannot be imported."""
    _write_skewed_tree(tmp_path, skewed=True)

    # Both files are valid Python -- the syntax guard sees nothing wrong.
    # NOTE: patch update_cmd's global, not hermes_main's. Both modules expose
    # the name, but _validate_critical_files_syntax reads the one in its own
    # module. Patching the re-export leaves the real list in place, the stub
    # files are never looked at, and the guard returns a vacuous (True, None,
    # None) that would make this test pass no matter what the code did.
    monkeypatch.setattr(
        update_cmd, "_UPDATE_CRITICAL_FILES", ("consumer.py", "provider/thing.py")
    )
    syntax_ok, _, _ = update_cmd._validate_critical_files_syntax(tmp_path)
    assert syntax_ok, "sanity: the skewed tree must parse cleanly"

    # The import guard catches it.
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)
    assert ok is False
    assert module == "consumer"
    assert error is not None and "SHARED_NAME" in error


def test_import_guard_passes_on_consistent_tree(monkeypatch, tmp_path):
    _write_skewed_tree(tmp_path, skewed=False)
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    assert update_cmd._validate_critical_modules_import(tmp_path) == (True, None, None)


def test_import_guard_ignores_non_import_errors(monkeypatch, tmp_path):
    """A module that raises at import time for config/env reasons is not
    update breakage -- the guard must not roll back a good update."""
    (tmp_path / "consumer.py").write_text(
        "raise RuntimeError('no API key configured')\n"
    )
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, _, _ = update_cmd._validate_critical_modules_import(tmp_path)
    assert ok is True


def test_import_guard_can_report_non_import_errors(monkeypatch, tmp_path):
    """Stash restore can compare runtime failures before and after apply."""
    (tmp_path / "consumer.py").write_text("raise RuntimeError('broken config')\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(
        tmp_path, report_runtime_errors=True
    )

    assert ok is False
    assert module == "consumer"
    assert error == "broken config"


def test_import_guard_can_report_missing_third_party_dependency(
    monkeypatch, tmp_path
):
    """Stash comparison must see newly introduced missing dependencies."""
    (tmp_path / "consumer.py").write_text("import totally_not_installed_pkg\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(
        tmp_path, report_runtime_errors=True
    )

    assert ok is False
    assert module == "consumer"
    assert error is not None and "totally_not_installed_pkg" in error


def test_import_failure_comparison_preserves_exception_type(monkeypatch, tmp_path):
    source = tmp_path / "consumer.py"
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    source.write_text("raise RuntimeError('stopped')\n")
    runtime_failure = update_cmd._critical_module_import_failures(
        tmp_path, report_runtime_errors=True
    )
    source.write_text("raise SystemExit('stopped')\n")
    terminating_failure = update_cmd._critical_module_import_failures(
        tmp_path, report_runtime_errors=True
    )

    assert runtime_failure == {"consumer": ("RuntimeError", "stopped")}
    assert terminating_failure == {"consumer": ("SystemExit", "stopped")}


def test_import_guard_reports_probe_termination_when_comparing_states(
    monkeypatch, tmp_path
):
    """A terminating import is unsafe when validating a restored stash."""
    (tmp_path / "consumer.py").write_text("import os\nos._exit(7)\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(
        tmp_path, report_runtime_errors=True
    )

    assert ok is False
    assert module == "critical-module probe"
    assert error and "7" in error


def test_import_guard_reports_probe_termination_by_default(monkeypatch, tmp_path):
    """A missing health marker must not classify a terminated probe as healthy."""
    (tmp_path / "consumer.py").write_text("import os\nos._exit(9)\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"
    assert error and "9" in error


def test_import_guard_reports_system_exit_by_default(monkeypatch, tmp_path):
    """Catchable terminating imports must not complete with a healthy marker."""
    (tmp_path / "consumer.py").write_text("raise SystemExit('stopped')\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "consumer"
    assert error == "stopped"


def test_import_guard_does_not_accept_forged_static_marker(monkeypatch, tmp_path):
    """Imported stdout cannot impersonate the per-probe completion marker."""
    (tmp_path / "consumer.py").write_text(
        "import os, sys\n"
        "sys.stdout.write('__HERMES_IMPORT_HEALTH__[]')\n"
        "sys.stdout.flush()\n"
        "os._exit(7)\n"
    )
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"
    assert error and "7" in error


def test_import_guard_rejects_malformed_health_payload(monkeypatch, tmp_path):
    class Result:
        returncode = 0
        stdout = ""

    def malformed(cmd, **_kwargs):
        marker = cmd[-1].split("sys.stdout.write('\\n", 1)[1].split("'", 1)[0]
        Result.stdout = f"{marker}{{}}"
        return Result()

    monkeypatch.setattr(update_cmd_deps, "bounded_probe_run", malformed)

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"


def test_import_guard_reports_probe_timeout(monkeypatch, tmp_path):

    def timeout(*_args, **_kwargs):
        return None

    monkeypatch.setattr(update_cmd_deps, "bounded_probe_run", timeout)

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"


def test_untracked_enumeration_failure_is_visible(monkeypatch, tmp_path, capsys):
    class Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *_a, **_kw: Result())

    assert update_cmd._git_untracked_paths(["git"], tmp_path) is None




@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec-bit PermissionError")
def test_import_guard_is_non_fatal_when_probe_cannot_run(tmp_path):
    """A venv interpreter that exists but cannot be executed must not read as a hung probe:
    spawn failure stays advisory (real ``bounded_probe_run``, real ``Popen``)."""
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/bin/sh\nexit 0\n")
    venv_python.chmod(0o644)  # present, not executable -> Popen raises PermissionError
    assert update_cmd._validate_critical_modules_import(tmp_path) == (True, None, None)


# ---------------------------------------------------------------------------
# partial_update_hint
# ---------------------------------------------------------------------------



@pytest.mark.parametrize(
    "exc",
    [
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),
        ValueError("unrelated"),
        ImportError("third-party broke"),
    ],
)
def test_hint_stays_silent_for_unrelated_failures(exc):
    """Missing third-party deps and non-import errors have different
    remediation -- claiming a partial update would misdirect the user."""
    if isinstance(exc, ImportError) and not isinstance(exc, ModuleNotFoundError):
        exc.name = "requests"
    assert partial_update_hint(exc) == []


def test_import_guard_prefers_the_project_venv_interpreter(monkeypatch, tmp_path):
    """``hermes update`` can run under a different Python than the install's.

    Probing ``sys.executable`` would then validate a tree the user never
    actually runs -- the same reasoning behind ``_venv_core_imports_healthy``.
    On Windows (the platform this guard exists for) the driving interpreter
    and the venv interpreter routinely differ.
    """
    bin_dir = "Scripts" if update_cmd._m()._is_windows() else "bin"
    name = "python.exe" if update_cmd._m()._is_windows() else "python"
    venv_python = tmp_path / "venv" / bin_dir / name
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["interpreter"] = cmd[0]
        seen["cwd"] = kwargs["cwd"]

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(update_cmd_deps, "bounded_probe_run", fake_run)
    update_cmd._validate_critical_modules_import(tmp_path)

    assert seen["interpreter"] == str(venv_python)
    assert seen["cwd"] == str(tmp_path)


def test_import_guard_ignores_missing_third_party_dependency(monkeypatch, tmp_path):
    """A new third-party requirement is not a partially-updated tree.

    On the git path this guard runs BEFORE the dependency sync, so a release
    that adds a dependency would otherwise look like breakage and trigger a
    spurious `git reset --hard` rollback of a perfectly good update.
    """
    (tmp_path / "consumer.py").write_text("import totally_not_installed_pkg\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    assert update_cmd._validate_critical_modules_import(tmp_path) == (True, None, None)


def test_import_guard_rejects_module_satisfied_only_by_inherited_pythonpath(
    monkeypatch, tmp_path
):
    """A stale checkout on PYTHONPATH must not stand in for a candidate module.

    The probe child used to inherit the updater's environment wholesale, so
    with PYTHONPATH pointing at an older tree the critical module imported
    fine from there and a candidate lacking it entirely read as healthy
    (#115032).
    """
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "hermes_stale_supply.py").write_text("VALUE = 'from the stale tree'\n")

    monkeypatch.setattr(
        update_cmd, "_UPDATE_CRITICAL_MODULES", ("hermes_stale_supply",)
    )
    monkeypatch.setattr(
        update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("hermes_stale_supply",)
    )
    monkeypatch.setenv("PYTHONPATH", str(stale))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "hermes_stale_supply"
    assert error is not None and "hermes_stale_supply" in error


def test_import_guard_accepts_candidate_with_foreign_pythonpath(monkeypatch, tmp_path):
    """The env scrub must not overreach: a candidate that carries the module
    still passes while a foreign PYTHONPATH is set (#115032)."""
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "hermes_stale_supply.py").write_text("VALUE = 'stale'\n")
    (tmp_path / "hermes_stale_supply.py").write_text("VALUE = 'candidate'\n")

    monkeypatch.setattr(
        update_cmd, "_UPDATE_CRITICAL_MODULES", ("hermes_stale_supply",)
    )
    monkeypatch.setattr(
        update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("hermes_stale_supply",)
    )
    monkeypatch.setenv("PYTHONPATH", str(stale))

    assert update_cmd._validate_critical_modules_import(tmp_path) == (True, None, None)


def test_import_guard_flags_missing_first_party_module(monkeypatch, tmp_path):
    """A missing *first-party* module IS skew — the update dropped a file."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "__init__.py").write_text("")
    (tmp_path / "consumer.py").write_text("import tools.nonexistent_module\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    monkeypatch.setattr(update_cmd_deps, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)
    assert ok is False
    assert module == "consumer"
    assert error is not None and "tools.nonexistent_module" in error


@pytest.mark.parametrize("modname", ["agents", "agentops", "toolsets_x", "hermesx"])
def test_hint_does_not_claim_partial_update_for_lookalike_third_party(modname):
    """``startswith`` would match third-party ``agents``/``agentops`` and blame
    our updater for someone else's import error."""
    exc = ImportError("boom")
    exc.name = modname
    assert partial_update_hint(exc) == []


@pytest.mark.parametrize("modname", ["tools.todo_tool", "agent.context_compressor",
                                     "hermes_constants", "hermes_cli.config", "cli"])
def test_hint_fires_for_each_first_party_root(modname):
    exc = ImportError("cannot import name 'X'")
    exc.name = modname
    assert partial_update_hint(exc), f"expected guidance for {modname}"


