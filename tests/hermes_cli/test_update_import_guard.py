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

from pathlib import Path

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd
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
    syntax_ok, _, _ = hermes_main._validate_critical_files_syntax(tmp_path)
    assert syntax_ok, "sanity: the skewed tree must parse cleanly"

    # The import guard catches it.
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
    ok, module, error = hermes_main._validate_critical_modules_import(tmp_path)
    assert ok is False
    assert module == "consumer"
    assert error is not None and "SHARED_NAME" in error


def test_import_guard_passes_on_consistent_tree(monkeypatch, tmp_path):
    _write_skewed_tree(tmp_path, skewed=False)
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    assert hermes_main._validate_critical_modules_import(tmp_path) == (True, None, None)


def test_import_guard_ignores_non_import_errors(monkeypatch, tmp_path):
    """A module that raises at import time for config/env reasons is not
    update breakage -- the guard must not roll back a good update."""
    (tmp_path / "consumer.py").write_text(
        "raise RuntimeError('no API key configured')\n"
    )
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, _, _ = hermes_main._validate_critical_modules_import(tmp_path)
    assert ok is True


def test_import_guard_can_report_non_import_errors(monkeypatch, tmp_path):
    """Stash restore can compare runtime failures before and after apply."""
    (tmp_path / "consumer.py").write_text("raise RuntimeError('broken config')\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = hermes_main._validate_critical_modules_import(
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

    ok, module, error = hermes_main._validate_critical_modules_import(
        tmp_path, report_runtime_errors=True
    )

    assert ok is False
    assert module == "consumer"
    assert error is not None and "totally_not_installed_pkg" in error


def test_import_failure_comparison_preserves_exception_type(monkeypatch, tmp_path):
    source = tmp_path / "consumer.py"
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
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

    ok, module, error = hermes_main._validate_critical_modules_import(
        tmp_path, report_runtime_errors=True
    )

    assert ok is False
    assert module == "critical-module probe"
    assert error == "terminated before reporting import health (exit code 7)"


def test_import_guard_reports_probe_termination_by_default(monkeypatch, tmp_path):
    """A missing health marker must not classify a terminated probe as healthy."""
    (tmp_path / "consumer.py").write_text("import os\nos._exit(9)\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = hermes_main._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"
    assert error == "terminated before reporting import health (exit code 9)"


def test_import_guard_reports_system_exit_by_default(monkeypatch, tmp_path):
    """Catchable terminating imports must not complete with a healthy marker."""
    (tmp_path / "consumer.py").write_text("raise SystemExit('stopped')\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = hermes_main._validate_critical_modules_import(tmp_path)

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

    ok, module, error = hermes_main._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"
    assert error == "terminated before reporting import health (exit code 7)"


def test_import_guard_rejects_malformed_health_payload(monkeypatch, tmp_path):
    class Result:
        returncode = 0
        stdout = ""

    def malformed(cmd, **_kwargs):
        marker = cmd[-1].split("sys.stdout.write('\\n", 1)[1].split("'", 1)[0]
        Result.stdout = f"{marker}{{}}"
        return Result()

    monkeypatch.setattr(update_cmd.subprocess, "run", malformed)

    ok, module, error = hermes_main._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"
    assert error == "reported malformed import health data"


def test_import_guard_reports_probe_timeout(monkeypatch, tmp_path):
    import subprocess

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["python", "-c", "probe"], 120)

    monkeypatch.setattr(update_cmd.subprocess, "run", timeout)

    ok, module, error = hermes_main._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "critical-module probe"
    assert error == "timed out before reporting import health"


def test_untracked_enumeration_failure_is_visible(monkeypatch, tmp_path, capsys):
    class Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *_a, **_kw: Result())

    assert update_cmd._git_untracked_paths(["git"], tmp_path) is None
    assert "Could not enumerate untracked files" in capsys.readouterr().out


def test_import_guard_is_non_fatal_when_probe_cannot_run(monkeypatch, tmp_path):
    """If we can't spawn the probe, don't block the user's update."""

    def boom(*_a, **_kw):
        raise OSError("cannot spawn")

    monkeypatch.setattr(update_cmd.subprocess, "run", boom)
    assert update_cmd._validate_critical_modules_import(tmp_path) == (True, None, None)


# ---------------------------------------------------------------------------
# partial_update_hint
# ---------------------------------------------------------------------------

def test_hint_fires_for_first_party_import_error():
    exc = ImportError("cannot import name 'TODO_INJECTION_HEADER'")
    exc.name = "tools.todo_tool"

    hint = partial_update_hint(exc)

    assert hint, "expected recovery guidance for a first-party ImportError"
    assert any("hermes update" in line for line in hint)


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

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)
    update_cmd._validate_critical_modules_import(tmp_path)

    assert seen["interpreter"] == str(venv_python)


def test_import_guard_ignores_missing_third_party_dependency(monkeypatch, tmp_path):
    """A new third-party requirement is not a partially-updated tree.

    On the git path this guard runs BEFORE the dependency sync, so a release
    that adds a dependency would otherwise look like breakage and trigger a
    spurious `git reset --hard` rollback of a perfectly good update.
    """
    (tmp_path / "consumer.py").write_text("import totally_not_installed_pkg\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    assert update_cmd._validate_critical_modules_import(tmp_path) == (True, None, None)


def test_import_guard_flags_missing_first_party_module(monkeypatch, tmp_path):
    """A missing *first-party* module IS skew — the update dropped a file."""
    (tmp_path / "consumer.py").write_text("import tools.nonexistent_module\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

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
                                     "hermes_constants", "cli"])
def test_hint_fires_for_each_first_party_root(modname):
    exc = ImportError("cannot import name 'X'")
    exc.name = modname
    assert partial_update_hint(exc), f"expected guidance for {modname}"


def test_probe_and_hint_share_one_first_party_definition():
    """The guard that BLOCKS and the hint that EXPLAINS must never disagree.

    These started as two hand-maintained lists and immediately diverged:
    `cli` was first-party to the hint but not the probe, and `hermesx`
    (third-party) matched the probe's loose `startswith("hermes")`. A user
    could get a rollback with no explanation, or an explanation with no
    detection. Both now derive from FIRST_PARTY_MODULE_ROOTS; this test
    fails if either grows a private copy.
    """
    from hermes_constants import FIRST_PARTY_MODULE_ROOTS, is_first_party_module

    captured = {}

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def capture(cmd, **_kw):
        captured["probe"] = cmd[-1]
        return _Result()

    real_run = update_cmd.subprocess.run
    update_cmd.subprocess.run = capture
    try:
        update_cmd._validate_critical_modules_import("/tmp")
    finally:
        update_cmd.subprocess.run = real_run

    probe_src = captured["probe"]
    # Every first-party root must appear in the probe's injected tuple.
    for root in FIRST_PARTY_MODULE_ROOTS:
        assert repr(root) in probe_src, f"{root} missing from the probe's root set"
    # And the hint must agree on each of them.
    for root in FIRST_PARTY_MODULE_ROOTS:
        assert is_first_party_module(f"{root}.anything")
    # Lookalikes stay out of both.
    for lookalike in ("agents", "agentops", "toolsets_x", "hermesx"):
        assert not is_first_party_module(lookalike)
