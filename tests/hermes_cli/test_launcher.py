"""Tests for the top-level `./hermes` launcher script."""

import builtins
import runpy
import shlex
import subprocess
import sys
import types
from pathlib import Path


def test_launcher_delegates_to_argparse_entrypoint(monkeypatch):
    """`./hermes` should use `hermes_cli.main`, not the legacy Fire wrapper."""
    launcher_path = Path(__file__).resolve().parents[2] / "hermes"
    called = []

    fake_main_module = types.ModuleType("hermes_cli.main")

    def fake_main():
        called.append("hermes_cli.main")

    fake_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "hermes_cli.main", fake_main_module)

    fake_cli_module = types.ModuleType("cli")

    def legacy_cli_main(*args, **kwargs):
        raise AssertionError("launcher should not import cli.main")

    fake_cli_module.main = legacy_cli_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli_module)

    fake_fire_module = types.ModuleType("fire")

    def legacy_fire(*args, **kwargs):
        raise AssertionError("launcher should not invoke fire.Fire")

    fake_fire_module.Fire = legacy_fire
    monkeypatch.setitem(sys.modules, "fire", fake_fire_module)

    monkeypatch.setattr(sys, "argv", [str(launcher_path), "gateway", "status"])

    runpy.run_path(str(launcher_path), run_name="__main__")

    assert called == ["hermes_cli.main"]


def _launcher() -> Path:
    return Path(__file__).resolve().parents[2] / "hermes"


def test_launcher_fails_actionably_when_deps_are_missing():
    """Running under an interpreter with none of Hermes' dependencies — a
    bare uv-managed CPython, or a generated .desktop Exec= whose interpreter
    escaped the install venv — must name the running interpreter and the fix,
    never a bare `ModuleNotFoundError` traceback (#92882, #90292, #91504)."""
    result = subprocess.run(
        [sys.executable, "-S", str(_launcher()), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "hermes:" in result.stderr
    assert sys.executable in result.stderr
    assert "venv" in result.stderr.lower()


def test_launcher_still_runs_cli_when_deps_are_present():
    """The import guard must not change the healthy path: with dependencies
    installed, `./hermes --help` still exits 0 and prints usage."""
    result = subprocess.run(
        [sys.executable, str(_launcher()), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0
    assert "usage" in (result.stdout + result.stderr).lower()


def _launcher_namespace() -> dict:
    """The launcher's module namespace without running its `__main__` block."""
    return runpy.run_path(str(_launcher()))


def _import_failure_from(tmp_path: Path, rel_path: str, code: str):
    """`exec` `code` from a temp file so the raised `ModuleNotFoundError`'s
    innermost traceback frame is that file — mirroring a real import
    statement living there."""
    file = tmp_path / rel_path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(code)
    try:
        exec(compile(code, str(file), "exec"), {"__builtins__": builtins.__dict__})
    except ModuleNotFoundError as exc:
        return exc
    raise AssertionError(f"expected {code!r} to raise ModuleNotFoundError")


def test_classify_flags_hermes_cli_internal_import_as_bug(tmp_path):
    """A `ModuleNotFoundError` for a `hermes_cli.*` submodule raised from a
    file *inside* the package is an internal bug — the package failing to
    import its own code — not a missing dependency."""
    classify = _launcher_namespace()["_classify_import_failure"]
    pkg_dir = tmp_path / "hermes_cli"

    exc = _import_failure_from(
        tmp_path,
        "hermes_cli/broken.py",
        "import hermes_cli.definitely_missing_submodule",
    )

    missing, internal = classify(exc, pkg_dir=pkg_dir)
    assert missing == "hermes_cli.definitely_missing_submodule"
    assert internal is True


def test_classify_reports_missing_third_party_dep(tmp_path):
    """A missing external module (yaml, dotenv, ...) is a missing dependency
    even when the failing import statement sits inside hermes_cli."""
    classify = _launcher_namespace()["_classify_import_failure"]
    pkg_dir = tmp_path / "hermes_cli"

    exc = _import_failure_from(
        tmp_path,
        "hermes_cli/config.py",
        "import definitely_missing_dependency",
    )

    missing, internal = classify(exc, pkg_dir=pkg_dir)
    assert missing == "definitely_missing_dependency"
    assert internal is False


def test_classify_hermes_cli_name_from_outside_is_not_internal(tmp_path):
    """The same `hermes_cli.*`-named failure raised from *outside* the
    package (e.g. the launcher itself) is not an internal bug."""
    classify = _launcher_namespace()["_classify_import_failure"]
    pkg_dir = tmp_path / "hermes_cli"

    exc = _import_failure_from(
        tmp_path,
        "outside.py",
        "import hermes_cli.definitely_missing_submodule",
    )

    missing, internal = classify(exc, pkg_dir=pkg_dir)
    assert missing == "hermes_cli.definitely_missing_submodule"
    assert internal is False


def test_classify_attributes_re_raised_import_to_cause_chain(tmp_path):
    """An import failure re-raised via `raise ... from err` keeps the original
    import site on `__cause__`. Classification must use the deepest traceback
    in that chain — here a frame inside hermes_cli — not the innermost frame
    of the re-raise site (which lives outside the package and would wrongly
    demote an internal bug to a missing dependency)."""
    classify = _launcher_namespace()["_classify_import_failure"]
    pkg_dir = tmp_path / "hermes_cli"

    inner = tmp_path / "hermes_cli" / "broken.py"
    inner.parent.mkdir(parents=True, exist_ok=True)
    inner.write_text("import hermes_cli.definitely_missing_submodule\n")

    outer = tmp_path / "re_raise_site.py"  # deliberately outside the package
    outer_code = (
        "import builtins\n"
        "try:\n"
        f"    exec(compile({inner.read_text()!r}, {str(inner)!r}, 'exec'),"
        " {'__builtins__': builtins.__dict__})\n"
        "except ModuleNotFoundError as err:\n"
        "    raise ModuleNotFoundError(err.name) from err\n"
    )
    try:
        exec(
            compile(outer_code, str(outer), "exec"),
            {"__builtins__": builtins.__dict__},
        )
    except ModuleNotFoundError as exc:
        wrapped = exc
    else:
        raise AssertionError("expected a re-raised ModuleNotFoundError")

    assert wrapped.__cause__ is not None
    missing, internal = classify(wrapped, pkg_dir=pkg_dir)
    assert missing == "hermes_cli.definitely_missing_submodule"
    assert internal is True


def test_classify_normalizes_unnamed_module_error_to_module_token():
    """When `exc.name` is empty, the raw `No module named 'x'` message must be
    normalized to just the module token so the `missing dependency:` report
    field stays machine-greppable."""
    classify = _launcher_namespace()["_classify_import_failure"]

    exc = ModuleNotFoundError("No module named 'definitely_missing_dependency'")
    assert exc.name is None

    missing, internal = classify(exc)
    assert missing == "definitely_missing_dependency"
    assert internal is False


def test_report_launcher_failure_missing_dependency_field_is_bare_token(capsys):
    """The `missing dependency:` field must carry just the module token, never
    the whole `No module named 'x'` sentence."""
    ns = _launcher_namespace()
    globs = ns["_report_launcher_failure"].__globals__
    globs["_find_install_venv_python"] = lambda checkout: None

    exc = ModuleNotFoundError("No module named 'definitely_missing_dependency'")
    missing, internal = ns["_classify_import_failure"](exc)
    ns["_report_launcher_failure"](missing, internal=internal)

    err = capsys.readouterr().err
    assert "missing dependency:  definitely_missing_dependency" in err
    assert "missing dependency:  No module named" not in err


def test_report_launcher_failure_fix_command_is_shell_quoted(monkeypatch, capsys):
    """argv entries containing spaces must be shell-quoted in the suggested
    fix so the printed command copy-pastes correctly (`--model "a b"` must
    not flatten into `--model a b`)."""
    ns = _launcher_namespace()
    ns["_report_launcher_failure"].__globals__["_find_install_venv_python"] = (
        lambda checkout: "/opt/hermes/venv/bin/python"
    )
    monkeypatch.setattr(
        sys, "argv", [str(_launcher()), "gateway", "run", "--model", "a b"]
    )

    ns["_report_launcher_failure"]("yaml")

    err = capsys.readouterr().err
    fix_lines = [line for line in err.splitlines() if line.startswith("  fix:")]
    assert len(fix_lines) == 1
    command = fix_lines[0][len("  fix:") :].strip()
    expected_prefix = f"/opt/hermes/venv/bin/python {_launcher()} "
    assert command.startswith(expected_prefix)
    assert command[len(expected_prefix) :] == shlex.join([
        "gateway",
        "run",
        "--model",
        "a b",
    ])
    # The argv portion must round-trip through a POSIX shell: 'a b' stays one
    # argument, not two.
    assert shlex.split(command[len(expected_prefix) :]) == [
        "gateway",
        "run",
        "--model",
        "a b",
    ]


def test_report_launcher_failure_distinguishes_internal_bug(capsys):
    """The internal-bug report must say 'internal error', never claim a
    missing dependency or suggest the venv fix."""
    report = _launcher_namespace()["_report_launcher_failure"]
    exc = ModuleNotFoundError("No module named 'hermes_cli.typo'")
    exc.name = "hermes_cli.typo"

    report("hermes_cli.typo", internal=True, exc=exc)

    err = capsys.readouterr().err
    assert "hermes_cli.typo" in err
    assert "internal error" in err.lower()
    assert "missing dependency" not in err.lower()


def test_launcher_flags_internal_import_bug_not_missing_deps(tmp_path):
    """End-to-end: a checkout whose hermes_cli imports its own missing
    submodule must exit with an 'internal error' diagnostic and the real
    traceback — it must never be misreported as a missing dependency."""
    checkout = tmp_path / "checkout"
    pkg = checkout / "hermes_cli"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text(
        "import hermes_cli.definitely_missing_submodule\ndef main():\n    pass\n"
    )
    script = checkout / "hermes"
    script.write_text(_launcher().read_text())

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "internal error" in result.stderr.lower()
    assert "missing dependency" not in result.stderr.lower()
    assert "hermes_cli.definitely_missing_submodule" in result.stderr
