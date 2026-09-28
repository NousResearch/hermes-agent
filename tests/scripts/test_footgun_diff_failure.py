"""``--diff <ref>`` failures must not masquerade as clean scans.

``scripts/check-windows-footguns.py --diff <ref>`` used to swallow every
``git diff`` failure (unresolvable ref, no merge base in a shallow
checkout, missing git) into an empty file list, printing the success line
and exiting 0 — a false-clean receipt indistinguishable from an empty
range. See #125997.

The end-to-end cases run the linter's ``__main__`` path against this
repository's own git checkout (driven via runpy so the exit-status
contract itself is exercised); the git-failure variants drive
``get_diff_files`` with a stubbed ``check_output``.
"""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LINTER_PATH = REPO_ROOT / "scripts" / "check-windows-footguns.py"


def _load_linter_module():
    spec = importlib.util.spec_from_file_location("check_windows_footguns", LINTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_windows_footguns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def linter():
    return _load_linter_module()


def _run_main(argv: list[str]) -> tuple[object, str, str]:
    """Execute the linter's ``__main__`` path in-process.

    Returns (SystemExit.code, stdout, stderr) so tests assert on the
    same exit-status contract a shell caller sees.
    """
    import io

    out, err = io.StringIO(), io.StringIO()
    old_argv, old_out, old_err = sys.argv, sys.stdout, sys.stderr
    sys.argv = ["check-windows-footguns.py", *argv]
    sys.stdout, sys.stderr = out, err
    try:
        code = runpy.run_path(str(LINTER_PATH), run_name="__main__")
        raise AssertionError("linter __main__ did not exit")
    except SystemExit as e:
        exit_code = e.code
    finally:
        sys.argv, sys.stdout, sys.stderr = old_argv, old_out, old_err
    return exit_code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# End-to-end — real git, this checkout
# ---------------------------------------------------------------------------


def test_unresolvable_ref_is_an_error_not_a_clean_scan():
    code, out, err = _run_main(["--diff", "no-such-ref-125997"])
    assert code == 2
    assert "no-such-ref-125997" in err
    assert "No Windows footguns found" not in out


def test_legitimately_empty_range_stays_clean_but_distinguishable():
    code, out, err = _run_main(["--diff", "HEAD"])
    assert code == 0
    assert "nothing changed vs HEAD" in out


# ---------------------------------------------------------------------------
# get_diff_files level — stubbed git failures
# ---------------------------------------------------------------------------


def _stub_check_output(monkeypatch, *, error):
    def fake_check_output(argv, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)


def test_no_merge_base_error_names_the_remedy(linter, monkeypatch):
    exc = subprocess.CalledProcessError(
        returncode=128,
        cmd="git diff",
        stderr="fatal: main...HEAD: no merge base\n",
    )
    _stub_check_output(monkeypatch, error=exc)
    with pytest.raises(linter.DiffRefError) as excinfo:
        linter.get_diff_files("main")
    assert "no merge base" in str(excinfo.value)
    assert "--unshallow" in str(excinfo.value)


def test_other_git_failure_surfaces_stderr(linter, monkeypatch):
    exc = subprocess.CalledProcessError(
        returncode=128,
        cmd="git diff",
        stderr="fatal: ambiguous argument 'nope...HEAD': unknown revision\n",
    )
    _stub_check_output(monkeypatch, error=exc)
    with pytest.raises(linter.DiffRefError) as excinfo:
        linter.get_diff_files("nope")
    assert "unknown revision" in str(excinfo.value)


def test_missing_git_binary_is_an_error(linter, monkeypatch):
    _stub_check_output(monkeypatch, error=FileNotFoundError("git"))
    with pytest.raises(linter.DiffRefError) as excinfo:
        linter.get_diff_files("main")
    assert "git executable not found" in str(excinfo.value)


def test_main_maps_diff_error_to_exit_code_2(linter, monkeypatch, capsys):
    exc = subprocess.CalledProcessError(
        returncode=128,
        cmd="git diff",
        stderr="fatal: main...HEAD: no merge base\n",
    )
    _stub_check_output(monkeypatch, error=exc)
    assert linter.main(["--diff", "main"]) == 2
    captured = capsys.readouterr()
    assert "--unshallow" in captured.err
    assert captured.out == ""
