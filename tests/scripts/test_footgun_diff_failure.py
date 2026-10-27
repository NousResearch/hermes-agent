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
# main() level — the empty-input receipt and the non-empty success path
# ---------------------------------------------------------------------------


def test_nothing_staged_stays_clean_but_distinguishable(linter, monkeypatch, capsys):
    # "Nothing staged" is the same class of input as an empty --diff range:
    # a legitimate empty input, not a failed computation, so it must stay
    # exit 0 with a distinguishable success line rather than exit 2 (which
    # the docstring reserves for "--diff <ref> could not be computed").
    monkeypatch.setattr(linter, "get_staged_files", lambda: [])
    assert linter.main([]) == 0
    captured = capsys.readouterr()
    assert "nothing staged (0 files scanned)" in captured.out
    assert "Pass --all" in captured.out


def test_nonempty_diff_range_still_reports_hits(linter, monkeypatch, capsys):
    # The empty-range early return must not swallow the success path: a
    # non-empty diff that contains a real hit still reports it and exits 1.
    hit = REPO_ROOT / "_tmp_footgun_hit_125997.py"
    hit.write_text("with open('a.txt', 'r') as fh:\n    pass\n", encoding="utf-8")
    try:
        monkeypatch.setattr(linter, "get_diff_files", lambda ref: [hit])
        assert linter.main(["--diff", "base"]) == 1
    finally:
        hit.unlink(missing_ok=True)
    captured = capsys.readouterr()
    assert "open() without encoding=" in captured.out
    assert "1 Windows footgun(s) found" in captured.err


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
