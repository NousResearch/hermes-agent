"""Regression tests for update-failure stage attribution (issue #85840).

The whole ``hermes update`` runs under one ``try``: a failure in the
Python-dependency install (``uv/pip install -e .``) reached the handler as a
bare ``CalledProcessError`` and was printed as "Git update failed" even though
the git pull had already succeeded — sending users to diagnose git when the
real cause was the deps step (e.g. the ``cryptography._rust.pyd`` self-lock of
#83569). ``_describe_update_failure`` classifies the failing command instead of
assuming git, and surfaces any captured stderr/stdout tail.
"""

import subprocess

from hermes_cli.update_cmd import _describe_update_failure


def _err(cmd, *, stderr=None, output=None, code=2):
    return subprocess.CalledProcessError(code, cmd, output=output, stderr=stderr)


def test_uv_pip_install_is_python_deps_not_git():
    label, _ = _describe_update_failure(
        _err(["C:\\hermes\\bin\\uv.exe", "pip", "install", "-e", "."])
    )
    assert label == "Python dependency install failed"


def test_venv_pip_module_install_is_python_deps():
    label, _ = _describe_update_failure(
        _err(["/venv/bin/python", "-m", "pip", "install", "-e", ".[all]"])
    )
    assert label == "Python dependency install failed"


def test_git_command_is_git_update():
    label, _ = _describe_update_failure(_err(["git", "pull", "--ff-only"]))
    assert label == "Git update failed"


def test_git_exe_basename_is_git_update():
    label, _ = _describe_update_failure(
        _err(["C:\\Program Files\\Git\\cmd\\git.exe", "reset", "--hard"])
    )
    assert label == "Git update failed"


def test_unknown_command_is_generic_step():
    label, _ = _describe_update_failure(_err(["node", "build.js"]))
    assert label == "Update step failed"


def test_stderr_tail_is_surfaced():
    tail = "\n".join(f"line {i}" for i in range(1, 20))
    _, detail = _describe_update_failure(
        _err(["uv", "pip", "install", "-e", "."], stderr=tail)
    )
    # Only the last 12 lines are kept, and the earliest surviving one is line 8.
    assert "line 19" in detail
    assert "line 8" in detail
    assert "line 7" not in detail


def test_bytes_stderr_is_decoded():
    _, detail = _describe_update_failure(
        _err(["uv", "pip", "install", "-e", "."], stderr=b"os error 5: Access is denied")
    )
    assert "Access is denied" in detail


def test_output_used_when_stderr_absent():
    _, detail = _describe_update_failure(
        _err(["uv", "pip", "install", "-e", "."], output="fallback stdout tail")
    )
    assert detail == "fallback stdout tail"


def test_no_captured_output_yields_empty_detail():
    _, detail = _describe_update_failure(_err(["uv", "pip", "install", "-e", "."]))
    assert detail == ""
