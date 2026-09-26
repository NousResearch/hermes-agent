"""A venv built for another CPython minor must not shadow the running one.

Windows venvs carry compiled extension modules tagged with the interpreter ABI
(``_pydantic_core.cp311-win_amd64.pyd``). Overlaying a 3.11 venv's
``site-packages`` onto a 3.14 interpreter lets the pure-Python part of a package
import while its compiled submodule vanishes, so the failure surfaces far from
its cause as ``No module named 'pydantic_core._pydantic_core'`` — which is how it
first showed up: two silent cron failures before anyone looked at the venv.

``_venv_matches_running_python`` reads the minor from ``pyvenv.cfg`` and is
deliberately fail-open: an unreadable, absent or unparsable config counts as a
match, because wrongly stripping a venv the install depends on is worse than
tolerating one that may mismatch.
"""

import sys

import pytest

from gateway.run import _venv_matches_running_python


def _venv(tmp_path, body: str | None, *, name: str = "venv"):
    """A venv dir, optionally carrying a ``pyvenv.cfg``."""
    d = tmp_path / name
    d.mkdir()
    if body is not None:
        (d / "pyvenv.cfg").write_text(body, encoding="utf-8")
    return d


def _running() -> str:
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


# --- mismatch: the case that broke production -------------------------------

def test_rejects_venv_built_for_another_minor(tmp_path):
    """The 3.11-vs-3.14 overlay that produced the pydantic_core ImportError."""
    other = "3.11" if sys.version_info[:2] != (3, 11) else "3.12"
    assert _venv_matches_running_python(
        _venv(tmp_path, f"version_info = {other}.9.final.0\n")) is False


def test_rejects_on_bare_version_key(tmp_path):
    """Some tools write ``version`` rather than ``version_info``."""
    other = "3.11" if sys.version_info[:2] != (3, 11) else "3.12"
    assert _venv_matches_running_python(
        _venv(tmp_path, f"version = {other}.9\n")) is False


# --- match: must keep working ----------------------------------------------

def test_accepts_matching_minor(tmp_path):
    assert _venv_matches_running_python(
        _venv(tmp_path, f"version_info = {_running()}.0.final.0\n")) is True


def test_accepts_matching_minor_ignoring_patch(tmp_path):
    """Patch level is ABI-compatible; only the minor decides."""
    assert _venv_matches_running_python(
        _venv(tmp_path, f"version_info = {_running()}.99\n")) is True


def test_reads_key_case_and_space_insensitively(tmp_path):
    """``pyvenv.cfg`` is written by several tools; spacing and case vary."""
    assert _venv_matches_running_python(
        _venv(tmp_path, f"  VERSION_INFO={_running()}.4\nhome = C:\\Python\n")) is True


# --- fail-open: never strip a venv on a parse failure ----------------------

@pytest.mark.parametrize("body", [
    None,                        # no pyvenv.cfg at all
    "",                          # empty file
    "home = C:\\Python314\n",    # no version key
    "version_info = \n",         # key present, value empty
    "version_info = threetwelve\n",   # unparsable
    "version_info = 3\n",        # too few components to read a minor
])
def test_fails_open_when_version_cannot_be_read(tmp_path, body):
    assert _venv_matches_running_python(_venv(tmp_path, body)) is True


def test_fails_open_when_config_is_unreadable(tmp_path, monkeypatch):
    """An OSError on read must not strip the venv the install runs on."""
    d = _venv(tmp_path, f"version_info = {_running()}.0\n")

    def boom(*_a, **_k):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.read_text", boom)
    assert _venv_matches_running_python(d) is True
