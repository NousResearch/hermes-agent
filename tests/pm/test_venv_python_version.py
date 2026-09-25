"""A venv's Python version must be readable whatever wrote its ``pyvenv.cfg``.

``uv venv`` (every Hermes-managed environment) records ``version_info``; ``venv``/``virtualenv``
record ``version``. Reading only one spelling made uv venvs unidentifiable, so callers fell back
to the *calling* interpreter's version and composed a site-packages path the venv never had.
"""

from pathlib import Path

import pytest

from pm.environments import venv_python_version


def _venv(tmp_path: Path, cfg: str | None) -> Path:
    venv = tmp_path / "venv"
    venv.mkdir(exist_ok=True)
    if cfg is not None:
        (venv / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    return venv


def test_uv_style_version_info_is_read(tmp_path):
    """Real ``uv venv`` output from a Hermes install."""
    venv = _venv(
        tmp_path,
        "home = C:\\Users\\x\\AppData\\Local\\hermes\\tools\\python-3.14.7-win32-x64\n"
        "implementation = CPython\n"
        "uv = 0.12.3\n"
        "version_info = 3.14.7\n"
        "include-system-site-packages = false\n"
        "relocatable = true\n",
    )

    assert venv_python_version(venv) == (3, 14)


def test_virtualenv_style_version_is_read(tmp_path):
    venv = _venv(tmp_path, "home = /usr/bin\nversion = 3.11.9\n")

    assert venv_python_version(venv) == (3, 11)


def test_missing_marker_returns_none(tmp_path):
    assert venv_python_version(_venv(tmp_path, None)) is None


def test_unparsable_value_returns_none(tmp_path):
    assert venv_python_version(_venv(tmp_path, "version_info = unknown\n")) is None


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX path layout only")
def test_site_packages_uses_the_venvs_own_version(tmp_path):
    """The regression the helper exists for: not the caller's interpreter version."""
    venv = _venv(tmp_path, "version_info = 3.11.9\n")
    (venv / "lib" / "python3.11").mkdir(parents=True)

    from pm.environments import site_packages

    assert site_packages(venv) == venv / "lib" / "python3.11" / "site-packages"
