import functools
import inspect
import logging
import sys

import pytest

IS_PYPY = '__pypy__' in sys.builtin_module_names


setup_py = """\
from setuptools import setup

setup(
    name="test_logging",
    version="0.0"
)
"""


@pytest.mark.parametrize(
    ('flags', 'expected_level'), [([], "INFO"), (["--verbose"], "DEBUG")]
)
def test_verbosity_level(tmp_path, monkeypatch, flags, expected_level):
    """Make sure the correct verbosity level is set (issue #3038)"""
    import setuptools  # noqa: F401  # import setuptools to monkeypatch distutils

    import distutils  # <- load distutils after all the patches take place

    logger = logging.Logger(__name__)
    monkeypatch.setattr(logging, "root", logger)
    unset_log_level = logger.getEffectiveLevel()
    assert logging.getLevelName(unset_log_level) == "NOTSET"

    setup_script = tmp_path / "setup.py"
    setup_script.write_text(setup_py, encoding="utf-8")
    dist = distutils.core.run_setup(setup_script, stop_after="init")
    dist.script_args = flags + ["sdist"]
    dist.parse_command_line()  # <- where the log level is set
    log_level = logger.getEffectiveLevel()
    log_level_name = logging.getLevelName(log_level)
    assert log_level_name == expected_level


def flaky_on_pypy(func):
    @functools.wraps(func)
    def _func():
        try:
            func()
        except AssertionError:  # pragma: no cover
            if IS_PYPY:
                msg = "Flaky monkeypatch on PyPy (#4124)"
                pytest.xfail(f"{msg}. Original discussion in #3707, #3709.")
            raise

    return _func


@flaky_on_pypy
def test_patching_does_not_cause_problems():
    # Ensure `dist.log` is only patched if necessary

    import _distutils_hack

    import setuptools.logging

    from distutils import dist

    setuptools.logging.configure()

    if _distutils_hack.enabled():
        # Modern logging infra, no problematic patching.
        assert dist.__file__ is None or "setuptools" in dist.__file__
        assert isinstance(dist.log, logging.Logger)
    else:
        assert inspect.ismodule(dist.log)
