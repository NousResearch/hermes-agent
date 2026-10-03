"""Regression tests for #127387: legacy venv shadows rpds.

On an install that still has a legacy in-tree ``venv`` built for an older
interpreter, ``_ensure_windows_gateway_venv_imports`` injected that venv's
``Lib/site-packages`` unconditionally. A ``rpds-py`` wheel built for the old
ABI ships only a ``cp311`` extension, so under the running runtime the
pure-Python ``rpds`` layer imports fine and the compiled part fails with
``No module named 'rpds.rpds'`` -- breaking every MCP tool call that
validates against a declared ``outputSchema``.

The guard: a venv whose ``pyvenv.cfg`` declares a major.minor different
from the running interpreter must never be injected. Unknown/unparseable
versions stay allowed (legacy behavior for exotic layouts).
"""
import os
import sys

import pytest


def _make_venv(tmp_path, name, version):
    venv_dir = tmp_path / name
    (venv_dir / "Lib" / "site-packages").mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text(
        "home = C:\\Python\ninclude-system-site-packages = false\n"
        "version = %s\n" % version,
        encoding="utf-8",
    )
    return venv_dir


def _call_ensure(monkeypatch, venv_dir):
    from gateway.run import _ensure_windows_gateway_venv_imports

    # No committed PM generation here: this test drives the leftover-venv path.
    try:
        import pm.environments as pm_environments
    except ImportError:
        pm_environments = None
    if pm_environments is not None:
        monkeypatch.setattr(
            pm_environments, "committed_venv", lambda _root: None, raising=False
        )

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("VIRTUAL_ENV", str(venv_dir))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    before = [os.path.normcase(p) for p in sys.path]
    _ensure_windows_gateway_venv_imports()
    after = [os.path.normcase(p) for p in sys.path]
    added = [p for p in after if p not in before]
    venv_root = os.path.normcase(str(venv_dir.resolve()))
    return added, venv_root


def test_mismatched_venv_is_not_injected(tmp_path, monkeypatch):
    """A venv built for another interpreter must not land on sys.path."""
    other = "%d.%d" % (sys.version_info.major, sys.version_info.minor + 1)
    venv_dir = _make_venv(tmp_path, "legacy-venv", other)
    added, venv_root = _call_ensure(monkeypatch, venv_dir)
    assert not any(p.startswith(venv_root) for p in added), (
        "py%s venv site-packages must not shadow the running py%d.%d runtime"
        % (other, sys.version_info.major, sys.version_info.minor)
    )
    pythonpath = os.environ.get("PYTHONPATH", "")
    entries = pythonpath.split(os.pathsep) if pythonpath else []
    assert not any(
        os.path.normcase(e).startswith(venv_root) for e in entries
    ), "mismatched venv must not be exported via PYTHONPATH either"


def test_matching_venv_still_injected(tmp_path, monkeypatch):
    """The guard must not break the intended path: same-version venv works."""
    same = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
    venv_dir = _make_venv(tmp_path, "current-venv", same)
    added, venv_root = _call_ensure(monkeypatch, venv_dir)
    assert any(p.startswith(venv_root) for p in added), (
        "same-interpreter venv must still be injected"
    )


def test_unversioned_venv_stays_allowed(tmp_path, monkeypatch):
    """No pyvenv.cfg (unknown version) keeps legacy allow behavior."""
    venv_dir = tmp_path / "odd-venv"
    (venv_dir / "Lib" / "site-packages").mkdir(parents=True)
    added, venv_root = _call_ensure(monkeypatch, venv_dir)
    assert any(p.startswith(venv_root) for p in added), (
        "venv without version info must keep legacy behavior"
    )
