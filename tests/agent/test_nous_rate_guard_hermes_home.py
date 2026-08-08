"""Tests for _state_path's HERMES_HOME awareness in agent/nous_rate_guard.py.

Regression for the ImportError fallback path which previously hardcoded
``~/.hermes`` and ignored ``HERMES_HOME``, breaking Docker deployments
(``HERMES_HOME=/opt/data``) and non-default profiles
(``HERMES_HOME=~/.hermes/profiles/<name>``).  See issue #18594 and the
narrowed extraction of issue #20633.
"""

import sys
from pathlib import Path


def _reload_rate_guard():
    """Force a fresh import so module-level state can't leak between tests."""
    sys.modules.pop("agent.nous_rate_guard", None)
    from agent import nous_rate_guard
    return nous_rate_guard


def test_state_path_respects_hermes_home_env(tmp_path, monkeypatch):
    """When HERMES_HOME is set, _state_path returns a path under it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    rg = _reload_rate_guard()
    path = Path(rg._state_path())

    assert tmp_path in path.parents
    assert path.name == rg._STATE_FILENAME
    assert path.parent.name == rg._STATE_SUBDIR


def test_state_path_respects_hermes_home_when_import_fails(tmp_path, monkeypatch):
    """If hermes_constants cannot be imported, the env var still wins."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Block the import inside _state_path's try clause.
    monkeypatch.setitem(sys.modules, "hermes_constants", None)

    rg = _reload_rate_guard()
    path = Path(rg._state_path())

    assert tmp_path in path.parents
    assert path.name == rg._STATE_FILENAME


def test_state_path_falls_back_to_home_when_env_unset_and_import_fails(monkeypatch):
    """No env var + ImportError → ~/.hermes/<subdir>/<file>."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setitem(sys.modules, "hermes_constants", None)

    rg = _reload_rate_guard()
    path = Path(rg._state_path())

    assert path == Path.home() / ".hermes" / rg._STATE_SUBDIR / rg._STATE_FILENAME


def test_state_path_ignores_blank_hermes_home(monkeypatch):
    """An empty/whitespace HERMES_HOME falls through to ~/.hermes."""
    monkeypatch.setenv("HERMES_HOME", "   ")
    monkeypatch.setitem(sys.modules, "hermes_constants", None)

    rg = _reload_rate_guard()
    path = Path(rg._state_path())

    assert path == Path.home() / ".hermes" / rg._STATE_SUBDIR / rg._STATE_FILENAME
