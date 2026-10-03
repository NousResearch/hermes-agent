"""The updater must not silently run against a dev clone while the live install sits still.

``hermes update`` operates on whatever checkout imports ``hermes_cli``. A dev-clone venv on PATH
(direnv + the repo's tracked ``.envrc`` ``use flake`` exports ``VIRTUAL_ENV``) makes bare ``hermes``
resolve to the dev clone, so the update runs there and prints a success line carrying a sha — while
the live install silently stays on the old commit (observed 2026-09-26: the live install sat one
commit behind after two runs reported success).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.update_cmd_common import foreign_install_shadow_warning


def _checkout(root: Path) -> Path:
    """A directory shaped like a real Hermes checkout (carries the marker files)."""
    (root / "hermes_cli").mkdir(parents=True, exist_ok=True)
    (root / "hermes_cli" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname = \"hermes-agent\"\n", encoding="utf-8")
    return root


@pytest.fixture
def layouts(tmp_path, monkeypatch):
    """A dev clone + a live layout, with HERMES_REAL_HOME pointed at tmp."""
    dev = _checkout(tmp_path / "repos" / "hermes-agent")
    live = _checkout(tmp_path / "home" / ".hermes" / "hermes-agent")
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path / "home"))
    return dev, live


def _leave_pytest(monkeypatch) -> None:
    """Drop the pytest guard the way a real invocation does.

    Must run in the TEST BODY, not a fixture: pytest re-sets ``PYTEST_CURRENT_TEST`` for the call
    phase after setup finishes, so a fixture-time delete is overwritten before the helper reads it.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert "PYTEST_CURRENT_TEST" not in os.environ


def test_dev_clone_warns_and_names_the_live_install(layouts, monkeypatch):
    dev, live = layouts
    _leave_pytest(monkeypatch)

    warning = foreign_install_shadow_warning(dev)

    assert warning is not None
    assert str(dev) in warning
    assert str(live) in warning
    # The remedy must be copy-pasteable and bypass the shadowing venv.
    assert "env -u VIRTUAL_ENV" in warning
    assert f"{live}/venv/bin/hermes update" in warning


def test_live_install_does_not_warn(layouts, monkeypatch):
    _dev, live = layouts
    _leave_pytest(monkeypatch)

    assert foreign_install_shadow_warning(live) is None


def test_no_live_layout_is_silent(tmp_path, monkeypatch):
    """A corp-style install (launcher straight at ~/repos/hermes-agent) must not be nagged."""
    dev = _checkout(tmp_path / "repos" / "hermes-agent")
    monkeypatch.setenv("HERMES_REAL_HOME", str(tmp_path / "empty-home"))
    _leave_pytest(monkeypatch)

    assert foreign_install_shadow_warning(dev) is None


def test_non_checkout_directory_is_silent(tmp_path, layouts, monkeypatch):
    """A directory that merely shares the name must not be claimed as a shadowing install."""
    _dev, _live = layouts
    _leave_pytest(monkeypatch)
    impostor = tmp_path / "not-a-checkout"
    impostor.mkdir()

    assert foreign_install_shadow_warning(impostor) is None


def test_never_warns_under_pytest(layouts, monkeypatch):
    """A suite running in the dev clone is normal — the guard must not fire during tests."""
    dev, _live = layouts
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/x.py::test_y (call)")

    assert foreign_install_shadow_warning(dev) is None
