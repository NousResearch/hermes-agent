"""A test run must not borrow the live checkout's install state (PER-590).

The suite runs with an isolation ``HERMES_HOME``. A deployed checkout carries an install stamp,
so PM resolves its store through ``owning_home_root``; when another data root installed that
checkout -- the default install running its own suite -- the lookup reaches the REAL home's
``installs/<key>/facts.json``, launcher and store interpreter. A test process is not a borrowing
launch (there is no per-task home to serve), so the lookup keeps to the active home and the
real-home guard stays honest. Copies under ``tmp_path`` keep the real contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pm import environments
from pm.environments import install_key, owning_home_root, store_root

pytestmark = pytest.mark.platforms("any")


def _checkout(parent: Path) -> Path:
    root = parent / "hermes-agent"
    root.mkdir(parents=True)
    (root / "install-stamp.json").write_text(
        json.dumps({"updateMechanism": "self", "commit": "abc"}), encoding="utf-8")
    return root


def _state(data_root: Path, checkout: Path) -> Path:
    facts = data_root / "installs" / install_key(checkout) / "facts.json"
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text("{}", encoding="utf-8")
    return facts


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """The ``install.sh`` layout, with this module's live checkout aimed at a temp copy."""
    import hermes_constants

    data = tmp_path / ".hermes"
    root = _checkout(data)
    _state(data, root)
    monkeypatch.setattr(environments, "_LIVE_CHECKOUT", root)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: data)
    borrower = tmp_path / "borrower"
    monkeypatch.setenv("HERMES_HOME", str(borrower))
    monkeypatch.setenv("HERMES_TEST_ISOLATION", str(borrower))
    monkeypatch.delenv("HERMES_TEST_ALLOW_LIVE_INSTALL", raising=False)
    return root, data, borrower


def test_the_live_checkout_is_not_borrowed_under_pytest(layout):
    root, _data, borrower = layout
    assert owning_home_root(root) is None
    assert store_root(root) == borrower / "tools"


def test_the_opt_in_restores_the_borrowing_contract(layout, monkeypatch):
    root, data, _borrower = layout
    monkeypatch.setenv("HERMES_TEST_ALLOW_LIVE_INSTALL", "1")
    assert owning_home_root(root) == data
    assert store_root(root) == data / "tools"


def test_a_copy_outside_the_live_checkout_still_borrows(tmp_path, monkeypatch):
    """The gate names the executing checkout; a copy a test launched borrows as in production."""
    import hermes_constants

    data = tmp_path / ".hermes"
    copy = _checkout(data)
    _state(data, copy)
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: data)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "borrower"))
    assert copy.resolve() != Path(environments._LIVE_CHECKOUT).resolve()
    assert owning_home_root(copy) == data


def test_a_product_launch_still_borrows(layout, monkeypatch):
    """No test marker: the borrowing launch keeps today's behaviour."""
    root, data, _borrower = layout
    for name in environments._TEST_PROCESS_ENV:
        monkeypatch.setenv(name, "")
    assert owning_home_root(root) == data
    assert store_root(root) == data / "tools"
