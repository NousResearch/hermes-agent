"""A store under a registered symlinked profile home names itself by that profile.

The home of a profile may be a symlink outside ``<root>/profiles/`` (an encrypted or
off-disk home); its store then sits at a resolved path carrying none of the
``<root>/profiles/<name>`` shape. ``_own_profile_name`` must find the name by identity,
or legacy-row validation refuses every row the profile stamped (#149776).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_state import SessionDB

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Symlinks require elevated privileges on Windows"
)


def _linked_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    offdisk = tmp_path / "offdisk"
    offdisk.mkdir()
    link = root / "profiles" / "locked"
    link.symlink_to(offdisk)
    return root, offdisk, link


def _pin_roots(monkeypatch, tmp_path: Path, root: Path) -> None:
    import hermes_constants

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)


def test_store_at_a_linked_home_names_itself_by_the_registered_profile(tmp_path, monkeypatch):
    root, offdisk, _link = _linked_layout(tmp_path)
    _pin_roots(monkeypatch, tmp_path, root)

    with SessionDB(offdisk / "state.db") as db:
        assert db._own_profile_name() == "locked"


def test_store_at_the_root_keeps_its_default_name(tmp_path, monkeypatch):
    root, _offdisk, _link = _linked_layout(tmp_path)
    _pin_roots(monkeypatch, tmp_path, root)

    with SessionDB(root / "state.db") as db:
        assert db._own_profile_name() == "default"
