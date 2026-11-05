"""``store_owner_name`` answers what the SQLite backend's store-owner copy answers, over the default root's profile
tree: ``default`` for the root, a valid profile name for a folder directly under ``<root>/profiles``, else None."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import hermes_constants
from hermes_constants import get_default_hermes_root
from persistence.sqlite.conversation._store.sessions import _own_profile_name
from persistence_assembly import store_owner_name


def _backend_copy(folder):
    return _own_profile_name(SimpleNamespace(db_path=folder / "state.db"))


def test_store_owner_name_matches_the_backend_copy_over_the_profile_tree(tmp_path):
    root = get_default_hermes_root()
    cases = [
        (root, "default"),
        (root / "profiles" / "alpha", "alpha"),
        (root / "profiles" / "Not.A.Name", None),
        (root / "profiles" / "alpha" / "nested", None),
        (tmp_path / "elsewhere", None),
    ]
    for folder, expected in cases:
        folder.mkdir(parents=True, exist_ok=True)
        # CONTRACT: the assembly's rule and the backend's copy give the answer the legacy store gives.
        assert (store_owner_name(folder), _backend_copy(folder)) == (expected, expected), folder


def test_a_failing_root_lookup_answers_none_and_logs_at_debug(monkeypatch, caplog):
    folder = get_default_hermes_root()

    def no_root():
        raise RuntimeError("no default root")

    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", no_root)
    with caplog.at_level(logging.DEBUG, logger="hermes_state"):
        # CONTRACT: a failure answers None, never raises.
        assert (store_owner_name(folder), _backend_copy(folder)) == (None, None)
    messages = [record.getMessage() for record in caplog.records if record.name == "hermes_state"]
    assert messages == ["own-profile derivation failed"] * 2
