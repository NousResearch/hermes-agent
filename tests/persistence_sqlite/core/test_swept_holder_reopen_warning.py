"""A read or write through a live handle after the shutdown sweep closed its store logs the reopen warning main's
store logged, under the ``hermes_state`` logger, once per swept store, and answers as before. The parity case runs
main's store through the same sweep and compares the warnings."""

from __future__ import annotations

import logging

import pytest

from persistence_assembly import close_all_stores, open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_scope
from tests.persistence_sqlite.conftest import profile_home

_WARNING = ("state.db connection for {} was closed while a {} was still in flight — reopening "
            "(teardown/worker race, #94736)")


def _home(name: str, mode: str):
    home = profile_home(name)
    (home / "config.yaml").write_text(f"database:\n  journal_mode: {mode}\n")
    return home


def _reopen_warnings(caplog, home) -> list[str]:
    return [r.getMessage().replace(str(home), "<home>") for r in caplog.records
            if r.name == "hermes_state" and "reopening" in r.getMessage()]


@pytest.mark.parametrize("mode", ["wal", "delete"])
def test_a_read_after_the_sweep_logs_the_reopen_once(mode, caplog):
    home = _home("alpha", mode)
    store = open_persistence(profile_scope(home=home))
    assert store.sessions.get("nope") is None
    close_all_stores()

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        assert store.sessions.get("nope") is None
        assert open_persistence(profile_scope(home=home)).sessions.get("nope") is None
    assert _reopen_warnings(caplog, home) == [_WARNING.format("<home>/state.db", "read")]


@pytest.mark.parametrize("mode", ["wal", "delete"])
def test_a_write_after_the_sweep_logs_the_reopen_as_a_write(mode, caplog):
    home = _home("alpha", mode)
    store = open_persistence(profile_scope(home=home))
    assert store.sessions.get("nope") is None
    close_all_stores()

    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        seed_session(store, "s1", "cli")
        assert store.sessions.get("s1") is not None
    assert _reopen_warnings(caplog, home) == [_WARNING.format("<home>/state.db", "write")]


def test_a_first_open_logs_no_reopen(caplog):
    home = _home("alpha", "wal")
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        assert open_persistence(profile_scope(home=home)).sessions.get("nope") is None
    assert _reopen_warnings(caplog, home) == []


def _main_warnings(home, caplog) -> list[str]:
    with legacy_scope():
        registry = legacy_import("hermes_state_registry")
        db = registry.acquire(home / "state.db")
        assert db.get_session("nope") is None
        registry.close_all()
        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            assert db.get_session("nope") is None
            assert db.get_session("nope") is None
        registry.close_all()
    return _reopen_warnings(caplog, home)


def _head_warnings(home, caplog) -> list[str]:
    store = open_persistence(profile_scope(home=home))
    assert store.sessions.get("nope") is None
    close_all_stores()
    with caplog.at_level(logging.WARNING, logger="hermes_state"):
        assert store.sessions.get("nope") is None
        assert store.sessions.get("nope") is None
    return _reopen_warnings(caplog, home)


@pytest.mark.parametrize("mode", ["wal", "delete"])
def test_the_warnings_match_main_after_the_sweep(mode, caplog):
    main = _main_warnings(_home("main", mode), caplog)
    caplog.clear()
    head = _head_warnings(_home("head", mode), caplog)

    assert main == [_WARNING.format("<home>/state.db", "read")]
    assert head == main
