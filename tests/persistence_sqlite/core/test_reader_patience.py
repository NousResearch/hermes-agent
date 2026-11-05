"""The read-only reader's patience: how long its open and each of its reads wait for a store another connection holds
exclusively, before the read raises ``StoreContention``."""

from __future__ import annotations

import time

import pytest

from persistence import StoreContention
from persistence_assembly import open_persistence, open_persistence_reader, open_store_admin, profile_scope
from tests.fixtures.storage_holder import read_busy_timeout_seam
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite.conftest import profile_home


def _store(name: str):
    home = profile_home(name)
    with open_persistence(profile_scope(home=home), eager=True) as store:
        seed_session(store, "s", "cli")
    return home / "state.db"


def _held(path):
    return open_store_admin(path, read_only=False).exclusive_repair_guard()


def _waited(read) -> float:
    start = time.monotonic()
    with pytest.raises(StoreContention):
        read()
    return time.monotonic() - start


def _about(waited: float, patience_s: float) -> bool:
    return patience_s * 0.8 <= waited < patience_s + 0.75


@pytest.mark.parametrize("patience_s", [0.3, 1.0, 2.0])
def test_the_open_waits_the_patience_the_reader_is_given(patience_s):
    path = _store("alpha")
    with _held(path), open_persistence_reader(path, patience_s=patience_s) as reader:
        assert _about(_waited(reader.sessions.first_session_started_at), patience_s)


def test_each_read_waits_the_patience_the_reader_is_given():
    path = _store("alpha")
    open_store_admin(path, read_only=False).set_journal_mode("delete")
    with open_persistence_reader(path, patience_s=0.5) as reader:
        assert reader.sessions.first_session_started_at() is not None
        with _held(path):
            assert _about(_waited(reader.messages.has_any_message), 0.5)
            assert _about(_waited(reader.sessions.first_session_started_at), 0.5)
        assert reader.messages.has_any_message() is False


def test_without_a_patience_the_reader_waits_the_backend_read_wait(monkeypatch):
    monkeypatch.setattr(*read_busy_timeout_seam(), 0.4)
    path = _store("alpha")
    with _held(path), open_persistence_reader(path) as reader:
        assert _about(_waited(reader.sessions.first_session_started_at), 0.4)
