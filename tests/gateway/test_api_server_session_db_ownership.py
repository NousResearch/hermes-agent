"""SessionDB ownership coverage for the API server profile cache."""

import asyncio
import threading

from gateway.platforms.api_server import APIServerAdapter
from hermes_state import get_shared_session_db, release_or_close
from hermes_state_registry import stats


def _bare_adapter() -> APIServerAdapter:
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._session_db = None
    adapter._session_dbs = {}
    adapter._session_db_cache_lock = threading.Lock()
    adapter._session_db_cache_closed = False
    adapter._session_db_lock = None
    return adapter


def test_profile_cache_borrows_and_releases_shared_session_dbs(tmp_path):
    adapter = _bare_adapter()
    homes = [tmp_path / "default", tmp_path / "profiles" / "work"]
    for home in homes:
        home.mkdir(parents=True)

    baseline = stats()
    sibling_dbs = []
    try:
        sibling_dbs.extend(get_shared_session_db(home / "state.db") for home in homes)
        api_dbs = [adapter._open_and_cache_session_db(home) for home in homes]

        assert api_dbs == sibling_dbs
        assert api_dbs[0] is not api_dbs[1]
        assert adapter._open_and_cache_session_db(homes[0]) is api_dbs[0]
        borrowed = stats()
        assert borrowed["live_generations"] == baseline["live_generations"] + 2
        assert borrowed["retired_generations"] == baseline["retired_generations"]
        assert borrowed["total_refcounts"] == baseline["total_refcounts"] + 4

        adapter._close_cached_session_dbs()

        assert stats()["total_refcounts"] == baseline["total_refcounts"] + 2
        assert all(db._conn is not None for db in sibling_dbs)
    finally:
        adapter._close_cached_session_dbs()
        for db in sibling_dbs:
            release_or_close(db)

    assert stats() == baseline


def test_async_profile_cache_keeps_single_flight(monkeypatch, tmp_path):
    adapter = _bare_adapter()
    home = tmp_path / "default"
    home.mkdir()
    acquire_calls = []

    class FakeSessionDB:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    fake_db = FakeSessionDB()

    def acquire(db_path):
        acquire_calls.append(db_path)
        return fake_db

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_state.get_shared_session_db", acquire)

    async def resolve_concurrently():
        return await asyncio.gather(
            *(adapter._ensure_session_db_async() for _ in range(4))
        )

    dbs = asyncio.run(resolve_concurrently())

    assert dbs == [fake_db] * 4
    assert acquire_calls == [home / "state.db"]

    adapter._close_cached_session_dbs()
    assert fake_db.close_calls == 1
