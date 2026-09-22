"""Pool state queries and management calls must run under ``self._lock``.

Shard of ``tests/agent/test_credential_pool.py`` (2K-law fracture, Part of
#79999, #78647).  The extracted block is byte-verbatim; the shared
``_write_auth_store`` helper stays in the parent module.
"""

import pytest

from tests.agent.test_credential_pool import _write_auth_store


def _load_two_ok_pool(tmp_path, monkeypatch):
    """A pool with two OK anthropic entries, current = cred-1."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1", "label": "primary", "auth_type": "api_key",
                        "priority": 0, "source": "manual", "access_token": "***",
                        "last_status": "ok", "last_status_at": None, "last_error_code": None,
                    },
                    {
                        "id": "cred-2", "label": "secondary", "auth_type": "api_key",
                        "priority": 1, "source": "manual", "access_token": "***",
                        "last_status": "ok", "last_status_at": None, "last_error_code": None,
                    },
                ]
            },
        },
    )
    from agent.credential_pool import load_pool

    return load_pool("anthropic")


def _fresh_entry(pool):
    """A copy of the pool's first entry under a new id, for add_entry()."""
    from dataclasses import replace as dc_replace

    return dc_replace(pool.entries()[0], id="cred-new")


class TestCredentialPoolQueryLocking:
    """Public pool-state methods must run under ``self._lock``.

    ``has_available``/``peek``/``current``/``entries`` all touch
    ``self._entries`` (and ``_available_entries`` even prunes + persists),
    and the management surface (``has_credentials``/``reset_statuses``/
    ``remove_index``/``resolve_target``/``add_entry``) reads or rebinds
    ``self._entries`` and persists auth.json, so they must all hold the
    same lock every mutating entry point uses.  A naive fix would deadlock
    because the lock is non-reentrant and ``peek`` calls ``current`` +
    ``_available_entries``; these tests guard both the no-deadlock and the
    actually-locked properties.
    """

    def test_query_methods_do_not_deadlock(self, tmp_path, monkeypatch):
        pool = _load_two_ok_pool(tmp_path, monkeypatch)
        pool.select()  # set a current entry

        # peek() internally calls current() + _available_entries(); if any of
        # these re-acquired the non-reentrant lock we'd hang here forever.
        assert pool.current() is not None
        assert pool.peek() is not None
        assert pool.has_available() is True
        assert pool.has_credentials() is True
        assert pool.resolve_target("cred-1")[1] is not None
        # (env may seed extra singleton entries; just assert ours are present)
        assert {"cred-1", "cred-2"} <= {e.id for e in pool.entries()}
        # try_refresh_matching's no-hint branch resolves the current entry
        # while already holding the lock — must use _current_unlocked(), not
        # current(), or it deadlocks on the non-reentrant lock (found when
        # rebasing this fix over the #69843 salvage which added the method).
        pool.try_refresh_matching()

    @pytest.mark.parametrize(
        "method,get_args",
        [
            ("has_available", lambda pool: ()),
            ("peek", lambda pool: ()),
            ("current", lambda pool: ()),
            ("entries", lambda pool: ()),
            ("has_credentials", lambda pool: ()),
            ("reset_statuses", lambda pool: ()),
            ("resolve_target", lambda pool: ("cred-1",)),
            ("remove_index", lambda pool: (1,)),
            ("add_entry", lambda pool: (_fresh_entry(pool),)),
        ],
    )
    def test_query_method_acquires_lock(self, tmp_path, monkeypatch, method, get_args):
        import threading

        pool = _load_two_ok_pool(tmp_path, monkeypatch)
        pool.select()
        args = get_args(pool)

        inner = pool._lock

        class _InstrumentedLock:
            """Probe that records acquire attempts, so the test can prove the
            worker actually reached ``self._lock`` before asserting that it
            blocks (a plain timed wait passes spuriously if the worker is
            simply never scheduled)."""

            def __init__(self):
                self.attempted = threading.Event()

            def acquire(self, *args, **kwargs):
                self.attempted.set()
                return inner.acquire(*args, **kwargs)

            def release(self):
                inner.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *exc):
                self.release()

        probe = _InstrumentedLock()
        pool._lock = probe

        done = threading.Event()

        def _call():
            getattr(pool, method)(*args)
            done.set()

        # Hold the real lock (without tripping the probe), then fire the query
        # on another thread. If the method acquires self._lock (as it must),
        # it blocks until we release.
        inner.acquire()
        try:
            worker = threading.Thread(target=_call, daemon=True)
            worker.start()
            assert probe.attempted.wait(timeout=2.0), (
                f"{method}() never attempted to acquire self._lock"
            )
            assert not done.wait(timeout=0.5), (
                f"{method}() returned while the pool lock was held — it is not "
                f"blocking on self._lock"
            )
        finally:
            inner.release()

        assert done.wait(timeout=2.0), f"{method}() did not complete after lock release"
