"""Regression tests for the #91912 salvage — credential-pool handlers off-loop
and the bounded Copilot token exchange.

The 2026-08-22 incident: ``GET /api/credentials/pool`` ran ``load_pool()`` on
the uvicorn event loop; for the copilot provider that reaches
``urllib.request.urlopen`` whose ``timeout`` does not bound ``getaddrinfo``,
so a networkless host froze the whole dashboard backend for 17 minutes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from hermes_cli import copilot_auth


# ---------------------------------------------------------------------------
# _urlopen_bounded
# ---------------------------------------------------------------------------


class TestUrlopenBounded:
    def test_returns_response_when_worker_completes(self):
        sentinel = object()
        with patch("urllib.request.urlopen", return_value=sentinel):
            assert copilot_auth._urlopen_bounded("req", 1.0) is sentinel

    def test_reraises_worker_exception(self):
        with patch("urllib.request.urlopen", side_effect=OSError("boom")):
            with pytest.raises(OSError, match="boom"):
                copilot_auth._urlopen_bounded("req", 1.0)

    def test_hard_cap_fires_on_hung_resolver_and_closes_late_response(self, monkeypatch):
        """A urlopen that hangs past timeout + grace must raise TimeoutError
        promptly, and when the abandoned worker later *succeeds* it must close
        the response instead of leaking the socket."""
        monkeypatch.setattr(copilot_auth, "_DNS_GRACE_SECONDS", 0.05)
        release = threading.Event()
        closed = threading.Event()

        class _LateResponse:
            def close(self):
                closed.set()

        def hung_urlopen(req, timeout):
            release.wait(timeout=5)
            return _LateResponse()

        with patch("urllib.request.urlopen", side_effect=hung_urlopen):
            started = time.monotonic()
            with pytest.raises(TimeoutError, match="hard cap"):
                copilot_auth._urlopen_bounded("req", 0.05)
            elapsed = time.monotonic() - started
            assert elapsed < 2.0
            release.set()
            assert closed.wait(timeout=2), "late response was not closed"


# ---------------------------------------------------------------------------
# single-flight exchange
# ---------------------------------------------------------------------------


class TestExchangeSingleFlight:
    @pytest.fixture(autouse=True)
    def _clean_caches(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        copilot_auth._jwt_cache.clear()
        copilot_auth._exchange_failure_cache.clear()
        copilot_auth._exchange_locks.clear()
        yield
        copilot_auth._jwt_cache.clear()
        copilot_auth._exchange_failure_cache.clear()
        copilot_auth._exchange_locks.clear()

    def test_concurrent_callers_share_one_exchange(self, monkeypatch):
        """N concurrent callers for the same token must perform ONE network
        exchange; the rest wait on the lock and hit the populated cache."""
        monkeypatch.setattr(copilot_auth, "_load_jwt_from_disk", lambda fp: None)
        monkeypatch.setattr(copilot_auth, "_save_jwt_to_disk", lambda *a, **k: None)
        calls = []
        gate = threading.Event()

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"token": "tid=1;exp=9", "expires_at": 4102444800}'

        def fake_bounded(req, timeout):
            calls.append(threading.get_ident())
            gate.wait(timeout=5)  # hold the first exchange open while others queue
            return _Resp()

        monkeypatch.setattr(copilot_auth, "_urlopen_bounded", fake_bounded)

        results = []
        threads = [
            threading.Thread(target=lambda: results.append(copilot_auth.exchange_copilot_token("ghu_" + "x" * 30)))
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        time.sleep(0.2)  # let every caller reach the lock
        assert len(calls) == 1
        gate.set()
        for t in threads:
            t.join(timeout=5)

        assert len(calls) == 1
        assert len(results) == 8
        assert {r[0] for r in results} == {"tid=1;exp=9"}

    def test_waiters_observe_negative_cache_after_failed_exchange(self, monkeypatch):
        """When the single in-flight exchange fails, queued callers must not
        each start their own exchange — they see the negative cache."""
        monkeypatch.setattr(copilot_auth, "_load_jwt_from_disk", lambda fp: None)
        monkeypatch.setattr(copilot_auth, "_EXCHANGE_MAX_ATTEMPTS", 1)
        calls = []
        gate = threading.Event()

        def fake_bounded(req, timeout):
            calls.append(1)
            gate.wait(timeout=5)
            raise TimeoutError("hard cap")

        monkeypatch.setattr(copilot_auth, "_urlopen_bounded", fake_bounded)
        errors = []

        def run():
            try:
                copilot_auth.exchange_copilot_token("ghu_" + "y" * 30)
            except ValueError as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=run) for _ in range(5)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        gate.set()
        for t in threads:
            t.join(timeout=5)

        assert len(calls) == 1
        assert len(errors) == 5
        assert any("recently failed" in e for e in errors)

    def test_negative_cache_short_circuits_before_taking_the_lock(self, monkeypatch):
        """While one exchange holds the lock, a caller whose fingerprint is
        already in the failure cache must raise immediately rather than park
        an executor thread behind the holder."""
        fp = copilot_auth._token_fingerprint("ghu_" + "z" * 30)
        copilot_auth._exchange_failure_cache[fp] = time.time() + 60
        lock = copilot_auth._exchange_lock_for(fp)
        lock.acquire()  # simulate an in-flight holder
        try:
            started = time.monotonic()
            with pytest.raises(ValueError, match="recently failed"):
                copilot_auth.exchange_copilot_token("ghu_" + "z" * 30)
            assert time.monotonic() - started < 0.5
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# web_server credential-pool handlers off the loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_credential_pool_runs_off_event_loop(monkeypatch):
    import hermes_cli.auth as auth_mod
    from hermes_cli import web_server

    loop_thread = threading.get_ident()
    seen = {}

    def fake_read_pool(*args, **kwargs):
        seen["thread"] = threading.get_ident()
        return {}

    monkeypatch.setattr(auth_mod, "read_credential_pool", fake_read_pool)
    result = await web_server.list_credential_pool()

    assert result == {"providers": []}
    assert seen["thread"] != loop_thread


@pytest.mark.asyncio
async def test_list_credential_pool_keeps_loop_responsive(monkeypatch):
    """A 200 ms blocking pool read must not freeze a concurrent ticker."""
    import hermes_cli.auth as auth_mod
    from hermes_cli import web_server

    def slow_read(*args, **kwargs):
        time.sleep(0.5)
        return {}

    monkeypatch.setattr(auth_mod, "read_credential_pool", slow_read)

    gaps = []
    stop = asyncio.Event()

    async def ticker():
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    await web_server.list_credential_pool()
    stop.set()
    await t
    # 0.25 s threshold vs a 0.5 s blocking read: a regression (read on the
    # loop) trips it by 2x, while runner-noise descheduling would need >200 ms.
    assert max(gaps) < 0.25, f"event loop stalled for {max(gaps) * 1000:.0f} ms"
