"""GET /api/config must not block the event loop on _SKILLS_PROFILE_LOCK.

Regression for a captured 1.044s gateway stall: the endpoint entered
``_profile_scope`` (which acquires the process-wide ``_SKILLS_PROFILE_LOCK``)
directly on the asyncio event loop, so any slow lock-holder in a worker
thread froze every request and WebSocket in the gateway. The handler now
runs the scope + ``load_config()`` in ``asyncio.to_thread``.
"""

import asyncio
import threading
import time

import pytest


class TestGetConfigOffLoop:
    @pytest.fixture(autouse=True)
    def _home(self, _isolate_hermes_home):
        pass

    def test_get_config_returns_data(self):
        """The threaded path still returns the normalized, filtered config."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        resp = client.get("/api/config")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)
        assert not any(k.startswith("_") for k in body)

    def test_loop_stays_responsive_while_profile_lock_held(self):
        """Heartbeats on the request's event loop must keep ticking while
        another thread holds _SKILLS_PROFILE_LOCK and /api/config is in
        flight. Before the fix the handler blocked the loop for the full
        hold; now only the worker thread waits."""
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")
        from hermes_cli import web_server

        hold_s = 1.0
        release = threading.Event()

        def _holder():
            with web_server._SKILLS_PROFILE_LOCK:
                release.wait(hold_s)

        async def _scenario():
            holder = threading.Thread(target=_holder)
            holder.start()
            # Give the holder time to actually take the lock.
            await asyncio.sleep(0.05)

            ticks = 0

            async def _heartbeat(stop: asyncio.Event):
                # Tick COUNT is the starvation signal, not max gap: when the
                # handler blocks the loop, this task never runs at all during
                # the request, so a gap-based assertion passes vacuously.
                nonlocal ticks
                while not stop.is_set():
                    await asyncio.sleep(0.02)
                    ticks += 1

            stop = asyncio.Event()
            hb = asyncio.create_task(_heartbeat(stop))
            # Let the heartbeat task actually start before the request.
            await asyncio.sleep(0)
            transport = httpx.ASGITransport(app=web_server.app)
            try:
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    client.headers[web_server._SESSION_HEADER_NAME] = (
                        web_server._SESSION_TOKEN
                    )
                    resp = await client.get("/api/config")
            finally:
                stop.set()
                release.set()
                await hb
                holder.join()

            assert resp.status_code == 200
            return ticks

        ticks = asyncio.run(_scenario())
        # The request waits out the ~1s lock hold in a worker thread while
        # the loop keeps ticking (~50 ticks at 20ms). Pre-fix, the handler
        # blocked the loop for the whole hold and the heartbeat got ~0
        # ticks. Threshold is generous so slow CI machines don't flake.
        assert ticks >= 10, (
            f"event loop heartbeat only ticked {ticks} time(s) while "
            "_SKILLS_PROFILE_LOCK was held — /api/config is blocking the "
            "loop again"
        )


class TestRouterOffLoop:
    """Mounted routers (skills/mcp/tools) hold the same locks — they must not
    do it on the event loop either (same bug class as /api/config)."""

    @pytest.fixture(autouse=True)
    def _home(self, _isolate_hermes_home):
        pass

    def test_get_skills_loop_stays_responsive_while_profile_lock_held(self):
        try:
            import httpx
        except ImportError:
            pytest.skip("httpx not installed")
        from hermes_cli import web_server

        hold_s = 1.0
        release = threading.Event()

        def _holder():
            with web_server._SKILLS_PROFILE_LOCK:
                release.wait(hold_s)

        async def _scenario():
            holder = threading.Thread(target=_holder)
            holder.start()
            await asyncio.sleep(0.05)

            ticks = 0

            async def _heartbeat(stop: asyncio.Event):
                nonlocal ticks
                while not stop.is_set():
                    await asyncio.sleep(0.02)
                    ticks += 1

            stop = asyncio.Event()
            hb = asyncio.create_task(_heartbeat(stop))
            await asyncio.sleep(0)
            transport = httpx.ASGITransport(app=web_server.app)
            try:
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    client.headers[web_server._SESSION_HEADER_NAME] = (
                        web_server._SESSION_TOKEN
                    )
                    resp = await client.get("/api/skills")
            finally:
                stop.set()
                release.set()
                await hb
                holder.join()

            assert resp.status_code == 200
            return ticks

        ticks = asyncio.run(_scenario())
        assert ticks >= 10, (
            f"event loop heartbeat only ticked {ticks} time(s) while "
            "_SKILLS_PROFILE_LOCK was held — GET /api/skills is blocking "
            "the loop"
        )


class TestConfigMutationLock:
    """Off-loop read-modify-write handlers must not lose concurrent updates.

    config.py's _CONFIG_LOCK covers each load/save individually; the span
    between them is serialized by web_server._CONFIG_MUTATION_LOCK. Two
    concurrent writers touching DIFFERENT keys must both survive."""

    @pytest.fixture(autouse=True)
    def _home(self, _isolate_hermes_home):
        pass

    def test_concurrent_distinct_updates_both_survive(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli import web_server
        from hermes_cli.config import load_config

        client = TestClient(web_server.app)
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

        # Interleave the two handlers' load→mutate→save spans deterministically:
        # gate the FIRST writer inside its mutation-lock hold, and assert the
        # second writer cannot slip its own load in between. With the lock,
        # writer B waits for writer A's full span; without it, B loads the
        # pre-A config and its save erases A's write.
        results = []

        def _put_theme():
            resp = client.put("/api/dashboard/theme", json={"name": "midnight"})
            results.append(("theme", resp.status_code))

        def _put_font():
            resp = client.put("/api/dashboard/font", json={"font": "jetbrains-mono"})
            results.append(("font", resp.status_code))

        # Widen the race window: make save_config slow so an unserialized
        # interleave is near-certain, not just possible.
        real_save = web_server.save_config

        def _slow_save(cfg, **kwargs):
            time.sleep(0.15)
            return real_save(cfg, **kwargs)

        threads = []
        try:
            web_server.save_config = _slow_save
            threads = [
                threading.Thread(target=_put_theme),
                threading.Thread(target=_put_font),
            ]
            for t in threads:
                t.start()
        finally:
            for t in threads:
                t.join()
            web_server.save_config = real_save

        assert all(code == 200 for _, code in results), results
        cfg = load_config()
        dashboard = cfg.get("dashboard") or {}
        # Both writes must be present — a lost update drops exactly one.
        assert dashboard.get("theme") == "midnight", (
            "theme write lost to a concurrent font write — "
            "read-modify-write span is not serialized"
        )
        assert dashboard.get("font") == "jetbrains-mono", (
            "font write lost to a concurrent theme write — "
            "read-modify-write span is not serialized"
        )
