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
import hermes_cli.config as _cfg_mod
import hermes_cli.web_server_profiles as _web_server_profiles


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
        acquired = threading.Event()

        def _holder():
            with _web_server_profiles._SKILLS_PROFILE_LOCK:
                acquired.set()
                release.wait(hold_s)

        async def _scenario():
            holder = threading.Thread(target=_holder)
            holder.start()
            # Deterministic gate: wait until the holder has actually taken the
            # lock (signalled from inside the `with` block) before issuing the
            # request — no sleep-and-hope race on slow CI machines.
            loop = asyncio.get_running_loop()
            assert await loop.run_in_executor(None, acquired.wait, 5), (
                "holder thread never acquired _SKILLS_PROFILE_LOCK"
            )

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
        acquired = threading.Event()

        def _holder():
            with _web_server_profiles._SKILLS_PROFILE_LOCK:
                acquired.set()
                release.wait(hold_s)

        async def _scenario():
            holder = threading.Thread(target=_holder)
            holder.start()
            # Deterministic gate: wait until the holder has actually taken the
            # lock before issuing the request (see TestGetConfigOffLoop).
            loop = asyncio.get_running_loop()
            assert await loop.run_in_executor(None, acquired.wait, 5), (
                "holder thread never acquired _SKILLS_PROFILE_LOCK"
            )

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

        # Race the two handlers' load→mutate→save spans. This is probabilistic,
        # not deterministically gated: the slow save_config below widens the
        # unserialized race window to ~150ms so a lost update is near-certain
        # without _CONFIG_MUTATION_LOCK, while remaining impossible with it.
        results = []

        def _put_theme():
            resp = client.put("/api/dashboard/theme", json={"name": "midnight"})
            results.append(("theme", resp.status_code))

        def _put_font():
            resp = client.put("/api/dashboard/font", json={"font": "jetbrains-mono"})
            results.append(("font", resp.status_code))

        # Widen the race window: make save_config slow so an unserialized
        # interleave is near-certain, not just possible.
        real_save = _cfg_mod.save_config

        def _slow_save(cfg, **kwargs):
            time.sleep(0.15)
            return real_save(cfg, **kwargs)

        threads = []
        try:
            _cfg_mod.save_config = _slow_save
            threads = [
                threading.Thread(target=_put_theme),
                threading.Thread(target=_put_font),
            ]
            for t in threads:
                t.start()
        finally:
            for t in threads:
                t.join()
            _cfg_mod.save_config = real_save

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

    def test_plugin_providers_put_serialized_against_other_writers(self):
        """PUT /api/dashboard/plugin-providers does config RMW through
        plugins_cmd._save_context_engine — it must hold the same
        _CONFIG_MUTATION_LOCK as every other config writer, or a concurrent
        locked writer's update gets erased by its stale save."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli import config as config_mod
        from hermes_cli import web_server
        from hermes_cli.config import load_config

        client = TestClient(web_server.app)
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

        results = []

        def _put_engine():
            resp = client.put(
                "/api/dashboard/plugin-providers", json={"context_engine": "builtin"}
            )
            results.append(("engine", resp.status_code))

        def _put_theme():
            resp = client.put("/api/dashboard/theme", json={"name": "midnight"})
            results.append(("theme", resp.status_code))

        # Slow down the engine writer's save (resolved at call time from
        # hermes_cli.config by _save_context_engine's function-local import)
        # so an unserialized theme write can land inside its RMW span and be
        # erased by the stale save. With the mutation lock the whole span is
        # serialized and both writes survive.
        real_save = config_mod.save_config

        def _slow_save(cfg, **kwargs):
            time.sleep(0.15)
            return real_save(cfg, **kwargs)

        threads = []
        try:
            config_mod.save_config = _slow_save
            t_engine = threading.Thread(target=_put_engine)
            t_theme = threading.Thread(target=_put_theme)
            threads = [t_engine, t_theme]
            t_engine.start()
            time.sleep(0.05)  # let the engine writer enter its RMW span first
            t_theme.start()
        finally:
            for t in threads:
                t.join()
            config_mod.save_config = real_save

        assert all(code == 200 for _, code in results), results
        cfg = load_config()
        assert (cfg.get("context") or {}).get("engine") == "builtin", (
            "context.engine write lost — plugin-providers RMW not serialized"
        )
        assert (cfg.get("dashboard") or {}).get("theme") == "midnight", (
            "theme write lost to a concurrent plugin-providers write — "
            "PUT /api/dashboard/plugin-providers is not holding "
            "_CONFIG_MUTATION_LOCK around its read-modify-write span"
        )
