"""PUT /api/memory/providers/{name}/config must hold the config mutation lock.

The handler runs off-loop via ``asyncio.to_thread`` and does a real
``load_config() -> mutate -> save_config()`` span (it sets ``memory.provider``),
plus writes through ``_write_memory_provider_config_values``. Without
``_CONFIG_MUTATION_LOCK`` a concurrent writer's update lands inside that span
and is erased by the stale save, which is the exact lost-update the lock was
introduced for.

Mirrors ``TestConfigMutationLock::test_plugin_providers_put_serialized_against_other_writers``.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


class TestMemoryProviderConfigLock:
    def test_memory_provider_put_serialized_against_other_writers(self):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli import config as config_mod
        from hermes_cli import web_server
        from hermes_cli.config import load_config

        client = TestClient(web_server.app)
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

        results: list[tuple[str, int]] = []

        def _put_memory():
            resp = client.put(
                "/api/memory/providers/fakeprov/config", json={"values": {"k": "v"}}
            )
            results.append(("memory", resp.status_code))

        def _put_theme():
            resp = client.put("/api/dashboard/theme", json={"name": "midnight"})
            results.append(("theme", resp.status_code))

        # web_server binds save_config at module import, so the slow wrapper
        # must replace THAT name, not hermes_cli.config's. Only the memory
        # writer's save is delayed (identified by its payload) so the theme
        # write can land inside the memory handler's read-modify-write span.
        real_save = web_server.save_config

        def _slow_save(cfg, **kwargs):
            if isinstance(cfg, dict) and (cfg.get("memory") or {}).get("provider") == "fakeprov":
                time.sleep(0.15)
            return real_save(cfg, **kwargs)

        threads: list[threading.Thread] = []
        # Neutralise provider resolution so the handler reaches its real
        # config read-modify-write span; that span is what is under test.
        with patch.object(web_server, "_require_valid_memory_provider_name", lambda *_a, **_k: None), \
             patch.object(web_server, "_load_memory_provider", lambda *_a, **_k: object()), \
             patch.object(web_server, "_write_memory_provider_config_values", lambda *_a, **_k: None), \
             patch.object(web_server, "_require_memory_provider_ready", lambda *_a, **_k: None), \
             patch.object(web_server, "_invalidate_plugins_hub_cache", lambda *_a, **_k: None):
            try:
                web_server.save_config = _slow_save
                t_mem = threading.Thread(target=_put_memory)
                t_theme = threading.Thread(target=_put_theme)
                threads = [t_mem, t_theme]
                t_mem.start()
                time.sleep(0.05)  # let the memory writer enter its RMW span first
                t_theme.start()
            finally:
                for t in threads:
                    t.join()
                web_server.save_config = real_save

        assert all(code == 200 for _, code in results), results
        cfg = load_config()
        assert (cfg.get("memory") or {}).get("provider") == "fakeprov", (
            "memory.provider write lost — the handler's RMW is not serialized"
        )
        assert (cfg.get("dashboard") or {}).get("theme") == "midnight", (
            "theme write lost to a concurrent memory-provider write — "
            "PUT /api/memory/providers/{name}/config is not holding "
            "_CONFIG_MUTATION_LOCK around its read-modify-write span"
        )


class TestOffLoopConfigWritersHoldTheLock:
    """Every off-loop config read-modify-write must serialize on the same lock.

    Two shapes run off the event loop and so lose its free serialization:
    work dispatched via ``asyncio.to_thread``, and plain ``def`` FastAPI route
    handlers, which the framework runs in its own threadpool. A scan keyed only
    on ``_run`` closures missed the latter entirely.
    """

    def test_the_known_off_loop_writers_are_wrapped(self):
        from hermes_cli import web_server

        for name in (
            "set_moa_models",
            "upsert_custom_endpoint",
            "activate_custom_endpoint",
            "delete_custom_endpoint",
            "_apply_model_assignment_sync",
        ):
            fn = getattr(web_server, name)
            assert hasattr(fn, "__wrapped__"), (
                f"{name} performs an off-loop config read-modify-write but is not "
                "serialized on _CONFIG_MUTATION_LOCK"
            )

    def test_the_wrapper_actually_holds_the_lock(self):
        from hermes_cli import web_server

        held = {}

        @web_server._serialized_config_write
        def _probe():
            held["locked"] = web_server._CONFIG_MUTATION_LOCK._is_owned()
            return "ok"

        assert _probe() == "ok"
        assert held["locked"] is True
        assert web_server._CONFIG_MUTATION_LOCK._is_owned() is False

    def test_fastapi_still_sees_the_real_signature(self):
        """FastAPI builds the request model from the signature; wraps must not hide it."""
        import inspect

        from hermes_cli import web_server

        params = inspect.signature(web_server.upsert_custom_endpoint).parameters
        assert "body" in params, "the decorator hid the handler's parameters from FastAPI"

    def test_no_unlocked_off_loop_writer_remains(self):
        """Scan the module rather than trusting a hand-kept list.

        Reachability is what matters: a sync ``@app.*`` handler and anything
        dispatched with ``to_thread`` both run off-loop. Anything else in the
        file is either loop-serialized or reached from inside a locked span.
        """
        import inspect
        import re

        from hermes_cli import web_server

        src = inspect.getsource(web_server)
        lines = src.split("\n")
        offenders = []
        for i, line in enumerate(lines):
            m = re.match(r"^def (\w+)\(", line)
            if not m:
                continue
            decorators = []
            j = i - 1
            while j >= 0 and lines[j].startswith("@"):
                decorators.append(lines[j])
                j -= 1
            if not any(d.startswith("@app.") for d in decorators):
                continue
            if any("_serialized_config_write" in d for d in decorators):
                continue
            end = len(lines)
            for k in range(i + 1, len(lines)):
                if lines[k] and not lines[k][0].isspace() and not lines[k].startswith("@"):
                    end = k
                    break
            body = "\n".join(lines[i:end])
            reads = re.search(r"\b(load_config|read_raw_config)\s*\(", body)
            writes = re.search(r"\bsave_config\s*\(", body)
            if reads and writes and "_CONFIG_MUTATION_LOCK" not in body:
                offenders.append(m.group(1))

        assert not offenders, (
            "sync FastAPI handlers doing an unserialized config read-modify-write: "
            f"{offenders}"
        )
