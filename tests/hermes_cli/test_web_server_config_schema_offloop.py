"""GET /api/config/schema must build the schema off the event loop.

Regression for a measured 3.6s gateway freeze: ``get_schema`` ran
``_schema_with_dynamic_provider_options()`` inline on the uvicorn event loop —
the same loop that serves the JSON-RPC chat WebSocket (``/api/ws``), the PTY
bridge and every other dashboard request — so an unrelated request arriving
during the build was not dispatched until it finished. The handler now takes the
same ``config_scoped_to_thread`` hop its sibling ``get_config`` already used.

Both halves of the claim are pinned without wall-clock sleeps: the build runs on
a worker thread, and an unrelated callback queued on the requesting loop runs to
completion *while* the build is still in flight (``call_soon_threadsafe`` is the
deterministic probe — inline, the loop is inside the build and cannot dispatch
it).
"""

import asyncio
import threading

import pytest


class TestConfigSchemaOffLoop:
    @pytest.fixture(autouse=True)
    def _home(self, _isolate_hermes_home):
        pass

    def test_schema_build_runs_off_the_loop_and_keeps_it_responsive(self, monkeypatch):
        httpx = pytest.importorskip("httpx")
        from hermes_cli import web_server
        from hermes_cli.web_routers import config_env

        observed = {}

        def _probe_schema():
            observed["thread"] = threading.current_thread()
            # Queue a no-op on the requesting loop and wait for it. The loop
            # dispatches it only if it is not stuck inside this build.
            progressed = threading.Event()
            observed["loop"].call_soon_threadsafe(progressed.set)
            observed["loop_was_free"] = progressed.wait(timeout=2.0)
            return [{"name": "probe", "type": "string"}]

        monkeypatch.setattr(
            config_env, "_schema_with_dynamic_provider_options", _probe_schema
        )

        async def _scenario():
            observed["loop"] = asyncio.get_running_loop()
            transport = httpx.ASGITransport(app=web_server.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                client.headers[web_server._SESSION_HEADER_NAME] = (
                    web_server._SESSION_TOKEN
                )
                return await client.get("/api/config/schema")

        resp = asyncio.run(_scenario())

        assert resp.status_code == 200
        assert resp.json()["fields"] == [{"name": "probe", "type": "string"}]
        assert observed["thread"] is not threading.main_thread(), (
            "the schema build ran on the event loop's thread — /api/config/schema "
            "blocks the loop for the whole build again"
        )
        assert observed["loop_was_free"] is True, (
            "the event loop could not dispatch an unrelated callback while the "
            "schema was being built — /api/config/schema is blocking the loop"
        )
