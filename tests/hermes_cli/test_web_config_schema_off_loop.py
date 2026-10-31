"""GET /api/config/schema builds the schema off the event loop (#147036).

The handler runs on the same loop that serves the JSON-RPC chat WebSocket
(``/api/ws``) and every other dashboard request, and
``_schema_with_dynamic_provider_options()`` rebuilds the schema on every
call — its memory-provider overlay walks plugin directories on disk
(``plugins/memory/__init__.py:_is_memory_provider_dir`` reads files and
``ast.parse``s every ``.py`` in a candidate directory). Measured on a
plugin-bearing install that is a ~3 s freeze of the loop per request.

Sibling handlers already hop to a worker thread (``get_config`` above it in
``config_env.py``, and ``config_scoped_to_thread`` across the other routers);
this pins the same guarantee for the schema route.
"""

import asyncio
import threading

import pytest

pytest.importorskip("fastapi")


class _Stop(Exception):
    pass


@pytest.mark.asyncio
async def test_schema_build_runs_off_the_event_loop(monkeypatch):
    from hermes_cli.web_routers import config_env

    loop_thread = threading.get_ident()
    seen = []

    def _fake_schema_build():
        seen.append(threading.get_ident())
        raise _Stop()

    monkeypatch.setattr(config_env, "_schema_with_dynamic_provider_options", _fake_schema_build)
    with pytest.raises(_Stop):
        await asyncio.wait_for(config_env.get_schema(), 10)
    assert seen, "the schema builder never ran"
    assert seen[0] != loop_thread, "the schema was built on the event loop thread"
