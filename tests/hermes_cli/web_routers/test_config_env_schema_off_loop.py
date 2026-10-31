"""Regression for #147036: config routes doing blocking work on the event loop.

``config_env.get_schema`` merged its discovery-driven provider options inline, so the
per-request schema walk plus the plugin-name scan ran on the uvicorn event loop that
also serves ``/api/ws`` and every other dashboard route — a slow scan froze the whole
gateway for its duration. ``get_egress_status`` had the same inline shape, reading the
profile's config block from disk on the loop. Both handlers now offload to a worker
thread via ``config_scoped_to_thread``, the same helper their sibling ``get_config``
uses.
"""

import asyncio
import threading

import pytest

from hermes_cli.web_routers import config_env


def test_get_schema_builds_off_the_loop_thread(monkeypatch):
    """The schema build runs on a worker thread, never the event-loop thread."""
    idents = {}
    built = {"voice.stt.provider": {"type": "str", "options": ["stub"]}}

    def _build():
        idents["build"] = threading.get_ident()
        # A worker thread runs no event loop; prove we are off it.
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return built

    monkeypatch.setattr(config_env, "_schema_with_dynamic_provider_options", _build)

    async def _drive():
        idents["loop"] = threading.get_ident()
        return await config_env.get_schema(None)

    result = asyncio.run(_drive())

    assert "build" in idents, "the schema build never ran"
    assert idents["build"] != idents["loop"], "the schema build ran on the event-loop thread"
    assert result["fields"] is built
    assert result["category_order"] == config_env._CATEGORY_ORDER


def test_get_egress_status_reads_off_the_loop_thread(monkeypatch):
    """The egress status read runs on a worker thread, never the event-loop thread."""
    from hermes_cli import proxy_cli

    idents = {}

    def _read():
        idents["read"] = threading.get_ident()
        # A worker thread runs no event loop; prove we are off it.
        with pytest.raises(RuntimeError):
            asyncio.get_running_loop()
        return "stub status text"

    monkeypatch.setattr(proxy_cli, "format_status_text", _read)

    async def _drive():
        idents["loop"] = threading.get_ident()
        return await config_env.get_egress_status(None)

    result = asyncio.run(_drive())

    assert "read" in idents, "the egress status read never ran"
    assert idents["read"] != idents["loop"], "the egress status read ran on the event-loop thread"
    assert result == {"text": "stub status text"}
