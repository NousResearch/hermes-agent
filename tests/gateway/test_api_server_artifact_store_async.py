"""Artifact store resolution for the one-shot artifact routes runs off the loop, single-flight."""

import asyncio
import threading

import pytest

from gateway.platforms.api_server import APIServerAdapter


def _bare_adapter():
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._browser_control_artifacts = {}
    adapter._browser_control_artifact_locks = {}
    return adapter


@pytest.mark.asyncio
async def test_artifact_store_constructed_once_off_loop_under_concurrency():
    adapter = _bare_adapter()
    built = []

    def slow_build(profile_key):
        built.append((profile_key, threading.current_thread() is threading.main_thread()))
        import time
        time.sleep(0.05)
        store = object()
        adapter._browser_control_artifacts[profile_key] = store
        return store

    adapter._artifact_store_for = slow_build

    stores = await asyncio.gather(*(adapter._artifact_store_for_async("p") for _ in range(5)))

    assert len(built) == 1, "racing first requests must not each build a store (receipts would be stranded)"
    assert built[0] == ("p", False), "construction must hop off the loop thread"
    assert len({id(s) for s in stores}) == 1
    # cache hit: no lock, no thread hop
    assert await adapter._artifact_store_for_async("p") is stores[0]
    assert len(built) == 1
