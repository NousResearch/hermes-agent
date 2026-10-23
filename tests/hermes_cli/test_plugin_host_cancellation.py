"""Hosted tools retain their caller's cancellation owner and session context.

No model, account, network service, or Reason executable is required.
"""

import json
import os
import threading

import pytest

from gateway.session_context import bound_session_key, clear_session_vars, set_session_vars
from hermes_cli.plugins import PluginManager
from tests.hermes_cli.test_plugin_host import _home_with_plugins
from tools.interrupt import acting_for_tid, set_interrupt
from tools.registry import registry


PROBE_PLUGIN = '''
import json, os

def register(ctx):
    def snapshot(args):
        ctx.dispatch_tool("host_cancellation_barrier", {"tag": args["tag"]})
        return json.dumps({"pid": os.getpid(), "interrupted": ctx.is_interrupted(),
                          "key": ctx.session_key(), "cwd": ctx.session_cwd(),
                          "surface": ctx.session_surface(),
                          "async_delivery": ctx.session_supports_async_delivery()})

    async def async_probe(args, **kwargs):
        return snapshot(args)

    for name, handler, asynchronous in (
        ("host_cancel_sync", snapshot, False),
        ("host_cancel_async", async_probe, True),
    ):
        ctx.register_tool(name=name, toolset="cancelprobe", handler=handler,
                          is_async=asynchronous, schema={"name": name, "description": "d",
                          "parameters": {"type": "object", "properties": {"tag": {"type": "string"}}}})
'''


@pytest.mark.platforms("any")
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("inherited_owner", [False, True], ids=["worker-owner", "nested-owner"])
def test_hosted_tool_cancellation_keeps_its_owner_and_session(
    tmp_path, monkeypatch, asynchronous, inherited_owner
):
    _home_with_plugins(tmp_path, monkeypatch, {"cancelprobe": PROBE_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    ready, release = threading.Event(), threading.Event()
    arrivals, results, restored, failures = set(), {}, {}, []
    arrival_lock = threading.Lock()
    outer_owner = threading.get_ident()
    cancel_target = None
    threads = {}

    def barrier(args, **kwargs):
        with arrival_lock:
            arrivals.add(args["tag"])
            if len(arrivals) == (1 if asynchronous else 2):
                ready.set()
        if not release.wait(10):
            raise TimeoutError("test did not release the hosted handler")
        return "{}"

    registry.register(name="host_cancellation_barrier", toolset="cancelprobe", handler=barrier,
                      schema={"name": "host_cancellation_barrier", "description": "d",
                              "parameters": {"type": "object", "properties": {}}},
                      scope=manager.scope_key)

    def invoke(tag):
        owner = outer_owner if tag == "cancelled" and inherited_owner else None
        owner_token = acting_for_tid.set(owner)
        cwd = tmp_path / tag
        cwd.mkdir()
        tokens = set_session_vars(platform="desktop", session_key=f"profile/{tag}",
                                  session_id=tag, cwd=str(cwd), async_delivery=True)
        try:
            name = "host_cancel_async" if asynchronous else "host_cancel_sync"
            results[tag] = json.loads(registry.dispatch(name, {"tag": tag}, scope=manager.scope_key))
            restored[tag] = (acting_for_tid.get(), bound_session_key())
        except BaseException as exc:
            failures.append(exc)
        finally:
            clear_session_vars(tokens)
            acting_for_tid.reset(owner_token)

    try:
        # A synchronous ctx RPC blocks the child's shared async loop; release the
        # async call before starting its sibling. Sync calls also verify concurrency.
        for tag in (("cancelled",) if asynchronous else ("cancelled", "sibling")):
            threads[tag] = threading.Thread(target=invoke, args=(tag,), daemon=True)
            threads[tag].start()
        assert ready.wait(10), failures
        cancel_target = outer_owner if inherited_owner else threads["cancelled"].ident
        set_interrupt(True, cancel_target, reason="user interrupt")
        release.set()
        for thread in threads.values():
            thread.join(timeout=10)
            assert not thread.is_alive()
        if asynchronous:
            set_interrupt(False, cancel_target)
            threads["sibling"] = threading.Thread(target=invoke, args=("sibling",), daemon=True)
            threads["sibling"].start()
            threads["sibling"].join(timeout=10)
            assert not threads["sibling"].is_alive()
        assert not failures
        assert results["cancelled"]["interrupted"] is True
        assert results["sibling"]["interrupted"] is False
        for tag, result in results.items():
            assert result["pid"] != os.getpid()
            assert result["key"] == f"profile/{tag}"
            assert result["cwd"] == str(tmp_path / tag)
            assert result["surface"] == "desktop"
            assert result["async_delivery"] is True
        assert restored["cancelled"] == (outer_owner if inherited_owner else None, "profile/cancelled")
        assert restored["sibling"] == (None, "profile/sibling")
    finally:
        release.set()
        for thread in threads.values():
            thread.join(timeout=5)
        if cancel_target is not None:
            set_interrupt(False, cancel_target)
        registry.deregister("host_cancellation_barrier", scope=manager.scope_key)
        manager.unload()
        manager._plugin_host().shutdown()
