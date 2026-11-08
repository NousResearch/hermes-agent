"""The review/side-question tool whitelist must survive dispatch onto the tool pool.

Regression for #151307: the whitelist lived in a ``threading.local`` while every tool call runs on
a pool worker submitted through ``propagate_context_to_thread`` (which copies ContextVars only), so
the worker read "no whitelist" — allow — and a background-review fork denied ``skill_manage`` could
still reach ``terminal``.
"""

import concurrent.futures
import threading

from hermes_cli import plugins
from tools.thread_context import propagate_context_to_thread

DENIED = "not in this thread's tool whitelist"


def _gate(tool_name: str):
    return plugins.get_pre_tool_call_block_message(tool_name, {})


def _isolate_whitelist(monkeypatch) -> None:
    """The whitelist must be the only thing on this path that can block or allow."""
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda hook_name, **kwargs: [])


def test_whitelist_reaches_a_dispatch_pool_worker(monkeypatch):
    _isolate_whitelist(monkeypatch)
    plugins.set_thread_tool_whitelist({"memory", "skill_manage"})
    try:
        assert _gate("terminal") is not None
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            denied, allowed = pool.submit(
                propagate_context_to_thread(lambda: (_gate("terminal"), _gate("memory")))
            ).result()
        assert denied is not None and DENIED in denied, (
            "a worker dispatched through propagate_context_to_thread lost the whitelist"
        )
        assert allowed is None
    finally:
        plugins.clear_thread_tool_whitelist()


def test_deny_message_reaches_a_dispatch_pool_worker(monkeypatch):
    _isolate_whitelist(monkeypatch)
    plugins.set_thread_tool_whitelist(set(), deny_msg_fmt="Side question denied {tool_name}")
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            message = pool.submit(propagate_context_to_thread(lambda: _gate("terminal"))).result()
        assert message == "Side question denied terminal"
    finally:
        plugins.clear_thread_tool_whitelist()


def test_clearing_leaves_pool_workers_unrestricted(monkeypatch):
    _isolate_whitelist(monkeypatch)
    plugins.set_thread_tool_whitelist({"memory"})
    plugins.clear_thread_tool_whitelist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(propagate_context_to_thread(lambda: _gate("terminal"))).result() is None


def test_plain_thread_does_not_inherit_the_whitelist(monkeypatch):
    """A thread nothing propagates into starts empty, exactly as a fresh thread always has."""
    _isolate_whitelist(monkeypatch)
    plugins.set_thread_tool_whitelist(set())
    try:
        seen = {}
        worker = threading.Thread(target=lambda: seen.update(message=_gate("terminal")))
        worker.start()
        worker.join()
        assert seen["message"] is None
    finally:
        plugins.clear_thread_tool_whitelist()
