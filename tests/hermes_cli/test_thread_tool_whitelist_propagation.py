"""Regression: the background-review tool whitelist must survive the hop into the tool worker.

Background-review forks restrict themselves to a small tool set. The whitelist is
installed with :func:`hermes_cli.plugins.set_thread_tool_whitelist` and enforced by
:func:`hermes_cli.plugins._get_pre_tool_call_directive_details`.

The enforcement point is ``agent/tool_executor.py::_pre_tool_block``, which runs
INSIDE the tool worker: the executor submits its worker through
``propagate_context_to_thread(self.run_worker)`` (``agent/tool_executor.py:1425``).

The whitelist used to live in a ``threading.local``. A ``threading.local`` is
per-thread, so the worker saw ``allowed = None`` and the gate never blocked —
a background-review fork could call tools the core meant to deny.

``ContextVar`` values DO cross that hop (``propagate_context_to_thread`` snapshots
``contextvars.copy_context()``), so the whitelist has to be carried in one.

The isolation contract stays intact: the whitelist must still not leak into
threads that were not spawned from the review context. The core test
``test_whitelist_is_thread_local`` covers the plain-thread case; this module
covers the worker case and the propagation semantics.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.plugins import (
    clear_thread_tool_whitelist,
    get_pre_tool_call_block_message,
    set_thread_tool_whitelist,
)
from tools.thread_context import propagate_context_to_thread

REVIEW_WHITELIST = {"memory", "skill_manage", "skill_view", "read_file", "search_files"}


def _block_message_in_worker(tool_name: str):
    """Ask the gate from a worker thread, exactly like the tool executor does."""

    def _in_worker():
        return get_pre_tool_call_block_message(tool_name, {})

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(propagate_context_to_thread(_in_worker)).result(timeout=30)


class TestThreadToolWhitelistWorkerPropagation:
    """The whitelist must be enforced where the tool actually runs."""

    @pytest.fixture(autouse=True)
    def _clean(self):
        clear_thread_tool_whitelist()
        yield
        clear_thread_tool_whitelist()

    def test_forbidden_tool_is_blocked_in_the_tool_worker(self):
        """RED for the old implementation: the worker saw no whitelist."""
        set_thread_tool_whitelist(set(REVIEW_WHITELIST))
        message = _block_message_in_worker("write_file")
        assert message is not None, (
            "the review whitelist was not enforced in the tool worker thread — "
            "a background-review fork could call tools the core meant to deny"
        )
        assert "write_file" in message

    def test_allowed_tool_passes_in_the_tool_worker(self):
        """The whitelist must not over-block: allowed tools still run."""
        set_thread_tool_whitelist(set(REVIEW_WHITELIST))
        for tool_name in sorted(REVIEW_WHITELIST):
            assert _block_message_in_worker(tool_name) is None, (
                f"{tool_name} is on the review whitelist but was blocked in the worker"
            )

    def test_no_whitelist_means_no_restriction(self):
        """Outside a background review the gate stays open (fail-open)."""
        clear_thread_tool_whitelist()
        assert _block_message_in_worker("write_file") is None

    def test_clearing_the_whitelist_ends_the_restriction(self):
        """The restriction ends with the review, in the worker too."""
        set_thread_tool_whitelist({"memory"})
        assert _block_message_in_worker("write_file") is not None
        clear_thread_tool_whitelist()
        assert _block_message_in_worker("write_file") is None

    def test_plain_thread_does_not_inherit_the_whitelist(self):
        """Isolation contract: an unrelated thread is not restricted.

        A plain ``threading.Thread`` does not copy the caller's context, so a
        ContextVar-based whitelist must not leak into it.
        """
        set_thread_tool_whitelist({"memory"})
        result = {}

        def worker():
            result["msg"] = get_pre_tool_call_block_message("write_file", {})

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result["msg"] is None, "the whitelist leaked into an unrelated thread"

    def test_worker_inherits_only_from_its_own_spawn_context(self):
        """A worker spawned from a restricted context inherits the whitelist.

        This is the propagation the fix relies on — the tool worker is spawned
        from the review context.
        """
        set_thread_tool_whitelist({"memory"})
        try:
            assert _block_message_in_worker("write_file") is not None
        finally:
            clear_thread_tool_whitelist()
