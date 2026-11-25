"""A submitted tool policy is a snapshot, and sibling async tasks own independent policies.

Related upstream work: #147390 and #151480. Worker propagation itself already exists on main.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.plugins import (
    clear_thread_tool_whitelist,
    get_pre_tool_call_directive,
    set_thread_tool_whitelist,
)
from tools.thread_context import propagate_context_to_thread


def test_submitted_tool_policy_is_an_immutable_snapshot():
    allowed = {"memory"}
    set_thread_tool_whitelist(allowed, "snapshot denies {tool_name}")
    try:
        worker = propagate_context_to_thread(lambda: get_pre_tool_call_directive("terminal", {}))
        allowed.add("terminal")
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(worker).result(timeout=10) == ("block", "snapshot denies terminal")
    finally:
        clear_thread_tool_whitelist()


@pytest.mark.asyncio
async def test_sibling_tasks_do_not_replace_or_clear_each_others_policy():
    first_ready, second_ready, first_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def first():
        set_thread_tool_whitelist({"memory"}, "first denies {tool_name}")
        first_ready.set()
        try:
            await second_ready.wait()
            assert get_pre_tool_call_directive("terminal", {}) == ("block", "first denies terminal")
        finally:
            clear_thread_tool_whitelist()
            first_done.set()

    async def second():
        await first_ready.wait()
        set_thread_tool_whitelist({"terminal"}, "second denies {tool_name}")
        second_ready.set()
        try:
            await first_done.wait()
            assert get_pre_tool_call_directive("memory", {}) == ("block", "second denies memory")
        finally:
            clear_thread_tool_whitelist()

    await asyncio.wait_for(asyncio.gather(first(), second()), timeout=10)
