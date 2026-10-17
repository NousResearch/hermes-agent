"""Regression tests: dispatcher-owned kanban workers keep the lifecycle tools.

_select_tool_names force-adds ``kanban`` for dispatcher-spawned workers so they
can complete/block their card, but ``disabled_toolsets`` is subtracted LAST
(#17309), which silently stripped that grant for any profile that disables the
kanban chat toolset. The worker then had no kanban tools, its CLI fallback is
fenced by the delegated-child guard, it exited rc=0 without a board call, and
the dispatcher read that as a protocol violation → respawn loop.

The fix re-grants the kanban toolset AFTER the subtraction, under the same
guards the force-add uses (real dispatcher-owned worker, not a delegated
child). Non-worker sessions must stay stripped.
"""
import os

import pytest

import model_tools as mt
from toolsets import resolve_toolset

KANBAN_TOOLS = set(resolve_toolset("kanban"))

WORKER_ENV = {
    "HERMES_KANBAN_TASK": "t_test",
    "HERMES_KANBAN_RUN_ID": "1",
    "HERMES_SESSION_SOURCE": "kanban",
    "HERMES_KANBAN_CLAIM_LOCK": "testlock",
}


def _set_worker_env(on: bool):
    for key in WORKER_ENV:
        if on:
            os.environ[key] = WORKER_ENV[key]
        else:
            os.environ.pop(key, None)


@pytest.fixture(autouse=True)
def _clean_env():
    saved = {k: os.environ.get(k) for k in WORKER_ENV}
    _set_worker_env(False)
    yield
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def test_worker_keeps_kanban_tools_when_profile_disables_kanban():
    """The bug: profile disables the kanban chat toolset; its dispatched
    worker still needs the lifecycle handoff (comment/complete/block)."""
    _set_worker_env(True)
    tools = mt._select_tool_names(["terminal"], ["kanban"], quiet_mode=True)
    assert KANBAN_TOOLS & tools, (
        "dispatcher worker lost kanban lifecycle tools to disabled_toolsets"
    )


def test_non_worker_stays_stripped():
    """No widening: a plain session with kanban disabled keeps zero kanban tools."""
    _set_worker_env(False)
    tools = mt._select_tool_names(["terminal"], ["kanban"], quiet_mode=True)
    assert not (KANBAN_TOOLS & tools)


def test_worker_without_disable_unchanged():
    """When kanban is not disabled the force-add already covered the worker;
    the re-grant must not double-grant anything extra beyond the toolset."""
    _set_worker_env(True)
    tools = mt._select_tool_names(["terminal"], [], quiet_mode=True)
    assert KANBAN_TOOLS <= tools
