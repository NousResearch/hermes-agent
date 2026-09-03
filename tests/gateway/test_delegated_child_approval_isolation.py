"""Delegated children must never enqueue or await gateway approvals under
their parent's session key.

Regression for nesquena/hermes-webui#6100: a delegated child running in a
gateway session resolved the parent's ``HERMES_SESSION_KEY`` (via the copied
context or the process env), so a dangerous child command enqueued a gateway
approval under the PARENT's key — surfacing a phantom "waiting for permission"
attention dot on the parent session.  ``delegated_child_context`` now rebinds
the child's approval authority at the worker boundary, so approval lookups fail
closed to a child-owned key instead of the parent's.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from agent.delegation_context import delegated_child_context
from gateway.session_context import _SESSION_KEY, _UNSET, get_session_env
from tools import approval as approval_mod
from tools.approval import (
    check_dangerous_command,
    detect_dangerous_command,
    get_current_session_key,
    register_gateway_notify,
    set_current_session_key,
)

PARENT_KEY = "agent:main:parent-chat"
CHILD_KEY = "subagent:child-session"
DANGEROUS_COMMAND = "sudo rm -rf /etc/hosts"


@pytest.fixture(autouse=True)
def _isolate_approval_and_session_state():
    """Pin env, contextvars, and the global approval registries per test."""
    saved_env = {
        name: os.environ.get(name)
        for name in ("HERMES_SESSION_KEY", "HERMES_SESSION_PLATFORM")
    }
    for name in saved_env:
        os.environ.pop(name, None)
    saved_session_key = _SESSION_KEY.get()
    _SESSION_KEY.set(_UNSET)
    saved_approval_key = approval_mod._approval_session_key.get()
    approval_mod._approval_session_key.set("")
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._pending.clear()
    approval_mod._session_approved.clear()
    approval_mod._permanent_approved.clear()
    yield
    for name, value in saved_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _SESSION_KEY.set(saved_session_key)
    approval_mod._approval_session_key.set(saved_approval_key)
    approval_mod._gateway_queues.clear()
    approval_mod._gateway_notify_cbs.clear()
    approval_mod._pending.clear()
    approval_mod._session_approved.clear()
    approval_mod._permanent_approved.clear()


def _bind_parent_approval_identity() -> None:
    """Model a parent gateway turn: both approval resolution layers are set, so
    ``copy_context()`` would carry both into a child worker thread."""
    set_current_session_key(PARENT_KEY)
    _SESSION_KEY.set(PARENT_KEY)


def test_child_context_rebinds_approval_key_away_from_parent():
    _bind_parent_approval_identity()

    with delegated_child_context("child-session"):
        assert get_current_session_key() == CHILD_KEY
        assert get_session_env("HERMES_SESSION_KEY") == CHILD_KEY

    # Parent identity restored after the child scope exits.
    assert get_current_session_key() == PARENT_KEY
    assert get_session_env("HERMES_SESSION_KEY") == PARENT_KEY


def test_child_context_never_falls_back_to_process_env_parent_key():
    # The original bug also fired when the worker thread received NO copied
    # context: get_session_env() fell back to the process-wide env the gateway
    # process sets, resolving the parent's key.
    os.environ["HERMES_SESSION_KEY"] = PARENT_KEY
    os.environ["HERMES_SESSION_PLATFORM"] = "webui"

    with delegated_child_context("child-session"):
        assert get_current_session_key() == CHILD_KEY
        assert get_session_env("HERMES_SESSION_KEY") == CHILD_KEY

    # The parent process env is untouched once the scope exits.
    assert get_session_env("HERMES_SESSION_KEY") == PARENT_KEY


def test_child_construction_without_session_id_clears_approval_key():
    # Child CONSTRUCTION enters delegated_child_context() before the child has
    # a session id; the approval key must still never resolve to the parent's.
    _bind_parent_approval_identity()

    with delegated_child_context():
        assert get_current_session_key() != PARENT_KEY
        assert get_session_env("HERMES_SESSION_KEY") != PARENT_KEY

    assert get_current_session_key() == PARENT_KEY


def test_copied_context_worker_thread_never_resolves_parent_key():
    # The real delegation boundary: copy_context() (parent's vars) + a fresh
    # worker thread + delegated_child_context(child.session_id).
    _bind_parent_approval_identity()
    os.environ["HERMES_SESSION_KEY"] = PARENT_KEY

    def worker() -> str:
        with delegated_child_context("child-session"):
            return get_current_session_key()

    ctx = copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        observed = pool.submit(ctx.run, worker).result()

    assert observed == CHILD_KEY
    assert observed != PARENT_KEY


def test_dangerous_child_command_does_not_enqueue_or_notify_under_parent_key():
    """Durable regression: a delegated child's dangerous command must not
    enqueue (or await) a gateway approval under the parent's session key."""
    os.environ["HERMES_SESSION_KEY"] = PARENT_KEY
    os.environ["HERMES_SESSION_PLATFORM"] = "webui"
    # The parent session has a live gateway notify callback: if the child
    # resolved the parent's key, _await_gateway_decision would fire it and
    # block the child's thread awaiting the parent's approval.
    notified: list = []
    register_gateway_notify(PARENT_KEY, lambda data: notified.append(data))

    is_dangerous, _, _ = detect_dangerous_command(DANGEROUS_COMMAND)
    assert is_dangerous, "test command must be flagged dangerous"

    with delegated_child_context("child-session"):
        result = check_dangerous_command(DANGEROUS_COMMAND, env_type="local")

    # Fail-closed: the command was NOT authorized...
    assert result.get("approved") is False
    # ...nothing was enqueued or awaited under the PARENT's key...
    assert PARENT_KEY not in approval_mod._gateway_queues
    assert PARENT_KEY not in approval_mod._pending
    assert notified == []
    # ...and anything recorded lives under the child's own key, never the
    # parent's.
    assert CHILD_KEY in approval_mod._pending
    assert approval_mod._pending[CHILD_KEY]["command"] == DANGEROUS_COMMAND
