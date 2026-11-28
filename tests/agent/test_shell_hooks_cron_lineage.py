"""Shell-hook payloads carry ``cron_session``: whether the call belongs to a cron run's lineage.

``session_id`` names whoever calls the tool, and a delegate_task child mints its own, so a
hook that scopes itself by ``session_id`` alone is bypassed by delegating. The flag is read
from the ``HERMES_CRON_SESSION`` ContextVar — which a child thread inherits through
``contextvars.copy_context()`` — and never from the process-env mirror, which is
last-writer-wins across concurrent sessions.
"""

from __future__ import annotations

import contextvars
import threading

import pytest

from agent.shell_hooks import _payload_fields
from gateway.session_context import _VAR_MAP


@pytest.fixture
def cron_var():
    var = _VAR_MAP["HERMES_CRON_SESSION"]
    token = var.set("")
    yield var
    var.reset(token)


def test_payload_marks_a_cron_run(cron_var):
    cron_var.set("1")
    assert _payload_fields({"tool_name": "terminal"})["cron_session"] == "1"


def test_payload_is_empty_outside_a_cron_run(cron_var):
    assert _payload_fields({"tool_name": "terminal"})["cron_session"] == ""


def test_a_delegated_child_inherits_the_flag(cron_var):
    """The child runs in a copied context with its own session id; the flag must survive."""
    cron_var.set("1")
    ctx = contextvars.copy_context()
    seen = {}

    def child():
        seen["payload"] = _payload_fields({"tool_name": "terminal", "session_id": "child-session"})

    t = threading.Thread(target=lambda: ctx.run(child))
    t.start(); t.join()

    assert seen["payload"]["session_id"] == "child-session"
    assert seen["payload"]["cron_session"] == "1"


def test_the_process_env_mirror_is_ignored(cron_var, monkeypatch):
    """Another session's env write must not mark this one as cron (false positive)."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    assert _payload_fields({"tool_name": "terminal"})["cron_session"] == ""
