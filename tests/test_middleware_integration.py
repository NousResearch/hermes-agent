"""Integration test: complete_task / fail_task trigger on_session_end hook.

Verifies the S3 chain wiring lands — specifically that middlewares registered
via `register_defaults()` get their `on_session_end` hook called when bus
tasks close via `core.complete_task` or `core.fail_task`.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from unittest import mock

import pytest

from agent_bus.middleware import BaseMiddleware, clear_registry, register


@pytest.fixture
def core_module(tmp_path, monkeypatch):
    """Reset agent_bus.core + storage with tmp DB (mirrors the finalizer fixture)."""
    db_path = tmp_path / f"agent_bus_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("AGENT_BUS_DB_PATH", str(db_path))

    from agent_bus import storage as _storage
    _storage._DB_CONN = None

    from agent_bus import core as _core
    monkeypatch.setattr(_core, "_slack_post_assignment", lambda *a, **k: (None, None))
    monkeypatch.setattr(_core, "_slack_reply", lambda *a, **k: True)
    monkeypatch.setattr(_core, "_notify_agent", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_openclaw", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_hermes_via_slack", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_user_of_outcome", lambda *a, **k: False)
    wiki_dir = tmp_path / "wiki_memory"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_core, "_wiki_memory_dir", lambda: wiki_dir)
    monkeypatch.setattr(_core, "_append_wiki_log", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_rebuild_agent_bus_moc", lambda: None)

    # Reset middleware registration so the test controls what runs
    clear_registry()
    # Re-enable lazy registration
    _core._middlewares_registered = False

    yield _core

    _storage._DB_CONN = None
    clear_registry()


def _assign(core, **over):
    defaults = dict(
        from_agent="hermes", to_agent="openclaw", goal="do the thing",
        priority="P2",
    )
    defaults.update(over)
    return core.assign_task(**defaults)


class TestCloseTriggersOnSessionEnd:
    def test_complete_fires_on_session_end(self, core_module):
        hook_calls: list[str] = []

        @register(order=500)
        class Spy(BaseMiddleware):
            name = "spy-on-close"

            def on_session_end(self, ctx):
                hook_calls.append(ctx.thread_id or "")
                return ctx

        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="finished")

        assert hook_calls == [t["task_id"]]

    def test_fail_fires_on_session_end(self, core_module):
        hook_calls: list[str] = []

        @register(order=500)
        class Spy(BaseMiddleware):
            name = "spy-on-fail"

            def on_session_end(self, ctx):
                hook_calls.append(ctx.thread_id or "")
                return ctx

        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.fail_task(task_id=t["task_id"], agent="openclaw", reason="blocked")

        assert hook_calls == [t["task_id"]]

    def test_master_switch_off_skips_chain(self, core_module, monkeypatch):
        hook_calls: list[str] = []

        @register(order=500)
        class Spy(BaseMiddleware):
            name = "spy-off"

            def on_session_end(self, ctx):
                hook_calls.append("ran")
                return ctx

        monkeypatch.setenv("HERMES_MIDDLEWARE_CHAIN", "off")
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")

        assert hook_calls == []

    def test_middleware_exception_does_not_break_close(self, core_module):
        @register(order=500, critical=False)
        class Buggy(BaseMiddleware):
            name = "buggy-spy"

            def on_session_end(self, ctx):
                raise RuntimeError("oops")

        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        # Must not raise — core catches middleware errors
        result = core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")
        assert result["status"] == "done"

    def test_idempotent_close_still_skips_chain(self, core_module):
        hook_calls: list[str] = []

        @register(order=500)
        class Spy(BaseMiddleware):
            name = "spy-idempotent"

            def on_session_end(self, ctx):
                hook_calls.append("ran")
                return ctx

        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="first")
        # Second close returns early; should NOT fire chain again
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="second")

        assert hook_calls == ["ran"]
