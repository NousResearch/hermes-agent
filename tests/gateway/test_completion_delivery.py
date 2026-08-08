"""Lifecycle-scoped gateway delivery regressions for terminal completions.

The gateway contract here is deliberately narrower than exactly-once: one live
GatewayRunner suppresses concurrent/replayed copies after successful adapter
injection, failed injection remains retryable, and durable async-delegation
state (when available) is acknowledged through its authoritative SQLite API.
"""

import asyncio
import json
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Any current/future durable compatibility path must stay in tmp state."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import tools.process_registry as pr_module

    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    return registry


def _runner(adapter, *, origins=None):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries=origins or {},
    )
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__("threading").Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    return runner


def _async_event(delegation_id="deleg_duplicate"):
    return {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": "agent:main:telegram:dm:12345:678",
        "goal": "Investigate flaky test",
        "status": "completed",
        "summary": "Found it",
        "api_calls": 1,
        "duration_seconds": 12.0,
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
        # PR #62479 stamps these on gateway-owned events. They must not
        # change the producer identity used for queue replay.
        "origin_profile": "default",
        "origin_hermes_home": "/tmp/hermes-default",
    }


def _completion_event(*, started_at, session_id="proc_reused"):
    return {
        "type": "completion",
        "session_id": session_id,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "started_at": started_at,
        "command": "echo done",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "done\n",
    }


def _stop_after_sleeps(monkeypatch, runner, count):
    sleep_calls = 0

    async def _bounded_sleep(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= count:
            runner._running = False

    monkeypatch.setattr(asyncio, "sleep", _bounded_sleep)


def test_duplicate_async_queue_replay_injects_once(monkeypatch, isolated_registry):
    """Byte-identical queue replays produce one turn in one gateway lifecycle."""
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(dict(_async_event()))
    isolated.put(dict(_async_event()))

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_awaited_once()


def test_unroutable_async_event_is_not_requeued_forever(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    event = _async_event("deleg_desktop_or_cli")
    event["session_key"] = "20260711_unparseable_ui_session"
    isolated.put(event)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=2)

    asyncio.run(runner._async_delegation_watcher(interval=0))

    adapter.handle_message.assert_not_awaited()
    assert isolated.empty()


def test_concurrent_claims_share_the_same_narrow_delivery_seam():
    """Concurrent consumers in one runner cannot both enter the adapter."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_injection(_event):
        entered.set()
        await release.wait()

    adapter = SimpleNamespace(handle_message=AsyncMock(side_effect=_blocked_injection))
    runner = _runner(adapter)
    event = _async_event()
    text = "completion"

    async def _exercise():
        first = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await entered.wait()
        second = asyncio.create_task(runner._deliver_completion_notification(text, dict(event)))
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first, second)

    assert sorted(asyncio.run(_exercise()), key=str) == [None, True]
    adapter.handle_message.assert_awaited_once()


def test_failed_async_injection_is_retried_and_only_success_is_acked(
    monkeypatch, isolated_registry,
):
    isolated = queue.Queue()
    monkeypatch.setattr(isolated_registry, "completion_queue", isolated)
    isolated.put(_async_event())

    adapter = SimpleNamespace(
        handle_message=AsyncMock(side_effect=[RuntimeError("temporary"), None])
    )
    runner = _runner(adapter)
    _stop_after_sleeps(monkeypatch, runner, count=3)

    from tools import async_delegation

    acknowledgements = []
    monkeypatch.setattr(
        async_delegation,
        "complete_completion_delivery",
        lambda delegation_id, _claim_id: acknowledgements.append(delegation_id) or True,
        raising=False,
    )

    asyncio.run(runner._async_delegation_watcher(interval=0))

    assert adapter.handle_message.await_count == 2
    assert acknowledgements == ["deleg_duplicate"]


def _persist_pending_completion(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    async_delegation._persist_completion(event, {
        "status": "completed",
        "summary": event["summary"],
    })


def test_explicit_kill_returns_output_before_consuming_notification(monkeypatch):
    import tools.process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_consumed",
        command="sleep 999",
        task_id="task",
        started_at=1.0,
        output_buffer="important terminal output\n",
        notify_on_complete=True,
    )
    session.process = MagicMock()
    session.process.pid = 4242
    registry._running[session.id] = session
    monkeypatch.setattr(registry, "_terminate_host_pid", lambda *_a, **_kw: None)
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(pr_module, "process_registry", registry)

    result = registry.kill_process(session.id)
    assert result["status"] == "killed"
    assert result["output"] == "important terminal output\n"
    assert registry.is_completion_consumed(session.id)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    adapter.handle_message.assert_not_awaited()


def test_process_tool_redacts_explicit_kill_output(monkeypatch):
    from tools import process_registry as pr_module

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_kill_redacted",
        command="printenv",
        task_id="task",
        started_at=1.0,
        output_buffer="PRIVATE_TOKEN=opaque-value\n",
        exited=True,
        exit_code=0,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)

    def _redact(result):
        assert result["output"] == "PRIVATE_TOKEN=opaque-value\n"
        result["output"] = "PRIVATE_TOKEN=<redacted>\n"
        return result

    monkeypatch.setattr(pr_module, "_redact_process_result", _redact)

    result = json.loads(pr_module._handle_process({
        "action": "kill",
        "session_id": session.id,
    }))
    assert result["output"] == "PRIVATE_TOKEN=<redacted>\n"


def test_autonomous_completion_redacts_real_command_and_output_secrets(monkeypatch):
    import agent.redact as redact_module
    import tools.process_registry as pr_module

    secret = "abc123randomopaquetokenvalue999"
    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_autonomous_redaction",
        command=f"printenv MY_SERVICE_TOKEN={secret}",
        task_id="task",
        started_at=1234.5,
        output_buffer=f"MY_SERVICE_TOKEN={secret}\nHOME=/home/user\n",
        exited=True,
        exit_code=0,
        notify_on_complete=True,
    )
    registry._finished[session.id] = session
    monkeypatch.setattr(pr_module, "process_registry", registry)
    monkeypatch.setattr(redact_module, "_REDACT_ENABLED", True)

    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(adapter)

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    asyncio.run(runner._run_process_watcher({
        "session_id": session.id,
        "check_interval": 0,
        "session_key": "agent:main:telegram:dm:123",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123",
        "notify_on_complete": True,
    }))

    delivered = adapter.handle_message.await_args.args[0]
    assert secret not in delivered.text
    assert "HOME=/home/user" in delivered.text


class TestSelfEchoGuard:
    """P0.e belt-and-braces: a genuine delegation whose payload is identical
    to the parent session's own recent assistant text must NOT be re-injected
    (the 2026-08-06 echo shape) — it is redirected to the direct outbound
    lane, and that redirect IS delivery: the durable claim completes, never
    releases. The guard fails open — it is not a delivery gate."""

    def _guard_runner(self, *, tail_content):
        runner = _runner(SimpleNamespace(handle_message=AsyncMock()))
        runner._session_db = SimpleNamespace(
            get_session=AsyncMock(return_value={"message_count": 40}),
            get_messages=AsyncMock(return_value=[
                {"role": "user", "content": "run the research"},
                {"role": "assistant", "content": tail_content},
            ]),
        )
        runner._classify_completion_target = AsyncMock(return_value="ok")
        runner._inject_watch_notification = AsyncMock(return_value=True)
        return runner

    def _claims(self, monkeypatch):
        import tools.async_delegation as ad

        calls = {"complete": 0, "release": 0, "drop": 0}
        monkeypatch.setattr(ad, "claim_completion_delivery", lambda *a: True)
        monkeypatch.setattr(
            ad, "complete_completion_delivery",
            lambda *a: calls.__setitem__("complete", calls["complete"] + 1),
        )
        monkeypatch.setattr(
            ad, "release_completion_delivery",
            lambda *a: calls.__setitem__("release", calls["release"] + 1),
        )
        monkeypatch.setattr(
            ad, "drop_completion_delivery",
            lambda *a: calls.__setitem__("drop", calls["drop"] + 1),
        )
        return calls

    def _evt(self, summary):
        evt = _async_event(delegation_id="deleg_echo_1")
        evt["summary"] = summary
        evt["parent_session_id"] = "sess-parent-1"
        return evt

    def test_self_echo_payload_guard_skips_injection(
        self, monkeypatch, isolated_registry, caplog,
    ):
        import logging

        # Whitespace differs from the stored row — the normalized compare
        # must still match (exact after collapse, nothing fuzzier).
        runner = self._guard_runner(
            tail_content="the full report\non every point",
        )
        calls = self._claims(monkeypatch)
        evt = self._evt("the full report on every point")

        with caplog.at_level(logging.WARNING, logger="gateway.run"):
            result = asyncio.run(
                runner._deliver_completion_notification("[envelope]", evt)
            )

        assert result is True
        assert runner._inject_watch_notification.call_count == 0
        assert "deleg_echo_1" in caplog.text
        redirected = isolated_registry.completion_queue.get_nowait()
        assert redirected["type"] == "sdk_background_result"
        assert redirected["payloads"] == ["the full report on every point"]
        assert redirected["_projected"] is True
        assert redirected["parent_session_id"] == "sess-parent-1"
        assert calls == {"complete": 1, "release": 0, "drop": 0}

    def test_novel_payload_injects_normally(
        self, monkeypatch, isolated_registry,
    ):
        runner = self._guard_runner(tail_content="an unrelated old answer")
        calls = self._claims(monkeypatch)
        evt = self._evt("a brand new subagent finding")

        result = asyncio.run(
            runner._deliver_completion_notification("[envelope]", evt)
        )

        assert result is True
        assert runner._inject_watch_notification.call_count == 1
        assert isolated_registry.completion_queue.empty()
        assert calls == {"complete": 1, "release": 0, "drop": 0}

    def test_guard_failure_fails_open_to_injection(
        self, monkeypatch, isolated_registry,
    ):
        runner = self._guard_runner(tail_content="whatever")
        runner._session_db.get_session = AsyncMock(
            side_effect=RuntimeError("db locked")
        )
        calls = self._claims(monkeypatch)
        evt = self._evt("some payload")

        result = asyncio.run(
            runner._deliver_completion_notification("[envelope]", evt)
        )

        assert result is True
        assert runner._inject_watch_notification.call_count == 1
        assert calls == {"complete": 1, "release": 0, "drop": 0}


class TestTerminalDropDurability:
    """P0.g: a completion whose parent session is permanently gone keeps its
    terminal drop (no auto-send — a rotated-away session can be a deliberate
    user statement like /new), but the PAYLOAD must survive the drop:
    projected into the transcript when the db can take it, else written to
    an orphaned-results file. Previously the drop discarded the only copy
    for legacy/id-less events. Nothing finished may ever become
    unrecoverable again."""

    def _claims(self, monkeypatch):
        import tools.async_delegation as ad

        calls = {"complete": 0, "release": 0, "drop": 0}
        monkeypatch.setattr(ad, "claim_completion_delivery", lambda *a: True)
        monkeypatch.setattr(
            ad, "complete_completion_delivery",
            lambda *a: calls.__setitem__("complete", calls["complete"] + 1),
        )
        monkeypatch.setattr(
            ad, "release_completion_delivery",
            lambda *a: calls.__setitem__("release", calls["release"] + 1),
        )
        monkeypatch.setattr(
            ad, "drop_completion_delivery",
            lambda *a: calls.__setitem__("drop", calls["drop"] + 1),
        )
        return calls

    def _evt(self):
        evt = _async_event(delegation_id="deleg_term_1")
        evt["parent_session_id"] = "sess-dead"
        return evt

    def test_terminal_drop_persists_payload_durably(
        self, monkeypatch, tmp_path, isolated_registry,
    ):
        # Projection path: session_db present — the payload lands in the
        # transcript (marked like a bg result so the continuity digest never
        # re-presents it); the drop itself and the claim disposition are
        # unchanged; no fallback file needed.
        adapter = SimpleNamespace(handle_message=AsyncMock())
        runner = _runner(adapter)
        runner._classify_completion_target = AsyncMock(return_value="terminal")
        runner._session_db = SimpleNamespace(append_message=AsyncMock())
        calls = self._claims(monkeypatch)

        result = asyncio.run(
            runner._deliver_completion_notification("[envelope]", self._evt())
        )

        assert result is None  # drop kept
        assert adapter.handle_message.call_count == 0  # no auto-send
        assert calls == {"complete": 0, "release": 0, "drop": 1}
        project = runner._session_db.append_message
        assert project.call_count == 1
        kw = project.call_args.kwargs
        assert kw["session_id"] == "sess-dead"
        assert kw["role"] == "assistant"
        assert kw["content"] == "Found it"
        assert kw["display_kind"] == "sdk_background_result"
        assert kw["display_metadata"]["orphaned"] == "terminal_drop"
        assert not (tmp_path / "orphaned-results").exists()

    def test_terminal_drop_without_db_writes_orphaned_results_file(
        self, monkeypatch, tmp_path, isolated_registry,
    ):
        adapter = SimpleNamespace(handle_message=AsyncMock())
        runner = _runner(adapter)
        runner._classify_completion_target = AsyncMock(return_value="terminal")
        runner._session_db = None
        calls = self._claims(monkeypatch)

        result = asyncio.run(
            runner._deliver_completion_notification("[envelope]", self._evt())
        )

        assert result is None
        assert adapter.handle_message.call_count == 0
        assert calls == {"complete": 0, "release": 0, "drop": 1}
        files = list((tmp_path / "orphaned-results").glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["payloads"] == ["Found it"]
        assert data["session_id"] == "sess-dead"
        assert data["reason"] == "terminal_drop"
        assert "sess-dead" in files[0].name
