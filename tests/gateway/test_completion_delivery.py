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


@pytest.mark.parametrize(
    ("event", "expected_identity"),
    [
        (
            {
                **_async_event("deleg_api_stable"),
                "session_key": "raw-api-target",
                "origin_session_id": "raw-api-target",
            },
            ("async_delegation", "deleg_api_stable", ""),
        ),
        (
            {
                **_completion_event(
                    started_at=1234.5,
                    session_id="proc_api_stable",
                ),
                "session_key": "raw-api-target",
                "origin_session_id": "raw-api-target",
                "platform": "",
                "chat_id": "",
            },
            ("completion", "proc_api_stable", 1234.5),
        ),
    ],
)
def test_apiserver_durable_completion_passes_stable_producer_identity(
    monkeypatch, event, expected_identity,
):
    """Replayable delegation/process events retain identity at self-post."""
    adapter = SimpleNamespace(supports_async_delivery=False)
    runner = _runner(adapter)
    runner.adapters = {Platform.API_SERVER: adapter}
    deliveries = []

    async def fake_deliver_wake(
        _adapter,
        *,
        text,
        session_id,
        producer_identity=None,
    ):
        deliveries.append({
            "text": text,
            "session_id": session_id,
            "producer_identity": producer_identity,
        })

    import gateway.wake as wake_mod

    monkeypatch.setattr(wake_mod, "deliver_wake", fake_deliver_wake)

    async def exercise():
        assert await runner._inject_watch_notification("done", dict(event)) is True
        assert await runner._inject_watch_notification("done", dict(event)) is True

    asyncio.run(exercise())

    assert [d["session_id"] for d in deliveries] == [
        "raw-api-target",
        "raw-api-target",
    ]
    assert [d["producer_identity"] for d in deliveries] == [
        expected_identity,
        expected_identity,
    ]


def _persist_pending_completion(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({
        "delegation_id": event["delegation_id"],
        "session_key": event["session_key"],
        "origin_ui_session_id": "",
        "origin_session_id": event.get("origin_session_id", ""),
        "parent_session_id": event.get("parent_session_id"),
        "dispatched_at": event["dispatched_at"],
    })
    async_delegation._persist_completion(event, {
        "status": "completed",
        "summary": event["summary"],
    })


def test_apiserver_durable_completion_recomputes_after_failed_agent_result(
    monkeypatch,
):
    """An outer durable replay must not inherit a cached failed agent run."""
    from aiohttp import web

    import gateway.platforms.api_server as api_server_mod
    import gateway.wake as wake_mod
    from tools import async_delegation

    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", ())
    monkeypatch.setattr(
        api_server_mod,
        "_idem_cache",
        api_server_mod._IdempotencyCache(),
    )

    event = {
        **_async_event("deleg_apiserver_retry"),
        "session_key": "raw-api-retry-session",
        "origin_session_id": "raw-api-retry-session",
    }
    _persist_pending_completion(event)

    response_statuses = []
    request_keys = []
    run_agent_calls = 0

    @web.middleware
    async def observe_status(request, handler):
        request_keys.append(request.headers.get("Idempotency-Key"))
        response = await handler(request)
        response_statuses.append(response.status)
        return response

    async def run_agent(**_kwargs):
        nonlocal run_agent_calls
        run_agent_calls += 1
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        if run_agent_calls == 1:
            return (
                {
                    "final_response": "",
                    "messages": [],
                    "api_calls": 1,
                    "completed": False,
                    "failed": True,
                    "error": "transient agent failure",
                },
                usage,
            )
        return (
            {
                "final_response": "completion accepted",
                "messages": [],
                "api_calls": 1,
                "completed": True,
            },
            usage,
        )

    async def exercise():
        adapter = api_server_mod.APIServerAdapter(
            api_server_mod.PlatformConfig(
                enabled=True,
                extra={"key": "test-key"},
            )
        )
        monkeypatch.setattr(
            adapter,
            "_ensure_session_db_async",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(adapter, "_run_agent", run_agent)

        app = web.Application(middlewares=[observe_status])
        app.router.add_post(
            "/v1/chat/completions",
            adapter._handle_chat_completions,
        )
        web_runner = web.AppRunner(app)
        await web_runner.setup()
        site = web.TCPSite(web_runner, "127.0.0.1", 0)
        await site.start()
        adapter._host = "127.0.0.1"
        adapter._port = site._server.sockets[0].getsockname()[1]

        delivery_runner = _runner(adapter)
        delivery_runner.adapters = {Platform.API_SERVER: adapter}
        try:
            first = await delivery_runner._deliver_completion_notification(
                "delegation complete",
                dict(event),
            )
            first_state = async_delegation.get_durable_delegation(
                event["delegation_id"]
            )
            second = await delivery_runner._deliver_completion_notification(
                "delegation complete",
                dict(event),
            )
            return first, first_state, second
        finally:
            await web_runner.cleanup()

    first, first_state, second = asyncio.run(exercise())

    assert first is False
    assert first_state["delivery_state"] == "pending"
    assert response_statuses[0] >= 500
    assert second is True
    assert response_statuses == [502, 200]
    assert request_keys[0]
    assert request_keys[0] == request_keys[1]
    assert run_agent_calls == 2
    final_state = async_delegation.get_durable_delegation(event["delegation_id"])
    assert final_state["delivery_state"] == "delivered"
    assert final_state["delivery_attempts"] == 2


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
