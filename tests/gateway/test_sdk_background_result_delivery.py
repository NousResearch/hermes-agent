"""Direct-outbound delivery of claude-agent-sdk background results.

The old lane wrapped the agent's own finished answer in a synthetic empty-id
async_delegation and re-injected it into the SAME session as a fake
delegation completion; the model recognized its own text, refused to "relay"
it, and the 2026-08-06 research report never left the box. These tests pin
the replacement: ``sdk_background_result`` events are sent DIRECTLY on the
platform outbound lane — adapter.handle_message is never involved, payloads
arrive as separate messages in order, and an event with no route is requeued,
never silently dropped.
"""

import asyncio
import queue
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, _drain_gateway_watch_events


@pytest.fixture(autouse=True)
def _fresh_ledger_db(tmp_path, monkeypatch):
    """Point the delivery ledger at a throwaway state.db per test."""
    import gateway.delivery_ledger as dl

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    return tmp_path / "state.db"


def _ledger_rows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM delivery_obligations ORDER BY created_at"
            )
        ]
    finally:
        conn.close()


def _adapter(**over):
    adapter = SimpleNamespace(
        supports_async_delivery=True,
        extract_media=lambda text: ([], text),
        extract_images=lambda text: ([], text),
        send=AsyncMock(),
        send_image=AsyncMock(),
        handle_message=AsyncMock(),
    )
    for key, value in over.items():
        setattr(adapter, key, value)
    return adapter


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner.session_store = SimpleNamespace(
        _ensure_loaded=lambda: None,
        _entries={},
    )
    runner._session_source_cache = {}
    runner._thread_metadata_for_source = lambda source, mid=None: None
    runner._session_db = SimpleNamespace(append_message=AsyncMock())
    return runner


def _event(**over):
    evt = {
        "type": "sdk_background_result",
        "payloads": ["Research landed — writing up.", "the full report"],
        "session_key": "agent:main:telegram:dm:12345:678",
        "parent_session_id": "sess-bg-1",
        "model": "m",
        "dispatched_at": 1000.0,
        "completed_at": 1012.0,
    }
    evt.update(over)
    return evt


def test_bg_result_main_agent_sent_outbound_never_reinjected():
    adapter = _adapter()
    runner = _runner(adapter)
    evt = _event()

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is True
    assert adapter.handle_message.call_count == 0
    assert adapter.send.call_count == 2
    for call in adapter.send.call_args_list:
        content = call.kwargs["content"]
        assert "[ASYNC DELEGATION COMPLETE" not in content
        assert "[USER IS WAITING" not in content
        assert call.kwargs["chat_id"] == "12345"


def test_bg_result_payloads_sent_in_order_without_directive():
    adapter = _adapter()
    runner = _runner(adapter)
    evt = _event(payloads=["one", "two", "three"])

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is True
    sent = [c.kwargs["content"] for c in adapter.send.call_args_list]
    assert sent == ["one", "two", "three"]
    assert not any("[USER IS WAITING" in s for s in sent)


def test_bg_result_unroutable_event_requeued_not_dropped():
    # Unparseable/empty session_key: no source -> retryable False, not None.
    adapter = _adapter()
    runner = _runner(adapter)
    assert asyncio.run(
        runner._deliver_sdk_background_result(_event(session_key=""))
    ) is False
    assert adapter.send.call_count == 0

    # Adapter for the platform missing entirely -> False too.
    runner_no_adapter = _runner(None)
    assert asyncio.run(
        runner_no_adapter._deliver_sdk_background_result(_event())
    ) is False

    # Non-push adapter (api_server-style) delivers by re-injection — no
    # deliverable route on this lane.
    non_push = _adapter(supports_async_delivery=False)
    runner_non_push = _runner(non_push)
    assert asyncio.run(
        runner_non_push._deliver_sdk_background_result(_event())
    ) is False
    assert non_push.send.call_count == 0

    # Empty payloads: nothing to send — dropped as None, never retried.
    assert asyncio.run(
        runner._deliver_sdk_background_result(_event(payloads=[]))
    ) is None


def test_bg_result_partial_failure_requeues_remaining():
    adapter = _adapter()
    adapter.send = AsyncMock(side_effect=[None, RuntimeError("flaky network")])
    runner = _runner(adapter)
    evt = _event(payloads=["one", "two", "three"])

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is False
    # The delivered prefix is trimmed so the retry sends only the remainder.
    assert evt["payloads"] == ["two", "three"]


def test_bg_result_projected_into_transcript(_fresh_ledger_db):
    # D3: the incident's report never entered state.db — FTS-invisible, one
    # session retire away from unrecoverable. EVERY payload of the burst
    # must project, intermediates included, marked for digest exclusion.
    adapter = _adapter()
    runner = _runner(adapter)
    evt = _event()

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is True
    project = runner._session_db.append_message
    assert project.call_count == 2
    for call, payload in zip(
        project.call_args_list,
        ["Research landed — writing up.", "the full report"],
    ):
        assert call.kwargs["session_id"] == "sess-bg-1"
        assert call.kwargs["role"] == "assistant"
        assert call.kwargs["content"] == payload
        assert call.kwargs["display_kind"] == "sdk_background_result"

    # Save-first: an UNROUTABLE event still projects — the text becomes
    # durable even while delivery is stuck.
    runner2 = _runner(_adapter())
    evt2 = _event(session_key="")
    assert asyncio.run(runner2._deliver_sdk_background_result(evt2)) is False
    assert runner2._session_db.append_message.call_count == 2


def test_bg_result_registers_delivery_ledger_obligation(_fresh_ledger_db):
    adapter = _adapter()
    runner = _runner(adapter)
    evt = _event()

    delivered = asyncio.run(runner._deliver_sdk_background_result(evt))

    assert delivered is True
    rows = _ledger_rows(_fresh_ledger_db)
    assert len(rows) == 2
    contents = {r["content"] for r in rows}
    assert contents == {"Research landed — writing up.", "the full report"}
    assert all(r["state"] == "delivered" for r in rows)
    assert all(r["session_key"] == "agent:main:telegram:dm:12345:678" for r in rows)
    assert all(r["platform"] == "telegram" for r in rows)


def test_requeued_event_does_not_double_project(_fresh_ledger_db):
    adapter = _adapter()
    adapter.send = AsyncMock(side_effect=[None, RuntimeError("flaky"), None])
    runner = _runner(adapter)
    evt = _event(payloads=["one", "two"])

    # First pass: payload 1 sends, payload 2 fails -> requeue contract.
    assert asyncio.run(runner._deliver_sdk_background_result(evt)) is False
    assert evt["payloads"] == ["two"]
    assert runner._session_db.append_message.call_count == 2

    # Retry of the SAME event: no re-projection, remainder delivers.
    assert asyncio.run(runner._deliver_sdk_background_result(evt)) is True
    assert runner._session_db.append_message.call_count == 2

    rows = {r["content"]: r["state"] for r in _ledger_rows(_fresh_ledger_db)}
    # The failed payload's obligation was restarted on retry and ends
    # delivered; no duplicate rows exist (same obligation id both passes).
    assert rows == {"one": "delivered", "two": "delivered"}


def test_post_turn_drain_leaves_sdk_background_result_on_queue():
    # The post-turn watch drain owns only watch events; it must requeue an
    # sdk_background_result rather than consuming it (a consumed event is a
    # silently lost finished result).
    q = queue.Queue()
    evt = _event()
    q.put(evt)
    q.put({"type": "watch_match", "pattern": "x", "output": "y"})

    watch_events = _drain_gateway_watch_events(q)

    assert [e.get("type") for e in watch_events] == ["watch_match"]
    remaining = []
    while not q.empty():
        remaining.append(q.get_nowait())
    assert remaining == [evt]


def test_projection_failure_writes_fallback_file_and_delivery_proceeds(
    monkeypatch, tmp_path,
):
    # P0.g: when the transcript projection is unavailable (no session_db /
    # no parent) or fails, the payload previously lived ONLY in the
    # in-memory queue — a gateway restart erased it. It must gain a durable
    # copy under ~/.hermes/orphaned-results/ while the delivery attempt
    # proceeds unchanged.
    import json as _json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Branch A: session_db unavailable.
    adapter = _adapter()
    runner = _runner(adapter)
    runner._session_db = None
    evt = _event()
    assert asyncio.run(runner._deliver_sdk_background_result(evt)) is True
    assert adapter.send.call_count == 2  # delivery proceeded
    files = sorted((tmp_path / "orphaned-results").glob("*.json"))
    assert len(files) == 1
    data = _json.loads(files[0].read_text(encoding="utf-8"))
    assert data["payloads"] == [
        "Research landed — writing up.", "the full report",
    ]
    assert data["reason"] == "projection_unavailable"
    assert data["session_id"] == "sess-bg-1"

    # A requeue-style retry of the SAME event never double-writes
    # (_projected is stamped once per event lifetime).
    assert asyncio.run(runner._deliver_sdk_background_result(evt)) is True
    assert len(list((tmp_path / "orphaned-results").glob("*.json"))) == 1

    # Branch B: db present but every append fails.
    adapter_b = _adapter()
    runner_b = _runner(adapter_b)
    runner_b._session_db = SimpleNamespace(
        append_message=AsyncMock(side_effect=RuntimeError("db locked"))
    )
    evt_b = _event(payloads=["only payload"])
    assert asyncio.run(runner_b._deliver_sdk_background_result(evt_b)) is True
    assert adapter_b.send.call_count == 1
    files = sorted((tmp_path / "orphaned-results").glob("*.json"))
    assert len(files) == 2
    newest = [
        f for f in files
        if _json.loads(f.read_text(encoding="utf-8"))["reason"]
        == "projection_failed"
    ]
    assert len(newest) == 1
    assert _json.loads(newest[0].read_text(encoding="utf-8"))["payloads"] == [
        "only payload"
    ]
