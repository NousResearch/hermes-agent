"""An admission observer that registers structured tool_start_callback/tool_complete_callback
sinks (as /v1/runs does) must still receive ``tool.progress`` (live MCP progress ticks have no
other delivery path), while the producer's own ``tool.started``/``tool.completed`` duplicate of
those two stays suppressed -- that suppression exists precisely so a structured observer is not
double-fed the same lifecycle moment by both its dedicated sink and the raw progress relay.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from gateway.session_api_turn import observe_api_turn, publish_api_tool_event, wire_api_observers
from hermes_state_runtime import RuntimeStoreError


class _FakeEventStream:
    def __init__(self, admission_id: str):
        self.lock = threading.Lock()
        self.execution = {"admission_id": admission_id}


def _fake_admission(monkeypatch, *, admission_id="adm-1", session_id="sess-1",
                     live_admission_id=None, generation_ok=True):
    # Keep this test to the observer registration/forwarding seam: no real SessionAuthority/DB.
    monkeypatch.setattr(
        "hermes_state_runtime.get_session_admission",
        lambda db, admission_id=None: {"status": "started", "outcome": "queued"},
    )
    monkeypatch.setattr("gateway.session_api_turn.admission_result", lambda db, admission_id: None)
    # The CURRENT live execution's admission_id (what _api_observers routes by) defaults to the
    # one this observer registers under; a caller passes a different one to model a newer
    # admission superseding this session while an old observer is still mid-await.
    live = SimpleNamespace(event_stream=_FakeEventStream(live_admission_id or admission_id))

    def check_approval_generation(sid, gen):
        if not generation_ok:
            raise RuntimeStoreError("stale_generation")

    authority = SimpleNamespace(
        db=None, waiters={}, api_observers={}, sessions={session_id: live},
        check_approval_generation=check_approval_generation,
        _publish_pending=lambda ref: None, _schedule=lambda ref: None,
    )
    ref = SimpleNamespace(session_id=session_id)
    row = {"admission_id": admission_id, "status": "queued"}
    return authority, ref, row


@pytest.mark.asyncio
async def test_admission_observer_forwards_tool_progress_without_duplicating_lifecycle(monkeypatch):
    authority, ref, row = _fake_admission(monkeypatch)
    admitted = (authority, ref, row)

    structured = []
    progress_events = []
    task = asyncio.create_task(observe_api_turn(
        admitted, stream_delta_callback=lambda *_: None,
        tool_start_callback=lambda *a: structured.append(("start", a)),
        tool_complete_callback=lambda *a: structured.append(("complete", a)),
        tool_progress_callback=lambda *a, **kw: progress_events.append((a, kw))))
    await asyncio.sleep(0)  # let observe_api_turn register its observer and suspend on the waiter

    agent = SimpleNamespace(tool_progress_callback=None, status_callback=None, interim_assistant_callback=None)
    wire_api_observers(agent, (authority, ref.session_id, 7), want_interim=False)
    agent.tool_progress_callback(
        "tool.progress", "fetch", "halfway", None,
        tool_call_id="call-1", progress=1.0, total=2.0, percent=50)
    agent.tool_progress_callback("tool.started", "fetch", "preview", None, tool_call_id="call-1")
    # The structured sinks get tool.started/tool.completed through their own dedicated path
    # (publish_api_tool_event), proving the lifecycle moment still reaches the observer --
    # just not twice, through tool_progress_callback as well.
    publish_api_tool_event(authority, ref.session_id, 7, "tool.start", "call-1", "fetch", {})

    authority.waiters[row["admission_id"]].set_result(None)
    with pytest.raises(RuntimeStoreError):
        await asyncio.wait_for(task, 5)

    assert progress_events == [
        (("tool.progress", "fetch", "halfway", None),
         {"tool_call_id": "call-1", "progress": 1.0, "total": 2.0, "percent": 50})]
    # tool.started reached the observer exactly once, via tool_start_callback -- never
    # duplicated through tool_progress_callback (which only forwarded tool.progress above).
    assert structured == [("start", ("call-1", "fetch", {}))]


@pytest.mark.parametrize(
    ("live_admission_id", "generation_ok", "expect_delivered"),
    [("adm-1", True, True), ("adm-other", True, False), ("adm-1", False, False)],
    ids=["matching_admission_and_generation", "other_admission_excluded", "stale_generation_excluded"],
)
@pytest.mark.asyncio
async def test_tool_progress_routes_only_to_the_exact_live_admission_and_generation(
    monkeypatch, live_admission_id, generation_ok, expect_delivered
):
    """``_api_observers`` looks up registered observers by the session's CURRENT live
    admission_id, not the one this observer registered under: a newer admission superseding the
    session (or a generation that no longer matches, e.g. a stop/retry) must exclude a still-
    awaiting observer, never cross-deliver another admission's tool.progress ticks to it."""
    authority, ref, row = _fake_admission(
        monkeypatch, admission_id="adm-1", live_admission_id=live_admission_id, generation_ok=generation_ok)
    admitted = (authority, ref, row)

    progress_events = []
    task = asyncio.create_task(observe_api_turn(
        admitted, stream_delta_callback=lambda *_: None,
        tool_progress_callback=lambda *a, **kw: progress_events.append((a, kw))))
    await asyncio.sleep(0)

    agent = SimpleNamespace(tool_progress_callback=None, status_callback=None, interim_assistant_callback=None)
    wire_api_observers(agent, (authority, ref.session_id, 7), want_interim=False)
    agent.tool_progress_callback(
        "tool.progress", "fetch", "halfway", None,
        tool_call_id="call-1", progress=1.0, total=2.0, percent=50)

    authority.waiters[row["admission_id"]].set_result(None)
    with pytest.raises(RuntimeStoreError):
        await asyncio.wait_for(task, 5)

    expected = [(("tool.progress", "fetch", "halfway", None),
                  {"tool_call_id": "call-1", "progress": 1.0, "total": 2.0, "percent": 50})]
    assert progress_events == (expected if expect_delivered else [])


@pytest.mark.parametrize(
    ("emit_progress", "is_error", "total", "percent"),
    [
        (True, False, 2.0, 50),
        (True, True, None, None),
        (False, False, None, None),
        (False, True, None, None),
    ],
    ids=["completed_ok_with_total", "completed_error_missing_total",
         "nonreporting_completed_ok", "nonreporting_completed_error"],
)
@pytest.mark.asyncio
async def test_progress_only_observer_gets_full_lifecycle_and_unsynthesized_progress(
    monkeypatch, emit_progress, is_error, total, percent
):
    """An observer with no structured tool_start_callback/tool_complete_callback (the session
    chat stream) has no other delivery path for tool lifecycle: it must see the producer's own
    tool.started/tool.completed (with its failure verdict) verbatim. A server reporting
    progress with no total is still reporting (missing_total is its own case, not
    'nonreporting'). A genuinely nonreporting server (emit_progress=False) never ticks
    tool.progress at all -- the observer gets exactly start/complete, no invented tick."""
    authority, ref, row = _fake_admission(monkeypatch)
    admitted = (authority, ref, row)

    progress_events = []
    task = asyncio.create_task(observe_api_turn(
        admitted, stream_delta_callback=lambda *_: None,
        tool_progress_callback=lambda *a, **kw: progress_events.append((a, kw))))
    await asyncio.sleep(0)

    agent = SimpleNamespace(tool_progress_callback=None, status_callback=None, interim_assistant_callback=None)
    wire_api_observers(agent, (authority, ref.session_id, 7), want_interim=False)

    agent.tool_progress_callback("tool.started", "fetch", "fetching url", None, tool_call_id="call-1")
    if emit_progress:
        agent.tool_progress_callback(
            "tool.progress", "fetch", "working", None,
            tool_call_id="call-1", progress=1.0, total=total, percent=percent)
    agent.tool_progress_callback(
        "tool.completed", "fetch", None, None, tool_call_id="call-1", is_error=is_error)

    authority.waiters[row["admission_id"]].set_result(None)
    with pytest.raises(RuntimeStoreError):
        await asyncio.wait_for(task, 5)

    expected = [(("tool.started", "fetch", "fetching url", None), {"tool_call_id": "call-1"})]
    if emit_progress:
        expected.append((("tool.progress", "fetch", "working", None),
                          {"tool_call_id": "call-1", "progress": 1.0, "total": total, "percent": percent}))
    expected.append((("tool.completed", "fetch", None, None), {"tool_call_id": "call-1", "is_error": is_error}))
    assert progress_events == expected
