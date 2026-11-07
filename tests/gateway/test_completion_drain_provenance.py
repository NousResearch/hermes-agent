"""Consumed completion drains must preserve merged or unproven input."""
import asyncio
from dataclasses import replace

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import merge_pending_message_event
from gateway.platforms.event import MessageType
from gateway.run import GatewayRunner
from tests.gateway.test_pending_drain_race import _make_adapter, _make_event, _sk
from tools import process_registry as processes


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.setattr(processes, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    adapter = _make_adapter()
    runner = GatewayRunner(GatewayConfig())
    source = _make_event().source
    monkeypatch.setattr(runner, "_build_process_event_source", lambda _: source)
    monkeypatch.setattr(runner, "_resolve_injection_adapter", lambda *_: adapter)
    adapter._active_sessions[_sk()] = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event)

    adapter.set_message_handler(handler)
    return registry, runner, adapter, processed


async def queue_completion(rig, count=1, kind="completion"):
    registry, runner, adapter, _ = rig
    events = []
    for i in range(count):
        session = processes.ProcessSession(id=f"proc_drain_{i}", command="build", started_at=123.0, session_key=_sk(),
                                           exited=True, exit_code=0, output_buffer=f"result {i}\n")
        registry._running[session.id] = session
        events.append(dict(type=kind, session_id=session.id, session_key=_sk(),
                           started_at=session.started_at, exit_code=0, output=session.output_buffer))
    if count == 1:
        assert await runner._inject_watch_notification("background process finished", events[0]) is True
    else:
        # Exercise the real coalescing flush without waiting for a timing window.
        key = ("drain-provenance",)
        futures = [asyncio.get_running_loop().create_future() for _ in events]
        runner._completion_notification_batches[key] = [
            ("background process finished", event, future) for event, future in zip(events, futures)]
        runner._completion_notification_batch_window = 0
        await runner._flush_process_completion_batch(key)
        assert all(future.result() is True for future in futures)
    return adapter._pending_messages[_sk()], events


async def drain(rig):
    _, _, adapter, _ = rig
    queued = adapter._pending_messages.pop(_sk())
    adapter._spawn_drain_task(queued, _sk())
    try:
        await asyncio.wait_for(asyncio.shield(adapter._session_tasks[_sk()]), timeout=5)
        assert _sk() not in adapter._active_sessions
        assert _sk() not in adapter._session_tasks
        assert _sk() not in adapter._pending_messages
    finally:
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("merged", ["text", "photo", "document", "same-text"])
async def test_consumed_completion_keeps_merged_human_input(rig, merged):
    registry, _, adapter, processed = rig
    queued, events = await queue_completion(rig)
    human = _make_event(text="human question" if merged != "same-text" else queued.text)
    if merged in {"photo", "document"}:
        human.message_type = MessageType.PHOTO if merged == "photo" else MessageType.DOCUMENT
        human.media_urls = ["/human/attachment"]
        human.media_types = ["image/png" if merged == "photo" else "application/pdf"]
    merge_pending_message_event(adapter._pending_messages, _sk(), human, merge_text=True)
    expected_text, expected_media = queued.text, list(queued.media_urls)
    registry.read_log(events[0]["session_id"])
    await drain(rig)
    assert processed == [queued], "consumed synthetic metadata discarded merged human input"
    assert processed[0].text == expected_text and human.text in processed[0].text
    assert processed[0].media_urls == expected_media


@pytest.mark.asyncio
@pytest.mark.parametrize("count,consumed", [(1, 1), (2, 1), (2, 2), (1, 0)])
async def test_unchanged_completion_consumption_and_poll(rig, count, consumed):
    registry, _, _, processed = rig
    queued, events = await queue_completion(rig, count)
    for i, event in enumerate(events):
        action = registry.wait if i < consumed else registry.poll
        action(event["session_id"])
        assert registry.is_completion_consumed(event["session_id"]) == (i < consumed)
    await drain(rig)
    assert processed == ([] if consumed == count else [queued])


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["missing", "invalid", "metadata-only", "rewritten", "media-types",
                                          "external", "ids-string", "ids-null", "replaced-process", "foreign-session"])
async def test_unproven_completion_is_preserved(rig, corruption):
    registry, _, adapter, processed = rig
    queued, events = await queue_completion(rig)
    registry.read_log(events[0]["session_id"])
    actions = {
        "missing": lambda: queued.__dict__.pop("_gateway_process_completion_provenance", None),
        "invalid": lambda: setattr(queued, "_gateway_process_completion_provenance", {"text": queued.text}),
        "metadata-only": lambda: replace(queued),
        "rewritten": lambda: setattr(queued, "text", queued.text + " rewritten content"),
        "media-types": lambda: setattr(queued, "media_types", ["image/png"]),
        "external": lambda: setattr(queued, "internal", False),
        "ids-string": lambda: queued.metadata.update(gateway_process_session_ids=events[0]["session_id"]),
        "ids-null": lambda: queued.metadata.update(gateway_process_session_ids=[None]),
        "replaced-process": lambda: registry._running.update({events[0]["session_id"]: processes.ProcessSession(
            id=events[0]["session_id"], command="replacement", started_at=456.0, session_key=_sk(), exited=True)}),
        "foreign-session": lambda: setattr(registry._running[events[0]["session_id"]], "session_key",
                                           "agent:other:telegram:dm:42"),
    }
    result = actions[corruption]()
    if corruption == "metadata-only":
        queued = result
        adapter._pending_messages[_sk()] = queued
    await drain(rig)
    assert processed == [queued]


@pytest.mark.asyncio
async def test_consumed_marker_does_not_drop_watch_or_user_event(rig):
    registry, _, adapter, processed = rig
    watch, events = await queue_completion(rig, kind="watch_match")
    registry.read_log(events[0]["session_id"])
    await drain(rig)
    human = _make_event("next real question")
    await adapter.handle_message(human)
    await asyncio.wait_for(asyncio.shield(adapter._session_tasks[_sk()]), timeout=5)
    assert processed == [watch, human]
    assert _sk() not in adapter._active_sessions


@pytest.mark.asyncio
async def test_suppression_advances_late_queue_and_releases_ownership(rig):
    registry, _, adapter, processed = rig
    completion, events = await queue_completion(rig)
    registry.read_log(events[0]["session_id"])
    adapter._pending_messages.pop(_sk())
    human = _make_event("next queued human")
    adapter._pending_messages[_sk()] = human
    adapter._spawn_drain_task(completion, _sk())
    owner = adapter._session_tasks[_sk()]
    await asyncio.wait_for(asyncio.shield(owner), timeout=5)
    next_owner = adapter._session_tasks.get(_sk())
    if next_owner is not None:
        await asyncio.wait_for(asyncio.shield(next_owner), timeout=5)
    assert processed == [human]
    assert _sk() not in adapter._active_sessions and _sk() not in adapter._session_tasks
    await adapter.cancel_background_tasks()
