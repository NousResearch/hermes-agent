"""A usage-limit rerun runs the exact request it stored, as the person who sent it, in the session
that owns it (gateway/run_turn_usage_hold.py).

Contract: the queue entry carries the request text and its sender; at the reset nothing is
re-derived from the session (whose origin may name someone else, and whose latest request may be
newer). A local Desktop chat reruns through its session authority; an image turn is never promised
a rerun the stored text could not honour; the terminal frame says whether a rerun is scheduled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from types import SimpleNamespace

import pytest

from agent.quota_resume import resume_note
from agent.usage_hold import record_hold
from gateway import run_turn_usage_hold as uh
from gateway.config import Platform
from gateway.session import SessionSource

ROUTE = {"provider": "custom", "base_url": "http://127.0.0.1:9/v1"}
KEY = "agent:main:telegram:group:1"
ASK = "summarize the doc"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _source(user, platform=Platform.TELEGRAM, chat_id="1"):
    return SessionSource(platform=platform, chat_id=chat_id, chat_type="group", user_id=user, user_name=user)


def _ctx(source, key=KEY, message=ASK):
    cfg = {"desktop": {"quota_resume": {"enabled": True, "max_attempts": 2}}}
    return SimpleNamespace(session_key=key, session_id="sess-1", message=message, source=source,
                           scheduled_heartbeat=False, persist_user_display_kind=None,
                           persist_user_message=None, user_config=cfg)


class _Adapter:
    def __init__(self):
        self.events = []

    async def handle_message(self, event):
        self.events.append(event)


class _Store:
    def __init__(self, entries):
        import threading
        self._lock = threading.Lock()
        self._entries = entries

    def _ensure_loaded_locked(self):
        return None


def _runner(transcript=(), *, adapter=None, images=(), key=KEY, origin=None):
    state = SimpleNamespace(persistent=SimpleNamespace(native_image_paths=list(images)))
    entry = SimpleNamespace(session_id="sess-1", origin=origin or _source("alice"), transport_profile=None)

    async def load_transcript(_sid):
        return list(transcript)

    @contextlib.asynccontextmanager
    async def scope(_source):
        yield

    return SimpleNamespace(
        _refresh_fallback_model=lambda: None, _peek_session_state=lambda _k: state,
        session_store=_Store({key: entry}),
        async_session_store=SimpleNamespace(load_transcript=load_transcript),
        _is_session_running=lambda _k: False, _restored_source=lambda e: e.origin,
        _async_profile_scope_for_source=scope, _delivery_adapter_for=lambda _s: adapter,
        _resume_owner_authorized=lambda _k, _s: True, _session_key_for_source=lambda _s: key,
    )


def _hold():
    until = time.time() + 60
    record_hold("custom", until=until, base_url=ROUTE["base_url"], source="provider_error")
    return until + 3600


def _held_tail(text=ASK):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": uh.QUEUED_NOTICE}]


def test_rerun_runs_the_stored_request_as_its_sender_not_the_session_origin(home):
    """Alice opened the shared thread; Bob's message was held. The rerun is Bob's request, as Bob."""
    due = _hold()
    adapter = _Adapter()
    runner = _runner(_held_tail(), adapter=adapter, origin=_source("alice"))
    assert uh.held_turn_result(runner, _ctx(_source("bob")), ROUTE)["failed_turn_notice"] == uh.QUEUED_NOTICE

    assert asyncio.run(uh.rerun_due(runner, [home], now=due)) == 1
    (event,) = adapter.events
    assert event.source.user_id == "bob"
    assert event.text == resume_note(ASK)
    # Rebound to the live adapter: a source rebuilt from disk after a restart has no transport of its
    # own, and the multiplexed inbound path refuses an event whose transport owner it cannot find.
    assert event.source._transport_adapter_ref() is adapter


def test_a_newer_request_is_never_rerun_in_place_of_the_held_one(home):
    due = _hold()
    adapter = _Adapter()
    runner = _runner(_held_tail("a newer question"), adapter=adapter)
    uh.held_turn_result(runner, _ctx(_source("bob")), ROUTE)

    assert asyncio.run(uh.rerun_due(runner, [home], now=due)) == 0
    assert adapter.events == []
    assert json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig")) == {}


def test_an_image_turn_is_told_the_reset_but_never_promised_a_rerun(home):
    _hold()
    result = uh.held_turn_result(_runner(images=["/cache/photo.jpg"]), _ctx(_source("bob")), ROUTE)
    assert "failed_turn_notice" not in result
    assert result["quota_resume"]["scheduled"] is False
    assert not (home / uh.QUEUE_FILE).exists()


def test_local_desktop_chat_queues_and_reruns_through_its_session_owner(home, monkeypatch):
    """A canonical local Desktop chat: the gateway's session authority owns it, so the rerun is a
    producer turn admitted into its FIFO on the live session's own source."""
    import gateway.session_authorities as authorities
    import gateway.session_automation as automation
    import gateway.session_local_recovery as local_recovery

    key, chat = "agent:main:local:dm:desk-1", "desk-1"
    live = SimpleNamespace(route=key, source=_source("local", Platform.LOCAL, chat))
    authority = SimpleNamespace(sessions={chat: live})
    local_adapter = object()
    admitted = []

    async def admit_route_wake(runner, adapter, source, text, producer, metadata=None):
        admitted.append((adapter, source, text, producer))
        return True

    monkeypatch.setattr(authorities, "active_authority", lambda _r: authority)
    monkeypatch.setattr(automation, "admit_route_wake", admit_route_wake)
    monkeypatch.setattr(local_recovery, "restore_local_session", lambda _a, _c: None)
    monkeypatch.setattr(local_recovery, "local_adapter_map", lambda _a: {Platform.LOCAL: local_adapter})

    due = _hold()
    runner = _runner(_held_tail(), key=key, origin=live.source)
    result = uh.held_turn_result(runner, _ctx(_source("local", Platform.LOCAL, chat), key=key), ROUTE)
    assert result["failed_turn_notice"] == uh.QUEUED_NOTICE
    assert result["quota_resume"]["scheduled"] is True

    assert asyncio.run(uh.rerun_due(runner, [home], now=due)) == 1
    assert admitted == [(local_adapter, live.source, resume_note(ASK), uh.PRODUCER)]


def test_owned_messaging_rerun_refuses_a_committed_turn_from_another_sender(home, monkeypatch):
    """Under the session authority the producer turn borrows the route's last committed human
    envelope; it is admitted only when that envelope is the stored sender's."""
    import gateway.session_authorities as authorities
    import gateway.session_automation as automation

    admitted = []

    async def admit_route_wake(runner, adapter, source, text, producer, metadata=None):
        admitted.append(source.user_id)
        return True

    envelope = {"source": {"user_id": "carol", "chat_id": "1"}, "provenance": {}}
    monkeypatch.setattr(authorities, "active_authority", lambda _r: SimpleNamespace(sessions={}))
    monkeypatch.setattr(automation, "admit_route_wake", admit_route_wake)
    monkeypatch.setattr(automation, "committed_route_envelope", lambda _a, _e: envelope)

    due = _hold()
    runner = _runner(_held_tail(), adapter=_Adapter())
    uh.held_turn_result(runner, _ctx(_source("bob")), ROUTE)
    assert asyncio.run(uh.rerun_due(runner, [home], now=due)) == 0
    assert admitted == []

    envelope["source"]["user_id"] = "bob"
    uh.held_turn_result(runner, _ctx(_source("bob")), ROUTE)
    assert asyncio.run(uh.rerun_due(runner, [home], now=due)) == 1
    assert admitted == ["bob"]


def test_a_provider_refusal_with_a_reported_reset_queues_its_rerun(home):
    plan = {"eligible": True, "resume_at": time.time() + 600, "provider": "custom", "source": "provider_error"}
    result = {"failed": True, "quota_resume": plan, "messages": [{"role": "user", "content": ASK}]}
    uh.schedule_failed_turn(_runner(), _ctx(_source("bob")), result)
    assert result["failed_turn_notice"] == uh.QUEUED_NOTICE and plan["scheduled"] is True
    queued = json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig"))[KEY]
    assert queued["text"] == ASK and queued["sender"]["user_id"] == "bob"

    # The provider was sent image parts: the stored text cannot carry them.
    (home / uh.QUEUE_FILE).unlink()
    plan.pop("scheduled")
    images = {"failed": True, "quota_resume": plan,
              "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}
    uh.schedule_failed_turn(_runner(), _ctx(_source("bob")), images)
    assert plan["scheduled"] is False and "failed_turn_notice" not in images
    assert not (home / uh.QUEUE_FILE).exists()


def test_terminal_frame_carries_the_rerun_verdict_in_the_declared_shape():
    """The local Desktop frame (``completion_payload``) names the plan, including a model-only scope."""
    from gateway.session_settlement_recovery import completion_payload
    from tui_gateway.contracts.events import QuotaResume

    plan = {"eligible": True, "resume_at": time.time() + 600, "provider": "anthropic",
            "source": "usage_api", "scope": "model", "scheduled": True}
    frame = completion_payload({"admission_id": "a1", "request_id": "r1"}, {"outcome": "failed"},
                               "limit reached", {"result": {"failed": True, "quota_resume": plan}})
    assert frame["quota_resume"] == plan
    # The declared wire shape (``message.complete.quota_resume``) admits every field the plan carries.
    assert QuotaResume.model_validate(frame["quota_resume"]).scope == "model"


def test_an_unwritable_queue_keeps_the_provider_refusal_and_promises_nothing(home, monkeypatch):
    def full_disk(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(uh, "atomic_json_write", full_disk)
    plan = {"eligible": True, "resume_at": time.time() + 600, "provider": "custom", "source": "provider_error"}
    result = {"failed": True, "final_response": "limit hit", "quota_resume": plan,
              "messages": [{"role": "user", "content": ASK}]}
    uh.schedule_failed_turn(_runner(), _ctx(_source("bob")), result)
    assert plan["scheduled"] is False and "failed_turn_notice" not in result
    assert result["final_response"] == "limit hit"
