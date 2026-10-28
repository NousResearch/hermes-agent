"""Gateway admission while the turn's provider is at a usage limit (gateway/run_turn_usage_hold.py).

Contract: a held provider with no fallback is never called — the turn answers at once with the
local reset time; an opted-in turn is queued durably and re-sent exactly once at the reset, only
while the session is still on that held turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from types import SimpleNamespace

import pytest

from agent.usage_hold import record_hold
from gateway.config import Platform
from gateway import run_turn_usage_hold as uh
from agent.quota_resume import resume_note


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _source(user="alice", platform=Platform.TELEGRAM, chat_id="1"):
    from gateway.session import SessionSource
    return SessionSource(platform=platform, chat_id=chat_id, chat_type="dm", user_id=user, user_name=user)


def _ctx(*, opted_in=True, max_attempts=2, **over):
    cfg = {"desktop": {"quota_resume": {"enabled": opted_in, "max_attempts": max_attempts}}}
    base = dict(session_key="agent:main:telegram:dm:1", session_id="sess-1", message="summarize the doc",
                source=_source(), scheduled_heartbeat=False, persist_user_display_kind=None,
                persist_user_message=None, user_config=cfg)
    base.update(over)
    return SimpleNamespace(**base)


def _runner(fallback=None, images=()):
    state = SimpleNamespace(persistent=SimpleNamespace(native_image_paths=list(images)))
    return SimpleNamespace(_refresh_fallback_model=lambda: fallback, _peek_session_state=lambda _k: state)


ROUTE = {"provider": "custom", "base_url": "http://127.0.0.1:9/v1"}


def test_held_provider_answers_now_and_a_fallback_chain_runs_the_turn(home):
    assert uh.held_turn_result(_runner(), _ctx(), ROUTE) is None  # no hold: the turn runs
    until = time.time() + 1800
    record_hold("custom", until=until, base_url=ROUTE["base_url"], source="provider_error")

    result = uh.held_turn_result(_runner(), _ctx(opted_in=False), ROUTE)
    assert result["failed"] and result["api_calls"] == 0 and result["failure_reason"] == "rate_limit"
    assert result["failure_resets_at"] == until
    assert time.strftime("%H:%M", time.localtime(until)) in result["final_response"]
    assert result["agent_persisted"] is False  # the gateway must write the user row itself
    assert "failed_turn_notice" not in result  # not opted in: "send it again" guidance stays
    assert not (home / uh.QUEUE_FILE).exists()

    assert uh.held_turn_result(_runner([{"provider": "nous"}]), _ctx(), ROUTE) is None


def test_opted_in_turn_is_queued_once_per_session_and_bounded(home):
    now = time.time()
    record_hold("custom", until=now + 600, base_url=ROUTE["base_url"], source="provider_error")
    result = uh.held_turn_result(_runner(), _ctx(), ROUTE)
    assert result["failed_turn_notice"] == uh.QUEUED_NOTICE
    queue = json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig"))
    assert list(queue) == ["agent:main:telegram:dm:1"]
    assert queue["agent:main:telegram:dm:1"]["state"] == "pending"

    # A rerun that is held again counts as an attempt; past max_attempts nothing is queued.
    queue["agent:main:telegram:dm:1"].update(state="claimed", attempt=2, claimed_at=now)
    (home / uh.QUEUE_FILE).write_text(json.dumps(queue), encoding="utf-8")
    assert "failed_turn_notice" not in uh.held_turn_result(_runner(), _ctx(), ROUTE)
    assert json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig")) == {}

    # A reset beyond the wait cap, a media turn, or an internal turn is never queued.
    record_hold("custom", until=now + 30 * 3600, base_url=ROUTE["base_url"], source="provider_error")
    assert uh.queue_rerun(_runner(), _ctx(), {"until": now + 30 * 3600, "provider": "custom"}) is None
    hold = {"until": now + 600, "provider": "custom"}
    assert uh.queue_rerun(_runner(), _ctx(persist_user_message=[{"type": "image_url"}]), hold) is None
    assert uh.queue_rerun(_runner(), _ctx(persist_user_display_kind="cron_delivery"), hold) is None
    # A LOCAL one-shot has no session owner to admit a rerun into: told the reset, never promised one.
    assert uh.queue_rerun(_runner(), _ctx(source=_source(platform=Platform.LOCAL)), hold) is None


def test_rerun_lands_one_grace_after_the_reported_reset(home):
    """The hold already carries the plan's grace; the rerun must not wait a second grace."""
    from agent.quota_resume import DEFAULT_GRACE_SECONDS, plan_quota_resume
    from agent.usage_hold import active_hold, record_from_plan

    now = time.time()
    plan = plan_quota_resume(failure_reason="rate_limit", error_context={"reset_at": now + 600}, now=now)
    record_from_plan(plan.to_dict(), provider="custom", base_url=ROUTE["base_url"])
    hold = active_hold("custom", ROUTE["base_url"])
    assert uh.queue_rerun(_runner(), _ctx(), hold, now=now) == pytest.approx(now + 600 + DEFAULT_GRACE_SECONDS, abs=1)


class _Store:
    def __init__(self, entries):
        import threading
        self._lock = threading.Lock()
        self._entries = entries

    def _ensure_loaded_locked(self):
        return None


class _Adapter:
    def __init__(self):
        self.sent = []
        self.events = []

    async def handle_message(self, event):
        self.sent.append(event.text)
        self.events.append(event)


def _rerun_runner(transcript, *, session_id="sess-1", adapter=None, running=False,
                  route_key="agent:main:telegram:dm:1", origin_user="alice"):
    entry = SimpleNamespace(session_id=session_id, origin=_source(origin_user), transport_profile=None)

    async def load_transcript(_sid):
        return transcript

    @contextlib.asynccontextmanager
    async def scope(_source):
        yield

    return SimpleNamespace(
        session_store=_Store({"agent:main:telegram:dm:1": entry}),
        async_session_store=SimpleNamespace(load_transcript=load_transcript),
        _is_session_running=lambda _k: running,
        _restored_source=lambda e: e.origin,
        _async_profile_scope_for_source=scope,
        _delivery_adapter_for=lambda _s: adapter,
        _resume_owner_authorized=lambda _k, _s: True,
        _session_key_for_source=lambda _s: route_key,
    )


def _queued(home, *, due=True):
    now = time.time()
    record_hold("custom", until=now + 60, base_url=ROUTE["base_url"], source="provider_error")
    uh.queue_rerun(_runner(), _ctx(), {"until": now + 60, "provider": "custom"}, now=now)
    queued_at = json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig"))["agent:main:telegram:dm:1"]["queued_at"]
    tail = [{"role": "user", "content": "summarize the doc"},
            {"role": "assistant", "content": uh.QUEUED_NOTICE, "timestamp": queued_at + 1}]
    return tail, (now + 3600 if due else now)


def test_rerun_fires_exactly_once_and_survives_a_restart(home):
    tail, at = _queued(home)
    adapter = _Adapter()
    runner = _rerun_runner(tail, adapter=adapter)
    assert asyncio.run(uh.rerun_due(runner, [home], now=at - 7200)) == 0  # not due yet
    # A fresh pass reads only the file: nothing in memory carries the queue across a restart.
    assert asyncio.run(uh.rerun_due(runner, [home], now=at)) == 1
    assert adapter.sent == [resume_note("summarize the doc")]
    assert asyncio.run(uh.rerun_due(runner, [home], now=at)) == 0  # claimed: never re-sent
    assert adapter.sent == [resume_note("summarize the doc")]


@pytest.mark.parametrize("change", ["new_session", "moved_on", "running", "origin_reroutes"])
def test_rerun_is_dropped_when_the_turn_identity_changed(home, change):
    tail, at = _queued(home)
    adapter = _Adapter()
    if change == "moved_on":
        tail = tail + [{"role": "user", "content": "never mind"}, {"role": "assistant", "content": "ok"}]
    runner = _rerun_runner(tail, adapter=adapter, session_id="sess-2" if change == "new_session" else "sess-1",
                           running=change == "running",
                           route_key="agent:main:webhook:other" if change == "origin_reroutes" else "agent:main:telegram:dm:1")
    assert asyncio.run(uh.rerun_due(runner, [home], now=at)) == 0
    assert adapter.sent == []
    assert json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig")) == {}


def test_offline_adapter_keeps_the_rerun_pending(home):
    tail, at = _queued(home)
    assert asyncio.run(uh.rerun_due(_rerun_runner(tail, adapter=None), [home], now=at)) == 0
    assert json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig"))["agent:main:telegram:dm:1"]["state"] == "pending"
    adapter = _Adapter()
    assert asyncio.run(uh.rerun_due(_rerun_runner(tail, adapter=adapter), [home], now=at)) == 1
    assert adapter.sent == [resume_note("summarize the doc")]


def test_a_rerun_refused_again_spends_its_remaining_attempts(home):
    """The rerun is an internal producer turn: it may re-queue its own request, nothing else may."""
    from gateway.response_filters import INTERNAL_NOTIFICATION_DISPLAY_KIND
    now = time.time()
    key, ask = "agent:main:telegram:dm:1", "summarize the doc"
    (home / uh.QUEUE_FILE).write_text(json.dumps({key: {
        "session_key": key, "session_id": "sess-1", "state": "claimed", "claimed_at": now,
        "attempt": 1, "queued_at": now - 600, "text": ask}}), encoding="utf-8")
    hold = {"until": now + 600, "provider": "custom"}
    internal = INTERNAL_NOTIFICATION_DISPLAY_KIND

    assert uh.queue_rerun(_runner(), _ctx(message="a cron delivery", persist_user_display_kind=internal), hold) is None
    assert uh.queue_rerun(_runner(), _ctx(message=resume_note(ask), persist_user_display_kind=internal), hold)
    queued = json.loads((home / uh.QUEUE_FILE).read_text(encoding="utf-8-sig"))[key]
    assert (queued["state"], queued["attempt"], queued["text"]) == ("pending", 2, ask)
