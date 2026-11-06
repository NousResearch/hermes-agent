"""Cron deliver=origin on a non-push surface (#69304).

An api_server turn binds ``async_delivery=False``: its adapter's ``send()`` is a stub, so a
job that captured ``origin.platform="api_server"`` ran fine (``last_status=ok``) while every
fire recorded ``last_delivery_error`` and nothing reached the creator. Both legs are covered:
a non-push session never stamps an origin (so the creation-time local-only notice fires), and
a job already stamped with such an origin resolves to the configured home channel instead.

Exception: a request that declared a server-history consumer (``session_history_delivery="1"``,
the gate async ``delegate_task`` uses) stamps a session origin, and the result is appended to that
session's transcript through ``gateway.wake.persist_delegation_delivery`` — no push, no agent turn.
"""

import asyncio
import threading
from pathlib import Path

import pytest


from gateway.session_context import clear_session_vars, set_session_vars


def test_non_push_session_stamps_no_origin_and_gets_creation_notice():
    from tools.cronjob_job_args import _local_delivery_notice, _origin_from_env

    tokens = set_session_vars(platform="api_server", chat_id="desk-1", session_key="desk-1",
                              async_delivery=False)
    try:
        origin = _origin_from_env()
        notice = _local_delivery_notice({"id": "j", "deliver": "origin", "origin": origin}, None)
    finally:
        clear_session_vars(tokens)
    assert origin is None
    assert notice

    # Control: a push-capable platform keeps its origin.
    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _origin_from_env()["platform"] == "telegram"
    finally:
        clear_session_vars(tokens)


def test_stamped_api_server_origin_falls_back_to_home_channel(monkeypatch):
    from cron import scheduler_delivery as sd

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    job = {"id": "old", "deliver": "origin", "origin": {"platform": "api_server", "chat_id": "desk-1"}}
    targets = sd._resolve_delivery_targets(job)
    assert [(t["platform"], t["chat_id"], t["_resolved_from"]) for t in targets] == [
        ("telegram", "12345", "origin_fallback")]
    # Control: a push-capable origin is delivered as written.
    job["origin"] = {"platform": "discord", "chat_id": "42"}
    assert sd._resolve_delivery_targets(job)[0]["platform"] == "discord"


def test_non_push_session_creation_notice_names_home_channel_fallback(monkeypatch):
    """With a home channel configured the rerouted job must still tell the creating client where
    the report goes (the api_server client never sees it); a push-capable session stays silent."""
    from tools.cronjob_job_args import _local_delivery_notice, _origin_from_env

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    tokens = set_session_vars(platform="api_server", chat_id="desk-1", session_key="desk-1",
                              async_delivery=False)
    try:
        notice = _local_delivery_notice({"id": "j", "deliver": "origin", "origin": _origin_from_env()}, None)
    finally:
        clear_session_vars(tokens)
    assert notice and "telegram:12345" in notice

    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _local_delivery_notice(
            {"id": "j", "deliver": "origin", "origin": _origin_from_env()}, None) is None
    finally:
        clear_session_vars(tokens)


def test_suggestions_accept_origin_shares_the_non_push_guard():
    from hermes_cli.suggestions_cmd import _resolve_origin

    tokens = set_session_vars(platform="api_server", chat_id="desk-1", session_key="desk-1",
                              async_delivery=False)
    try:
        assert _resolve_origin() is None
    finally:
        clear_session_vars(tokens)
    tokens = set_session_vars(platform="telegram", chat_id="777", session_key="tg", async_delivery=True)
    try:
        assert _resolve_origin()["chat_id"] == "777"
    finally:
        clear_session_vars(tokens)


# ── Server-history API sessions: the result lands in the scheduling session's transcript ──────────

def _history_session_vars(chat_id="web-1"):
    return set_session_vars(platform="api_server", chat_id=chat_id, session_key=chat_id,
                            async_delivery=False, session_history_delivery="1")


def _session_job(sid="web-1", **extra):
    job = {"id": "checkin", "name": "checkin", "deliver": "origin", "execution_id": "exec-1",
           "origin": {"platform": "api_server", "chat_id": sid, "session_history": True}}
    job.update(extra)
    return job


class _SessionAdapter:
    """The api_server adapter's shape as far as cron delivery can see it: no push, a SessionDB."""

    supports_async_delivery = False

    def __init__(self, db):
        self._db = db

    def _ensure_session_db(self):
        return self._db

    async def send(self, *args, **kwargs):  # the stub every other lane would have hit
        raise AssertionError("api_server send() must never be called for a session delivery")


@pytest.fixture()
def session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmp_path) / "state.db")
    db.create_session("web-1", source="api_server")
    db.append_message("web-1", "user", content="check me in when it opens")
    db.append_message("web-1", "assistant", content="scheduled", finish_reason="stop")
    return db


@pytest.fixture()
def gateway_loop():
    """A gateway event loop on its own thread, as the cron ticker sees it."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _adapters(db):
    from gateway.config import Platform

    return {Platform.API_SERVER: _SessionAdapter(db)}


@pytest.fixture()
def deliver(session_db, gateway_loop):
    from cron.scheduler_delivery import _deliver_result

    return lambda job, content: _deliver_result(
        job, content, adapters=_adapters(session_db), loop=gateway_loop)


def test_history_session_stamps_session_origin_and_resolves_to_it(monkeypatch):
    """The declared-history request is the one api_server case whose client reads the transcript:
    it gets an origin, deliver=origin resolves to that session (not a home channel), and the
    creation notice says where the result goes instead of calling the job local-only."""
    from cron import scheduler_delivery as sd
    from tools.cronjob_job_args import _local_delivery_notice, _origin_from_env

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    tokens = _history_session_vars()
    try:
        origin = _origin_from_env()
        job = {"id": "j", "deliver": "origin", "origin": origin}
        notice = _local_delivery_notice(job, None)
    finally:
        clear_session_vars(tokens)
    assert origin == {"platform": "api_server", "chat_id": "web-1", "session_history": True}
    assert [(t["platform"], t["chat_id"], t["_resolved_from"]) for t in sd._resolve_delivery_targets(job)] == [
        ("api_server", "web-1", "origin_session")]
    assert notice and "session's history" in notice and "local-only" not in notice


def test_deliver_result_appends_to_scheduling_session_once(session_db, deliver):
    out = deliver(_session_job(), "Check-in is open.")
    assert out is None
    rows = session_db.get_messages("web-1")
    assert len(rows) == 3
    row = rows[-1]
    assert row["role"] == "user" and row["display_kind"] == "async_delegation_complete"
    assert "Check-in is open." in row["content"]
    assert row["display_metadata"]["delegation_id"] == "cron:checkin:exec-1"

    # A retried delivery of the same fire is recorded once (the delegation row's own idempotency key).
    assert deliver(_session_job(), "Check-in is open.") is None
    assert len(session_db.get_messages("web-1")) == 3


def test_session_delivery_redacts_like_every_other_lane(session_db, deliver):
    secret = "sk-" + "A" * 32
    assert deliver(_session_job(), f"token {secret}") is None
    row = session_db.get_messages("web-1")[-1]
    assert row["display_kind"] == "async_delegation_complete" and "token" in row["content"]
    assert secret not in row["content"]


def test_session_delivery_waits_out_a_live_turn(session_db, deliver, monkeypatch):
    """The append is refused while the user is mid-turn in that session; the delivery waits for the
    turn to end instead of recording the result as failed."""
    from cron import scheduler_delivery_session as sds

    monkeypatch.setattr(sds, "_BUSY_RETRY_DELAYS_SECONDS", (0.2, 0.2, 0.2, 0.2, 0.2))
    assert session_db.try_acquire_session_turn_lease("web-1", "live-turn")
    threading.Timer(0.3, session_db.release_session_turn_lease, args=("web-1", "live-turn")).start()
    assert deliver(_session_job(), "Check-in is open.") is None
    assert "Check-in is open." in session_db.get_messages("web-1")[-1]["content"]


def test_session_delivery_reports_a_session_that_stays_busy(session_db, deliver, monkeypatch):
    from cron import scheduler_delivery_session as sds

    monkeypatch.setattr(sds, "_BUSY_RETRY_DELAYS_SECONDS", (0.01, 0.01))
    assert session_db.try_acquire_session_turn_lease("web-1", "live-turn")
    out = deliver(_session_job(), "Check-in is open.")
    assert out and "web-1" in out and "busy" in out
    assert len(session_db.get_messages("web-1")) == 2


def test_session_delivery_reports_a_missing_session_or_adapter(session_db, deliver):
    from cron.scheduler_delivery import _deliver_result

    out = deliver(_session_job(sid="gone"), "x")
    assert out and "gone" in out
    # `hermes cron run` with no gateway: no api_server adapter or loop to write through.
    out = _deliver_result(_session_job(), "x", adapters=None, loop=None)
    assert out and "API server" in out
    out = _deliver_result(_session_job(), "x", adapters=_adapters(session_db), loop=None)
    assert out and "API server" in out
    assert len(session_db.get_messages("web-1")) == 2


def test_legacy_api_server_origin_without_history_flag_still_falls_back(session_db, monkeypatch):
    """Jobs stamped before #69304 carry no history declaration; they keep the home-channel path."""
    from cron import scheduler_delivery as sd

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "12345")
    job = _session_job()
    del job["origin"]["session_history"]
    assert [t["platform"] for t in sd._resolve_delivery_targets(job)] == ["telegram"]


def test_session_delivery_from_inside_a_running_event_loop(session_db, gateway_loop):
    """A caller that is itself inside a running loop (where a bare ``asyncio.run`` raises) still
    lands the row: the write runs on the gateway loop that owns the adapter."""
    from cron.scheduler_delivery_session import deliver_to_session, session_target
    from gateway.config import load_gateway_config

    job = _session_job()

    async def caller():
        return deliver_to_session(job, session_target(job), "Check-in is open.",
                                  adapters=_adapters(session_db), loop=gateway_loop,
                                  config=load_gateway_config())

    assert asyncio.run(caller()) is None
    assert "Check-in is open." in session_db.get_messages("web-1")[-1]["content"]
