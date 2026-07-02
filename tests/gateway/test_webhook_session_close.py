"""Invariant test: a completed webhook delivery closes its session.

Regression guard for the ghost-session leak.  Webhook deliveries create a
unique one-shot session (``delivery_id`` baked into the session key), but the
adapter historically fired ``handle_message`` without ever ending the session.
``SessionDB.prune_sessions`` only reaps rows where ``ended_at IS NOT NULL``, so
every webhook session stayed unprunable and state.db grew without bound (this
was the primary driver of the SQLite lock-contention gateway outage).

The invariant asserted here is a *behavior contract*, not a snapshot: once a
webhook delivery's agent run completes, the session row for that delivery must
have ``ended_at`` set — mirroring how a cron run closes its session with
``end_session(..., "cron_complete")``.  We exercise the REAL close path
(``WebhookAdapter._run_delivery_and_close`` → ``_end_webhook_session`` →
``SessionDB.end_session``) against a REAL ``SessionStore`` + ``SessionDB`` on a
temp HERMES_HOME, so an integration regression can't hide behind a mock.
"""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


def _make_adapter(routes, **extra_kw) -> WebhookAdapter:
    extra = {"host": "127.0.0.1", "port": 0, "routes": routes}
    extra.update(extra_kw)
    config = PlatformConfig(enabled=True, extra=extra)
    return WebhookAdapter(config)


class _FakeRunner:
    """Minimal gateway runner surface the webhook close path depends on.

    Wires a real ``SessionStore`` (which owns a real ``SessionDB``) and reuses
    that same ``SessionDB`` as ``_session_db`` so the row created at routing
    time is the row the close path ends — exactly the wiring the live gateway
    has (``self.session_store`` + ``self._session_db``).
    """

    def __init__(self, store: SessionStore):
        self.session_store = store
        self._session_db = store._db

    def _session_key_for_source(self, source: SessionSource) -> str:
        return self.session_store._generate_session_key(source)


@pytest.mark.asyncio
async def test_completed_webhook_delivery_closes_its_session(tmp_path):
    """After a webhook run finishes, its session row has ended_at set."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config = GatewayConfig(
        platforms={Platform.WEBHOOK: PlatformConfig(enabled=True)}
    )
    store = SessionStore(sessions_dir=sessions_dir, config=config)
    assert store._db is not None, "test requires a real SessionDB"
    runner = _FakeRunner(store)

    adapter = _make_adapter(
        {
            "alerts": {
                "secret": _INSECURE_NO_AUTH,
                "prompt": "Alert: {message}",
                "deliver": "log",
            }
        }
    )
    adapter.gateway_runner = runner

    # The gateway creates the session row when it routes the inbound event to
    # the agent.  Simulate that inside handle_message so the close path has a
    # real row to reap, and capture the session_id for the assertion.
    created = {}

    async def _fake_handle_message(event: MessageEvent) -> None:
        entry = store.get_or_create_session(event.source)
        created["session_id"] = entry.session_id

    adapter.handle_message = _fake_handle_message

    delivery_id = "alert-close-001"
    session_chat_id = f"webhook:alerts:{delivery_id}"
    source = adapter.build_source(
        chat_id=session_chat_id,
        chat_name="webhook/alerts",
        chat_type="webhook",
        user_id="webhook:alerts",
        user_name="alerts",
    )
    event = MessageEvent(
        text="Alert: server on fire",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"message": "server on fire"},
        message_id=delivery_id,
    )

    # Run the exact wrapper the adapter now schedules on delivery.
    await adapter._run_delivery_and_close(event, session_chat_id)

    session_id = created["session_id"]
    row = store._db.get_session(session_id)
    assert row is not None

    # INVARIANT: a completed webhook session must be closed so prune can reap it.
    assert row["ended_at"] is not None, (
        "webhook session was never closed — ended_at is NULL, so "
        "prune_sessions can never reap it (the ghost-session leak)"
    )
    assert row["end_reason"] == "webhook_complete"

    # And the closed row is actually prunable, unlike the pre-fix leak.
    pruned = store._db.prune_sessions(older_than_days=0, source="webhook")
    assert pruned >= 1
    store._db.close()


@pytest.mark.asyncio
async def test_webhook_session_closed_even_when_agent_run_raises(tmp_path):
    """A failing agent run still closes the session (finally-path)."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config = GatewayConfig(
        platforms={Platform.WEBHOOK: PlatformConfig(enabled=True)}
    )
    store = SessionStore(sessions_dir=sessions_dir, config=config)
    runner = _FakeRunner(store)

    adapter = _make_adapter(
        {"alerts": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "deliver": "log"}}
    )
    adapter.gateway_runner = runner

    created = {}

    async def _boom(event: MessageEvent) -> None:
        # Row exists (routing happened) before the run blows up mid-turn.
        entry = store.get_or_create_session(event.source)
        created["session_id"] = entry.session_id
        raise RuntimeError("agent exploded mid-run")

    adapter.handle_message = _boom

    delivery_id = "alert-fail-001"
    session_chat_id = f"webhook:alerts:{delivery_id}"
    source = adapter.build_source(
        chat_id=session_chat_id,
        chat_name="webhook/alerts",
        chat_type="webhook",
        user_id="webhook:alerts",
        user_name="alerts",
    )
    event = MessageEvent(
        text="x",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={},
        message_id=delivery_id,
    )

    with pytest.raises(RuntimeError):
        await adapter._run_delivery_and_close(event, session_chat_id)

    row = store._db.get_session(created["session_id"])
    assert row is not None
    assert row["ended_at"] is not None, (
        "session left open after a failed webhook run — the leak persists "
        "on the error path"
    )
    assert row["end_reason"] == "webhook_complete"
    store._db.close()
