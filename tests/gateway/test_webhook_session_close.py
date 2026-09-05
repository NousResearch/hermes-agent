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
``end_session(..., "cron_complete")``.

CRITICAL: these tests go through the REAL ``handle_message`` →
``_process_message_background`` → ``on_processing_complete`` pipeline (only the
runner-side ``_message_handler`` is stubbed, exactly the seam the live gateway
injects).  ``handle_message`` is fire-and-forget — it spawns the background
task and returns before the run starts — so any close bolted around
``handle_message`` itself runs BEFORE the session row exists and silently
no-ops.  A test that fakes ``handle_message`` to create the row synchronously
masks exactly that bug (the first version of this fix shipped that way).
"""

import asyncio
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from gateway.session import SessionSource, SessionStore


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


def _make_store(tmp_path) -> SessionStore:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    config = GatewayConfig(
        platforms={Platform.WEBHOOK: PlatformConfig(enabled=True)}
    )
    store = SessionStore(sessions_dir=sessions_dir, config=config)
    assert store._db is not None, "test requires a real SessionDB"
    return store


def _make_event(
    adapter: WebhookAdapter,
    delivery_id: str,
    text: str,
    *,
    completion_script: str | None = None,
) -> MessageEvent:
    session_chat_id = f"webhook:alerts:{delivery_id}"
    source = adapter.build_source(
        chat_id=session_chat_id,
        chat_name="webhook/alerts",
        chat_type="webhook",
        user_id="webhook:alerts",
        user_name="alerts",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"message": text},
        message_id=delivery_id,
        metadata={
            "webhook_route": "alerts",
            "webhook_completion_script": completion_script,
        },
    )


async def _drain_background_tasks(adapter: WebhookAdapter, timeout: float = 5.0) -> None:
    """Wait for the adapter's spawned processing task(s) to finish."""
    deadline = asyncio.get_event_loop().time() + timeout
    while adapter._background_tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.02)
    # One extra tick for done-callbacks to run.
    await asyncio.sleep(0.05)


def _write_completion_script(tmp_path, monkeypatch, body: str) -> str:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script = scripts_dir / "complete.py"
    script.write_text(body, encoding="utf-8")
    return script.name


def _get_db(store: SessionStore):
    assert store._db is not None
    return store._db


def _get_session_row(store: SessionStore, event: MessageEvent):
    session_key = store._generate_session_key(event.source)
    session_id = store.peek_session_id(session_key)
    assert session_id is not None
    row = _get_db(store).get_session(session_id)
    assert row is not None
    return row


@pytest.mark.asyncio
async def test_completed_webhook_delivery_closes_its_session(tmp_path):
    """After a webhook run finishes (REAL dispatch path), ended_at is set."""
    store = _make_store(tmp_path)
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

    # Stub the RUNNER-side handler (the seam the live gateway injects) — the
    # adapter's own handle_message / _process_message_background pipeline runs
    # for real, including the fire-and-forget task spawn and the
    # on_processing_complete hook.  The handler creates the session row, just
    # like GatewayRunner._handle_message does at routing time.
    created = {}

    async def _message_handler(event: MessageEvent):
        entry = store.get_or_create_session(event.source)
        created["session_id"] = entry.session_id
        return ""  # webhook deliver=log — nothing to send back

    adapter._message_handler = _message_handler

    event = _make_event(adapter, "alert-close-001", "Alert: server on fire")

    # Exactly what _handle_webhook schedules.
    await adapter.handle_message(event)
    # handle_message is fire-and-forget: the session must NOT be expected to
    # exist yet.  (Guards against reintroducing a close wrapped around
    # handle_message itself, which ran before the row existed and no-op'd.)
    await _drain_background_tasks(adapter)

    session_id = created["session_id"]
    row = _get_db(store).get_session(session_id)
    assert row is not None

    # INVARIANT: a completed webhook session must be closed so prune can reap it.
    assert row["ended_at"] is not None, (
        "webhook session was never closed — ended_at is NULL, so "
        "prune_sessions can never reap it (the ghost-session leak)"
    )
    assert row["end_reason"] == "webhook_complete"

    # And the closed row is actually prunable, unlike the pre-fix leak.
    pruned = _get_db(store).prune_sessions(older_than_days=0, source="webhook")
    assert pruned >= 1
    _get_db(store).close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_error", "expected_outcome"),
    [(False, "success"), (True, "failure")],
)
async def test_completion_script_receives_terminal_outcome(
    tmp_path, monkeypatch, handler_error, expected_outcome
):
    output_path = tmp_path / "completion.json"
    script_name = _write_completion_script(
        tmp_path,
        monkeypatch,
        "import json, os, pathlib, sys\n"
        "payload = json.load(sys.stdin)\n"
        "pathlib.Path(os.environ['COMPLETION_OUTPUT']).write_text(json.dumps(payload))\n",
    )
    monkeypatch.setenv("COMPLETION_OUTPUT", str(output_path))
    store = _make_store(tmp_path)
    adapter = _make_adapter({})
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event: MessageEvent):
        store.get_or_create_session(event.source)
        if handler_error:
            raise RuntimeError("agent failed")
        return ""

    adapter._message_handler = _message_handler
    event = _make_event(
        adapter,
        f"terminal-{expected_outcome}",
        "payload; touch should-not-run",
        completion_script=script_name,
    )

    await adapter.handle_message(event)
    await _drain_background_tasks(adapter)

    envelope = json.loads(output_path.read_text(encoding="utf-8"))
    assert envelope == {
        "version": 1,
        "route": "alerts",
        "outcome": expected_outcome,
        "delivery_id": f"terminal-{expected_outcome}",
        "payload": {"message": "payload; touch should-not-run"},
    }
    assert not (tmp_path / "should-not-run").exists()
    row = _get_session_row(store, event)
    assert row["end_reason"] == "webhook_complete"
    _get_db(store).close()


@pytest.mark.asyncio
async def test_cancelled_webhook_does_not_run_completion_script(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "completion.json"
    script_name = _write_completion_script(
        tmp_path,
        monkeypatch,
        "import os, pathlib, sys\n"
        "sys.stdin.read()\n"
        "pathlib.Path(os.environ['COMPLETION_OUTPUT']).write_text('called')\n",
    )
    monkeypatch.setenv("COMPLETION_OUTPUT", str(output_path))
    store = _make_store(tmp_path)
    adapter = _make_adapter({})
    adapter.gateway_runner = _FakeRunner(store)
    started = asyncio.Event()

    async def _message_handler(event: MessageEvent):
        store.get_or_create_session(event.source)
        started.set()
        await asyncio.Event().wait()

    adapter._message_handler = _message_handler
    event = _make_event(
        adapter,
        "cancelled-001",
        "interrupted",
        completion_script=script_name,
    )

    await adapter.handle_message(event)
    await started.wait()
    task = adapter._session_tasks[store._generate_session_key(event.source)]
    adapter._expected_cancelled_tasks.add(task)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not output_path.exists()
    row = _get_session_row(store, event)
    assert row["end_reason"] == "webhook_complete"
    _get_db(store).close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script_body",
    [
        "raise SystemExit(3)\n",
        "import time; time.sleep(2)\n",
    ],
)
async def test_completion_script_failure_does_not_block_session_close(
    tmp_path, monkeypatch, script_body
):
    script_name = _write_completion_script(tmp_path, monkeypatch, script_body)
    store = _make_store(tmp_path)
    adapter = _make_adapter({}, script_timeout_seconds=1)
    adapter.gateway_runner = _FakeRunner(store)

    async def _message_handler(event: MessageEvent):
        store.get_or_create_session(event.source)
        return ""

    adapter._message_handler = _message_handler
    event = _make_event(
        adapter,
        "script-failure-001",
        "complete",
        completion_script=script_name,
    )

    await adapter.handle_message(event)
    await _drain_background_tasks(adapter)

    row = _get_session_row(store, event)
    assert row["end_reason"] == "webhook_complete"
    _get_db(store).close()


