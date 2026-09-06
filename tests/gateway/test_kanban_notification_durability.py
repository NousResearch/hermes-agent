
from hermes_cli import kanban_db_connect, kanban_db_notify
"""Durable per-event Kanban notification delivery regressions.

These tests use the real board/subscription state boundary and the real gateway
watcher with an in-process transport.  They never contact a messaging service.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from plugins.platforms.discord.adapter import DiscordAdapter


class ControlledAdapter:
    def __init__(self, outcomes=None, accepted_tokens=None):
        self.outcomes = list(outcomes or [])
        self.accepted_tokens = accepted_tokens if accepted_tokens is not None else set()
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        metadata = dict(metadata or {})
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if outcome == "failed":
            return SendResult(
                success=False,
                error="confirmed transport rejection",
                raw_response={"delivery_rejected": True},
            )
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata})
        token = metadata.get("delivery_token")
        if token:
            self.accepted_tokens.add(token)
        if outcome == "ambiguous":
            return SendResult(
                success=False,
                error="connection closed after submit",
                raw_response={"delivery_ambiguous": True},
            )
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def reconcile_delivery(self, chat_id, delivery_token, metadata=None):
        if delivery_token in self.accepted_tokens:
            return SendResult(success=True, message_id="reconciled-message")
        return SendResult(success=False, error="definitively absent")

    async def handle_message(self, event):
        self.handled.append(event)


def _runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.DISCORD: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


async def _one_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _tick(monkeypatch, runner):
    runner._running = True
    asyncio.run(_one_tick(monkeypatch, runner))


@pytest.fixture()
def subscribed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "delivery.db"))
    kanban_db_connect.init_db()
    with kanban_db_connect.connect_closing() as conn:
        task_id = kb.create_task(conn, title="durable stop", assignee="operator")
        kanban_db_notify.add_notify_sub(
            conn,
            task_id=task_id,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            chat_type="thread",
            delivery_mode="notify",
        )
    return task_id


def _emit(task_id, reason):
    with kanban_db_connect.connect_closing() as conn:
        kb._append_event(
            conn,
            task_id,
            "blocked",
            {"reason": reason, "kind": "needs_input"},
        )


def _delivery_rows(task_id):
    with kanban_db_connect.connect_closing() as conn:
        return kanban_db_notify.list_notify_deliveries(
            conn,
            task_id=task_id,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
        )


def test_missing_required_review_handoff_is_actionable_and_not_replayed(
    subscribed, monkeypatch
):
    with kanban_db_connect.connect_closing() as conn:
        conn.execute(
            "UPDATE tasks SET review_requirement = ? WHERE id = ?",
            ('{"required":true,"owner":"techlead"}', subscribed),
        )
        task = kb.claim_task(conn, subscribed, claimer="operator:1")
        assert task is not None
        assert kb.complete_task(
            conn,
            subscribed,
            summary="implementation ready",
            expected_run_id=task.current_run_id,
        )

    adapter = ControlledAdapter()
    runner = _runner(adapter)
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1
    notice = adapter.sent[0]["text"]
    assert "REVIEW HANDOFF" in notice
    assert "@techlead" in notice
    assert "exactly one independent review" in notice
    assert "done" not in notice.lower()

    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1


def test_existing_canonical_review_notifies_handoff_then_approval(
    subscribed, monkeypatch
):
    with kanban_db_connect.connect_closing() as conn:
        review_id = kb.create_task(
            conn,
            title="Independent review",
            assignee="reviewer",
            parents=[subscribed],
            created_by="techlead",
        )
        task = kb.claim_task(conn, subscribed, claimer="operator:1")
        assert task is not None
        assert kb.complete_task(
            conn,
            subscribed,
            summary="implementation ready",
            metadata={
                "review_requirement": {
                    "required": True,
                    "owner": "techlead",
                    "review_task_id": review_id,
                }
            },
            expected_run_id=task.current_run_id,
        )

    adapter = ControlledAdapter()
    runner = _runner(adapter)
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1
    assert "REVIEW HANDOFF" in adapter.sent[0]["text"]
    assert "not final acceptance" in adapter.sent[0]["text"]
    assert "Ready: no" in adapter.sent[0]["text"]
    assert "implementation ready for independent review" in adapter.sent[0]["text"]

    with kanban_db_connect.connect_closing() as conn:
        review = kb.claim_task(conn, review_id, claimer="reviewer:1")
        assert review is not None
        assert kb.complete_task(
            conn,
            review_id,
            summary="independent review approved",
            expected_run_id=review.current_run_id,
        )
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 2
    assert "done" in adapter.sent[1]["text"].lower()


def test_ordinary_non_code_completion_remains_final(subscribed, monkeypatch):
    with kanban_db_connect.connect_closing() as conn:
        task = kb.claim_task(conn, subscribed, claimer="operator:1")
        assert task is not None
        assert kb.complete_task(
            conn,
            subscribed,
            summary="research summary delivered",
            expected_run_id=task.current_run_id,
        )
    adapter = ControlledAdapter()
    _tick(monkeypatch, _runner(adapter))
    assert len(adapter.sent) == 1
    assert "done" in adapter.sent[0]["text"].lower()
    assert "REVIEW HANDOFF" not in adapter.sent[0]["text"]


def test_process_loss_after_staging_before_send_replays_on_restart(
    subscribed, monkeypatch
):
    _emit(subscribed, "restart boundary")
    with kanban_db_connect.connect_closing() as conn:
        staged = kanban_db_notify.stage_unseen_notify_deliveries_for_sub(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            kinds=["blocked"],
        )
        assert len(staged) == 1
    # The staging process disappears before it invokes transport.  A fresh
    # watcher must consume the durable pending row even though no in-memory
    # claim survives.
    adapter = ControlledAdapter()
    _tick(monkeypatch, _runner(adapter))
    assert len(adapter.sent) == 1
    assert "restart boundary" in adapter.sent[0]["text"]
    assert _delivery_rows(subscribed) == []


def test_partial_batch_failure_retries_only_the_unacknowledged_event(
    subscribed, monkeypatch
):
    _emit(subscribed, "first event")
    _emit(subscribed, "second event")
    first = ControlledAdapter(["ok", "failed"])
    _tick(monkeypatch, _runner(first))
    assert ["first event" in item["text"] for item in first.sent] == [True]

    recovered = ControlledAdapter()
    _tick(monkeypatch, _runner(recovered))
    assert len(recovered.sent) == 1
    assert "second event" in recovered.sent[0]["text"]
    assert "first event" not in recovered.sent[0]["text"]
    _tick(monkeypatch, _runner(recovered))
    assert len(recovered.sent) == 1


def test_ambiguous_acceptance_reconciles_instead_of_duplicate_send(
    subscribed, monkeypatch
):
    _emit(subscribed, "ambiguous acceptance")
    accepted_tokens = set()
    ambiguous = ControlledAdapter(["ambiguous"], accepted_tokens)
    _tick(monkeypatch, _runner(ambiguous))
    assert len(ambiguous.sent) == 1
    assert _delivery_rows(subscribed)[0]["state"] == "ambiguous"

    restarted = ControlledAdapter(accepted_tokens=accepted_tokens)
    _tick(monkeypatch, _runner(restarted))
    assert restarted.sent == [], "reconciliation must not post a duplicate"
    assert _delivery_rows(subscribed) == []


def test_two_watchers_do_not_reconcile_or_send_while_first_send_is_in_flight(
    subscribed, monkeypatch,
):
    """The durable send claim must cover the entire adapter await."""

    class InFlightAdapter(ControlledAdapter):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.send_count = 0
            self.reconcile_count = 0

        async def send(self, chat_id, text, metadata=None):
            self.send_count += 1
            self.started.set()
            await self.release.wait()
            return await super().send(chat_id, text, metadata)

        async def reconcile_delivery(self, chat_id, delivery_token, metadata=None):
            self.reconcile_count += 1
            return SendResult(success=False, error="definitively absent")

    async def exercise():
        real_sleep = asyncio.sleep

        async def accelerated_sleep(delay):
            await real_sleep(0.01 if delay == 1 else 0)

        monkeypatch.setattr(asyncio, "sleep", accelerated_sleep)
        with kanban_db_connect.connect_closing() as conn:
            conn.execute(
                "UPDATE kanban_notify_subs SET delivery_mode = 'notify+wake' "
                "WHERE task_id = ?",
                (subscribed,),
            )
            conn.commit()
            kb._append_event(
                conn,
                subscribed,
                "crashed",
                {"error": "overlap boundary"},
            )
        adapter = InFlightAdapter()
        first = _runner(adapter)
        second = _runner(adapter)
        first_task = asyncio.create_task(first._kanban_notifier_watcher(interval=1))
        await asyncio.wait_for(adapter.started.wait(), timeout=2)
        second_task = asyncio.create_task(second._kanban_notifier_watcher(interval=1))
        await real_sleep(0.1)
        assert adapter.send_count == 1
        assert adapter.reconcile_count == 0
        assert adapter.handled == [], "an in-flight notice must not wake early"
        first._running = False
        second._running = False
        adapter.release.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2)
        assert len(adapter.handled) == 1

    asyncio.run(exercise())
    assert _delivery_rows(subscribed) == []


def test_ack_failure_after_accepted_send_reconciles_without_resending(
    subscribed, monkeypatch
):
    _emit(subscribed, "accepted before ack failed")
    accepted_tokens = set()
    adapter = ControlledAdapter(accepted_tokens=accepted_tokens)
    runner = _runner(adapter)
    real_ack = runner._kanban_ack_delivery
    failed_once = False

    def fail_before_ack(*args, **kwargs):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("fault injected before acknowledgement transaction")
        return real_ack(*args, **kwargs)

    runner._kanban_ack_delivery = fail_before_ack
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1
    assert _delivery_rows(subscribed)[0]["state"] == "ambiguous"

    recovered = ControlledAdapter(accepted_tokens=accepted_tokens)
    _tick(monkeypatch, _runner(recovered))
    assert recovered.sent == []
    assert _delivery_rows(subscribed) == []


def test_marker_survives_nonce_loss_across_real_discord_adapter_boundary(
    subscribed, monkeypatch
):
    with kanban_db_connect.connect_closing() as conn:
        conn.execute(
            "UPDATE kanban_notify_subs SET chat_id = '555', thread_id = '777' "
            "WHERE task_id = ?",
            (subscribed,),
        )
        conn.commit()

    def marker_rows():
        with kanban_db_connect.connect_closing() as conn:
            return kanban_db_notify.list_notify_deliveries(
                conn,
                task_id=subscribed,
                platform="discord",
                chat_id="555",
                thread_id="777",
            )

    _emit(subscribed, "durable marker acceptance")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    accepted_messages = []

    async def send_message(**kwargs):
        message = SimpleNamespace(
            id=777,
            nonce=None,
            content=kwargs["content"],
            author=SimpleNamespace(id=42),
        )
        accepted_messages.append(message)
        return message

    async def history(**_kwargs):
        for message in accepted_messages:
            yield message

    channel = SimpleNamespace(send=send_message, history=history)
    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: channel if channel_id == 777 else None,
        fetch_channel=AsyncMock(return_value=channel),
        user=SimpleNamespace(id=42),
    )
    adapter._record_discord_response = lambda **_kwargs: None
    real_send = adapter.send
    ambiguous_once = True

    async def accept_then_lose_response(chat_id, text, metadata=None):
        nonlocal ambiguous_once
        result = await real_send(chat_id, text, metadata=metadata)
        if ambiguous_once:
            ambiguous_once = False
            return SendResult(
                success=False,
                error="response lost after acceptance",
                raw_response={"delivery_ambiguous": True},
            )
        return result

    adapter.send = accept_then_lose_response
    _tick(monkeypatch, _runner(adapter))
    assert len(accepted_messages) == 1
    row = marker_rows()[0]
    assert accepted_messages[0].content.splitlines()[-1] == (
        f"[kanban-delivery:{row['delivery_token']}]"
    )
    assert row["state"] == "ambiguous"

    _tick(monkeypatch, _runner(adapter))
    assert len(accepted_messages) == 1, "durable marker must prevent a duplicate"
    assert marker_rows() == []


def test_legacy_nonce_only_ambiguous_delivery_parks_without_resend(
    subscribed, monkeypatch
):
    with kanban_db_connect.connect_closing() as conn:
        conn.execute(
            "UPDATE kanban_notify_subs SET chat_id = '555', thread_id = '777' "
            "WHERE task_id = ?",
            (subscribed,),
        )
        conn.commit()
    _emit(subscribed, "legacy nonce ambiguity")
    with kanban_db_connect.connect_closing() as conn:
        kanban_db_notify.stage_unseen_notify_deliveries_for_sub(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="555",
            thread_id="777",
            kinds=["blocked"],
        )
        conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'ambiguous', "
            "delivery_identity = NULL WHERE task_id = ?",
            (subscribed,),
        )
        token = conn.execute(
            "SELECT delivery_token FROM kanban_notify_deliveries WHERE task_id = ?",
            (subscribed,),
        ).fetchone()[0]
        conn.commit()

    async def history(**_kwargs):
        yield SimpleNamespace(
            id=778,
            nonce=token,
            content="old notice without a durable marker",
            author=SimpleNamespace(id=42),
        )

    channel = SimpleNamespace(history=history)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: channel if channel_id == 777 else None,
        fetch_channel=AsyncMock(return_value=channel),
        user=SimpleNamespace(id=42),
    )

    _tick(monkeypatch, _runner(adapter))

    with kanban_db_connect.connect_closing() as conn:
        rows = kanban_db_notify.list_notify_deliveries(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="555",
            thread_id="777",
        )
    assert len(rows) == 1
    assert rows[0]["state"] == "parked"
    assert rows[0]["attempt_count"] == 0


def test_reconciliation_ack_exception_releases_owner_in_same_process(
    subscribed, monkeypatch
):
    _emit(subscribed, "reconciliation ack fault")
    accepted_tokens = set()
    adapter = ControlledAdapter(["ambiguous"], accepted_tokens)
    runner = _runner(adapter)
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1

    real_ack = runner._kanban_ack_delivery
    failed_once = False

    def fail_reconciliation_ack(*args, **kwargs):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("transient reconciliation acknowledgement fault")
        return real_ack(*args, **kwargs)

    runner._kanban_ack_delivery = fail_reconciliation_ack
    _tick(monkeypatch, runner)
    assert _delivery_rows(subscribed)[0]["state"] == "ambiguous"

    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1
    assert _delivery_rows(subscribed) == []


def test_reconciliation_claim_and_stale_send_claim_are_fenced(subscribed):
    _emit(subscribed, "fencing boundary")
    with kanban_db_connect.connect_closing() as conn:
        kanban_db_notify.stage_unseen_notify_deliveries_for_sub(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            kinds=["blocked"],
        )
        event_id = _delivery_rows(subscribed)[0]["event_id"]
        send_claim = kanban_db_notify.begin_notify_delivery(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
        )
        assert isinstance(send_claim, str) and send_claim
        assert kanban_db_notify.mark_notify_delivery_ambiguous(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
            claim_token=send_claim,
            error="response lost",
        )
        reconcile_claim = kanban_db_notify.claim_notify_reconciliation(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
        )
        assert isinstance(reconcile_claim, str) and reconcile_claim
        assert kanban_db_notify.claim_notify_reconciliation(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
        ) is None
        assert kanban_db_notify.acknowledge_notify_delivery(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
            claim_token=reconcile_claim,
            message_id="accepted",
        )
        assert not kanban_db_notify.retry_notify_delivery(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
            claim_token=send_claim,
            error="stale rejection",
        )
    assert _delivery_rows(subscribed) == []


def test_restart_reconciles_provably_abandoned_inflight_claim(
    subscribed, monkeypatch
):
    _emit(subscribed, "process died after send claim")
    with kanban_db_connect.connect_closing() as conn:
        kanban_db_notify.stage_unseen_notify_deliveries_for_sub(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            kinds=["blocked"],
        )
        event_id = _delivery_rows(subscribed)[0]["event_id"]
        assert kanban_db_notify.begin_notify_delivery(
            conn,
            task_id=subscribed,
            platform="discord",
            chat_id="origin-channel",
            thread_id="origin-thread",
            event_id=event_id,
        )
        local_host = kb._claimer_id().rsplit(":", 1)[0]
        conn.execute(
            "UPDATE kanban_notify_deliveries SET claim_owner = ? "
            "WHERE event_id = ?",
            (f"{local_host}:999999999:dead-process", event_id),
        )
        conn.commit()

    adapter = ControlledAdapter()
    runner = _runner(adapter)
    _tick(monkeypatch, runner)
    assert adapter.sent == [], "first restart tick must reconcile before retry"
    assert _delivery_rows(subscribed)[0]["state"] == "pending"
    _tick(monkeypatch, runner)
    assert len(adapter.sent) == 1
    assert _delivery_rows(subscribed) == []


def test_ambiguous_acceptance_without_reconciliation_is_parked(
    subscribed, monkeypatch, caplog
):
    class UnreconcilableAdapter(ControlledAdapter):
        reconcile_delivery = None

    _emit(subscribed, "cannot prove acceptance")
    ambiguous = UnreconcilableAdapter(["ambiguous"])
    _tick(monkeypatch, _runner(ambiguous))
    assert len(ambiguous.sent) == 1

    with caplog.at_level("ERROR", logger="gateway.run"):
        _tick(monkeypatch, _runner(UnreconcilableAdapter()))
    rows = _delivery_rows(subscribed)
    assert len(rows) == 1
    assert rows[0]["state"] == "parked"
    assert "operator decision required" in caplog.text


def test_prolonged_confirmed_failures_preserve_subscription_and_recover(
    subscribed, monkeypatch
):
    _emit(subscribed, "long outage")
    failing = ControlledAdapter(["failed"] * 20)
    runner = _runner(failing)
    for _ in range(15):
        _tick(monkeypatch, runner)
    with kanban_db_connect.connect_closing() as conn:
        assert len(kanban_db_notify.list_notify_subs(conn, subscribed)) == 1
    assert len(_delivery_rows(subscribed)) == 1

    healthy = ControlledAdapter()
    _tick(monkeypatch, _runner(healthy))
    assert len(healthy.sent) == 1
    assert "long outage" in healthy.sent[0]["text"]
    assert _delivery_rows(subscribed) == []
