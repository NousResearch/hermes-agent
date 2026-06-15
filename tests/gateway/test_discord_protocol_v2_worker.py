from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway import discord_protocol_v2_worker as worker_module
from gateway.discord_identity_registry import DiscordIdentityMetadata, DiscordIdentityRegistry
from gateway.discord_protocol_v2_outbox import (
    DiscordProtocolV2ClientBinding,
    DiscordProtocolV2OutboxSender,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store, response_idempotency_key
from gateway.discord_protocol_v2_worker import DiscordProtocolV2Worker, GatewayRunnerDiscordV2Invoker
from gateway.session import SessionSource


@dataclass
class _FakeEntry:
    session_key: str
    session_id: str
    origin: SessionSource
    created_at: str = "2026-01-01T00:00:00Z"
    updated_at: str = "2026-01-01T00:00:00Z"


class _FakeSessionStore:
    def __init__(self) -> None:
        self.entries: dict[str, _FakeEntry] = {}

    def get_or_create_session_for_key(
        self,
        session_key: str,
        source: SessionSource,
        force_new: bool = False,
    ) -> _FakeEntry:
        if force_new or session_key not in self.entries:
            self.entries[session_key] = _FakeEntry(
                session_key=session_key,
                session_id=f"session-{len(self.entries) + 1}",
                origin=source,
            )
        return self.entries[session_key]

    def bind_session_key(
        self,
        session_key: str,
        session_id: str,
        source: SessionSource,
    ) -> _FakeEntry:
        self.entries[session_key] = _FakeEntry(
            session_key=session_key,
            session_id=session_id,
            origin=source,
        )
        return self.entries[session_key]

    def load_transcript(self, session_id: str) -> list:
        return []


class _FakeDiscordChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, *, content: str, reference=None):
        assert reference is None
        self.sent.append(content)
        return SimpleNamespace(id=f"bohumil-discord-{len(self.sent)}")


class _FakeBohumilDiscordClient:
    def __init__(self) -> None:
        self.channel = _FakeDiscordChannel()

    def get_channel(self, channel_id: int):
        assert channel_id == 100
        return self.channel


def _registry() -> DiscordIdentityRegistry:
    identity = DiscordIdentityMetadata(
        agent_id="bohumil",
        hermes_profile="bohumil-profile",
        discord_application_id="app-bohumil",
        discord_bot_user_id="bot-bohumil",
        token_secret_ref="secret://discord/bohumil-token",
        capabilities=("chat",),
        allowed_scopes={"guild_ids": ["guild-1"]},
        enabled=True,
    )
    return DiscordIdentityRegistry(
        enabled=True,
        mode="listen_only",
        identities={"bohumil": identity},
        active_agent_ids={"bohumil"},
        secret_resolver=None,
    )


def _seed_store(store: DiscordProtocolV2Store) -> dict:
    store.upsert_identity(
        agent_id="bohumil",
        hermes_profile="bohumil-profile",
        discord_application_id="app-bohumil",
        discord_bot_user_id="bot-bohumil",
        token_secret_ref="secret://discord/bohumil-token",
        capabilities=["chat"],
        scopes={"guild_ids": ["guild-1"]},
        enabled=True,
    )
    store.upsert_topic(
        topic_id="guild-1/channel-1/root",
        guild_id="guild-1",
        channel_id="channel-1",
        title="general",
        state={"mode": "listen_only"},
    )
    return store.create_discord_inbound_deliveries(
        discord_message_id="discord-message-1",
        guild_id="guild-1",
        channel_id="channel-1",
        topic_id="guild-1/channel-1/root",
        author_id="human-1",
        author_kind="human",
        target_agent_ids=["bohumil"],
        route_reason="mention",
        payload={
            "discord_message_id": "discord-message-1",
            "content": "hello bohumil",
            "author_id": "human-1",
        },
    )[0]


def _worker(store: DiscordProtocolV2Store, invoker):
    return DiscordProtocolV2Worker(
        store=store,
        identity_registry=_registry(),
        session_store=_FakeSessionStore(),
        invoker=invoker,
        worker_id="test-worker",
        lease_seconds=30,
    )


@pytest.mark.asyncio
async def test_worker_invokes_fake_agent_and_enqueues_one_outbox_response(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        calls = []

        async def fake_agent(context):
            calls.append(context)
            assert context.delivery["delivery_key"] == delivery["delivery_key"]
            assert context.identity.hermes_profile == "bohumil-profile"
            assert context.message_text == "hello bohumil"
            assert context.session_source.platform.value == "discord"
            assert context.session_source.chat_id == "channel-1"
            assert context.metadata["target_agent_id"] == "bohumil"
            assert "Target agent: bohumil" in context.channel_prompt
            return {"final_response": "ahoj z workeru"}

        worker = _worker(store, fake_agent)

        result = await worker.run_once()
        replay = await worker.run_once()

        assert result is not None
        assert result.status == "completed"
        assert replay is None
        assert len(calls) == 1
        assert store.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"
        assert store.count_rows("outbox_deliveries") == 1
        outbox = result.outbox_delivery
        assert outbox["idempotency_key"] == response_idempotency_key(
            delivery["delivery_key"], "bohumil"
        )
        assert outbox["source_inbound_delivery_key"] == delivery["delivery_key"]
        assert outbox["delivery_kind"] == "response"
        assert json.loads(outbox["payload_json"])["content"] == "ahoj z workeru"


@pytest.mark.asyncio
async def test_active_bohumil_fake_path_invokes_once_sends_once_and_maps(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        store.upsert_identity(
            agent_id="bohumil",
            hermes_profile="bohumil-profile",
            discord_application_id="app-bohumil",
            discord_bot_user_id="bot-bohumil",
            token_secret_ref="secret://discord/bohumil-token",
            capabilities=["chat"],
            scopes={"guild_ids": ["guild-1"]},
            enabled=True,
        )
        store.upsert_topic(
            topic_id="guild-1/100/root",
            guild_id="guild-1",
            channel_id="100",
            title="active-general",
            state={"mode": "active"},
        )
        delivery = store.create_discord_inbound_deliveries(
            discord_message_id="discord-inbound-bohumil-1",
            guild_id="guild-1",
            channel_id="100",
            topic_id="guild-1/100/root",
            author_id="human-1",
            author_kind="human",
            target_agent_ids=["bohumil"],
            route_reason="mention",
            mentions=["bohumil"],
            payload={
                "discord_message_id": "discord-inbound-bohumil-1",
                "content": "ahoj Bohumile",
                "author_id": "human-1",
            },
        )[0]
        invocations = []

        async def fake_invoker(context):
            invocations.append(context)
            assert context.delivery["target_agent_id"] == "bohumil"
            assert context.message_text == "ahoj Bohumile"
            return {"final_response": "deterministic Bohumil response"}

        worker_result = await _worker(store, fake_invoker).run_once(target_agent_id="bohumil")
        assert worker_result is not None
        assert worker_result.status == "completed"
        assert worker_result.outbox_delivery is not None
        assert len(invocations) == 1
        assert store.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"
        assert store.count_rows("outbox_deliveries") == 1

        client = _FakeBohumilDiscordClient()
        resolver_calls = []

        def resolve_client(agent_id: str):
            resolver_calls.append(agent_id)
            assert agent_id == "bohumil"
            return DiscordProtocolV2ClientBinding(
                client=client,
                source_client_agent_id="bohumil",
                author_bot_user_id="bot-bohumil",
            )

        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=resolve_client,
            worker_id="bohumil-fake-sender",
        )
        sent = await sender.run_once()
        duplicate = await sender.run_once()

        assert sent is not None
        assert sent.status == "sent"
        assert duplicate is None
        assert resolver_calls == ["bohumil"]
        assert client.channel.sent == ["deterministic Bohumil response"]
        outbox_id = worker_result.outbox_delivery["outbox_delivery_id"]
        assert store.get_outbox_delivery(outbox_id)["status"] == "sent"
        assert store.count_rows("outbox_parts") == 1
        mapped = store.get_message_map("bohumil-discord-1")
        assert mapped is not None
        assert mapped["direction"] == "outbound"
        assert mapped["agent_id"] == "bohumil"
        assert mapped["source_client_agent_id"] == "bohumil"
        assert mapped["author_bot_user_id"] == "bot-bohumil"

        assert await _worker(store, fake_invoker).run_once(target_agent_id="bohumil") is None
        assert len(invocations) == 1
        replay = await sender.deliver_outbox(worker_result.outbox_delivery)
        assert replay.status == "already_sent"
        assert client.channel.sent == ["deterministic Bohumil response"]


@pytest.mark.asyncio
async def test_worker_retries_expired_lease_left_by_crashed_worker(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        crashed_lease = store.lease_next_inbound(
            lease_owner="crashed-worker",
            lease_seconds=-1,
        )
        assert crashed_lease["status"] == "leased"

        calls = 0

        def fake_agent(context):
            nonlocal calls
            calls += 1
            assert context.delivery["attempts"] == 2
            return "recovered response"

        result = await _worker(store, fake_agent).run_once()

        assert result is not None
        assert result.status == "completed"
        assert calls == 1
        assert store.get_inbound_delivery(delivery["delivery_key"])["attempts"] == 2
        assert store.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"
        assert store.count_rows("outbox_deliveries") == 1


@pytest.mark.asyncio
async def test_worker_replay_after_existing_outbox_does_not_duplicate(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        key = response_idempotency_key(delivery["delivery_key"], "bohumil")
        existing = store.create_outbox_delivery(
            idempotency_key=key,
            target_agent_id="bohumil",
            topic_id="guild-1/channel-1/root",
            channel_id="channel-1",
            delivery_kind="response",
            source_inbound_delivery_key=delivery["delivery_key"],
            payload={"content": "already queued"},
        )

        def should_not_run(context):  # pragma: no cover - assertion path
            raise AssertionError("invoker must not run when response is already queued")

        result = await _worker(store, should_not_run).run_once()
        completed_replay = await _worker(store, should_not_run).process_delivery(
            store.get_inbound_delivery(delivery["delivery_key"])
        )
        replay = await _worker(store, should_not_run).run_once()

        assert result is not None
        assert result.status == "already_enqueued"
        assert completed_replay.status == "already_enqueued"
        assert completed_replay.outbox_delivery == existing
        assert result.outbox_delivery == existing
        assert replay is None
        assert store.count_rows("outbox_deliveries") == 1
        assert store.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"


@pytest.mark.asyncio
async def test_gateway_runner_invoker_suppresses_inline_sends_and_enqueues_outbox(tmp_path, monkeypatch):
    class RecordingAdapter:
        def __init__(self) -> None:
            self.sends = []

        async def send(self, *args, **kwargs):
            self.sends.append((args, kwargs))
            return SimpleNamespace(success=True, message_id="inline-send")

    class FakeConfig:
        group_sessions_per_user = True
        thread_sessions_per_user = False

        def get_connected_platforms(self):
            return [Platform.DISCORD]

        def get_home_channel(self, platform):
            return None

    class FakeRunner:
        def __init__(self) -> None:
            self.session_store = _FakeSessionStore()
            self.config = FakeConfig()
            self.adapters = {Platform.DISCORD: RecordingAdapter()}
            self.calls = []

        def _begin_session_run_generation(self, session_key: str) -> int:
            return 1

        async def _run_agent(self, **kwargs):
            self.calls.append(kwargs)
            assert kwargs["hermes_profile"] == "bohumil-profile"
            assert kwargs["hermes_home"] == "/profiles/bohumil-profile"
            assert kwargs["suppress_inline_delivery"] is True
            assert os.environ.get("HERMES_PROFILE") == "legacy-profile"
            if not kwargs["suppress_inline_delivery"]:  # pragma: no cover - regression guard
                await self.adapters[Platform.DISCORD].send("channel-1", "inline")
            return {"final_response": "queued response only"}

    monkeypatch.setenv("HERMES_PROFILE", "legacy-profile")
    monkeypatch.setenv("HERMES_HOME", "/legacy/home")
    monkeypatch.setattr(
        worker_module,
        "_resolve_hermes_profile_home",
        lambda profile: f"/profiles/{profile}",
    )
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed_store(store)
        runner = FakeRunner()
        worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=runner.session_store,
            invoker=GatewayRunnerDiscordV2Invoker(runner),
            worker_id="test-worker",
            lease_seconds=30,
        )

        result = await worker.run_once()

        assert result is not None
        assert result.status == "completed"
        assert runner.calls
        assert runner.adapters[Platform.DISCORD].sends == []
        assert store.count_rows("outbox_deliveries") == 1
        assert json.loads(result.outbox_delivery["payload_json"])["content"] == "queued response only"
        assert os.environ["HERMES_PROFILE"] == "legacy-profile"
        assert os.environ["HERMES_HOME"] == "/legacy/home"


@pytest.mark.asyncio
async def test_stale_worker_cannot_complete_new_owner_lease(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        stale_lease = store.lease_next_inbound(lease_owner="worker-a", lease_seconds=-1)
        fresh_lease = store.lease_next_inbound(lease_owner="worker-b", lease_seconds=30)
        assert stale_lease["delivery_key"] == delivery["delivery_key"]
        assert fresh_lease["lease_owner"] == "worker-b"

        invocations = []

        def stale_invocation(context):  # pragma: no cover - regression guard
            invocations.append(context)
            return "late response"

        stale_worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=stale_invocation,
            worker_id="worker-a",
            lease_seconds=30,
        )

        result = await stale_worker.process_delivery(stale_lease)

        current = store.get_inbound_delivery(delivery["delivery_key"])
        assert result.status == "lease_lost"
        assert result.outbox_delivery is None
        assert invocations == []
        assert current["status"] == "leased"
        assert current["lease_owner"] == "worker-b"
        assert store.count_rows("outbox_deliveries") == 0

        owner_worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=lambda context: "fresh response",
            worker_id="worker-b",
            lease_seconds=30,
        )
        owner_result = await owner_worker.process_delivery(fresh_lease)
        assert owner_result.status == "completed"
        assert store.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"
        assert store.count_rows("outbox_deliveries") == 1


@pytest.mark.asyncio
async def test_stale_worker_cannot_retry_new_owner_lease(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        stale_lease = store.lease_next_inbound(lease_owner="worker-a", lease_seconds=-1)
        fresh_lease = store.lease_next_inbound(lease_owner="worker-b", lease_seconds=30)
        assert stale_lease is not None
        assert fresh_lease is not None

        invocations = []

        def fail_invocation(context):  # pragma: no cover - regression guard
            invocations.append(context)
            raise RuntimeError("boom")

        stale_worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=fail_invocation,
            worker_id="worker-a",
            lease_seconds=30,
        )

        result = await stale_worker.process_delivery(stale_lease)
        assert result.status == "lease_lost"
        assert result.outbox_delivery is None
        assert invocations == []

        current = store.get_inbound_delivery(delivery["delivery_key"])
        assert fresh_lease["lease_owner"] == "worker-b"
        assert current["status"] == "leased"
        assert current["lease_owner"] == "worker-b"
        assert store.count_rows("outbox_deliveries") == 0


@pytest.mark.asyncio
async def test_worker_does_not_enqueue_outbox_after_losing_lease_mid_invocation(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        leased = store.lease_next_inbound(lease_owner="worker-a", lease_seconds=30)
        assert leased is not None
        assert leased["delivery_key"] == delivery["delivery_key"]

        def lose_lease_then_respond(context):
            store.conn.execute(
                """
                UPDATE inbound_deliveries
                SET status = 'leased', lease_owner = 'worker-b',
                    state_version = state_version + 1
                WHERE delivery_key = ?
                """,
                (context.delivery["delivery_key"],),
            )
            store.conn.commit()
            return "late response after stolen lease"

        worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=lose_lease_then_respond,
            worker_id="worker-a",
            lease_seconds=30,
        )

        result = await worker.process_delivery(leased)

        current = store.get_inbound_delivery(delivery["delivery_key"])
        assert current is not None
        assert result.status == "lease_lost"
        assert result.outbox_delivery is None
        assert current["status"] == "leased"
        assert current["lease_owner"] == "worker-b"
        assert store.count_rows("outbox_deliveries") == 0


@pytest.mark.asyncio
async def test_invoke_with_lease_heartbeat_external_cancel_awaits_child_cleanup(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        leased = store.lease_next_inbound(lease_owner="worker-a", lease_seconds=30)
        assert leased is not None
        started = asyncio.Event()
        cleanup_done = asyncio.Event()

        async def cancellable_agent(_context):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                cleanup_done.set()

        worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=cancellable_agent,
            worker_id="worker-a",
            lease_seconds=30,
        )
        context = worker._build_invocation_context(leased)
        task = asyncio.create_task(worker._invoke_with_lease_heartbeat(leased["delivery_key"], context))
        await asyncio.wait_for(started.wait(), timeout=1.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)

        assert cleanup_done.is_set()


@pytest.mark.asyncio
async def test_inbound_lease_heartbeat_prevents_duplicate_invocation_during_long_run(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        delivery = _seed_store(store)
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        async def slow_agent(context):
            calls.append(context.delivery["delivery_key"])
            started.set()
            await release.wait()
            return "single response"

        first_worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=slow_agent,
            worker_id="worker-a",
            lease_seconds=1,
        )
        second_worker = DiscordProtocolV2Worker(
            store=store,
            identity_registry=_registry(),
            session_store=_FakeSessionStore(),
            invoker=lambda context: "duplicate response",
            worker_id="worker-b",
            lease_seconds=1,
        )

        task = asyncio.create_task(first_worker.run_once())
        await asyncio.wait_for(started.wait(), timeout=1.0)
        await asyncio.sleep(1.2)

        assert await second_worker.run_once() is None

        release.set()
        result = await asyncio.wait_for(task, timeout=1.0)

        assert result is not None
        assert result.status == "completed"
        assert calls == [delivery["delivery_key"]]
        assert store.count_rows("outbox_deliveries") == 1
        assert store.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"


def test_gateway_run_agent_suppressed_queued_followup_has_no_inline_delivery_regression():
    run_py = Path(__file__).parents[2] / "gateway" / "run.py"
    text = run_py.read_text(encoding="utf-8")

    first_response_guard = (
        "if first_response and not _already_streamed "
        "and not suppress_inline_delivery:"
    )
    assert first_response_guard in text
    assert (
        "_followup_adapter = None if suppress_inline_delivery "
        "else self.adapters.get(source.platform)"
    ) in text
    assert (
        "hermes_home=hermes_home,\n"
        "            )\n\n"
        "        from run_agent import AIAgent"
    ) in text
    assert (
        "event_message_id=next_message_id,\n"
        "                    channel_prompt=next_channel_prompt,\n"
        "                    suppress_inline_delivery=suppress_inline_delivery,\n"
        "                    hermes_profile=hermes_profile,\n"
        "                    hermes_home=hermes_home,"
    ) in text
