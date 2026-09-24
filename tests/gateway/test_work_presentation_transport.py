"""Focused proposal delivery proof through the real fenced Telegram adapter."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest

from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars
from gateway.session_identity import canonical_identity
from gateway.surface_scope import TelegramRoute, TelegramSurfaceScope
from gateway.work_presentation import TrustedWorkAudience, WorkPresentationService
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.transport_admission import operation_admission


class State:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = json.loads(json.dumps(value))


class Bot:
    id = 123456

    def __init__(self, *, sends=(), edits=()):
        self.send_modes, self.edit_modes = list(sends), list(edits)
        self.send_attempts, self.edit_attempts = [], []
        self.sent, self.edited = [], []
        self.started, self.release = asyncio.Event(), asyncio.Event()

    def arm(self):
        self.started, self.release = asyncio.Event(), asyncio.Event()

    async def _operation(self, kind, kwargs):
        modes = self.send_modes if kind == "send" else self.edit_modes
        attempts = self.send_attempts if kind == "send" else self.edit_attempts
        attempts.append(dict(kwargs))
        mode = modes.pop(0) if modes else "ok"
        if mode == "dispatched_block":
            operation_admission.get().dispatched = True
            self.started.set()
            await self.release.wait()
        elif mode == "fence":
            self.started.set()
            await self.release.wait()
            admission = operation_admission.get()
            admission.transport_entered = True
            admission.check()
        elif mode == "failed":
            raise BadRequest("synthetic refusal")
        elif mode == "unknown":
            raise OSError("synthetic unconfirmed dispatch")
        target = self.sent if kind == "send" else self.edited
        target.append(dict(kwargs))
        return SimpleNamespace(message_id=701 if kind == "send" else kwargs["message_id"])

    async def send_message(self, **kwargs):
        return await self._operation("send", kwargs)

    async def edit_message_text(self, **kwargs):
        return await self._operation("edit", kwargs)


def make_service(tmp_path, monkeypatch, bot, *, state=None, home=None):
    home = home or tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    route = TelegramRoute("default", "telegram", "-100", "7")
    service = WorkPresentationService.__new__(WorkPresentationService)
    service.profile, service.home, service.plugin_id = "default", home, "synthetic-plugin"
    service.scope, service.state = TelegramSurfaceScope(frozenset({route}), frozenset()), state or State()
    service.active, service.binding = True, None
    service.lock, service._plan_mode, service._begin_new = threading.RLock(), {}, set()
    service._card_renderer = lambda snapshot: {
        "text": f"{snapshot.title}\n{snapshot.status}", "links": (),
    }
    service._card_lanes, service._transport_bindings = {}, {}
    service._enabled = lambda: None
    monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: tmp_path)

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="synthetic-token", typing_indicator=False))
    adapter._bot = bot
    adapter._send_path_degraded = False
    source = SessionSource(
        Platform.TELEGRAM, "-100", chat_type="group", user_id="42", thread_id="7")
    runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=False), _primary_profile_name="default",
        _owning_profile=lambda candidate, platform: (True, "default"),
    )
    canonical_identity(source, runner=runner, adapter=adapter, primary_home=home)
    assert service.bind_transport_from_ingress(source, adapter) is True
    audience = TrustedWorkAudience("default", "telegram", bot.id, "-100", "7", 42)
    plan = tmp_path / ".hermes" / "plans" / "current.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("private plan content")
    return service, adapter, source, audience, plan


def publish(service, audience, plan, summary):
    tokens = set_session_vars(
        platform="telegram", chat_id=audience.chat_id, chat_type="group",
        thread_id=audience.thread_id, user_id=str(audience.actor_id),
        profile=audience.profile, session_id="session-1", work_audience=audience,
    )
    try:
        return service.publish_proposal(
            source_path=str(plan), presentation={"summary": summary})
    finally:
        clear_session_vars(tokens)


def transition(service, audience, ref, **kwargs):
    tokens = set_session_vars(
        platform="telegram", chat_id=audience.chat_id, chat_type="group",
        thread_id=audience.thread_id, user_id=str(audience.actor_id),
        profile=audience.profile, session_id="session-1", work_audience=audience,
    )
    try:
        return service.transition_proposal(ref, expected_revision=ref.revision, **kwargs)
    finally:
        clear_session_vars(tokens)


def delivery(service, ref):
    return service._load()["proposals"][ref.proposal_id].get("delivery", {})


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_real_adapter_native_command_edits_confirmed_card_after_turn(tmp_path, monkeypatch):
    bot = Bot()
    service, adapter, source, audience, plan = make_service(tmp_path, monkeypatch, bot)
    ref = publish(service, audience, plan, "First safe plan")
    await wait_for(lambda: delivery(service, ref).get("delivered_revision") == ref.revision)
    manager = SimpleNamespace(_work_presentation_registration=service)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    result = await asyncio.to_thread(GatewayRunner._native_plan_transition, source, "reject")
    assert result == "Plan changes requested. Publish a revised brief before approval."
    await wait_for(lambda: delivery(service, ref).get("delivered_revision") == ref.revision + 1)
    assert len(bot.sent) == 1 and len(bot.edited) == 1
    assert bot.edited[0]["message_id"] == bot.sent[0].get("message_id", 701) == 701
    assert not any(source.__class__.__name__ == "TodoSource" for source in adapter._live_todo_sources)


@pytest.mark.asyncio
async def test_approval_requires_displayed_revision_and_cannot_mutate_brief(tmp_path, monkeypatch):
    bot = Bot(edits=("dispatched_block", "ok"))
    service, _, _, audience, plan = make_service(tmp_path, monkeypatch, bot)
    first = publish(service, audience, plan, "Plan A")
    await wait_for(lambda: delivery(service, first).get("delivered_revision") == first.revision)
    bot.arm()
    revised = publish(service, audience, plan, "Plan B")
    await bot.started.wait()
    with pytest.raises(ValueError, match="confirmed displayed"):
        transition(service, audience, revised, status="approved")
    bot.release.set()
    await wait_for(lambda: delivery(service, revised).get("delivered_revision") == revised.revision)
    with pytest.raises(ValueError, match="cannot change"):
        transition(service, audience, revised, status="approved",
                   presentation={"summary": "different"})
    approved = transition(service, audience, revised, status="approved")
    await wait_for(lambda: delivery(service, approved).get("delivered_revision") == approved.revision)
    record = service._load()["proposals"][approved.proposal_id]
    assert approved.status == "approved" and record["presentation"] == {"summary": "Plan B"}


@pytest.mark.parametrize(("mode", "state"), [("failed", "failed"), ("unknown", "unknown")])
@pytest.mark.asyncio
async def test_failed_or_unknown_revision_cannot_be_approved(tmp_path, monkeypatch, mode, state):
    bot = Bot(edits=(mode,))
    service, _, _, audience, plan = make_service(tmp_path, monkeypatch, bot)
    first = publish(service, audience, plan, "Plan A")
    await wait_for(lambda: delivery(service, first).get("state") == "sent")
    revised = publish(service, audience, plan, "Plan B")
    await wait_for(lambda: delivery(service, revised).get("state") == state)
    with pytest.raises(ValueError, match="confirmed displayed"):
        transition(service, audience, revised, status="approved")


@pytest.mark.asyncio
async def test_epoch_revocation_at_write_fence_sends_no_bytes(tmp_path, monkeypatch):
    bot = Bot(sends=("fence",))
    service, adapter, _, audience, plan = make_service(tmp_path, monkeypatch, bot)
    ref = publish(service, audience, plan, "Fenced plan")
    await bot.started.wait()
    adapter._live_todo_epoch = "replacement-epoch"
    bot.release.set()
    await wait_for(lambda: delivery(service, ref).get("state") == "failed")
    assert not bot.sent


@pytest.mark.asyncio
async def test_unknown_create_survives_reopen_without_duplicate(tmp_path, monkeypatch):
    shared = State()
    first_bot = Bot(sends=("unknown",))
    first, _, _, audience, plan = make_service(tmp_path, monkeypatch, first_bot, state=shared)
    ref = publish(first, audience, plan, "Uncertain create")
    await wait_for(lambda: delivery(first, ref).get("state") == "unknown")
    assert len(first_bot.send_attempts) == 1 and not first_bot.sent

    first.active = False
    second_bot = Bot()
    second, _, _, _, _ = make_service(
        tmp_path, monkeypatch, second_bot, state=shared, home=first.home)
    await asyncio.sleep(0.05)
    assert not second_bot.send_attempts
    assert delivery(second, ref).get("state") == "unknown"


@pytest.mark.asyncio
async def test_unknown_edit_quarantines_newer_revision(tmp_path, monkeypatch):
    shared = State()
    first_bot = Bot(edits=("unknown",))
    first, _, _, audience, plan = make_service(tmp_path, monkeypatch, first_bot, state=shared)
    initial = publish(first, audience, plan, "Plan A")
    await wait_for(lambda: delivery(first, initial).get("state") == "sent")
    uncertain = publish(first, audience, plan, "Plan B")
    await wait_for(lambda: delivery(first, uncertain).get("state") == "unknown")
    assert len(first_bot.edit_attempts) == 1

    first.active = False
    second_bot = Bot()
    second, _, _, _, _ = make_service(
        tmp_path, monkeypatch, second_bot, state=shared, home=first.home)
    newer = publish(second, audience, plan, "Plan C")
    await asyncio.sleep(0.05)

    assert not second_bot.send_attempts and not second_bot.edit_attempts
    assert delivery(second, newer).get("state") == "unknown"


@pytest.mark.asyncio
async def test_new_revision_while_create_pending_edits_confirmed_message(tmp_path, monkeypatch):
    bot = Bot(sends=("dispatched_block",), edits=("ok",))
    service, _, _, audience, plan = make_service(tmp_path, monkeypatch, bot)
    first = publish(service, audience, plan, "Plan A")
    await bot.started.wait()
    revised = publish(service, audience, plan, "Plan B")
    bot.release.set()
    await wait_for(lambda: delivery(service, revised).get("delivered_revision") == revised.revision)
    assert len(bot.sent) == 1 and len(bot.edited) == 1
    assert bot.edited[0]["message_id"] == 701
    assert bot.edited[0]["text"] == "Plan\nrevised"


@pytest.mark.asyncio
async def test_linked_task_identical_payload_advances_confirmed_revision(tmp_path, monkeypatch):
    bot = Bot()
    service, _, _, audience, plan = make_service(tmp_path, monkeypatch, bot)
    first = publish(service, audience, plan, "Plan A")
    await wait_for(lambda: delivery(service, first).get("state") == "sent")
    approved = transition(service, audience, first, status="approved")
    await wait_for(lambda: delivery(service, approved).get("delivered_revision") == approved.revision)

    db = tmp_path / "kanban.db"
    publication = {
        "version": 1,
        "audience": {
            "profile": audience.profile, "platform": audience.platform,
            "bot_id": audience.bot_id, "chat_id": audience.chat_id,
            "thread_id": audience.thread_id, "actor_id": audience.actor_id,
        },
        "presentation": {"summary": "Safe task brief"}, "steps": [], "run_id": None,
    }
    with kbc.connect(db) as conn:
        task_id = kb.create_task(
            conn, title="Linked task", assignee="worker", publication=publication)
        identity = __import__(
            "hermes_cli.kanban_db_surface", fromlist=["get_task_source"],
        ).get_task_source(conn, task_id)
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: db)
    tokens = set_session_vars(
        platform="telegram", chat_id=audience.chat_id, chat_type="group",
        thread_id=audience.thread_id, user_id=str(audience.actor_id),
        profile=audience.profile, session_id="session-1", work_audience=audience,
    )
    try:
        linked = service.link_task(
            approved, expected_revision=approved.revision, board="default",
            task_id=task_id, task_incarnation=identity.task_incarnation)
    finally:
        clear_session_vars(tokens)

    await wait_for(lambda: delivery(service, linked).get("delivered_revision") == linked.revision)
    assert len(bot.sent) == 1 and len(bot.edited) == 1
    assert delivery(service, linked)["state"] == "sent"
    assert delivery(service, linked)["message_id"] == "701"
