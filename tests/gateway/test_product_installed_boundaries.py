"""Focused host boundaries exercised by the installed Telegram experience."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.kanban_surfaces import CardHandle, CardSource
from gateway.live_todo import DeliveryOutcome, DeliveryStatus
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import (
    clear_session_vars,
    current_work_session_id,
    set_session_vars,
)
from gateway.surface_scope import TelegramRoute, TelegramSurfaceScope
from gateway.task_read import TaskReadDenied, TaskReadService
from gateway.work_presentation import TrustedWorkAudience, WorkPresentationService
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli import kanban_db_surface as receipts


class _State:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = json.loads(json.dumps(value))


def _presentation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    scope = TelegramSurfaceScope(
        frozenset({TelegramRoute("default", "telegram", "-100", "7")}),
        frozenset(),
    )
    service = WorkPresentationService.__new__(WorkPresentationService)
    service.profile, service.home, service.plugin_id = "default", home, "synthetic-plugin"
    service.scope, service.state = scope, _State()
    service.active, service.binding = True, None
    service.lock, service._plan_mode, service._begin_new = threading.RLock(), {}, set()
    service._card_renderer, service._card_lanes, service._transport_bindings = None, {}, {}
    service._enabled = lambda: None
    monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: tmp_path)
    return service


def _session(audience, session_id="session-1"):
    return set_session_vars(
        platform="telegram", chat_id=audience.chat_id, chat_type="group",
        thread_id=audience.thread_id or "", user_id=str(audience.actor_id),
        profile=audience.profile, session_id=session_id, work_audience=audience,
    )


def test_proposal_wire_is_json_safe_bounded_and_mode_restore_keeps_identity(tmp_path, monkeypatch):
    service = _presentation(tmp_path, monkeypatch)
    audience = TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)
    plan = tmp_path / ".hermes" / "plans" / "current.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("synthetic plan")
    summary = "Useful plan " + "x" * 300

    tokens = _session(audience)
    try:
        service.begin_proposal()
        service.set_plan_mode(True)
        first = service.publish_proposal(
            source_path=str(plan), presentation={"title": "Useful plan", "summary": summary})
        service._plan_mode.clear()  # process restart loses only the ephemeral marker
        service.set_plan_mode(True)  # native Plan Mode restores the same exact session
        second = service.publish_proposal(
            source_path=str(plan), presentation={
                "title": "Useful plan revised", "summary": summary + " revised"})
    finally:
        clear_session_vars(tokens)

    projected, _ = service.project_proposal(
        ("default", "proposals", second.proposal_id, second.incarnation))
    assert 0 < first.incarnation <= 2**53 - 1
    assert (second.proposal_id, second.incarnation, second.revision) == (
        first.proposal_id, first.incarnation, first.revision + 1)
    assert projected["title"] == "Useful plan revised"
    assert projected["presentation"]["summary"] == summary + " revised"
    with pytest.raises(ValueError, match="too long"):
        tokens = _session(audience)
        try:
            service.publish_proposal(
                source_path=str(plan), presentation={"title": "x" * 121, "summary": summary})
        finally:
            clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_same_payload_revision_advances_without_edit(tmp_path):
    db = tmp_path / "kanban.db"
    with closing(kbc.connect(db)) as conn:
        task_id = kb.create_task(conn, title="Synthetic task")
        first = receipts.get_task_source(conn, task_id)
        receipt = receipts.ensure_delivery_receipt(
            conn, task_id=task_id, task_incarnation=first.task_incarnation,
            desired_revision=first.current_revision, platform="telegram", chat_id="-100",
            thread_id="7", notifier_profile="default", renderer_version="task-card-1",
        )
        from gateway.live_todo import rendered_payload_hash
        clean_links = [{"text": "Open details", "url": "https://tasks.example.test/open"}]
        renderer_hash = rendered_payload_hash("Current card", [clean_links])
        conn.execute(
            "UPDATE kanban_delivery_receipts SET state='sent', destination_message_id='701', "
            "delivered_revision=?, renderer_hash=? WHERE id=?",
            (first.current_revision, renderer_hash, receipt.id),
        )
        kb.add_comment(conn, task_id, "synthetic", "next revision")
        second = receipts.get_task_source(conn, task_id)
        receipts.ensure_delivery_receipt(
            conn, task_id=task_id, task_incarnation=second.task_incarnation,
            desired_revision=second.current_revision, platform="telegram", chat_id="-100",
            thread_id="7", notifier_profile="default", renderer_version="task-card-1",
        )

    calls = []

    async def deliver(source, text):
        calls.append((source.message_id, text))
        return DeliveryOutcome(DeliveryStatus.DELIVERED, message_id="701")

    source = object.__new__(CardSource)
    source.admitted = lambda: True
    source.data = {"db_path": str(db), "receipt_id": receipt.id}
    source.registration = SimpleNamespace(token="owner", decisions=None)
    source.sub = {"notifier_profile": "default"}
    source.adapter = SimpleNamespace(deliver_live_todo=deliver)
    source.lease = source.message_id = source.last_outcome = None
    snapshot = SimpleNamespace(revision=second.current_revision, task_id=task_id)
    links = ({"label": "Open details", "url": "https://tasks.example.test/open"},)

    outcome = await CardHandle(source).deliver(snapshot, "Current card", links=links)

    assert outcome.status == DeliveryStatus.SKIPPED and calls == []
    assert outcome.reason == "confirmed card payload unchanged"
    with closing(kbc.connect(db)) as conn:
        assert receipts.get_delivery_receipt(conn, receipt.id).delivered_revision == second.current_revision


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [True, False])
async def test_native_plugin_command_binds_exact_persisted_session(monkeypatch, existing):
    from hermes_cli import plugins as plugin_module

    source = SessionSource(
        platform=Platform.TELEGRAM, user_id="42", chat_id="-100",
        thread_id="7", chat_type="group", profile="default",
    )
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="synthetic-token")})
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(supports_async_delivery=True)}
    runner._hm_quick_commands = lambda: {}
    session_key = "agent:default:telegram:synthetic"
    runner._session_key_for_source = lambda value: session_key
    created = []

    def get_or_create(value):
        created.append(value)
        return SimpleNamespace(session_key=session_key, session_id="exact-session")

    runner.session_store = SimpleNamespace(
        peek_session_id=lambda key: "exact-session" if existing else None,
        get_or_create_session=get_or_create,
    )
    event = SimpleNamespace(get_command_args=lambda: "on")

    async def handler(raw):
        return f"{raw}:{current_work_session_id()}"

    monkeypatch.setattr(
        plugin_module, "get_plugin_command_handler",
        lambda name: handler if name == "planmode" else None,
    )
    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(
        event, source, "planmode")

    assert (handled, result, command) == (True, "on:exact-session", "planmode")
    assert created == ([] if existing else [source])
    assert current_work_session_id() is None


@pytest.mark.asyncio
async def test_work_brief_read_rechecks_current_route_policy(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    scope_value = {"routes": [{
        "profile": "default", "platform": "telegram", "chat_id": "-100", "thread_id": "7",
    }], "task_resources": []}
    scope = TelegramSurfaceScope(
        frozenset({TelegramRoute("default", "telegram", "-100", "7")}), frozenset())

    def write(*, work_briefs=True, current_scope=scope_value):
        payload = {"plugins": {"enabled": ["synthetic-plugin"], "entries": {
            "synthetic-plugin": {"settings": {
                "enabled": True, "task_detail": True, "work_briefs": work_briefs,
                "scope": current_scope,
            }},
        }}}
        (home / "config.yaml").write_text(yaml.safe_dump(payload))

    write()
    service = TaskReadService.__new__(TaskReadService)
    service.home, service.profile, service.plugin_id = home, "default", "synthetic-plugin"
    service.active, service.binding = True, (object(), object())
    service.scope, service.proposal_provider = scope, object()
    audience = TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)

    class Bot:
        async def get_chat_member(self, chat_id, actor):
            return SimpleNamespace(status="administrator" if actor == 123 else "member")

    await service._member(Bot(), 42, 123, audience)
    write(current_scope={"routes": [{
        "profile": "default", "platform": "telegram", "chat_id": "-200", "thread_id": "7",
    }], "task_resources": []})
    with pytest.raises(TaskReadDenied):
        await service._member(Bot(), 42, 123, audience)
    write(work_briefs=False)
    with pytest.raises(TaskReadDenied):
        await service._member(Bot(), 42, 123, audience)
