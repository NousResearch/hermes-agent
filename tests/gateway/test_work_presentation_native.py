"""Focused synthetic proof for the native proposal-to-task publication path."""
from contextlib import closing
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.session_context import clear_session_vars, set_session_vars
from gateway.surface_scope import TaskResource, TelegramRoute, TelegramSurfaceScope
from gateway.task_read import TaskReadDenied, TaskReadService, _clean_title
from gateway.work_presentation import TrustedWorkAudience, WorkPresentationService
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from hermes_cli.kanban_publication import project_task_publication
from tools.kanban_tools import _handle_create


class State:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = json.loads(json.dumps(value))


def service(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    route = TelegramRoute("default", "telegram", "-100", "7")
    scope = TelegramSurfaceScope(frozenset({route}), frozenset())
    item = WorkPresentationService.__new__(WorkPresentationService)
    item.profile, item.home, item.plugin_id = "default", home, "synthetic-plugin"
    item.scope, item.state = scope, State()
    item.active, item.binding = True, None
    item.lock, item._plan_mode, item._begin_new = threading.RLock(), {}, set()
    item._card_renderer, item._card_lanes, item._transport_bindings = None, {}, {}
    item._enabled = lambda: None
    monkeypatch.setattr("agent.runtime_cwd.resolve_agent_cwd", lambda: tmp_path)
    return item, TrustedWorkAudience("default", "telegram", 123, "-100", "7", 42)


def bind(audience, *, session="s_1"):
    return set_session_vars(
        platform="telegram", chat_id=audience.chat_id, chat_type="group",
        thread_id=audience.thread_id or "", user_id=str(audience.actor_id),
        profile=audience.profile, session_id=session, work_audience=audience,
    )


def confirm_card(presentation, ref, message_id="701"):
    state = presentation._load()
    state["proposals"][ref.proposal_id]["delivery"] = {
        "attempt_id": "confirmed", "revision": ref.revision,
        "delivered_revision": ref.revision, "state": "sent", "message_id": message_id,
    }
    presentation.state.set(presentation.STATE_KEY, state)


def test_native_plan_approval_creates_links_and_projects_published_task(tmp_path, monkeypatch):
    presentation, audience = service(tmp_path, monkeypatch)
    plan = tmp_path / ".hermes" / "plans" / "current.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("RAW PLAN CONTENT MUST NOT BE PROJECTED")
    db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(presentation.home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(presentation.home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    manager = SimpleNamespace(_work_presentation_registration=presentation)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("gateway.work_presentation.trusted_audience_for_source", lambda source: audience)
    identity = SimpleNamespace(runtime_home=presentation.home)
    monkeypatch.setattr("gateway.session_identity.identity_of", lambda source: identity)

    tokens = bind(audience)
    try:
        ref = presentation.publish_proposal(
            source_path=str(plan), presentation={
                "title": "Safe plan", "summary": "Safe plan summary"})
    finally:
        clear_session_vars(tokens)
    confirm_card(presentation, ref)
    result = GatewayRunner._native_plan_transition(object(), "approve")
    assert result == "Plan approved. It is awaiting task creation and execution."

    tokens = bind(audience)
    try:
        created = json.loads(_handle_create({
            "title": "Useful task", "body": "PRIVATE TASK BODY MUST NOT PROJECT",
            "assignee": "worker", "presentation": {
                "summary": "Useful public summary", "current_step": "Starting safely"},
            "steps": [{"content": "First bounded step", "status": "in_progress"}],
        }))
    finally:
        clear_session_vars(tokens)
    assert created["ok"] is True and created["proposal_linked"] is True
    assert created["status"] == "ready"
    assert created["execution"] == {
        "owner": "dispatcher", "state": "ready", "run_id": None}

    with closing(kbc.connect(db)) as conn:
        source = __import__("hermes_cli.kanban_db_surface", fromlist=["get_task_source"])
        identity = source.get_task_source(conn, created["task_id"])
        publication = project_task_publication(conn, created["task_id"], identity.current_revision)
        assert publication["publication_stale"] is False
        assert publication["audience"] == audience
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (created["task_id"],)).fetchone()[0]
        assert "PRIVATE TASK BODY" not in event and "RAW PLAN CONTENT" not in event

    monkeypatch.delenv("HERMES_KANBAN_DB")
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: db)
    reader = TaskReadService.__new__(TaskReadService)
    reader.profile, reader.proposal_provider = "default", presentation
    projected, projected_audience, _ = reader._project(
        ("default", "default", created["task_id"], identity.task_incarnation))
    assert projected["kind"] == "task" and projected["presentation"]["summary"] == "Useful public summary"
    assert projected_audience == audience
    proposal, _ = presentation.project_proposal(
        ("default", "proposals", ref.proposal_id, ref.incarnation))
    assert proposal["status"] == "approved"
    assert proposal["linked_tasks"][0]["task_id"] == created["task_id"]


def test_plan_mode_marker_is_session_bound_and_sources_reject_userinfo(tmp_path, monkeypatch):
    presentation, audience = service(tmp_path, monkeypatch)
    plan = tmp_path / ".hermes" / "plans" / "current.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("plan")
    tokens = bind(audience, session="first")
    try:
        presentation.set_plan_mode(True)
        first = presentation.publish_proposal(
            source_path=str(plan), presentation={"summary": "Enforced plan"})
        assert presentation.project_proposal(
            ("default", "proposals", first.proposal_id, first.incarnation))[0]["plan_mode"] == "enforced"
    finally:
        clear_session_vars(tokens)
    tokens = bind(audience, session="replacement")
    try:
        second = presentation.publish_proposal(
            source_path=str(plan), presentation={"summary": "Prompt plan"})
        assert presentation.project_proposal(
            ("default", "proposals", second.proposal_id, second.incarnation))[0]["plan_mode"] == "prompt"
        with pytest.raises(ValueError):
            presentation.publish_proposal(source_path=str(plan), presentation={
                "summary": "unsafe", "result": {"summary": "done", "sources": [
                    {"label": "bad", "url": "https://user:pass@example.test/path"}]}})
        safe = presentation.publish_proposal(
            source_path=str(plan), presentation={"summary": "A task-like plan"})
        assert presentation.project_proposal(
            ("default", "proposals", safe.proposal_id, safe.incarnation))[0]["title"] == "Plan"
        assert _clean_title("task-like title sk-synthetic-secret") == "task-like title [redacted]"
    finally:
        clear_session_vars(tokens)


def test_work_briefs_do_not_bypass_current_membership_with_a_legacy_grant(monkeypatch):
    app = object()
    owner = SimpleNamespace(config=SimpleNamespace(token="synthetic-token"),
                            _bot=object(), _live_todo_epoch="epoch")
    runner = SimpleNamespace(
        _primary_profile_name="default", _profile_adapters={},
        _authorization_adapter=lambda platform, profile: owner,
    )
    api = SimpleNamespace(gateway_runner=runner)
    scope = TelegramSurfaceScope(
        frozenset({TelegramRoute("default", "telegram", "-100", "7")}),
        frozenset({TaskResource("default", "t_12345678")}),
    )
    grant = {"actor": 42, "bot_id": 123, "profile": "default", "board": "default",
             "task_id": "t_12345678", "task_incarnation": 1, "permissions": ["read"]}
    reader = TaskReadService.__new__(TaskReadService)
    reader.active, reader.profile, reader.scope = True, "default", scope
    reader.plugin_id, reader.proposal_provider = "synthetic-plugin", object()
    reader.binding = (app, api)
    reader._policy = lambda: ({}, [grant], 300)
    monkeypatch.setattr("gateway.task_read.verify_init_data", lambda *args, **kwargs: {
        "actor": 42, "bot_id": 123,
    })
    selected = ("default", "default", "t_12345678", 1)
    assert reader._authorize(app, "signed", selected)[-1] is False
    reader.proposal_provider = None
    assert reader._authorize(app, "signed", selected)[-1] is True


@pytest.mark.asyncio
async def test_membership_is_fresh_and_temporary_failure_is_typed(tmp_path, monkeypatch):
    presentation, audience = service(tmp_path, monkeypatch)
    reader = TaskReadService.__new__(TaskReadService)
    reader.profile, reader.scope = "default", presentation.scope
    reader.proposal_provider = None

    class Bot:
        async def get_chat_member(self, chat_id, actor):
            return SimpleNamespace(status="administrator" if actor == audience.bot_id else "member")

    await reader._member(Bot(), 99, audience.bot_id, audience)

    class Offline:
        async def get_chat_member(self, chat_id, actor):
            raise OSError("synthetic outage")

    with pytest.raises(TaskReadDenied) as failed:
        await reader._member(Offline(), 99, audience.bot_id, audience)
    assert failed.value.reason == "temporary"

    class Left:
        async def get_chat_member(self, chat_id, actor):
            return SimpleNamespace(status="administrator" if actor == audience.bot_id else "left")

    with pytest.raises(TaskReadDenied) as denied:
        await reader._member(Left(), 99, audience.bot_id, audience)
    assert denied.value.reason == "unavailable"
