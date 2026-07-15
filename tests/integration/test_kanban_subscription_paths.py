"""Cross-path acceptance coverage for Kanban create-time subscriptions.

These tests exercise production creator entry points against a real temporary
Hermes home and SQLite board. Assertions stay at the public behavior seam:
created task ids, dependency links, and persisted notification rows/events.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import config as hermes_config
from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _configure(**settings):
    config = hermes_config.load_config()
    config["kanban"].update(settings)
    hermes_config.save_config(config)


def _subscriptions(conn, task_id):
    return {
        (
            row["platform"],
            row["chat_id"],
            row["thread_id"],
            row["user_id"],
            row["notifier_profile"],
        )
        for row in kb.list_notify_subs(conn, task_id)
    }


def _dashboard_client():
    plugin_file = (
        Path(__file__).resolve().parents[2]
        / "plugins"
        / "kanban"
        / "dashboard"
        / "plugin_api.py"
    )
    module_name = "hermes_dashboard_plugin_kanban_subscription_acceptance"
    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/kanban")
    return TestClient(app)


def test_cli_create_keeps_one_subscription_and_event_set_across_idempotent_retry(
    kanban_home,
):
    """The CLI exposes explicit targets even with automatic policy disabled."""
    _configure(
        auto_subscribe_on_create=False,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
    )
    command = (
        "create 'idempotent CLI task' --notify Telegram:chat-1:topic-1 "
        "--idempotency-key acceptance-cli --json"
    )

    first = json.loads(kanban_cli.run_slash(command))
    with kb.connect() as conn:
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (first["id"],)
        ).fetchone()[0]

    retried = json.loads(kanban_cli.run_slash(command))
    with kb.connect() as conn:
        after_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (first["id"],)
        ).fetchone()[0]
        subscription_rows = kb.list_notify_subs(conn, first["id"])
        subscriptions = _subscriptions(conn, first["id"])

    assert retried["id"] == first["id"]
    assert after_events == before_events
    assert len(subscription_rows) == 1
    assert subscriptions == {("telegram", "chat-1", "topic-1", None, "default")}


@pytest.mark.asyncio
async def test_gateway_create_preserves_structured_origin_and_policy_precedence(
    kanban_home,
):
    """Gateway ambient, explicit, and opt-out intent all reach shared creation."""
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "gateway-owner"

    async def create(command):
        event = SimpleNamespace(
            text=f"/kanban {command}",
            source=SimpleNamespace(
                platform=Platform.TELEGRAM,
                chat_id="ambient-chat",
                thread_id="ambient-thread",
                user_id="ambient-user",
            ),
        )
        return json.loads(
            await GatewayRunner._handle_kanban_command(runner, cast(Any, event))
        )

    _configure(
        auto_subscribe_on_create=True,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
    )
    ambient = await create("create 'gateway ambient' --json")

    _configure(auto_subscribe_on_create=False)
    explicit = await create("create 'gateway explicit' --notify discord:channel-1 --json")
    quiet = await create(
        "create 'gateway quiet' --notify discord:channel-2 --no-subscribe --json"
    )

    with kb.connect() as conn:
        assert _subscriptions(conn, ambient["id"]) == {
            (
                "telegram",
                "ambient-chat",
                "ambient-thread",
                "ambient-user",
                "gateway-owner",
            )
        }
        assert _subscriptions(conn, explicit["id"]) == {
            ("discord", "channel-1", "", None, "gateway-owner")
        }
        assert _subscriptions(conn, quiet["id"]) == set()


@pytest.mark.parametrize(
    ("automatic", "depth", "expected"),
    [
        (True, 0, {("default", "fallback")}),
        (True, 1, {("parent", "one")}),
        (True, "unlimited", {("parent", "one"), ("ancestor", "two")}),
        (False, "unlimited", set()),
    ],
)
def test_worker_create_uses_configured_depth_on_actual_parent_graph(
    kanban_home, monkeypatch, automatic, depth, expected,
):
    """The worker delegates the global gate and configured depth to policy."""
    from tools import kanban_tools

    _configure(
        auto_subscribe_on_create=automatic,
        notify_inherit_depth=depth,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
    )
    with kb.connect() as conn:
        ancestor = kb.create_task(conn, title="ancestor", assignee="worker")
        parent = kb.create_task(
            conn, title="parent", assignee="worker", parents=(ancestor,)
        )
        kb.add_notify_sub(conn, task_id=ancestor, platform="ancestor", chat_id="two")
        kb.add_notify_sub(conn, task_id=parent, platform="parent", chat_id="one")

    monkeypatch.setenv("HERMES_KANBAN_TASK", parent)
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    for name in (
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_THREAD_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    response = json.loads(
        kanban_tools._handle_create(
            {
                "title": f"worker child depth {depth}",
                "assignee": "peer",
                "parents": [parent],
            }
        )
    )

    assert response["ok"] is True
    with kb.connect() as conn:
        assert kb.parent_ids(conn, response["task_id"]) == [parent]
        assert {
            (row["platform"], row["chat_id"])
            for row in kb.list_notify_subs(conn, response["task_id"])
        } == expected


def test_decomposer_uses_sibling_graph_and_global_automatic_gate(kanban_home):
    """Decomposition never treats its root as a hidden subscription parent."""
    _configure(
        auto_subscribe_on_create=True,
        notify_inherit_depth=1,
        notify_default_targets=[
            {
                "platform": "default",
                "chat_id": "fallback",
                "notifier_profile": "default-owner",
            }
        ],
    )
    with kb.connect() as conn:
        root = kb.create_task(conn, title="triage enabled", triage=True)
        kb.add_notify_sub(conn, task_id=root, platform="root", chat_id="must-not-inherit")
        enabled_children = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[{"title": "first"}, {"title": "second", "parents": [0]}],
        )
        assert enabled_children is not None
        assert kb.parent_ids(conn, enabled_children[0]) == []
        assert kb.parent_ids(conn, enabled_children[1]) == [enabled_children[0]]
        assert all(
            _subscriptions(conn, task_id)
            == {("default", "fallback", "", None, "default-owner")}
            for task_id in enabled_children
        )

    _configure(auto_subscribe_on_create=False)
    with kb.connect() as conn:
        root = kb.create_task(conn, title="triage disabled", triage=True)
        kb.add_notify_sub(conn, task_id=root, platform="root", chat_id="must-not-inherit")
        disabled_children = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[{"title": "disabled child"}],
        )
        assert disabled_children is not None
        assert _subscriptions(conn, disabled_children[0]) == set()


def test_dashboard_create_is_explicit_only(kanban_home):
    """Browser requests never become ambient messaging destinations."""
    _configure(
        auto_subscribe_on_create=False,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
    )
    client = _dashboard_client()

    implicit = client.post("/api/plugins/kanban/tasks", json={"title": "dashboard implicit"})
    explicit = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "dashboard explicit",
            "notification_targets": [
                {
                    "platform": " Telegram ",
                    "chat_id": " chat-1 ",
                    "thread_id": " topic-1 ",
                    "user_id": " user-1 ",
                    "notifier_profile": " owner-1 ",
                }
            ],
        },
    )
    quiet = client.post(
        "/api/plugins/kanban/tasks",
        json={
            "title": "dashboard quiet",
            "notification_targets": [{"platform": "discord", "chat_id": "channel-1"}],
            "no_subscribe": True,
        },
    )

    assert implicit.status_code == explicit.status_code == quiet.status_code == 200
    with kb.connect() as conn:
        assert _subscriptions(conn, implicit.json()["task"]["id"]) == set()
        assert _subscriptions(conn, explicit.json()["task"]["id"]) == {
            ("telegram", "chat-1", "topic-1", "user-1", "owner-1")
        }
        assert _subscriptions(conn, quiet.json()["task"]["id"]) == set()


@pytest.mark.asyncio
async def test_gateway_swarm_subscribes_root_only_and_retry_adds_no_rows_or_events(
    kanban_home,
):
    """Gateway swarm origin reaches its root without flooding internal cards."""
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    _configure(auto_subscribe_on_create=True)
    runner = object.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "gateway-owner"
    event = SimpleNamespace(
        text=(
            "/kanban swarm 'Research, verify, and synthesize.' "
            "--worker researcher:Research --verifier reviewer "
            "--synthesizer writer --idempotency-key acceptance-swarm --json"
        ),
        source=SimpleNamespace(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            thread_id="topic-1",
            user_id="user-1",
        ),
    )

    first = json.loads(
        await GatewayRunner._handle_kanban_command(runner, cast(Any, event))
    )

    with kb.connect() as conn:
        before_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        before_subscriptions = conn.execute(
            "SELECT COUNT(*) FROM kanban_notify_subs"
        ).fetchone()[0]

    retried = json.loads(
        await GatewayRunner._handle_kanban_command(runner, cast(Any, event))
    )

    with kb.connect() as conn:
        after_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        after_subscriptions = conn.execute(
            "SELECT COUNT(*) FROM kanban_notify_subs"
        ).fetchone()[0]

        assert retried == first
        assert after_events == before_events
        assert after_subscriptions == before_subscriptions == 1
        assert _subscriptions(conn, first["root_id"]) == {
            (
                "telegram",
                "chat-1",
                "topic-1",
                "user-1",
                "gateway-owner",
            )
        }
        for task_id in [
            *first["worker_ids"],
            first["verifier_id"],
            first["synthesizer_id"],
        ]:
            assert _subscriptions(conn, task_id) == set()
