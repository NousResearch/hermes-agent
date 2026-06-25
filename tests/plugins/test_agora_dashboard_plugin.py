"""Tests for the Ágora dashboard plugin backend (plugins/agora/dashboard/plugin_api.py).

The plugin mounts as /api/plugins/agora/ inside the dashboard's FastAPI app,
but here we attach its router to a bare FastAPI instance so we can test the
REST surface without spinning up the whole dashboard.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load plugins/agora/dashboard/plugin_api.py and return (module, router)."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "agora" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_agora_test",
        plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, mod.router


@pytest.fixture
def agora_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty Ágora DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("AGORA_TMUX_WAKE_ENABLED", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Reset per-process DB init flag so each test re-initialises against the new home.
    mod = sys.modules.get("hermes_dashboard_plugin_agora_test")
    if mod is not None:
        mod._db_init_path = None
    return home


@pytest.fixture
def client(agora_home):
    app = FastAPI()
    router_mod, router = _load_plugin_router()
    app.include_router(router, prefix="/api/plugins/agora")

    # Tests mount the router directly, which bypasses the plugin manager's
    # discovery/registration phase. Register the Kanban lifecycle hooks so
    # integration tests that exercise kanban_db.complete_task/block_task also
    # exercise the Ágora callbacks.
    import hermes_cli.plugins as _plugins

    pm = _plugins.get_plugin_manager()
    if not pm.has_hook("kanban_task_completed"):
        ctx = _plugins.PluginContext(
            manifest=_plugins.PluginManifest(
                name="agora-dashboard",
                provides_hooks=["kanban_task_completed", "kanban_task_blocked"],
            ),
            manager=pm,
        )
        router_mod.register(ctx)

    return TestClient(app)


def test_db_path_uses_shared_root_in_profile_mode(tmp_path, monkeypatch):
    """Ágora DB is shared across profiles, matching the Kanban board."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "agent-qa"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    assert pa._db_path() == root / "agora.db"
    with pa._connect() as conn:
        conn.execute("SELECT 1")

    assert (root / "agora.db").exists()
    assert not (profile_home / "agora.db").exists()


# ---------------------------------------------------------------------------
# Channels (defaults + CRUD)
# ---------------------------------------------------------------------------


def test_channels_default_seed(client):
    r = client.get("/api/plugins/agora/channels")
    assert r.status_code == 200
    slugs = {c["slug"] for c in r.json()["channels"]}
    assert slugs == {"praca", "planejamento", "decisoes", "incidentes", "profarma"}


def test_create_channel_and_get(client):
    r = client.post(
        "/api/plugins/agora/channels",
        json={"slug": "test-channel", "name": "Test", "description": "d"},
    )
    assert r.status_code == 200
    assert r.json()["channel"]["slug"] == "test-channel"

    r = client.get("/api/plugins/agora/channels/test-channel")
    assert r.status_code == 200
    assert r.json()["channel"]["name"] == "Test"


def test_create_channel_duplicate_rejected(client):
    client.post("/api/plugins/agora/channels", json={"slug": "dup", "name": "Dup"})
    r = client.post(
        "/api/plugins/agora/channels", json={"slug": "dup", "name": "Dup 2"}
    )
    assert r.status_code == 409


def test_create_channel_invalid_slug(client):
    r = client.post(
        "/api/plugins/agora/channels", json={"slug": "Bad Slug!", "name": "Bad"}
    )
    assert r.status_code == 400


def test_create_channel_empty_slug_rejected(client):
    r = client.post("/api/plugins/agora/channels", json={"slug": "", "name": "Bad"})
    assert r.status_code == 400


def test_create_channel_whitespace_slug_rejected(client):
    r = client.post("/api/plugins/agora/channels", json={"slug": "   ", "name": "Bad"})
    assert r.status_code == 400


def test_create_channel_whitespace_name_rejected(client):
    r = client.post(
        "/api/plugins/agora/channels", json={"slug": "noname", "name": "\t\n"}
    )
    assert r.status_code == 400


def test_create_channel_empty_name_rejected(client):
    r = client.post(
        "/api/plugins/agora/channels", json={"slug": "noname", "name": "   "}
    )
    assert r.status_code == 400


def test_cleanup_emptyname_channel_moves_content(client):
    client.post(
        "/api/plugins/agora/channels",
        json={"slug": "emptyname", "name": "Empty Name", "description": "legacy"},
    )
    client.post(
        "/api/plugins/agora/channels/emptyname/messages",
        json={"body": "legacy message"},
    )

    empty = client.get("/api/plugins/agora/channels/emptyname").json()["channel"]
    praca = client.get("/api/plugins/agora/channels/praca").json()["channel"]
    thread = client.post(
        "/api/plugins/agora/threads",
        json={"channel_id": empty["id"], "title": "legacy thread"},
    ).json()["thread"]
    client.post(
        "/api/plugins/agora/channels/emptyname/messages",
        json={"body": "thread message", "thread_id": thread["id"]},
    )

    r = client.post("/api/plugins/agora/channels/cleanup-emptyname")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["moved_messages"] == 2
    assert data["moved_threads"] == 1
    assert data["target_slug"] == "praca"

    assert client.get("/api/plugins/agora/channels/emptyname").status_code == 404

    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) == 2
    assert {m["body"] for m in msgs} == {"legacy message", "thread message"}

    r = client.get(f"/api/plugins/agora/threads/{thread['id']}")
    assert r.json()["thread"]["channel_id"] == praca["id"]


def test_cleanup_emptyname_idempotent(client):
    client.post(
        "/api/plugins/agora/channels", json={"slug": "emptyname", "name": "Empty Name"}
    )
    r = client.post("/api/plugins/agora/channels/cleanup-emptyname")
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.post("/api/plugins/agora/channels/cleanup-emptyname")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["moved_messages"] == 0


def test_channel_creation_emits_event(client):
    before = client.get("/api/plugins/agora/events").json()["cursor"]
    client.post(
        "/api/plugins/agora/channels", json={"slug": "evented", "name": "Evented"}
    )
    r = client.get(f"/api/plugins/agora/events?since_id={before}")
    assert r.status_code == 200
    events = [e for e in r.json()["events"] if e["entity_type"] == "channel"]
    assert len(events) == 1
    assert events[0]["event_type"] == "created"
    assert events[0]["payload"]["slug"] == "evented"


def test_get_channel_missing(client):
    r = client.get("/api/plugins/agora/channels/no-such-channel")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Admin channels
# ---------------------------------------------------------------------------


def test_admin_create_channel(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "admin-novo", "name": "Admin Novo", "description": "admin channel"},
    )
    assert r.status_code == 201
    channel = r.json()["channel"]
    assert channel["slug"] == "admin-novo"
    assert channel["name"] == "Admin Novo"
    assert channel["description"] == "admin channel"


def test_admin_create_channel_caps_are_lowered(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "ADMIN-CAPS", "name": "Caps"},
    )
    assert r.status_code == 201
    assert r.json()["channel"]["slug"] == "admin-caps"


def test_admin_create_channel_folds_accents(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "são-paulo", "name": "São Paulo"},
    )
    assert r.status_code == 201
    assert r.json()["channel"]["slug"] == "sao-paulo"


def test_admin_create_channel_non_ascii_rejected(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "日本語", "name": "Japanese"},
    )
    assert r.status_code == 400
    assert "ascii" in r.json()["detail"].lower()


def test_admin_create_channel_too_long_rejected(client):
    long_slug = "a" * 65
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": long_slug, "name": "Too long"},
    )
    assert r.status_code == 400


def test_admin_create_channel_leading_hyphen_rejected(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "-start", "name": "Bad"},
    )
    assert r.status_code == 400


def test_admin_create_channel_trailing_underscore_rejected(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "end_", "name": "Bad"},
    )
    assert r.status_code == 400


def test_admin_create_channel_duplicate_rejected(client):
    client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "admin-dup", "name": "Dup"},
    )
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "admin-dup", "name": "Dup 2"},
    )
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


def test_admin_create_channel_invalid_name_rejected(client):
    r = client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "valid-slug", "name": "   "},
    )
    assert r.status_code == 400


def test_admin_create_channel_emits_event(client):
    before = client.get("/api/plugins/agora/events").json()["cursor"]
    client.post(
        "/api/plugins/agora/admin/channels",
        json={"slug": "admin-evented", "name": "Admin Evented"},
    )
    r = client.get(f"/api/plugins/agora/events?since_id={before}")
    assert r.status_code == 200
    events = [e for e in r.json()["events"] if e["entity_type"] == "channel"]
    assert len(events) == 1
    assert events[0]["event_type"] == "created"
    assert events[0]["payload"]["slug"] == "admin-evented"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_post_message_and_list(client):
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "hello agora",
            "author_type": "agent",
            "author_profile": "worker-1",
        },
    )
    assert r.status_code == 200
    msg = r.json()["message"]
    assert msg["body"] == "hello agora"
    assert msg["author_type"] == "agent"
    assert msg["author_profile"] == "worker-1"

    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    assert len(r.json()["messages"]) == 1
    assert r.json()["messages"][0]["body"] == "hello agora"


def test_post_message_empty_rejected(client):
    r = client.post("/api/plugins/agora/channels/praca/messages", json={"body": "   "})
    assert r.status_code == 400


def test_post_message_invalid_author_type(client):
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "x", "author_type": "robot"},
    )
    assert r.status_code == 400


def test_post_human_message_without_author_profile_defaults_to_human(client):
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "hello from dashboard", "author_type": "human"},
    )
    assert r.status_code == 200
    msg = r.json()["message"]
    assert msg["author_type"] == "human"
    assert msg["author_profile"] == "human"

    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    assert r.json()["messages"][0]["author_profile"] == "human"


def test_post_human_message_with_blank_author_profile_defaults_to_human(client):
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "blank profile", "author_type": "human", "author_profile": "   "},
    )
    assert r.status_code == 200
    assert r.json()["message"]["author_profile"] == "human"


def test_post_message_link_task(client):
    r = client.post(
        "/api/plugins/agora/channels/incidentes/messages",
        json={"body": "blocked", "linked_task_id": "t_abc123"},
    )
    assert r.status_code == 200
    assert r.json()["message"]["linked_task_id"] == "t_abc123"


def test_list_messages_since_id(client):
    r1 = client.post(
        "/api/plugins/agora/channels/praca/messages", json={"body": "first"}
    )
    first_id = r1.json()["message"]["id"]
    client.post("/api/plugins/agora/channels/praca/messages", json={"body": "second"})

    r = client.get(f"/api/plugins/agora/channels/praca/messages?since_id={first_id}")
    assert r.status_code == 200
    assert len(r.json()["messages"]) == 1
    assert r.json()["messages"][0]["body"] == "second"


def test_list_messages_has_more_sentinel(client):
    """A full page must not imply there are more pages; limit+1 sentinel must."""
    # Create exactly PAGE_SIZE messages in a fresh channel so the last page is
    # a full page and the legacy ``length >= PAGE_SIZE`` heuristic would lie.
    client.post("/api/plugins/agora/channels", json={"slug": "sentinel", "name": "Sentinel"})
    page_size = 5
    for i in range(page_size):
        client.post(
            "/api/plugins/agora/channels/sentinel/messages",
            json={"body": f"msg-{i}"},
        )

    # A full page of exactly ``page_size`` items should report has_more=False.
    r = client.get(f"/api/plugins/agora/channels/sentinel/messages?limit={page_size}")
    assert r.status_code == 200
    data = r.json()
    assert len(data["messages"]) == page_size
    assert data["has_more"] is False

    # Add one more message; now the first page of ``page_size`` should report
    # has_more=True because the sentinel row exists.
    client.post(
        "/api/plugins/agora/channels/sentinel/messages",
        json={"body": "msg-extra"},
    )
    r = client.get(f"/api/plugins/agora/channels/sentinel/messages?limit={page_size}")
    assert r.status_code == 200
    data = r.json()
    assert len(data["messages"]) == page_size
    assert data["has_more"] is True


def test_post_message_thread_not_in_channel_rejected(client):
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "x", "thread_id": 9999},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def test_create_thread_and_get(client):
    channel = client.get("/api/plugins/agora/channels/decisoes").json()["channel"]
    r = client.post(
        "/api/plugins/agora/threads",
        json={
            "channel_id": channel["id"],
            "title": "Should we ship?",
            "linked_task_id": "t_ship",
        },
    )
    assert r.status_code == 200
    thread = r.json()["thread"]
    assert thread["title"] == "Should we ship?"
    assert thread["linked_task_id"] == "t_ship"

    r = client.get(f"/api/plugins/agora/threads/{thread['id']}")
    assert r.status_code == 200
    assert r.json()["thread"]["id"] == thread["id"]
    assert r.json()["messages"] == []


def test_create_thread_empty_title_rejected(client):
    channel = client.get("/api/plugins/agora/channels/decisoes").json()["channel"]
    r = client.post(
        "/api/plugins/agora/threads", json={"channel_id": channel["id"], "title": "   "}
    )
    assert r.status_code == 400


def test_create_thread_missing_channel(client):
    r = client.post(
        "/api/plugins/agora/threads", json={"channel_id": 9999, "title": "x"}
    )
    assert r.status_code == 404


def test_thread_messages(client):
    channel = client.get("/api/plugins/agora/channels/planejamento").json()["channel"]
    thread = client.post(
        "/api/plugins/agora/threads",
        json={"channel_id": channel["id"], "title": "plan"},
    ).json()["thread"]

    client.post(
        "/api/plugins/agora/channels/planejamento/messages",
        json={"body": "in thread", "thread_id": thread["id"]},
    )
    client.post(
        "/api/plugins/agora/channels/planejamento/messages",
        json={"body": "in thread 2", "thread_id": thread["id"]},
    )

    r = client.get(f"/api/plugins/agora/threads/{thread['id']}")
    assert len(r.json()["messages"]) == 2

    r = client.get(
        f"/api/plugins/agora/channels/planejamento/messages?thread_id={thread['id']}"
    )
    assert len(r.json()["messages"]) == 2


def test_update_thread_status(client):
    channel = client.get("/api/plugins/agora/channels/profarma").json()["channel"]
    thread = client.post(
        "/api/plugins/agora/threads",
        json={"channel_id": channel["id"], "title": "topic"},
    ).json()["thread"]

    r = client.patch(
        f"/api/plugins/agora/threads/{thread['id']}",
        json={"status": "closed"},
    )
    assert r.status_code == 200
    assert r.json()["thread"]["status"] == "closed"

    r = client.patch(
        f"/api/plugins/agora/threads/{thread['id']}",
        json={"status": "invalid"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Agent status
# ---------------------------------------------------------------------------


def test_post_and_get_agent_status(client):
    r = client.post(
        "/api/plugins/agora/agents/status/worker-1",
        json={
            "state": "working",
            "current_task_id": "t_abc",
            "current_step": "writing tests",
            "status_text": "几乎完成",
            "pid": 1234,
            "run_id": 42,
            "metadata": {"model": "kimi-k2.7-code"},
        },
    )
    assert r.status_code == 200
    agent = r.json()["agent"]
    assert agent["state"] == "working"
    assert agent["current_task_id"] == "t_abc"
    assert agent["metadata"]["model"] == "kimi-k2.7-code"

    # Heartbeat updates state. Because we store epoch seconds, wait a full
    # second so the heartbeat timestamp strictly increases.
    before = agent["last_heartbeat_at"]
    time.sleep(1.05)
    r = client.post(
        "/api/plugins/agora/agents/status/worker-1",
        json={"state": "idle"},
    )
    assert r.status_code == 200
    assert r.json()["agent"]["state"] == "idle"
    assert r.json()["agent"]["last_heartbeat_at"] > before


def test_list_agent_status(client):
    client.post("/api/plugins/agora/agents/status/a", json={"state": "working"})
    client.post("/api/plugins/agora/agents/status/b", json={"state": "deliberating"})
    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    profiles = {a["profile"] for a in r.json()["agents"]}
    assert profiles == {"a", "b"}


def test_list_agent_status_includes_manifest_profiles(client, agora_home, monkeypatch):
    """Manifest-declared profiles appear even before the first heartbeat."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    agents_dir = agora_home / "agents.d"
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / "agent-frontend.yaml").write_text("profile: agent-frontend\n", encoding="utf-8")

    monkeypatch.setattr(router_mod, "_resolve_profile_pid", lambda profile: None)

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    agents = {a["profile"]: a for a in r.json()["agents"]}
    assert "agent-frontend" in agents
    assert agents["agent-frontend"]["state"] == "idle"
    assert agents["agent-frontend"]["metadata"]["source"] == "manifest"


def test_list_agent_status_forces_manifest_source_for_configured_profiles(
    client, monkeypatch
):
    """Configured profiles keep source=manifest even if stale row metadata says otherwise."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(router_mod, "_configured_profile_names", lambda: {"agent-frontend"})
    monkeypatch.setattr(router_mod, "_kanban_active_profiles", lambda: set())

    # Seed stale metadata through the public API (source=kanban-worker).
    seeded = client.post(
        "/api/plugins/agora/agents/status/agent-frontend",
        json={
            "state": "idle",
            "status_text": "offline",
            "metadata": {"source": "kanban-worker"},
        },
    )
    assert seeded.status_code == 200

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    agents = {a["profile"]: a for a in r.json()["agents"]}
    assert agents["agent-frontend"]["metadata"]["source"] == "manifest"


def test_list_agent_status_re_resolves_pid_for_configured_profile_without_pid(
    client, monkeypatch
):
    """Configured profile rows without pid should regain a live resolved pid."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(router_mod, "_configured_profile_names", lambda: {"agent-qa"})
    monkeypatch.setattr(router_mod, "_kanban_active_profiles", lambda: set())
    monkeypatch.setattr(router_mod, "_resolve_profile_pid", lambda profile: 42424 if profile == "agent-qa" else None)

    seeded = client.post(
        "/api/plugins/agora/agents/status/agent-qa",
        json={
            "state": "idle",
            "status_text": "offline",
            "metadata": {"source": "kanban-worker"},
            "pid": None,
        },
    )
    assert seeded.status_code == 200

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    agents = {a["profile"]: a for a in r.json()["agents"]}
    assert agents["agent-qa"]["metadata"]["source"] == "manifest"
    assert agents["agent-qa"]["pid"] == 42424


def test_summon_agent_endpoint_upserts_status_and_opens_terminal(client, monkeypatch):
    """/agents/{profile}/summon creates/opens terminal and refreshes status row."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]

    monkeypatch.setattr(
        router_mod,
        "_ensure_visible_agent_terminal",
        lambda profile, allow_known=True: {
            "ok": True,
            "profile": profile,
            "session": profile,
            "created": True,
        },
    )
    monkeypatch.setattr(
        router_mod,
        "_open_agent_tmux_terminal",
        lambda profile: {
            "ok": True,
            "profile": profile,
            "session": profile,
            "opened": True,
            "focused": False,
        },
    )
    monkeypatch.setattr(router_mod, "_resolve_profile_pid", lambda profile: 4242)

    r = client.post(
        "/api/plugins/agora/agents/agent-frontend/summon",
        json={"open_terminal": True, "state": "working"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    assert data["summon"]["created"] is True
    assert data["terminal"]["opened"] is True
    assert data["agent"]["profile"] == "agent-frontend"
    assert data["agent"]["pid"] == 4242


def test_get_agent_status_404(client):
    r = client.get("/api/plugins/agora/agents/status/nobody")
    assert r.status_code == 404


def test_post_agent_status_invalid_state(client):
    r = client.post("/api/plugins/agora/agents/status/x", json={"state": "napping"})
    assert r.status_code == 400


def test_post_agent_status_fills_missing_pid_from_profile_process(client, monkeypatch):
    """When no pid is supplied, the backend discovers the profile's local process."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(router_mod, "_resolve_profile_pid", lambda profile: 12345)

    r = client.post(
        "/api/plugins/agora/agents/status/agent-techlead",
        json={"state": "idle", "current_step": "ready"},
    )
    assert r.status_code == 200
    agent = r.json()["agent"]
    assert agent["pid"] == 12345


def test_post_agent_status_preserves_explicit_pid(client, monkeypatch):
    """An explicit pid from the caller is never overwritten by discovery."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(router_mod, "_resolve_profile_pid", lambda profile: 99999)

    r = client.post(
        "/api/plugins/agora/agents/status/agent-techlead",
        json={"state": "idle", "pid": 77777},
    )
    assert r.status_code == 200
    agent = r.json()["agent"]
    assert agent["pid"] == 77777


def test_post_agent_status_rejects_reserved_profile_human(client):
    """Reserved pseudo-profiles must not be promoted to agent rows."""
    r = client.post(
        "/api/plugins/agora/agents/status/human",
        json={"state": "idle", "current_step": "diag", "status_text": "diag status"},
    )
    assert r.status_code == 400
    assert "reserved" in r.json()["detail"].lower()


def test_post_agent_status_rejects_reserved_profile_with_explicit_pid(client):
    """An explicit pid must not bypass the reserved-profile guard."""
    r = client.post(
        "/api/plugins/agora/agents/status/human",
        json={"state": "idle", "pid": 12345},
    )
    assert r.status_code == 400


def test_get_agent_status_reserved_profile_not_found(client):
    r = client.get("/api/plugins/agora/agents/status/human")
    assert r.status_code == 404


def test_list_agent_status_excludes_reserved_profile(client, agora_home):
    """A stale reserved row in the DB must never appear in the dashboard."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]

    # Real agent row created through the API.
    client.post("/api/plugins/agora/agents/status/real-agent", json={"state": "working"})

    # Simulate a pre-existing stale row for the reserved ``human`` profile.
    conn = sqlite3.connect(str(router_mod._db_path()))
    try:
        conn.execute(
            """
            INSERT INTO agora_agent_status
                (profile, state, current_task_id, current_step, status_text,
                 last_heartbeat_at, pid, run_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("human", "idle", None, "diag", "diag status", int(time.time()), 136625, None, None),
        )
        conn.commit()
    finally:
        conn.close()

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    profiles = {a["profile"] for a in r.json()["agents"]}
    assert "real-agent" in profiles
    assert "human" not in profiles


def test_list_agent_status_clears_stale_dead_pid(client, monkeypatch):
    """Dead stored PIDs are nulled so non-invoked agents don't look active."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(router_mod, "_pid_is_alive", lambda pid: False)

    client.post(
        "/api/plugins/agora/agents/status/agent-backend",
        json={"state": "working", "pid": 99999},
    )

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    row = next(a for a in r.json()["agents"] if a["profile"] == "agent-backend")
    assert row["pid"] is None
    assert row["state"] == "idle"


def test_list_agent_status_clears_stale_worker_snapshot_when_pid_dead(client, monkeypatch):
    """Dead PID cleanup must also clear stale run/task fields from old worker snapshots."""
    router_mod = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(router_mod, "_pid_is_alive", lambda pid: False)

    client.post(
        "/api/plugins/agora/agents/status/agent-qa",
        json={
            "state": "working",
            "pid": 99999,
            "run_id": 171,
            "current_task_id": "t_275c6f16",
            "current_step": "run 171",
            "status_text": "stale worker snapshot",
            "metadata": {"source": "kanban-worker", "run_id": 171, "task_id": "t_275c6f16"},
        },
    )

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    row = next(a for a in r.json()["agents"] if a["profile"] == "agent-qa")
    assert row["pid"] is None
    assert row["state"] == "idle"
    assert row["run_id"] is None
    assert row["current_task_id"] is None
    assert row["current_step"] is None


def test_list_agent_status_worker_active_overrides_stale_status(client, monkeypatch):
    """A Kanban active worker must overwrite stale agora_agent_status state/pid/step."""
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    # Seed a stale snapshot: idle agent with old pid/step/status from a previous run.
    client.post(
        "/api/plugins/agora/agents/status/agent-frontend",
        json={
            "state": "idle",
            "current_task_id": "t_old",
            "current_step": "run 42",
            "status_text": "old task done",
            "pid": 11111,
            "run_id": 42,
        },
    )

    # Live worker telemetry comes from Kanban.
    monkeypatch.setattr(pa, "_kanban_active_profiles", lambda: {"agent-frontend"})
    monkeypatch.setattr(
        pa,
        "_kanban_active_worker",
        lambda profile: {
            "task_id": "t_b572c33d",
            "run_id": 162,
            "worker_pid": 5423,
            "started_at": 1782386201,
            "last_heartbeat_at": 1782386237,
            "task_title": "Estabilidade: worker ativo deve vencer agora_agent_status stale",
        },
    )

    r = client.get("/api/plugins/agora/agents/status")
    assert r.status_code == 200
    agent = next(a for a in r.json()["agents"] if a["profile"] == "agent-frontend")
    assert agent["state"] == "working"
    assert agent["current_task_id"] == "t_b572c33d"
    assert agent["current_step"] == "run 162"
    assert agent["status_text"] == "Estabilidade: worker ativo deve vencer agora_agent_status stale"
    assert agent["pid"] == 5423
    assert agent["run_id"] == 162

    # agora_agent_status itself must have been upserted with the worker truth.
    with pa._connect() as conn:
        row = conn.execute(
            "SELECT state, pid, run_id, current_step, current_task_id, status_text, metadata_json "
            "FROM agora_agent_status WHERE profile = ?",
            ("agent-frontend",),
        ).fetchone()
        assert row["state"] == "working"
        assert row["pid"] == 5423
        assert row["run_id"] == 162
        assert row["current_step"] == "run 162"
        assert row["current_task_id"] == "t_b572c33d"
        assert "stale" in row["status_text"]
        meta = json.loads(row["metadata_json"])
        assert meta.get("source") == "kanban-worker"


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def test_create_decision_and_list(client):
    channel = client.get("/api/plugins/agora/channels/decisoes").json()["channel"]
    thread = client.post(
        "/api/plugins/agora/threads",
        json={"channel_id": channel["id"], "title": "decide"},
    ).json()["thread"]

    r = client.post(
        "/api/plugins/agora/decisions",
        json={
            "thread_id": thread["id"],
            "proposal": "Use SQLite local",
            "decision": "approved",
            "rationale": "Simple and profile-safe",
            "decided_by": "human",
        },
    )
    assert r.status_code == 200
    decision = r.json()["decision"]
    assert decision["proposal"] == "Use SQLite local"
    assert decision["decided_by"] == "human"

    r = client.get("/api/plugins/agora/decisions")
    assert len(r.json()["decisions"]) == 1

    r = client.get(f"/api/plugins/agora/decisions?thread_id={thread['id']}")
    assert len(r.json()["decisions"]) == 1


def test_create_decision_missing_thread(client):
    r = client.post(
        "/api/plugins/agora/decisions",
        json={"thread_id": 9999, "proposal": "x", "decision": "y"},
    )
    assert r.status_code == 404


def test_create_decision_empty_fields(client):
    r = client.post(
        "/api/plugins/agora/decisions",
        json={"thread_id": 1, "proposal": "", "decision": ""},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_list_returns_mutations(client):
    r = client.get("/api/plugins/agora/events")
    assert r.status_code == 200
    assert r.json()["events"] == []
    assert r.json()["cursor"] == 0

    client.post("/api/plugins/agora/channels/praca/messages", json={"body": "event me"})

    r = client.get("/api/plugins/agora/events")
    assert len(r.json()["events"]) >= 1
    assert r.json()["events"][0]["event_type"] == "created"
    assert r.json()["events"][0]["entity_type"] == "message"


def test_events_since_id(client):
    client.post("/api/plugins/agora/channels/praca/messages", json={"body": "first"})
    r = client.get("/api/plugins/agora/events")
    cursor = r.json()["cursor"]

    client.post("/api/plugins/agora/channels/praca/messages", json={"body": "second"})
    r = client.get(f"/api/plugins/agora/events?since_id={cursor}")
    assert len(r.json()["events"]) >= 1
    bodies = {e["payload"]["channel_slug"] for e in r.json()["events"]}
    assert "praca" in bodies


# ---------------------------------------------------------------------------
# WebSocket events
# ---------------------------------------------------------------------------


def test_ws_events_rejects_when_token_required(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import hermes_cli
    import types

    def _fake_ws_auth_ok(ws):
        return ws.query_params.get("token", "") == "secret-xyz"

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=_fake_ws_auth_ok,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    _router_mod, router = _load_plugin_router()
    app.include_router(router, prefix="/api/plugins/agora")
    c = TestClient(app)

    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/agora/events"):
            pass
    assert exc.value.code == 1008

    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/plugins/agora/events?token=nope"):
            pass
    assert exc.value.code == 1008

    with c.websocket_connect("/api/plugins/agora/events?token=secret-xyz") as ws:
        assert ws is not None


def test_ws_events_stream_message_creation(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import hermes_cli
    import types

    stub = types.SimpleNamespace(
        _SESSION_TOKEN="secret-xyz",
        _ws_auth_ok=lambda ws: ws.query_params.get("token", "") == "secret-xyz",
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.web_server", stub)
    monkeypatch.setattr(hermes_cli, "web_server", stub, raising=False)

    app = FastAPI()
    _router_mod, router = _load_plugin_router()
    app.include_router(router, prefix="/api/plugins/agora")
    c = TestClient(app)

    # Shorten polling so the test receives events quickly.
    import plugins.agora.dashboard.plugin_api as pa

    monkeypatch.setattr(pa, "_EVENT_POLL_SECONDS", 0.05)

    with c.websocket_connect(
        "/api/plugins/agora/events?token=secret-xyz&since=0"
    ) as ws:
        c.post("/api/plugins/agora/channels/praca/messages", json={"body": "live"})
        payload = ws.receive_json()
        assert "events" in payload
        assert payload["cursor"] > 0
        assert any(e["entity_type"] == "message" for e in payload["events"])


def test_ws_events_swallows_cancellation_on_shutdown(tmp_path, monkeypatch):
    import asyncio

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import plugins.agora.dashboard.plugin_api as pa

    monkeypatch.setattr(pa, "_ws_upgrade_authorized", lambda ws: True)

    class _FakeWS:
        def __init__(self):
            self.query_params = {"token": "x", "since": "0"}
            self.accepted = False
            self.closed = False

        async def accept(self):
            self.accepted = True

        async def send_json(self, data):
            pass

        async def close(self, code=None):
            self.closed = True

    async def _run():
        ws = _FakeWS()
        task = asyncio.create_task(pa.stream_events(ws))
        await asyncio.sleep(0.05)
        assert ws.accepted is True
        task.cancel()
        result = await task
        return result, ws

    result, ws = asyncio.run(_run())
    assert result is None


# ---------------------------------------------------------------------------
# Mentions / mailbox
# ---------------------------------------------------------------------------


def test_extract_mentions():
    import plugins.agora.dashboard.plugin_api as pa

    assert pa._extract_mentions("oi @agent-frontend veja isso") == ["agent-frontend"]
    assert set(pa._extract_mentions("@all @agent-backend @todos")) == {
        "all",
        "agent-backend",
        "todos",
    }
    assert pa._extract_mentions("sem mencoes aqui") == []
    # Deduplicate while preserving order
    assert pa._extract_mentions("@a @a @b") == ["a", "b"]


def test_extract_mentions_ignores_quoted_mentions():
    import plugins.agora.dashboard.plugin_api as pa

    assert pa._extract_mentions("o comando '@todos' aqui é só texto") == []
    assert pa._extract_mentions('falando de "@all" sem acionar ninguém') == []
    assert pa._extract_mentions("use `@agent-frontend` como exemplo") == []
    assert pa._extract_mentions("avisar @agent-qa sobre '@todos'") == ["agent-qa"]


def test_quoted_todos_does_not_broadcast(client):
    client.post("/api/plugins/agora/agents/status/worker-a", json={"state": "idle"})
    client.post("/api/plugins/agora/agents/status/worker-b", json={"state": "idle"})

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "o comando '@todos' aqui é só exemplo", "author_profile": "human"},
    )
    assert r.status_code == 200

    for profile in ("worker-a", "worker-b"):
        resp = client.get(f"/api/plugins/agora/notifications?recipient={profile}")
        assert resp.status_code == 200
        assert resp.json()["notifications"] == []


def test_resolve_recipients_filters_literal_placeholders(client):
    """Handles like @perfil and @fallback are documentation tokens, not recipients."""
    import plugins.agora.dashboard.plugin_api as pa

    client.post("/api/plugins/agora/agents/status/worker-x", json={"state": "idle"})
    with pa._connect() as conn:
        recipients = pa._resolve_recipients(
            conn, ["perfil", "fallback", "agent-frontend", "worker-x", "all"]
        )
    assert recipients == {"agent-frontend", "worker-x"}


def test_placeholder_mentions_do_not_create_notifications(client):
    """System/docs messages citing @perfil or @fallback must not create rows."""
    client.post("/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"})

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@perfil e @fallback são só exemplos; @agent-frontend deve ver",
            "author_profile": "human",
        },
    )
    assert r.status_code == 200

    for placeholder in ("perfil", "fallback"):
        resp = client.get(f"/api/plugins/agora/notifications?recipient={placeholder}")
        assert resp.status_code == 200
        assert resp.json()["notifications"] == []

    real_resp = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert real_resp.status_code == 200
    assert len(real_resp.json()["notifications"]) == 1


def test_unknown_profile_mention_does_not_create_notification(client):
    """Arbitrary @handles that are neither known nor checked-in are silently ignored."""
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@some-random-profile please review",
            "author_profile": "human",
        },
    )
    assert r.status_code == 200

    resp = client.get("/api/plugins/agora/notifications?recipient=some-random-profile")
    assert resp.status_code == 200
    assert resp.json()["notifications"] == []


def test_mention_creates_notification(client):
    import plugins.agora.dashboard.plugin_api as pa

    # Ensure recipient is known as an active agent.
    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200
    msg_id = r.json()["message"]["id"]

    notif_resp = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert notif_resp.status_code == 200
    notifs = notif_resp.json()["notifications"]
    assert len(notifs) == 1
    assert notifs[0]["recipient"] == "agent-frontend"
    assert notifs[0]["message_id"] == msg_id
    assert notifs[0]["read_at"] is None


def test_delivered_tmux_mention_marks_notification_read_and_acked(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)
    monkeypatch.setattr(
        pa,
        "_tmux_send_message",
        lambda recipient, message: {
            "ok": True,
            "profile": recipient,
            "session": recipient,
            "created": False,
            "delivered": True,
        },
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200

    notif_resp = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert notif_resp.status_code == 200
    notif = notif_resp.json()["notifications"][0]
    assert notif["read_at"] is not None
    assert notif["ack_at"] is not None


def test_non_idle_tmux_mention_is_delivered_as_steer(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    sent = []

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "working"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)

    def _fake_send(recipient, message):
        sent.append((recipient, message))
        return {
            "ok": True,
            "profile": recipient,
            "session": recipient,
            "created": False,
            "delivered": True,
        }

    monkeypatch.setattr(pa, "_tmux_send_message", _fake_send)

    before = client.get("/api/plugins/agora/events").json()["cursor"]
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200

    assert sent
    assert sent[0][0] == "agent-frontend"
    assert sent[0][1].startswith("/steer ")
    assert "Ágora mention in #praca" in sent[0][1]

    notif = client.get(
        "/api/plugins/agora/notifications?recipient=agent-frontend"
    ).json()["notifications"][0]
    assert notif["read_at"] is not None
    assert notif["ack_at"] is not None

    events = client.get(f"/api/plugins/agora/events?since_id={before}").json()["events"]
    delivered = [
        e for e in events
        if e["entity_type"] == "wake_delivery" and e["event_type"] == "delivered"
    ][0]
    assert delivered["payload"]["delivery_mode"] == "steer"
    assert delivered["payload"]["agent_state"] == "working"


def test_idle_tmux_mention_is_delivered_as_plain_prompt(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    sent = []

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)

    def _fake_send(recipient, message):
        sent.append((recipient, message))
        return {
            "ok": True,
            "profile": recipient,
            "session": recipient,
            "created": False,
            "delivered": True,
        }

    monkeypatch.setattr(pa, "_tmux_send_message", _fake_send)

    before = client.get("/api/plugins/agora/events").json()["cursor"]
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200

    assert sent
    assert sent[0][0] == "agent-frontend"
    assert not sent[0][1].startswith("/steer ")
    assert sent[0][1].startswith("Ágora mention in #praca")

    events = client.get(f"/api/plugins/agora/events?since_id={before}").json()["events"]
    delivered = [
        e for e in events
        if e["entity_type"] == "wake_delivery" and e["event_type"] == "delivered"
    ][0]
    assert delivered["payload"]["delivery_mode"] == "prompt"
    assert delivered["payload"]["agent_state"] == "idle"


def test_active_kanban_worker_overrides_idle_agent_state_for_tmux(client, monkeypatch):
    """A Kanban active worker forces /steer even if agora_agent_status says idle."""
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    sent = []

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)
    monkeypatch.setattr(pa, "_kanban_active_profiles", lambda: {"agent-frontend"})

    def _fake_send(recipient, message):
        sent.append((recipient, message))
        return {
            "ok": True,
            "profile": recipient,
            "session": recipient,
            "created": False,
            "delivered": True,
        }

    monkeypatch.setattr(pa, "_tmux_send_message", _fake_send)

    before = client.get("/api/plugins/agora/events").json()["cursor"]
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200

    assert sent
    assert sent[0][0] == "agent-frontend"
    assert sent[0][1].startswith("/steer ")

    events = client.get(f"/api/plugins/agora/events?since_id={before}").json()["events"]
    delivered = [
        e for e in events
        if e["entity_type"] == "wake_delivery" and e["event_type"] == "delivered"
    ][0]
    assert delivered["payload"]["delivery_mode"] == "steer"
    assert delivered["payload"]["agent_state"] == "working"


def test_agent_state_helper_uses_active_profiles_override(client, monkeypatch):
    """_agent_state returns working when the profile is in the active set."""
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    with pa._connect() as conn:
        assert pa._agent_state(conn, "agent-frontend", active_profiles=set()) == "idle"
        assert pa._agent_state(conn, "agent-frontend", active_profiles={"agent-frontend"}) == "working"
        assert pa._agent_state(conn, "unknown-profile", active_profiles={"unknown-profile"}) == "working"


def test_failed_tmux_mention_delivery_leaves_notification_unread(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)
    monkeypatch.setattr(
        pa,
        "_tmux_send_message",
        lambda recipient, message: {
            "ok": False,
            "profile": recipient,
            "session": recipient,
            "reason": "send-keys-failed",
            "delivered": False,
        },
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200

    notif_resp = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert notif_resp.status_code == 200
    notif = notif_resp.json()["notifications"][0]
    assert notif["read_at"] is None
    assert notif["ack_at"] is None


def test_delivered_tmux_mention_emits_notification_read_event(client, monkeypatch):
    """A successful tmux wake-up surfaces a notification.read event with auto_ack."""
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)
    monkeypatch.setattr(
        pa,
        "_tmux_send_message",
        lambda recipient, message: {
            "ok": True,
            "profile": recipient,
            "session": recipient,
            "created": False,
            "delivered": True,
        },
    )

    before = client.get("/api/plugins/agora/events").json()["cursor"]
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200
    msg_id = r.json()["message"]["id"]

    events_resp = client.get(f"/api/plugins/agora/events?since_id={before}")
    assert events_resp.status_code == 200
    events = events_resp.json()["events"]
    read_events = [
        e
        for e in events
        if e["entity_type"] == "notification" and e["event_type"] == "read"
    ]
    assert len(read_events) == 1
    assert read_events[0]["payload"]["recipient"] == "agent-frontend"
    assert read_events[0]["payload"]["message_id"] == msg_id
    assert read_events[0]["payload"]["auto_ack"] is True

    delivered_events = [
        e
        for e in events
        if e["entity_type"] == "wake_delivery" and e["event_type"] == "delivered"
    ]
    assert len(delivered_events) == 1
    assert delivered_events[0]["payload"]["recipient"] == "agent-frontend"


def test_failed_tmux_mention_emits_wake_delivery_failed_event(client, monkeypatch):
    """A failed tmux wake-up emits wake_delivery.failed but no notification.read."""
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)
    monkeypatch.setattr(
        pa,
        "_tmux_send_message",
        lambda recipient, message: {
            "ok": False,
            "profile": recipient,
            "session": recipient,
            "reason": "send-keys-failed",
            "delivered": False,
        },
    )

    before = client.get("/api/plugins/agora/events").json()["cursor"]
    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend preciso de você",
            "author_profile": "agent-backend",
        },
    )
    assert r.status_code == 200

    events_resp = client.get(f"/api/plugins/agora/events?since_id={before}")
    assert events_resp.status_code == 200
    events = events_resp.json()["events"]
    read_events = [
        e
        for e in events
        if e["entity_type"] == "notification" and e["event_type"] == "read"
    ]
    assert read_events == []

    failed_events = [
        e
        for e in events
        if e["entity_type"] == "wake_delivery" and e["event_type"] == "failed"
    ]
    assert len(failed_events) == 1
    assert failed_events[0]["payload"]["recipient"] == "agent-frontend"


def test_partial_tmux_delivery_only_acks_successful_recipient(client, monkeypatch):
    """With mixed delivery results only the successfully woken recipient is auto-acked."""
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    client.post("/api/plugins/agora/agents/status/worker-a", json={"state": "idle"})
    client.post("/api/plugins/agora/agents/status/worker-b", json={"state": "idle"})
    monkeypatch.setattr(pa, "_tmux_wake_enabled", lambda: True)

    def _fake_send(recipient, message):
        delivered = recipient == "worker-a"
        return {
            "ok": delivered,
            "profile": recipient,
            "session": recipient,
            "created": False,
            "delivered": delivered,
        }

    monkeypatch.setattr(pa, "_tmux_send_message", _fake_send)

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "@all reunião", "author_profile": "coordenador"},
    )
    assert r.status_code == 200

    a_notif = client.get(
        "/api/plugins/agora/notifications?recipient=worker-a"
    ).json()["notifications"][0]
    b_notif = client.get(
        "/api/plugins/agora/notifications?recipient=worker-b"
    ).json()["notifications"][0]
    assert a_notif["read_at"] is not None
    assert a_notif["ack_at"] is not None
    assert b_notif["read_at"] is None
    assert b_notif["ack_at"] is None


def test_human_mention_without_author_profile_creates_notification_with_human_author(
    client,
):
    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "@agent-frontend preciso de você", "author_type": "human"},
    )
    assert r.status_code == 200
    msg = r.json()["message"]
    assert msg["author_profile"] == "human"

    notif_resp = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert notif_resp.status_code == 200
    notifs = notif_resp.json()["notifications"]
    assert len(notifs) == 1
    assert notifs[0]["author_profile"] == "human"


def test_all_mention_broadcasts(client):
    client.post("/api/plugins/agora/agents/status/worker-a", json={"state": "idle"})
    client.post("/api/plugins/agora/agents/status/worker-b", json={"state": "idle"})

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "@all reunião na praça", "author_profile": "coordenador"},
    )
    assert r.status_code == 200

    for profile in ("worker-a", "worker-b"):
        resp = client.get(f"/api/plugins/agora/notifications?recipient={profile}")
        assert len(resp.json()["notifications"]) == 1
        assert (
            resp.json()["notifications"][0]["body_snippet"] == "@all reunião na praça"
        )


def test_tech_lead_profile_is_wakeable():
    import plugins.agora.dashboard.plugin_api as pa

    assert "agent-techlead" in pa._PROFILE_SKILLS


def test_todos_broadcast_reaches_tech_lead(client):
    client.post(
        "/api/plugins/agora/agents/status/agent-backend", json={"state": "idle"}
    )
    client.post(
        "/api/plugins/agora/agents/status/agent-techlead", json={"state": "working"}
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "@todos reunião na praça", "author_profile": "felipi"},
    )
    assert r.status_code == 200

    resp = client.get("/api/plugins/agora/notifications?recipient=agent-techlead")
    assert resp.status_code == 200
    notifs = resp.json()["notifications"]
    assert len(notifs) == 1
    assert notifs[0]["body_snippet"] == "@todos reunião na praça"


def test_todos_broadcast_excludes_author(client):
    client.post("/api/plugins/agora/agents/status/agent-qa", json={"state": "idle"})
    client.post(
        "/api/plugins/agora/agents/status/agent-backend", json={"state": "working"}
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "@todos preciso de olhos", "author_profile": "agent-qa"},
    )
    assert r.status_code == 200

    # Author must not receive a notification about their own broadcast.
    author_resp = client.get("/api/plugins/agora/notifications?recipient=agent-qa")
    assert author_resp.status_code == 200
    assert author_resp.json()["notifications"] == []

    # Other agents still receive it.
    backend_resp = client.get(
        "/api/plugins/agora/notifications?recipient=agent-backend"
    )
    assert backend_resp.status_code == 200
    assert len(backend_resp.json()["notifications"]) == 1


def test_all_broadcast_excludes_author(client):
    client.post("/api/plugins/agora/agents/status/coordenador", json={"state": "idle"})
    client.post("/api/plugins/agora/agents/status/worker-b", json={"state": "idle"})

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={"body": "@all reunião na praça", "author_profile": "coordenador"},
    )
    assert r.status_code == 200

    author_resp = client.get("/api/plugins/agora/notifications?recipient=coordenador")
    assert author_resp.status_code == 200
    assert author_resp.json()["notifications"] == []

    worker_resp = client.get("/api/plugins/agora/notifications?recipient=worker-b")
    assert worker_resp.status_code == 200
    assert len(worker_resp.json()["notifications"]) == 1


def test_author_self_mention_is_excluded(client):
    client.post(
        "/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"}
    )

    r = client.post(
        "/api/plugins/agora/channels/praca/messages",
        json={
            "body": "@agent-frontend nota para mim mesmo",
            "author_profile": "agent-frontend",
        },
    )
    assert r.status_code == 200

    resp = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert resp.status_code == 200
    assert resp.json()["notifications"] == []


# ---------------------------------------------------------------------------
# Kanban completion integration
# ---------------------------------------------------------------------------


def test_kanban_task_completion_notifies_tech_lead(client, monkeypatch):
    """Completing a kanban task posts an Ágora delivery report and notifies agent-techlead."""
    from hermes_cli import kanban_db as kb

    # The core callback imports the real ``plugins.agora.dashboard.plugin_api``
    # module, whose DB-init flag needs to honour the per-test HERMES_HOME.
    import plugins.agora.dashboard.plugin_api as real_agora

    real_agora._db_init_path = None

    # Ensure agent-techlead is present so the @mention resolves cleanly.
    client.post(
        "/api/plugins/agora/agents/status/agent-techlead", json={"state": "idle"}
    )

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Ship feature X", assignee="agent-backend")
        ok = kb.complete_task(
            conn,
            task_id,
            summary="Feature shipped with 12 tests passing.",
            result="shipped",
            metadata={"artifacts": ["/tmp/report.pdf"], "tests_run": 12},
        )
        assert ok is True
    finally:
        conn.close()

    # Ágora message in the configured completion channel.
    r = client.get("/api/plugins/agora/channels/planejamento/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["author_type"] == "system"
    assert msg["author_profile"] == "kanban"
    assert msg["linked_task_id"] == task_id
    body = msg["body"]
    assert "@agent-techlead" in body
    assert "Ship feature X" in body
    assert task_id in body
    assert "agent-backend" in body
    assert "Feature shipped with 12 tests passing." in body
    assert "/tmp/report.pdf" in body

    # Mailbox notification for agent-techlead.
    r = client.get("/api/plugins/agora/notifications?recipient=agent-techlead")
    assert r.status_code == 200
    notifs = r.json()["notifications"]
    assert len(notifs) == 1
    assert notifs[0]["message_id"] == msg["id"]
    assert notifs[0]["recipient"] == "agent-techlead"


def test_kanban_task_completion_drops_unknown_task_id(client):
    """Synthetic completions for non-existent Kanban task ids must not post to Ágora."""
    from plugins.agora.dashboard.plugin_api import _on_kanban_task_completed

    _on_kanban_task_completed(
        task_id="t_3386de3f",
        title="Ship feature X",
        assignee="agent-backend",
        status="done",
        result="shipped",
        summary="Feature shipped with 12 tests passing.",
        metadata={"artifacts": ["/tmp/report.pdf"], "tests_run": 12},
        completed_at=0,
    )

    r = client.get("/api/plugins/agora/channels/planejamento/messages")
    assert r.status_code == 200
    assert r.json()["messages"] == []

    r = client.get("/api/plugins/agora/notifications?recipient=agent-techlead")
    assert r.status_code == 200
    assert r.json()["notifications"] == []


def test_format_delivery_report_warns_when_summary_missing():
    """A completed task with no summary/result surfaces a visible warning."""
    import plugins.agora.dashboard.plugin_api as pa

    body = pa._format_delivery_report(
        task_id="t_abc",
        title="No report task",
        assignee="worker",
        result=None,
        summary=None,
        metadata=None,
    )
    assert "@agent-techlead" in body
    assert "No report task" in body
    assert "nenhum relatório de entrega" in body.lower() or "Nenhum relatório" in body


def test_resolve_handoff_mention_prefers_explicit_handle():
    import plugins.agora.dashboard.plugin_api as pa

    assert pa._resolve_handoff_mention("@agent-frontend check this") == "agent-frontend"
    assert pa._resolve_handoff_mention("ask @agent-backend") == "agent-backend"
    assert pa._resolve_handoff_mention("cc @agent-techlead") == "agent-techlead"
    # First explicit mention wins.
    assert pa._resolve_handoff_mention("@agent-qa @agent-frontend") == "agent-qa"


def test_resolve_handoff_mention_infers_from_keywords():
    import plugins.agora.dashboard.plugin_api as pa

    assert pa._resolve_handoff_mention("need frontend review") == "agent-frontend"
    assert pa._resolve_handoff_mention("backend wiring required") == "agent-backend"
    assert pa._resolve_handoff_mention("waiting QA validation") == "agent-qa"
    assert pa._resolve_handoff_mention("need tech-lead decision") == "agent-techlead"


def test_resolve_handoff_mention_fallback_to_tech_lead():
    import plugins.agora.dashboard.plugin_api as pa

    assert pa._resolve_handoff_mention("") == "agent-techlead"
    assert pa._resolve_handoff_mention("stuck") == "agent-techlead"
    assert pa._resolve_handoff_mention(None) == "agent-techlead"


def test_kanban_task_blocked_notifies_expected_profile(client, monkeypatch):
    """Blocking a kanban task posts a handoff to Ágora and notifies the target profile."""
    from hermes_cli import kanban_db as kb

    import plugins.agora.dashboard.plugin_api as real_agora

    real_agora._db_init_path = None

    # Ensure target profile is known so notifications resolve.
    client.post("/api/plugins/agora/agents/status/agent-frontend", json={"state": "idle"})

    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="Mobile checkout overflow", assignee="agent-qa"
        )
        # Claim the task so it can be blocked from a running state.
        kb.claim_task(conn, task_id)
        ok = kb.block_task(
            conn,
            task_id,
            reason="@agent-frontend need responsive review: widget overflows on mobile",
        )
        assert ok is True
    finally:
        conn.close()

    # Ágora handoff message in the configured handoff channel (#praca).
    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    msgs = r.json()["messages"]
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["author_type"] == "system"
    assert msg["author_profile"] == "kanban"
    assert msg["linked_task_id"] == task_id
    body = msg["body"]
    assert "@agent-frontend" in body
    assert "Mobile checkout overflow" in body
    assert task_id in body
    assert "widget overflows on mobile" in body

    # Mailbox notification for the mentioned profile.
    r = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert r.status_code == 200
    notifs = r.json()["notifications"]
    assert len(notifs) == 1
    assert notifs[0]["message_id"] == msg["id"]
    assert notifs[0]["recipient"] == "agent-frontend"


def test_kanban_task_blocked_infers_backend_from_keyword(client):
    """When no @handle is given, blocked handoff infers the owner from keywords."""
    from hermes_cli import kanban_db as kb

    import plugins.agora.dashboard.plugin_api as real_agora

    real_agora._db_init_path = None
    client.post("/api/plugins/agora/agents/status/agent-backend", json={"state": "idle"})

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Wire API", assignee="agent-frontend")
        kb.claim_task(conn, task_id)
        kb.block_task(conn, task_id, reason="need backend to expose /orders endpoint")
    finally:
        conn.close()

    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    body = r.json()["messages"][0]["body"]
    assert "@agent-backend" in body

    r = client.get("/api/plugins/agora/notifications?recipient=agent-backend")
    assert r.status_code == 200
    assert len(r.json()["notifications"]) == 1


def test_kanban_task_blocked_fallback_to_tech_lead(client):
    """Blocked handoff with no clear owner falls back to @agent-techlead."""
    from hermes_cli import kanban_db as kb

    import plugins.agora.dashboard.plugin_api as real_agora

    real_agora._db_init_path = None
    client.post("/api/plugins/agora/agents/status/agent-techlead", json={"state": "idle"})

    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="Ambiguous blocker", assignee="agent-qa")
        kb.claim_task(conn, task_id)
        kb.block_task(conn, task_id, reason="stuck")
    finally:
        conn.close()

    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    body = r.json()["messages"][0]["body"]
    assert "@agent-techlead" in body

    r = client.get("/api/plugins/agora/notifications?recipient=agent-techlead")
    assert r.status_code == 200
    assert len(r.json()["notifications"]) == 1


def test_kanban_task_blocked_drops_unknown_task_id(client):
    """Synthetic block callbacks for non-existent task ids must not post to Ágora."""
    from plugins.agora.dashboard.plugin_api import _on_kanban_task_blocked

    _on_kanban_task_blocked(
        task_id="t_3386de3f",
        title="Ghost task",
        assignee="agent-backend",
        reason="blocked",
        blocked_at=0,
    )

    r = client.get("/api/plugins/agora/channels/praca/messages")
    assert r.status_code == 200
    assert r.json()["messages"] == []

    r = client.get("/api/plugins/agora/notifications?recipient=agent-frontend")
    assert r.status_code == 200
    assert r.json()["notifications"] == []


def test_notifications_count_and_read(client):
    client.post("/api/plugins/agora/agents/status/target", json={"state": "idle"})
    client.post(
        "/api/plugins/agora/channels/planejamento/messages",
        json={"body": "@target check this", "author_profile": "sender"},
    )

    count_resp = client.get("/api/plugins/agora/notifications/count?recipient=target")
    assert count_resp.json()["unread"] == 1
    assert count_resp.json()["total"] == 1

    notif_id = client.get("/api/plugins/agora/notifications?recipient=target").json()[
        "notifications"
    ][0]["id"]
    read_resp = client.post(f"/api/plugins/agora/notifications/{notif_id}/read")
    assert read_resp.status_code == 200
    assert read_resp.json()["notification"]["read_at"] is not None

    count_resp = client.get("/api/plugins/agora/notifications/count?recipient=target")
    assert count_resp.json()["unread"] == 0


# ---------------------------------------------------------------------------
# Profile DB migration (shared root hardening)
# ---------------------------------------------------------------------------


def _seed_legacy_profile_db(pa_mod, db_path: Path, messages: list[tuple[str, str, str, str]]) -> None:
    """Create a fully-initialised legacy per-profile agora.db at ``db_path``.

    Temporarily redirects the plugin's ``_db_path`` helper so ``_init_db``
    creates the schema in the profile DB instead of the shared root.
    """
    original_db_path = pa_mod._db_path
    pa_mod._db_path = lambda: db_path
    pa_mod._db_init_path = None
    try:
        pa_mod._init_db()
        with pa_mod._connect() as conn:
            now = int(time.time())
            for slug, body, author_type, author_profile in messages:
                row = conn.execute(
                    "SELECT id FROM agora_channels WHERE slug = ?", (slug,)
                ).fetchone()
                if row is None:
                    cur = conn.execute(
                        "INSERT INTO agora_channels (slug, name, description, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (slug, slug, "", now),
                    )
                    channel_id = cur.lastrowid
                else:
                    channel_id = row["id"]
                conn.execute(
                    "INSERT INTO agora_messages "
                    "(channel_id, thread_id, author_type, author_profile, body, linked_task_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (channel_id, None, author_type, author_profile, body, None, now),
                )
            conn.commit()
    finally:
        pa_mod._db_path = original_db_path
        pa_mod._db_init_path = None


def test_migrate_profile_dbs_moves_messages_to_shared_root(tmp_path, monkeypatch):
    """Legacy per-profile agora.db files are merged into the shared root DB."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "worker-a"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    _seed_legacy_profile_db(
        pa, profile_home / "agora.db", [("praca", "legacy hello", "agent", "worker-a")]
    )

    report = pa.migrate_profile_agora_dbs()
    assert report["ok"] is True
    assert report["errors"] == []
    assert report["skipped"] == []
    assert len(report["migrated"]) == 1
    assert report["migrated"][0]["profile"] == "worker-a"

    shared_db = root / "agora.db"
    assert shared_db.exists()

    conn = sqlite3.connect(str(shared_db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT body, author_profile FROM agora_messages WHERE author_profile = ?",
            ("worker-a",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["body"] == "legacy hello"
    finally:
        conn.close()

    # Source is renamed and backed up so it cannot be reprocessed accidentally.
    assert not (profile_home / "agora.db").exists()
    assert len(list(profile_home.glob("agora.db.*.bak"))) == 1
    assert len(list(profile_home.glob("agora.db.migrated"))) == 1


def test_migrate_profile_dbs_idempotent(tmp_path, monkeypatch):
    """Running the migration twice does not duplicate data."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "worker-a"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    _seed_legacy_profile_db(
        pa, profile_home / "agora.db", [("praca", "legacy hello", "agent", "worker-a")]
    )

    r1 = pa.migrate_profile_agora_dbs()
    assert len(r1["migrated"]) == 1

    r2 = pa.migrate_profile_agora_dbs()
    assert r2["ok"] is True
    assert r2["migrated"] == []
    assert r2["skipped"] == []

    conn = sqlite3.connect(str(root / "agora.db"))
    conn.row_factory = sqlite3.Row
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM agora_messages WHERE author_profile = ?", ("worker-a",)
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()


def test_migrate_profile_dbs_dry_run_leaves_source_intact(tmp_path, monkeypatch):
    """Dry-run reports the work but does not touch the source or target DBs."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "worker-a"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    source_db = profile_home / "agora.db"
    _seed_legacy_profile_db(pa, source_db, [("praca", "legacy hello", "agent", "worker-a")])

    report = pa.migrate_profile_agora_dbs(dry_run=True)
    assert report["ok"] is True
    assert report["dry_run"] is True
    assert len(report["migrated"]) == 1
    assert report["migrated"][0]["dry_run"] is True

    # Source untouched, no backup created, shared DB unchanged.
    assert source_db.exists()
    assert len(list(profile_home.glob("agora.db.*"))) == 0
    assert not (root / "agora.db").exists()


def test_migrate_profile_dbs_no_profiles_directory(tmp_path, monkeypatch):
    """If there is no profiles directory, migration is a no-op."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "worker-a"))
    pa._db_init_path = None

    report = pa.migrate_profile_agora_dbs()
    assert report["ok"] is True
    assert report["migrated"] == []
    assert report["skipped"] == []
    assert "no profiles directory" in report["note"].lower()


def test_migrate_profile_dbs_skips_empty_file(tmp_path, monkeypatch):
    """Empty (0-byte) per-profile agora.db files are ignored."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "empty-profile"
    profile_home.mkdir(parents=True)
    (profile_home / "agora.db").write_bytes(b"")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    report = pa.migrate_profile_agora_dbs()
    assert report["ok"] is True
    assert report["migrated"] == []
    assert len(report["skipped"]) == 1
    assert report["skipped"][0]["reason"] == "empty-file"
    assert not (root / "agora.db").exists()


def test_migrate_profile_dbs_preserves_existing_shared_data(tmp_path, monkeypatch):
    """Migration appends to the shared DB without overwriting existing rows."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "worker-a"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    # Pre-populate shared DB with a message of our own.
    with pa._connect() as conn:
        conn.execute(
            "INSERT INTO agora_messages "
            "(channel_id, thread_id, author_type, author_profile, body, linked_task_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, None, "agent", "shared-bot", "shared message", None, int(time.time())),
        )
        conn.commit()

    _seed_legacy_profile_db(
        pa, profile_home / "agora.db", [("praca", "legacy hello", "agent", "worker-a")]
    )

    pa.migrate_profile_agora_dbs()

    conn = sqlite3.connect(str(root / "agora.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT body FROM agora_messages ORDER BY body"
        ).fetchall()
        assert [r["body"] for r in rows] == ["legacy hello", "shared message"]
    finally:
        conn.close()


def test_migrate_profile_dbs_maps_channels_by_slug(tmp_path, monkeypatch):
    """Channels that exist in both source and target are merged by slug."""
    import plugins.agora.dashboard.plugin_api as pa

    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "worker-a"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    pa._db_init_path = None

    # Shared DB already has default #praca.
    _seed_legacy_profile_db(
        pa, profile_home / "agora.db", [("praca", "in praca", "agent", "worker-a")]
    )

    pa.migrate_profile_agora_dbs()

    conn = sqlite3.connect(str(root / "agora.db"))
    conn.row_factory = sqlite3.Row
    try:
        # There should only be one #praca channel.
        rows = conn.execute("SELECT id FROM agora_channels WHERE slug = ?", ("praca",)).fetchall()
        assert len(rows) == 1
        # Message should be in that single channel.
        msg = conn.execute(
            "SELECT channel_id FROM agora_messages WHERE author_profile = ?", ("worker-a",)
        ).fetchone()
        assert msg["channel_id"] == rows[0]["id"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Agent terminal
# ---------------------------------------------------------------------------


def test_open_terminal_rejects_when_tmux_missing(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    # Simulate tmux not installed, but terminals "available".
    monkeypatch.setattr(pa.shutil, "which", lambda binary: None)
    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal")
    assert r.status_code == 400
    data = r.json()["detail"]
    assert data["ok"] is False
    assert data["reason"] == "tmux-not-found"


def test_open_terminal_creates_tmux_for_unknown_profile_with_fallback_skills(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    created_sessions = []

    monkeypatch.setattr(pa, "_tmux_has_session", lambda session: False)
    monkeypatch.setattr(pa, "_tmux_session_has_clients", lambda session: False)
    monkeypatch.setattr(
        pa.shutil, "which", lambda binary: f"/usr/bin/{binary}" if binary == "ptyxis" else "/usr/bin/tmux"
    )

    def fake_run(args, **kwargs):
        if args[:3] == ["tmux", "new-session", "-d"]:
            created_sessions.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "unexpected")

    monkeypatch.setattr(pa.subprocess, "run", fake_run)
    monkeypatch.setattr(
        pa.subprocess,
        "Popen",
        lambda args, **kwargs: None,
    )

    r = client.post("/api/plugins/agora/agents/stranger/open-terminal")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ok"] is True
    assert payload["terminal"]["opened"] is True
    assert len(created_sessions) == 1
    assert "--skills hermes-agent" in " ".join(created_sessions[0])


def test_open_terminal_creates_session_and_opens_terminal(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    created_sessions = []
    opened_terminals = []

    monkeypatch.setattr(pa, "_tmux_has_session", lambda session: False)
    monkeypatch.setattr(pa, "_tmux_session_has_clients", lambda session: False)
    monkeypatch.setattr(
        pa.shutil, "which", lambda binary: f"/usr/bin/{binary}" if binary == "ptyxis" else "/usr/bin/tmux"
    )

    def fake_run(args, **kwargs):
        if args[:3] == ["tmux", "new-session", "-d"]:
            created_sessions.append(args)
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "unexpected")

    monkeypatch.setattr(pa.subprocess, "run", fake_run)
    monkeypatch.setattr(
        pa.subprocess,
        "Popen",
        lambda args, **kwargs: opened_terminals.append(args) or None,
    )

    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ok"] is True
    assert payload["terminal"]["opened"] is True
    assert payload["terminal"]["command"] == "ptyxis"
    assert len(created_sessions) == 1
    assert created_sessions[0][4] == "agent-frontend"
    assert len(opened_terminals) == 1
    assert opened_terminals[0][0] == "ptyxis"


def test_open_terminal_focuses_existing_client(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(pa, "_tmux_has_session", lambda session: True)
    monkeypatch.setattr(pa, "_tmux_session_has_clients", lambda session: True)
    monkeypatch.setattr(pa.shutil, "which", lambda binary: f"/usr/bin/{binary}")

    opened_terminals = []
    monkeypatch.setattr(
        pa.subprocess, "Popen", lambda args, **kwargs: opened_terminals.append(args) or None
    )

    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal")
    assert r.status_code == 200, r.text
    terminal = r.json()["terminal"]
    assert terminal["opened"] is False
    assert terminal["focused"] is True
    assert terminal["command"] is None
    assert opened_terminals == []


def test_open_terminal_503_when_no_terminal_available(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]
    monkeypatch.setattr(pa, "_tmux_has_session", lambda session: True)
    monkeypatch.setattr(pa, "_tmux_session_has_clients", lambda session: False)
    # tmux is installed, but none of the supported terminal emulators are.
    monkeypatch.setattr(
        pa.shutil, "which", lambda binary: "/usr/bin/tmux" if binary == "tmux" else None
    )

    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal")
    assert r.status_code == 503
    data = r.json()["detail"]
    assert data["ok"] is True
    assert data["opened"] is False
    assert data["reason"] == "no-terminal"


def test_open_terminal_empty_profile_rejected(client):
    r = client.post("/api/plugins/agora/agents/%20/open-terminal")
    assert r.status_code == 400
    assert "profile is required" in r.json()["detail"]


def test_open_terminal_defaults_to_profile_session_even_with_active_worker(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    monkeypatch.setattr(
        pa,
        "_kanban_active_worker",
        lambda profile: {
            "task_id": "t_active",
            "run_id": 123,
            "worker_pid": 9999,
        },
    )

    opened = []

    def _fake_open(profile, session=None):
        opened.append(session)
        return {
            "ok": True,
            "profile": profile,
            "session": session or profile,
            "opened": True,
            "focused": False,
            "command": "ptyxis",
        }

    monkeypatch.setattr(pa, "_open_agent_tmux_terminal", _fake_open)

    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ok"] is True
    assert payload["telemetry"]["mode"] == "profile-session"
    assert opened == [None]


def test_open_terminal_kanban_tail_target_prefers_worker_session(client, monkeypatch):
    pa = sys.modules["hermes_dashboard_plugin_agora_test"]

    monkeypatch.setattr(
        pa,
        "_kanban_active_worker",
        lambda profile: {
            "task_id": "t_active",
            "run_id": 777,
            "worker_pid": 4321,
        },
    )
    monkeypatch.setattr(
        pa,
        "_ensure_worker_telemetry_session",
        lambda profile, worker: {
            "ok": True,
            "session": "agent-frontend--w-t_active",
            "task_id": worker["task_id"],
            "run_id": worker["run_id"],
        },
    )

    opened = []

    def _fake_open(profile, session=None):
        opened.append(session)
        return {
            "ok": True,
            "profile": profile,
            "session": session,
            "opened": True,
            "focused": False,
            "command": "ptyxis",
        }

    monkeypatch.setattr(pa, "_open_agent_tmux_terminal", _fake_open)

    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal?target=kanban-tail")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ok"] is True
    assert payload["telemetry"]["mode"] == "kanban-tail"
    assert payload["telemetry"]["task_id"] == "t_active"
    assert opened == ["agent-frontend--w-t_active"]


def test_open_terminal_invalid_target_rejected(client):
    r = client.post("/api/plugins/agora/agents/agent-frontend/open-terminal?target=bogus")
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["reason"] == "invalid-target"
    assert set(detail["allowed"]) == {"profile-session", "kanban-tail", "auto"}

