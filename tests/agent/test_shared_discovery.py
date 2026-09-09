from contextvars import ContextVar
import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from agent.shared_discovery import (
    SharedDiscoveryScope,
    build_local_discovery_scope,
    discover_shared_references,
)
from agent.worker_interfaces import (
    InterfaceSelection, bind_worker_interface, normalize_worker_call,
    project_worker_tool_definitions,
)
from agent.worker_store import WorkerStore
from gateway import hosted_rooms
from hermes_state import SessionDB
from tui_gateway.session_discovery import build_gateway_discovery_scope


def _parent(db, scope, *, tools=()):
    return SimpleNamespace(
        session_id="owner", _session_db=db, _shared_discovery_scope=scope,
        _executable_tool_names=set(tools), _session_title_hint="",
    )


def test_absent_worker_schema_discovery_does_not_initialize_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = SessionDB(tmp_path / "state.db")
    scope = build_local_discovery_scope()
    parent = _parent(db, scope)
    with db._read_ctx() as conn:
        before = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'orchestration_%' ORDER BY name"
        )]

    result = discover_shared_references(parent)

    with db._read_ctx() as conn:
        after = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'orchestration_%' ORDER BY name"
        )]
    assert result["references"] == []
    assert before == after == []
    db.close()


def test_worker_and_run_references_keep_owner_and_style_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = SessionDB(tmp_path / "state.db")
    store = WorkerStore(db)
    store.ensure_schema()
    worker = store.create_worker("owner", profile="research")
    run = store.enqueue_run(worker["worker_id"], "owner", goal="synthetic")
    foreign = store.create_worker("other", profile="private")
    parent = _parent(db, build_local_discovery_scope())

    refs = discover_shared_references(parent)["references"]
    names = {item["reference"] for item in refs}
    assert names == {f"worker:{worker['worker_id']}", f"run:{run['run_id']}"}
    run_ref = next(item for item in refs if item["kind"] == "run")
    assert run_ref["worker"] == f"worker:{worker['worker_id']}"
    assert run_ref["scope"] == {"kind": "worker", "reference": f"worker:{worker['worker_id']}"}
    resolved_run = discover_shared_references(parent, f"run:{run['run_id']}")
    assert resolved_run["reference_detail"]["status"] == "PENDING"
    assert resolved_run["worker"]["reference"] == f"worker:{worker['worker_id']}"
    resolved_worker = discover_shared_references(parent, f"worker:{worker['worker_id']}")
    assert resolved_worker["runs"][0]["reference"] == f"run:{run['run_id']}"
    with pytest.raises(PermissionError, match="Unknown or unavailable"):
        discover_shared_references(parent, f"worker:{foreign['worker_id']}")

    from tools import delegate_tool
    import run_agent
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    dispatched = json.loads(run_agent.AIAgent._dispatch_delegate_task(
        parent, {"action": "discover", "reference": f"run:{run['run_id']}"},
    ))
    assert dispatched["reference_detail"]["reference"] == f"run:{run['run_id']}"

    for style, tool in (("codex", "worker_capabilities"), ("claude", "TaskCapabilities")):
        selection = bind_worker_interface(
            InterfaceSelection(style, "explicit", "experimental_unqualified", "fixture", "fixture"),
            [{"type": "function", "function": {"name": "delegate_task", "parameters": {}}}],
        )
        call = normalize_worker_call(selection, tool, {"reference": f"run:{run['run_id']}"})
        assert call.arguments == {"action": "discover", "reference": f"run:{run['run_id']}"}
        projected = project_worker_tool_definitions(
            [{"type": "function", "function": {"name": "delegate_task", "parameters": {}}}], selection,
        )
        capability = next(item for item in projected if item["function"]["name"] == tool)
        assert "reference" in capability["function"]["parameters"]["properties"]
    db.close()


def test_bot_and_task_references_revalidate_native_gates_and_pinned_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    teammate = home / "profiles" / "research"
    teammate.mkdir(parents=True)
    home.mkdir(exist_ok=True)
    (home / "profile.yaml").write_text("ui_meta:\n  hermes-bots: {}\n", encoding="utf-8")
    (teammate / "profile.yaml").write_text("description: Research\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    board_db = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    conn = connect(board_db)
    task_id = kb.create_task(conn, title="Synthetic task")
    conn.close()
    session_db = SimpleNamespace(db_path=home / "state.db")
    parent = _parent(session_db, build_local_discovery_scope(), tools={"kanban_list", "kanban_show"})
    parent._session_title_hint = "Bot Chat"

    refs = discover_shared_references(parent)["references"]
    names = {item["reference"] for item in refs}
    assert "bot:research" in names
    assert f"task:{task_id}" in names
    bot = discover_shared_references(parent, "bot:research")["reference_detail"]
    task = discover_shared_references(parent, f"task:{task_id}")["reference_detail"]
    assert bot["actions"] == ["message"]
    assert bot["scope"] == {"kind": "profile_roster", "profile": "default"}
    assert task["status"] == "ready"
    assert task["scope"] == {"kind": "kanban_board", "board": "default"}

    parent._session_title_hint = "Other"
    with pytest.raises(PermissionError, match="Unknown or unavailable"):
        discover_shared_references(parent, "bot:research")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "other.db"))
    with pytest.raises(PermissionError, match="Unknown or unavailable"):
        discover_shared_references(parent, f"task:{task_id}")


def test_room_grant_rechecks_session_policy_service_authority_and_participants(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    room_db = tmp_path / "rooms.db"
    hosted_rooms.create_room(
        room_db, room_id="room-synthetic", name="Synthetic room",
        members=[{"profile": "research", "handle": "research"}],
        authority_gateway_id="gateway-synthetic",
    )
    current_gateway = {"id": "gateway-synthetic"}
    monkeypatch.setattr(
        hosted_rooms, "local_authority_gateway_id_existing", lambda: current_gateway["id"])
    service = SimpleNamespace(
        db_path=room_db,
        runtime=SimpleNamespace(status=lambda: {"running": True, "stopping": False}),
    )
    from tui_gateway import methods_groups
    monkeypatch.setattr(methods_groups, "_service", service)
    cfg = {"orchestration": {"discovery": {"rooms": [{
        "id": "room-synthetic", "actions": ["inspect"], "participants": ["research"],
    }]}}}
    runtime_record = ContextVar("test_discovery_record", default=None)
    server = SimpleNamespace(
        _sessions={}, _sessions_lock=threading.RLock(),
        _current_runtime_session_record=runtime_record,
        _current_profile_name=lambda: "default", _load_cfg=lambda: cfg,
    )
    scope = build_gateway_discovery_scope(server, sid="runtime-1", cfg=cfg, source="tui")
    agent = SimpleNamespace(_shared_discovery_scope=scope)
    record = {"agent": agent, "discovery_scope": scope, "source": "tui"}
    server._sessions["runtime-1"] = record
    token = runtime_record.set(record)
    try:
        detail = discover_shared_references(agent, "room:room-synthetic")["reference_detail"]
        assert detail["participants"] == ["research"]
        assert detail["authority_epoch"] == 1
        assert detail["scope"] == {"kind": "gateway_room_grant"}

        hosted_rooms.create_room(
            room_db, room_id="room-foreign", name="Foreign room", members=[],
            authority_gateway_id="gateway-synthetic",
        )
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            discover_shared_references(agent, "room:room-foreign")

        cfg["orchestration"]["discovery"]["rooms"][0]["participants"] = []
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            discover_shared_references(agent, "room:room-synthetic")
        cfg["orchestration"]["discovery"]["rooms"][0]["participants"] = ["research"]

        methods_groups._service = SimpleNamespace(
            db_path=room_db,
            runtime=SimpleNamespace(status=lambda: {"running": True, "stopping": False}),
        )
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            discover_shared_references(agent, "room:room-synthetic")
        methods_groups._service = service

        with sqlite3.connect(room_db) as conn:
            conn.execute("UPDATE hosted_rooms SET members_json='[]' WHERE room_id='room-synthetic'")
            conn.commit()
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            discover_shared_references(agent, "room:room-synthetic")
        with sqlite3.connect(room_db) as conn:
            conn.execute(
                "UPDATE hosted_rooms SET members_json=? WHERE room_id='room-synthetic'",
                (json.dumps([{"profile": "research", "handle": "research"}]),),
            )
            conn.commit()

        current_gateway["id"] = "gateway-away"
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            discover_shared_references(agent, "room:room-synthetic")
        current_gateway["id"] = "gateway-synthetic"
        assert discover_shared_references(
            agent, "room:room-synthetic")["reference_detail"]["availability"] == "available"

        replacement_agent = SimpleNamespace()
        replacement_scope = build_gateway_discovery_scope(
            server, sid="runtime-1", cfg=cfg, source="tui")
        replacement_agent._shared_discovery_scope = replacement_scope
        replacement_record = {
            "agent": replacement_agent, "discovery_scope": replacement_scope, "source": "tui",
        }
        server._sessions["runtime-1"] = replacement_record
        replacement_token = runtime_record.set(replacement_record)
        try:
            with pytest.raises(PermissionError, match="Unknown or unavailable"):
                discover_shared_references(agent, "room:room-synthetic")
        finally:
            runtime_record.reset(replacement_token)
            server._sessions["runtime-1"] = record

        with sqlite3.connect(room_db) as conn:
            conn.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='room-synthetic'")
            conn.commit()
        with pytest.raises(PermissionError, match="Unknown or unavailable"):
            discover_shared_references(agent, "room:room-synthetic")
    finally:
        runtime_record.reset(token)

    # SID reuse or a compute-host record cannot inherit the prior grant.
    server._sessions["runtime-1"] = {"agent": agent, "discovery_scope": scope, "source": "tui"}
    with pytest.raises(PermissionError, match="Unknown or unavailable"):
        discover_shared_references(agent, "room:room-synthetic")
    server._sessions["compute"] = {"_compute_host_active": True, "source": "tui"}
    compute = build_gateway_discovery_scope(server, sid="compute", cfg=cfg, source="tui")
    assert compute.room_provider is None


def test_room_identity_reads_never_mint_or_repair_install_identity(tmp_path, monkeypatch):
    room_db = tmp_path / "rooms.db"
    authority = "install:" + "a" * 32
    hosted_rooms.create_room(
        room_db, room_id="room-synthetic", name="Synthetic room", members=[],
        authority_gateway_id=authority,
    )
    service = SimpleNamespace(
        db_path=room_db,
        runtime=SimpleNamespace(status=lambda: {"running": True, "stopping": False}),
    )
    from tui_gateway import methods_groups
    monkeypatch.setattr(methods_groups, "_service", service)
    assert methods_groups.get_hosted_room_service() is service
    from hermes_cli import install_identity
    original_read = install_identity.read_existing_install_id
    identity_reads = []

    def observed_identity_read(root=None):
        identity_reads.append(root)
        return original_read(root)

    monkeypatch.setattr(install_identity, "read_existing_install_id", observed_identity_read)
    cfg = {"orchestration": {"discovery": {"rooms": [{
        "id": "room-synthetic", "actions": ["inspect"], "participants": [],
    }]}}}
    server = SimpleNamespace(_sessions={}, _sessions_lock=threading.RLock())

    missing_home = tmp_path / "missing-home"
    monkeypatch.setenv("HERMES_HOME", str(missing_home))
    assert build_gateway_discovery_scope(
        server, sid="missing", cfg=cfg, source="tui").room_provider is None
    assert not missing_home.exists()

    malformed_home = tmp_path / "malformed-home"
    malformed_home.mkdir()
    malformed_identity = malformed_home / "install_id"
    malformed_identity.write_text("not-an-install-id\n", encoding="utf-8")
    malformed_before = (set(malformed_home.iterdir()), malformed_identity.read_bytes())
    monkeypatch.setenv("HERMES_HOME", str(malformed_home))
    assert build_gateway_discovery_scope(
        server, sid="malformed", cfg=cfg, source="tui").room_provider is None
    assert (set(malformed_home.iterdir()), malformed_identity.read_bytes()) == malformed_before

    unreadable_home = tmp_path / "unreadable-home"
    unreadable_home.mkdir()
    (unreadable_home / "install_id").mkdir()
    unreadable_before = set(unreadable_home.iterdir())
    monkeypatch.setenv("HERMES_HOME", str(unreadable_home))
    assert build_gateway_discovery_scope(
        server, sid="unreadable", cfg=cfg, source="tui").room_provider is None
    assert set(unreadable_home.iterdir()) == unreadable_before
    assert len(identity_reads) == 3


def test_existing_readonly_helpers_leave_missing_paths_absent(tmp_path):
    from hermes_cli.kanban_db_connect import connect_existing_readonly

    room_path = tmp_path / "missing-room.db"
    task_path = tmp_path / "missing-task.db"
    assert hosted_rooms.list_rooms_existing(room_path) == []
    assert hosted_rooms.room_state_existing(room_path, room_id="missing") is None
    assert connect_existing_readonly(task_path) is None
    assert not room_path.exists()
    assert not task_path.exists()
