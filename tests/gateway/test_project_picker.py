"""Contracts for the Telegram registered-project picker and its fresh-session boundary."""

import pytest

from gateway.project_picker import (
    list_projects_for_profile,
    picker_page,
    validate_workspace,
)


def _profile_projects(home, rows):
    from hermes_cli import projects_db

    home.mkdir(parents=True, exist_ok=True)
    with projects_db.connect_closing(home / "projects.db") as conn:
        ids = []
        for name, slug, folders in rows:
            pid = projects_db.create_project(
                conn, name=name, slug=slug, folders=folders
            )
            ids.append(pid)
        return ids


def test_registry_is_profile_scoped_and_picker_disambiguates_pages(tmp_path):
    home_a, home_b = tmp_path / "profile-a", tmp_path / "profile-b"
    rows = []
    for i in range(11):
        folder = tmp_path / f"workspace-{i}"
        folder.mkdir()
        rows.append(
            ("Shared" if i < 2 else f"Project {i}", f"project-{i}", [str(folder)])
        )
    ids = _profile_projects(home_a, rows)
    other = tmp_path / "other-workspace"
    other.mkdir()
    _profile_projects(home_b, [("Other profile", "other-profile", [str(other)])])
    rows_a = list_projects_for_profile(home_a)
    rows_b = list_projects_for_profile(home_b)
    assert {p["id"] for p in rows_a} == set(ids)
    assert [p["name"] for p in rows_b] == ["Other profile"]
    state = {"projects": rows_a, "folders": []}
    title, choices, controls = picker_page(state, 0)
    assert title == "Choose a Hermes Desktop project"
    assert len(choices) == 8
    assert all(
        len(f"pj:{'abcdefgh1234'}:{item['action']}".encode()) <= 64 for item in choices
    )
    next_control = next(item for item in controls if item["action"].startswith("page_"))
    title, second, _ = picker_page(state, int(next_control["action"].split("_")[1]))
    assert title and len(second) == 3
    shared = [
        item["label"] for item in choices + second if item["label"].startswith("Shared")
    ]
    assert len(shared) == 2 and len(set(shared)) == 2
    assert len({item["action"] for item in choices + second}) == 11


def test_multi_folder_picker_has_back_cancel_and_folder_paths_stay_opaque():
    state = {
        "projects": [],
        "folders": [
            {"path": "/home/a", "label": "Source"},
            {"path": "/home/b", "label": "Docs"},
        ],
    }
    title, choices, controls = picker_page(state, 0, folder=True)
    assert title == "Choose a workspace for this project"
    assert [item["label"] for item in choices] == ["Source", "Docs"]
    assert all("/home" not in item["action"] for item in choices + controls)
    assert {item["action"] for item in controls} >= {"back", "cancel"}


def test_workspace_must_still_exist_and_be_a_directory(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    assert validate_workspace(str(workspace)) == str(workspace.resolve())
    with pytest.raises(ValueError):
        validate_workspace(str(workspace / "removed"))
    file_path = tmp_path / "not-a-folder"
    file_path.touch()
    with pytest.raises(ValueError):
        validate_workspace(str(file_path))


def test_session_project_reset_persists_cwd_and_desktop_groups_fresh_session(
    tmp_path, monkeypatch
):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore, build_session_context
    from hermes_state import SessionDB
    from tui_gateway.project_tree import build_tree

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    home = tmp_path / ".hermes"
    home.mkdir()
    sessions_dir = home / "sessions"
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat", user_id="user", chat_type="dm"
    )
    store = SessionStore(sessions_dir, GatewayConfig())
    db = SessionDB(home / "state.db")
    store._db = db
    old = store.get_or_create_session(source)
    db.create_session(
        old.session_id,
        "telegram",
        session_key=old.session_key,
        chat_id="chat",
        user_id="user",
    )
    db.append_message(old.session_id, role="user", content="old transcript survives")
    workspace = tmp_path / "selected"
    workspace.mkdir()
    project_id = "p_stable"
    new = store.reset_session_for_project(
        old.session_key,
        source,
        expected_session_id=old.session_id,
        cwd=str(workspace),
        project_id=project_id,
    )
    assert new is not None and new.session_id != old.session_id
    assert store.peek_session_id(old.session_key) == new.session_id
    assert new.metadata == {"project_id": project_id}
    from gateway.project_session import PROJECT_MARKER_KEY

    assert db.get_session_model_config_value(new.session_id, PROJECT_MARKER_KEY) == {
        "project_id": project_id, "cwd": str(workspace),
    }
    assert db.get_session(old.session_id)["cwd"] is None
    assert db.get_session(new.session_id)["cwd"] == str(workspace)
    assert db.get_messages(old.session_id)[0]["content"] == "old transcript survives"
    assert db.get_messages(new.session_id) == []
    assert build_session_context(source, GatewayConfig(), new).session_id == new.session_id
    project = {
        "id": project_id,
        "name": "Selected",
        "primary_path": str(workspace),
        "folders": [{"path": str(workspace), "label": "Selected", "is_primary": True}],
    }
    tree = build_tree([project], [{"id": new.session_id, "cwd": str(workspace)}], [])
    node = next(p for p in tree["projects"] if p["id"] == project_id)
    assert new.session_id in node["sessionIds"]
    db.close()


def test_project_route_compare_and_swap_rejects_a_stale_picker(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    db = SessionDB(home / "state.db")
    store._db = db
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm"
    )
    first = store.get_or_create_session(source)
    second = store.reset_session(first.session_key)
    assert second is not None
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    rejected = store.reset_session_for_project(
        first.session_key,
        source,
        expected_session_id=first.session_id,
        cwd=str(cwd),
        project_id="stale",
    )
    assert rejected is None
    assert store.peek_session_id(first.session_key) == second.session_id
    db.close()


def test_fresh_gateway_agent_receives_the_session_workspace_cwd(tmp_path):
    from types import SimpleNamespace

    from agent.runtime_cwd import reset_session_cwd, set_session_cwd
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_execution import GatewaySessionAgentMixin

    workspace = tmp_path / "agent-workspace"
    workspace.mkdir()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="user")
    captured = {}
    context = SimpleNamespace(
        source=source,
        user_config={},
        AIAgent=lambda **kwargs: captured.update(kwargs) or kwargs,
        enabled_toolsets=[],
        disabled_toolsets=[],
        session_id="session",
        session_key="route",
    )
    runner = SimpleNamespace(
        _prefill_messages=None,
        _service_tier=None,
        _session_db=None,
        _refresh_fallback_model=lambda: None,
    )
    mixin = SimpleNamespace(_ctx=context, _runner=runner)
    token = set_session_cwd(str(workspace))
    try:
        GatewaySessionAgentMixin._build_fresh_agent(
            mixin,
            {"model": "test-model", "runtime": {}},
            "telegram",
            None,
            4,
            None,
            {},
            False,
        )
    finally:
        reset_session_cwd(token)
    assert captured["cwd"] == str(workspace)


@pytest.mark.parametrize("failure_site", ["create", "routing_save"])
def test_project_session_creation_failure_keeps_the_previous_route_and_transcript(
    tmp_path,
    monkeypatch,
    failure_site,
):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = SessionStore(home / "sessions", GatewayConfig())
    db = SessionDB(home / "state.db")
    store._db = db
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm"
    )
    old = store.get_or_create_session(source)
    db.create_session(old.session_id, "telegram", session_key=old.session_key)
    db.append_message(old.session_id, role="user", content="keep this transcript")
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    if failure_site == "create":
        monkeypatch.setattr(
            db,
            "create_session",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
    else:
        monkeypatch.setattr(
            store,
            "_persist_routing_data",
            lambda *_a: (_ for _ in ()).throw(OSError("routing write failed")),
        )
    with pytest.raises(OSError):
        store.reset_session_for_project(
            old.session_key,
            source,
            expected_session_id=old.session_id,
            cwd=str(cwd),
            project_id="project",
        )
    assert store.peek_session_id(old.session_key) == old.session_id
    assert db.get_messages(old.session_id)[0]["content"] == "keep this transcript"
    assert db.session_count() == 1
    db.close()
