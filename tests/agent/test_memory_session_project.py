"""Only the destination session's verified workspace can select a memory project."""

from dataclasses import asdict

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _init_memory
from hermes_cli import projects_db
from persistence import SessionId
from persistence.authority.bindings import NewSessionBinding, local_session_id
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
from tests.agent.test_memory_migration_notifications import StartupAgent
from tests.agent.test_memory_session_switch import _RecordingProvider


def _start(monkeypatch, store, session_id, *, platform="cli", skip_memory=False):
    provider = _RecordingProvider()
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: provider)
    agent = StartupAgent(platform)
    agent.session_id = session_id
    # AIAgent uses the writer interface as its session store when given that handle alone.
    agent._session_db = agent.persistence = store
    agent.session_cwd = None
    for name in _GATEWAY_IDENTITY_PARAMS:
        setattr(agent, f"_{name}", None)
    _init_memory(agent, {"memory": {"provider": provider.name}}, skip_memory, platform)
    return agent._memory_manager, provider


def test_project_attribution_uses_destination_session_and_owner_profile(tmp_path, monkeypatch):
    home, other_home = tmp_path / "owner", tmp_path / "other"
    workspace, nested = tmp_path / "workspace", tmp_path / "workspace" / "nested"
    nested.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with projects_db.connect_closing() as registry:
        outer = projects_db.create_project(registry, name="Outer", folders=[str(workspace)])
        inner = projects_db.create_project(registry, name="Inner", folders=[str(nested)])
    with projects_db.connect_closing(other_home / "projects.db") as registry:
        projects_db.create_project(registry, name="Other profile", folders=[str(workspace)])

    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.create(NewSession(session_id=SessionId("outer"), source="cli", cwd=str(workspace)))
        store.sessions.create(NewSession(session_id=SessionId("inner"), source="cli", cwd=str(nested)))
        store.sessions.create(NewSession(session_id=SessionId("fresh"), source="cli"))
        manager, provider = _start(monkeypatch, store, "outer")
        assert provider.initialize_calls[0]["session_project_id"] == outer
        assert "_session_store" not in provider.initialize_calls[0]

        manager.on_session_switch("inner", parent_session_id="outer", reason="resume", session_project_id=outer)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] == inner
        manager.commit_session_boundary_async([], new_session_id="fresh", parent_session_id="inner")
        assert manager.flush_pending(timeout=5)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] is None
        manager.shutdown_all()

        restarted, provider = _start(monkeypatch, store, "inner")
        assert provider.initialize_calls[0]["session_project_id"] == inner
        restarted.shutdown_all()

        nonprimary, provider = _start(monkeypatch, store, "inner", platform="cron")
        assert provider.initialize_calls[0]["session_project_id"] is None
        nonprimary.shutdown_all()
        skipped, provider = _start(monkeypatch, store, "inner", skip_memory=True)
        assert skipped is None and not provider.initialize_calls

        with open_persistence_reader(home) as reader:
            assert reader.sessions.get(SessionId("outer")) is not None
            unscoped, provider = _start(monkeypatch, reader, "outer")
            assert provider.initialize_calls[0]["session_project_id"] is None
            unscoped.shutdown_all()

        monkeypatch.setenv("HERMES_HOME", str(other_home))
        mismatch, provider = _start(monkeypatch, store, "outer")
        assert provider.initialize_calls[0]["session_project_id"] is None
        mismatch.shutdown_all()


def test_canonical_workspace_comes_from_receipt_and_project_must_be_unique(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_policy import build_policy

    home, workspace, decoy = tmp_path / "owner", tmp_path / "workspace", tmp_path / "decoy"
    workspace.mkdir()
    decoy.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with projects_db.connect_closing() as registry:
        project = projects_db.create_project(registry, name="Named", folders=[str(workspace)])
        projects_db.create_project(registry, name="Decoy", folders=[str(decoy)])

    with open_persistence(profile_scope(home=home)) as store:
        epoch = store.runtime.begin_epoch(instance_id="owner")
        owner_home = store.scope.hermes_home_key
        sid = local_session_id(owner_home, "human", "request")
        source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id="human", chat_type="dm")
        entry = SessionEntry(f"agent:main:local:dm:{sid}", sid, _now(), _now(), origin=source, platform=Platform.LOCAL)
        policy = build_policy(
            {"source": "cli", "cwd": str(workspace), "row_cwd": str(decoy), "toolsets": []}, {},
        )
        store.bindings.reserve(epoch=epoch, owner=None, binding=NewSessionBinding(
            profile_id=owner_home, principal_id="human", request_id="request", session_id=sid,
            route=entry.session_key, entry=entry.to_dict(), policy=asdict(policy), created_source="cli",
        ))
        assert store.sessions.get(sid).cwd == str(decoy)
        manager, provider = _start(monkeypatch, store, sid)
        assert provider.initialize_calls[0]["session_project_id"] == project

        successor = SessionId("reset-segment")
        store.bindings.reset_target(
            epoch=epoch, parent_session_id=sid, entry={**entry.to_dict(), "session_id": successor},
        )
        manager.on_session_switch(successor, parent_session_id=sid, reset=True)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] == project

        unrelated = SessionId("unbound-segment")
        store.sessions.create(NewSession(
            session_id=unrelated, source="cli", cwd=str(decoy), session_key=entry.session_key, chat_id=sid,
        ))
        manager.on_session_switch(unrelated)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] is None

        with projects_db.connect_closing() as registry:
            projects_db.create_project(
                registry, name="Duplicate", folders=[str(workspace)], allow_duplicate_path=True,
            )
        manager.on_session_switch(successor, rewound=True)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] is None
        manager.shutdown_all()
