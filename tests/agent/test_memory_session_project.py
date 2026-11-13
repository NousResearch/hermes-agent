"""Only the target session's verified workspace can select a memory project."""
from contextlib import closing
from dataclasses import asdict

from agent.agent_init import _GATEWAY_IDENTITY_PARAMS, _init_memory
from hermes_cli import projects_db
from hermes_state import SessionDB
from tests.agent.test_memory_migration_notifications import StartupAgent
from tests.agent.test_memory_session_switch import _RecordingProvider


def _start(monkeypatch, db, session_id, *, platform="cli", skip_memory=False):
    provider = _RecordingProvider()
    monkeypatch.setattr("plugins.memory.load_memory_provider", lambda name: provider)
    agent = StartupAgent(platform)
    agent.session_id = session_id
    agent._session_db = db
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

    with closing(SessionDB(home / "state.db")) as db:
        db.create_session("outer", source="cli", cwd=str(workspace))
        db.create_session("inner", source="cli", cwd=str(nested))
        db.create_session("fresh", source="cli")
        manager, provider = _start(monkeypatch, db, "outer")
        assert provider.initialize_calls[0]["session_project_id"] == outer
        assert "_session_db" not in provider.initialize_calls[0]

        manager.on_session_switch("inner", parent_session_id="outer", reason="resume")
        assert provider.switch_calls[-1]["extra"]["session_project_id"] == inner
        manager.commit_session_boundary_async([], new_session_id="fresh", parent_session_id="inner")
        assert manager.flush_pending(timeout=5)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] is None
        manager.shutdown_all()

        restarted, provider = _start(monkeypatch, db, "inner")
        assert provider.initialize_calls[0]["session_project_id"] == inner
        restarted.shutdown_all()

        nonprimary, provider = _start(monkeypatch, db, "inner", platform="cron")
        assert provider.initialize_calls[0]["session_project_id"] is None
        nonprimary.shutdown_all()
        skipped, provider = _start(monkeypatch, db, "inner", skip_memory=True)
        assert skipped is None and not provider.initialize_calls

        monkeypatch.setenv("HERMES_HOME", str(other_home))
        mismatch, provider = _start(monkeypatch, db, "outer")
        assert provider.initialize_calls[0]["session_project_id"] is None
        mismatch.shutdown_all()


def test_canonical_workspace_comes_from_receipt_and_project_must_be_unique(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_policy import build_policy
    from hermes_state_keys import local_identity
    from hermes_state_local import commit_local_session
    from hermes_state_runtime import begin_runtime_epoch

    home, workspace, decoy = tmp_path / "owner", tmp_path / "workspace", tmp_path / "decoy"
    workspace.mkdir()
    decoy.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with projects_db.connect_closing() as registry:
        project = projects_db.create_project(registry, name="Named", folders=[str(workspace)])
        projects_db.create_project(registry, name="Decoy", folders=[str(decoy)])
    with closing(SessionDB(home / "state.db")) as db:
        epoch = begin_runtime_epoch(db, instance_id="owner")
        sid = local_identity(str(home), "human", "request")
        source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id="human", chat_type="dm")
        entry = SessionEntry(f"agent:main:local:dm:{sid}", sid, _now(), _now(), origin=source, platform=Platform.LOCAL)
        policy = build_policy(
            {"source": "cli", "cwd": str(workspace), "row_cwd": str(decoy), "toolsets": []}, {},
        )
        commit_local_session(db, epoch=epoch, receipt={
            "profile_id": str(home), "principal_id": "human", "request_id": "request",
            "session_id": sid, "route": entry.session_key, "entry": entry.to_dict(), "policy": asdict(policy),
        })
        manager, provider = _start(monkeypatch, db, sid)
        assert provider.initialize_calls[0]["session_project_id"] == project

        manager.on_session_switch(sid, rewound=True)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] == project

        with projects_db.connect_closing() as registry:
            projects_db.create_project(
                registry, name="Duplicate", folders=[str(workspace)], allow_duplicate_path=True,
            )
        manager.on_session_switch(sid, rewound=True)
        assert provider.switch_calls[-1]["extra"]["session_project_id"] is None
        manager.shutdown_all()
