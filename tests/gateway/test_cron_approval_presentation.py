"""Cron approval prompts must reach the presenter through the real owner execution path."""

import contextlib
import json
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_owner_agent_presents_its_original_pending_invocation(tmp_path, monkeypatch, decision):
    from cron import scheduler
    from gateway import session_cron
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionRef
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = PluginManager()
    service = PluginContext(PluginManifest(name="guard", version="0.1.0"), manager).cron_approvals
    presented, retired = threading.Event(), threading.Event()
    requests, dispatches, audit = [], [], []
    deadline = time.monotonic() + 30

    @contextlib.contextmanager
    def present(request, lifetime):
        assert str(get_hermes_home().resolve()) == request.profile_home
        lifetime.check()
        requests.append(request)
        audit.append(decision)
        presented.set()
        try:
            yield
        finally:
            retired.set()

    service.register_presenter(present)

    class Agent:
        session_id = "owner-session"

        def run_conversation(self, prompt, *, task_id):
            with service.pending(session_id=self.session_id, external_session_id="audit-session",
                                 call_id="call-one", deadline=deadline, check=lambda: None):
                assert presented.wait(2), "Owner cron invocation never reached its gateway presenter"
                if audit == ["approve"]:
                    dispatches.append("original-call")
            return {"final_response": decision, "completed": True}

        def close(self):
            pass

    agent = Agent()
    monkeypatch.setattr(scheduler, "_prepare_job_prompt", lambda *args: (None, "prompt"))
    monkeypatch.setattr(scheduler, "_load_cron_job_config", lambda *args: SimpleNamespace(cfg={}, model="test"))
    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", lambda *args: SimpleNamespace(
        blocked=None, model="test", fallback_notice=None))
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda job: None)
    monkeypatch.setattr(scheduler, "_construct_cron_agent", lambda *args, **kwargs: agent)

    db = SessionDB(tmp_path / "state.db")
    db.create_session("owner-session", source="cron")
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path), instance_id="test", db=db,
                                 epoch=begin_runtime_epoch(db, instance_id="test"))
    authority.sessions["owner-session"] = LiveSession(SimpleNamespace(platform=None, user_id="cron-owner"), "route")
    try:
        admit_session_input(db, epoch=authority.epoch, principal_id="cron-owner", session_id="owner-session",
                            request_id="cron:daily:fire", payload={"text": ""})
        row = claim_session_input(db, epoch=authority.epoch, session_id="owner-session")
        policy = SimpleNamespace(request_json=json.dumps({
            "extra_prompt": None, "request_id": "cron:daily:fire", "cron_job": {"id": "daily", "prompt": "prompt"}}))
        result = await session_cron.execute(authority, SessionRef(str(tmp_path), "owner-session"), row, policy)
        assert result == decision
        assert retired.wait(2)
        assert dispatches == (["original-call"] if decision == "approve" else [])
        assert len(requests) == 1
        request = requests[0]
        assert (request.profile_home, request.job_id, request.execution_id, request.session_id,
                request.external_session_id, request.call_id, request.deadline) == (
            str(tmp_path), "daily", "fire", "owner-session", "audit-session", "call-one", deadline)
    finally:
        service.close()
        db.close()
