"""Tests for the durable profile-agent orchestration control plane."""

from __future__ import annotations

import json
from pathlib import Path

from agent.orchestration import AgentControlStore, AgentController
from tools.agent_control_tool import (
    AGENT_FORK_SCHEMA,
    AGENT_PROMPT_SCHEMA,
    AGENT_START_SCHEMA,
    _ADMIN_APPROVAL_POLICY_ENV,
    _approval_policy_from_env,
)


class FakeACPClient:
    prompts: list[tuple[str, str]] = []
    loaded_sessions: list[str] = []
    sessions_created = 0

    def __init__(self, profile: str, cwd: str, approval_policy: str = "deny"):
        self.profile = profile
        self.cwd = cwd
        self.approval_policy = approval_policy
        self.closed = False

    def connect(self):
        return "fake-auth"

    def new_session(self, cwd=None):
        type(self).sessions_created += 1
        return f"sid-{self.profile}-{type(self).sessions_created}"

    def resume_session(self, session_id, cwd=None):
        return session_id

    def load_session(self, session_id, cwd=None):
        self.loaded_sessions.append(session_id)
        return session_id

    def fork_session(self, session_id, cwd=None):
        return f"{session_id}-fork"

    def prompt(self, session_id, text, timeout=600.0):
        self.prompts.append((session_id, text))
        return {
            "stop_reason": "end_turn",
            "text": f"{self.profile}: {text}",
            "usage": {"totalTokens": 3},
        }

    def cancel(self, session_id):
        return None

    def close(self):
        self.closed = True


def _controller(tmp_path: Path) -> AgentController:
    FakeACPClient.prompts = []
    FakeACPClient.loaded_sessions = []
    FakeACPClient.sessions_created = 0
    store = AgentControlStore(tmp_path / "agent-control.db")
    return AgentController(store=store, client_factory=FakeACPClient)


def test_start_agent_creates_persistent_handle(tmp_path):
    controller = _controller(tmp_path)

    result = controller.start_agent(profile="default", cwd=str(tmp_path))

    assert result["ok"] is True
    agent = result["agent"]
    assert agent["profile"] == "default"
    assert agent["session_id"].startswith("sid-default")
    assert agent["status"] == "idle"


def test_start_agent_rejects_invalid_profile_name(tmp_path):
    controller = _controller(tmp_path)

    result = controller.start_agent(profile="../bad", cwd=str(tmp_path))

    assert result["ok"] is False
    assert "Invalid profile name" in result["error"]


def test_start_agent_rejects_missing_cwd(tmp_path):
    controller = _controller(tmp_path)

    result = controller.start_agent(profile="default", cwd=str(tmp_path / "missing"))

    assert result["ok"] is False
    assert "Working directory does not exist" in result["error"]


def test_start_agent_surfaces_client_start_errors(tmp_path):
    class BrokenClient(FakeACPClient):
        def connect(self):
            raise RuntimeError("acp unavailable")

    store = AgentControlStore(tmp_path / "agent-control.db")
    controller = AgentController(store=store, client_factory=BrokenClient)

    result = controller.start_agent(profile="default", cwd=str(tmp_path))

    assert result["ok"] is False
    assert "acp unavailable" in result["error"]


def test_start_agent_loads_existing_session_strictly(tmp_path):
    controller = _controller(tmp_path)

    result = controller.start_agent(
        profile="default",
        cwd=str(tmp_path),
        session_id="existing-session",
    )

    assert result["ok"] is True
    assert result["agent"]["session_id"] == "existing-session"
    assert FakeACPClient.loaded_sessions == ["existing-session"]


def test_start_agent_reuses_idempotency_key(tmp_path):
    controller = _controller(tmp_path)

    first = controller.start_agent(
        profile="default",
        cwd=str(tmp_path),
        idempotency_key="team:researcher",
    )
    second = controller.start_agent(
        profile="default",
        cwd=str(tmp_path),
        idempotency_key="team:researcher",
    )

    assert first["agent"]["id"] == second["agent"]["id"]
    assert second["reused"] is True


def test_prompt_agent_serializes_and_persists_result(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent_id = started["agent"]["id"]

    result = controller.prompt_agent(agent_id=agent_id, prompt="Summarize repo")

    assert result["ok"] is True
    assert result["response"] == "default: Summarize repo"
    assert FakeACPClient.prompts == [(started["agent"]["session_id"], "Summarize repo")]

    status = controller.status(agent_id=agent_id)
    assert status["agent"]["status"] == "idle"
    assert status["last_run"]["status"] == "completed"
    assert status["last_run"]["response"] == "default: Summarize repo"
    assert status["last_run"]["usage"] == {"totalTokens": 3}


def test_expired_lease_is_marked_error_on_status(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent_id = started["agent"]["id"]
    acquired = controller.store.acquire_handle_lease(
        agent_id,
        owner="stale-worker",
        ttl_seconds=-1,
    )
    assert acquired is True

    status = controller.status(agent_id=agent_id)

    assert status["agent"]["status"] == "error"
    assert status["agent"]["lease_owner"] is None
    assert status["agent"]["last_error"] == "agent_control lease expired"


def test_expired_lease_marks_running_run_error(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent = started["agent"]
    run = controller.store.create_run(
        handle_id=agent["id"],
        profile=agent["profile"],
        session_id=agent["session_id"],
        prompt="work",
    )
    acquired = controller.store.acquire_handle_lease(
        agent["id"],
        owner="stale-worker",
        ttl_seconds=-1,
    )
    assert acquired is True

    status = controller.status(agent_id=agent["id"])

    assert status["last_run"]["id"] == run["id"]
    assert status["last_run"]["status"] == "error"
    assert status["last_run"]["error"] == "agent_control lease expired"


def test_prompt_agent_returns_busy_without_creating_run_when_lease_held(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent_id = started["agent"]["id"]
    acquired = controller.store.acquire_handle_lease(
        agent_id,
        owner="other-worker",
        ttl_seconds=60,
    )
    assert acquired is True

    result = controller.prompt_agent(
        agent_id=agent_id,
        prompt="Do work",
        lease_wait_seconds=0,
    )

    assert result["ok"] is False
    assert result["run_id"] is None
    assert "busy" in result["error"]
    assert controller.store.last_run_for_handle(agent_id) is None


def test_session_lease_blocks_duplicate_handles_for_same_profile_session(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent = started["agent"]
    duplicate = controller.store.create_handle(
        profile=agent["profile"],
        session_id=agent["session_id"],
        cwd=agent["cwd"],
    )

    with controller.store.leased_session(
        agent["id"],
        profile=agent["profile"],
        session_id=agent["session_id"],
        owner="worker-a",
        ttl_seconds=60,
    ):
        result = controller.prompt_agent(
            agent_id=duplicate["id"],
            prompt="Do work",
            lease_wait_seconds=0,
        )

    assert result["ok"] is False
    assert result["run_id"] is None
    assert "agent session" in result["error"]
    assert "busy" in result["error"]
    assert controller.store.last_run_for_handle(duplicate["id"]) is None


def test_prompt_agent_rejects_missing_handle_cwd_before_run(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent_id = started["agent"]["id"]
    missing = tmp_path / "missing"
    controller.store.update_handle(agent_id, cwd=str(missing))

    result = controller.prompt_agent(agent_id=agent_id, prompt="Do work")

    assert result["ok"] is False
    assert "Working directory does not exist" in result["error"]
    assert controller.store.last_run_for_handle(agent_id) is None


def test_fork_agent_creates_new_handle_with_forked_session(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))

    forked = controller.fork_agent(agent_id=started["agent"]["id"])

    assert forked["ok"] is True
    assert forked["forked_from"] == started["agent"]["id"]
    assert forked["agent"]["session_id"] == f"{started['agent']['session_id']}-fork"


def test_fork_agent_takes_source_session_lease(tmp_path):
    controller = _controller(tmp_path)
    started = controller.start_agent(profile="default", cwd=str(tmp_path))
    agent = started["agent"]

    with controller.store.leased_session(
        agent["id"],
        profile=agent["profile"],
        session_id=agent["session_id"],
        owner="worker-a",
        ttl_seconds=60,
    ):
        result = controller.fork_agent(agent_id=agent["id"], lease_wait_seconds=0)

    assert result["ok"] is False
    assert "agent session" in result["error"]
    assert "busy" in result["error"]


def test_session_lease_is_keyed_by_profile_and_session(tmp_path):
    store = AgentControlStore(tmp_path / "agent-control.db")

    first = store.acquire_session_lease(
        profile="default",
        session_id="shared-session",
        handle_id="agent-a",
        owner="worker-a",
        ttl_seconds=60,
    )
    second_same_session = store.acquire_session_lease(
        profile="default",
        session_id="shared-session",
        handle_id="agent-b",
        owner="worker-b",
        ttl_seconds=60,
        wait_seconds=0,
    )
    second_other_session = store.acquire_session_lease(
        profile="default",
        session_id="other-session",
        handle_id="agent-c",
        owner="worker-b",
        ttl_seconds=60,
        wait_seconds=0,
    )

    assert first is True
    assert second_same_session is False
    assert second_other_session is True


def test_agent_control_tool_schema_does_not_expose_approval_policy():
    for schema in (AGENT_START_SCHEMA, AGENT_PROMPT_SCHEMA, AGENT_FORK_SCHEMA):
        properties = schema["parameters"]["properties"]
        assert "approval_policy" not in properties
        assert "allow_once" not in json.dumps(schema)


def test_agent_control_approval_policy_is_admin_env_only(monkeypatch):
    monkeypatch.delenv(_ADMIN_APPROVAL_POLICY_ENV, raising=False)
    assert _approval_policy_from_env() == "deny"

    monkeypatch.setenv(_ADMIN_APPROVAL_POLICY_ENV, "allow_once")
    assert _approval_policy_from_env() == "allow_once"

    monkeypatch.setenv(_ADMIN_APPROVAL_POLICY_ENV, "allow")
    assert _approval_policy_from_env() == "deny"
