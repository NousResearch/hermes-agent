"""Provider conversation identity survives cache rebuilds and process restarts."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import run_agent
from hermes_state import SessionDB
from agent.codex_runtime import _load_codex_thread_id, _store_codex_thread_id
from agent.transports.codex_app_server_session import (
    CodexAppServerError, CodexAppServerSession, TurnResult,
)


class StubClient:
    def __init__(self, resume_error=None, resume_payload=None):
        self.requests = []
        self._resume_error = resume_error
        self._resume_payload = resume_payload

    def initialize(self, **kwargs):
        pass

    def request(self, method, params, timeout=None):
        self.requests.append((method, params))
        if method == "thread/resume":
            if self._resume_error is not None:
                raise self._resume_error
            return self._resume_payload
        if method == "thread/start":
            return {"thread": {"id": "fresh-thread-1"}}
        raise AssertionError(f"unexpected method {method}")

    def close(self):
        pass


def session(client, resume_thread_id=None):
    return CodexAppServerSession(
        resume_thread_id=resume_thread_id,
        client_factory=lambda **kwargs: client,
    )


def test_resume_happy_path_never_starts_a_fresh_thread():
    client = StubClient(resume_payload={"thread": {"id": "old-thread"}})
    s = session(client, "old-thread")
    assert s.ensure_started() == s.ensure_started() == "old-thread"
    assert [m for m, _ in client.requests] == ["thread/resume"]


@pytest.mark.parametrize("payload", [{}, {"thread": {"id": "wrong-thread"}}])
def test_invalid_resume_never_starts_empty_thread(payload):
    client = StubClient(resume_payload=payload)
    with pytest.raises(CodexAppServerError):
        session(client, "old-thread").ensure_started()
    assert [m for m, _ in client.requests] == ["thread/resume"]


@pytest.mark.parametrize("error", [
    CodexAppServerError(code=-32600, message="missing rollout"),
    TimeoutError("temporary outage"),
])
def test_resume_failure_preserves_binding_for_retry(error):
    client = StubClient(resume_error=error)
    with pytest.raises(type(error)):
        session(client, "old-thread").ensure_started()
    assert [m for m, _ in client.requests] == ["thread/resume"]


def test_new_session_starts_fresh():
    client = StubClient()
    assert session(client).ensure_started() == "fresh-thread-1"
    assert [m for m, _ in client.requests] == ["thread/start"]


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "sessions.db")
    yield database
    database.close()


def agent_for(db, sid="sess-1", platform="discord"):
    db.create_session(sid, source=platform, model_config={"unrelated": "preserve"})
    return SimpleNamespace(_session_db=db, session_id=sid, platform=platform)


def test_binding_survives_new_database_connection(db):
    agent = agent_for(db)
    assert _load_codex_thread_id(agent) is None
    _store_codex_thread_id(agent, "thread-1")
    other = SessionDB(db.db_path)
    try:
        rebuilt = SimpleNamespace(_session_db=other, session_id=agent.session_id, platform="discord")
        assert _load_codex_thread_id(rebuilt) == "thread-1"
        assert other.get_session_model_config_value(agent.session_id, "unrelated") == "preserve"
    finally:
        other.close()


def test_concurrent_session_bindings_are_not_lost(db):
    agents = [agent_for(db, f"s-{i}") for i in range(24)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda a: _store_codex_thread_id(a, "thread-" + a.session_id), agents))
    assert [_load_codex_thread_id(a) for a in agents] == ["thread-" + a.session_id for a in agents]


def test_hygiene_cannot_clobber_binding(db):
    agent = agent_for(db)
    _store_codex_thread_id(agent, "thread-1")
    agent.platform = "gateway_hygiene"
    assert _load_codex_thread_id(agent) is None
    _store_codex_thread_id(agent, "clobber")
    agent.platform = "discord"
    assert _load_codex_thread_id(agent) == "thread-1"


def test_new_hermes_session_does_not_inherit_old_thread(db):
    _store_codex_thread_id(agent_for(db, "old"), "thread-old")
    assert _load_codex_thread_id(agent_for(db, "new")) is None


def test_changed_codex_home_refuses_resume(db, monkeypatch, tmp_path):
    agent = agent_for(db)
    _store_codex_thread_id(agent, "thread-1")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "other-home"))
    with pytest.raises(RuntimeError, match="profile/home"):
        _load_codex_thread_id(agent)


@pytest.mark.parametrize("binding", ["corrupt", {}, {"thread_id": 9}])
def test_corrupt_binding_is_not_treated_as_new_session(db, binding):
    agent = agent_for(db)
    db.patch_session_model_config(agent.session_id, {"codex_thread_binding": binding})
    with pytest.raises(RuntimeError):
        _load_codex_thread_id(agent)


def test_missing_database_row_is_not_silent_success(db):
    agent = SimpleNamespace(_session_db=db, session_id="missing", platform="discord")
    with pytest.raises(RuntimeError):
        _store_codex_thread_id(agent, "thread-1")


def make_agent(sid, db=None):
    return run_agent.AIAgent(
        session_id=sid, api_key="stub", base_url="https://stub.invalid",
        provider="openai", api_mode="codex_app_server", quiet_mode=True,
        skip_context_files=True, skip_memory=True, session_db=db,
    )


def test_real_agent_rebuild_resumes_before_sending_followup(monkeypatch, db):
    clients = []
    original_init = CodexAppServerSession.__init__
    def init(self, **kwargs):
        client = StubClient(resume_payload={"thread": {"id": "fresh-thread-1"}})
        clients.append(client)
        original_init(self, client_factory=lambda **kw: client, **kwargs)
    def run_turn(self, user_input, **kwargs):
        # Persistence must already be durable before user input reaches Codex.
        return TurnResult(final_text="done", thread_id=self._thread_id,
                          projected_messages=[{"role": "assistant", "content": "done"}])
    monkeypatch.setattr(CodexAppServerSession, "__init__", init)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", run_turn)
    with patch.object(run_agent.AIAgent, "_spawn_background_review"):
        first = make_agent("rebuild-test", db)
        result = first.run_conversation("Remember the word amber.")
        assert result["completed"], result
        assert _load_codex_thread_id(first) == "fresh-thread-1"
        first._codex_session.close()
        rebuilt = make_agent("rebuild-test", db)
        second = rebuilt.run_conversation("Which word?", conversation_history=result["messages"])
        assert second["completed"], second
    assert [[m for m, _ in c.requests] for c in clients] == [["thread/start"], ["thread/resume"]]


def test_persistence_failure_sends_no_user_turn(db):
    agent = make_agent("write-failure", db)
    with patch.object(CodexAppServerSession, "ensure_started", return_value="thread-1"), \
         patch.object(CodexAppServerSession, "run_turn") as turn, \
         patch.object(agent._session_db, "patch_session_model_config", side_effect=OSError("disk full")):
        result = agent.run_conversation("hello")
    assert result["completed"] is False
    turn.assert_not_called()


def test_unbound_existing_history_does_not_start_empty_thread():
    agent = make_agent("unbound")
    with patch.object(CodexAppServerSession, "ensure_started") as start:
        result = agent.run_conversation("what was that?", conversation_history=[
            {"role": "user", "content": "Remember amber"},
            {"role": "assistant", "content": "Remembered"},
        ])
    assert result["completed"] is False
    assert "no saved Codex thread binding" in result["final_response"]
    start.assert_not_called()


def test_automatic_rename_rebuild_keeps_provider_binding(db):
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, SessionContext, build_session_context_prompt
    from gateway.config import Platform
    source = SessionSource(platform=Platform.DISCORD, chat_id="chat", thread_id="thread",
                           chat_type="thread", chat_name="Remember amber", user_id="user")
    context = SessionContext(source=source, connected_platforms=[Platform.DISCORD], home_channels={})
    def signature():
        return GatewayRunner._agent_config_signature("model", {}, [], build_session_context_prompt(context))
    before = signature()
    first = agent_for(db, "rename")
    _store_codex_thread_id(first, "provider-thread")
    source.chat_name = "Remember a word"
    assert signature() != before  # The real gateway rebuild trigger.
    rebuilt = agent_for(db, "rename")
    client = StubClient(resume_payload={"thread": {"id": "provider-thread"}})
    assert session(client, _load_codex_thread_id(rebuilt)).ensure_started() == "provider-thread"
    assert [m for m, _ in client.requests] == ["thread/resume"]
