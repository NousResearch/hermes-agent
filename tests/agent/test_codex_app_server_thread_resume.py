"""#100531 — the codex app-server thread survives an AIAgent rebuild (API-server restart, per-request agents).

``CodexAppServerSession`` keeps the codex thread id in memory only, so every new ``AIAgent`` for the same
Hermes session used to ``thread/start`` an empty thread while Hermes' own transcript continued. The runtime
now publishes ``codex_thread_id`` into the session row's ``model_config`` once the turn's projected rows are
durable, the next agent for that session issues ``thread/resume`` for it, and a stored id codex cannot hand
back fails closed: fresh thread, binding dropped, one status-rail notice.
"""

from pathlib import Path

from agent.transports import codex_app_server_session as session_mod
from agent.transports.codex_app_server import CodexAppServerError
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_session

SID = "sess-codex-restart"


class _WireClient:
    """Minimal app-server stand-in: answers thread/start with a fresh id, thread/resume with the requested
    id (or refuses ids in ``dead``), and records every JSON-RPC method it saw."""

    dead: set[str] = set()
    instances: "list[_WireClient]" = []
    counter = 0

    def __init__(self, **kwargs):
        self.requests: list[tuple[str, dict]] = []
        _WireClient.instances.append(self)

    def initialize(self, **kwargs):
        return {}

    def request(self, method, params=None, timeout=30.0):
        params = params or {}
        self.requests.append((method, params))
        if method == "thread/resume":
            if params["threadId"] in _WireClient.dead:
                raise CodexAppServerError(code=-32600, message=f"no rollout found for thread id {params['threadId']}")
            return {"thread": {"id": params["threadId"]}}
        _WireClient.counter += 1
        return {"thread": {"id": f"thread-{_WireClient.counter}"}}

    def close(self):
        pass


def _agent(db, **kwargs):
    from run_agent import AIAgent
    agent = AIAgent(api_key="stub", base_url="https://stub.invalid", provider="openai", api_mode="codex_app_server",
                    quiet_mode=True, skip_context_files=True, skip_memory=True, persistence=db, session_id=SID, **kwargs)
    agent._spawn_background_review = lambda **kw: None
    return agent


def _run_turn(self, user_input, **kwargs):
    return TurnResult(final_text=f"echo {user_input}", thread_id=self._thread_id, turn_id="turn-1",
                      projected_messages=[{"role": "assistant", "content": f"echo {user_input}"}])


def test_rebuilt_agent_resumes_the_stored_codex_thread_and_an_unresumable_one_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(session_mod, "CodexAppServerClient", _WireClient)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", _run_turn)
    _WireClient.instances, _WireClient.dead, _WireClient.counter = [], set(), 0
    db = open_persistence(profile_scope(home=Path(tmp_path)))
    try:
        # Process 1: first turn publishes the binding only after the transcript is durable.
        first = _agent(db)
        assert first.run_conversation("Remember the word amber.")["completed"]
        assert db.sessions.get_model_config_value(SID, "codex_thread_id") == "thread-1"
        assert [r["content"] for r in transcript_rows(db, SID)][-1] == "echo Remember the word amber."

        # Process 2 (restart): a new AIAgent for the same session resumes that thread before turn/start.
        notices: list[str] = []
        def status_callback(kind, message):  # the lifecycle rail every surface renders; other kinds are noise
            if kind == "lifecycle":
                notices.append(str(message))
        second = _agent(db)
        second.status_callback = status_callback
        assert second.run_conversation("Which word?", conversation_history=[])["completed"]
        assert [m for m, _ in _WireClient.instances[1].requests] == ["thread/resume"]
        assert _WireClient.instances[1].requests[0][1]["threadId"] == "thread-1"
        assert notices == []

        # Control: the stored thread is gone on the codex side -> fresh thread, binding rotated, one notice.
        _WireClient.dead = {"thread-1"}
        third = _agent(db)
        third.status_callback = status_callback
        assert third.run_conversation("And now?", conversation_history=[])["completed"]
        assert [m for m, _ in _WireClient.instances[2].requests] == ["thread/resume", "thread/start"]
        assert db.sessions.get_model_config_value(SID, "codex_thread_id") == "thread-2"
        assert notices == ["Codex thread could not be resumed; starting a new one."]
    finally:
        db.close()


def test_the_binding_is_stored_and_read_through_the_writer_handle(tmp_path):
    """An agent with a writer's storage handle beside its store keeps the binding through the handle."""
    from types import SimpleNamespace

    import persistence_assembly
    from agent.codex_runtime import _store_codex_thread_id, _stored_codex_thread_id

    db = open_persistence(profile_scope(home=Path(tmp_path)))
    seed_session(db, SID, "cli")
    try:
        with persistence_assembly.open_persistence(persistence_assembly.profile_scope(home=tmp_path)) as handle:
            agent = SimpleNamespace(_session_db=db, session_id=SID, persistence=handle)
            _store_codex_thread_id(agent, "thread-9")
            assert db.sessions.get_model_config_value(SID, "codex_thread_id") == "thread-9"
            assert _stored_codex_thread_id(agent) == "thread-9"
            _store_codex_thread_id(agent, None)
            assert _stored_codex_thread_id(agent) is None
    finally:
        db.close()
