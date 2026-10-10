"""#100531 — the codex app-server thread survives an AIAgent rebuild (API-server restart, per-request agents).

``CodexAppServerSession`` keeps the codex thread id in memory only, so every new ``AIAgent`` for the same
Hermes session used to ``thread/start`` an empty thread while Hermes' own transcript continued. The runtime
now publishes ``codex_thread_id`` into the session row's ``model_config`` once the turn's projected rows are
durable, the next agent for that session issues ``thread/resume`` for it, and a stored id codex cannot hand
back fails closed: fresh thread, binding dropped, one status-rail notice.

#127103 — the binding rides with a transcript watermark (``codex_thread_watermark``: head id + count
at commit time). A recreated agent resumes only while the durable transcript still ends where the
binding left it (the one allowed new row is this turn's already-persisted user input). Rows added
past the watermark elsewhere — e.g. an intervening turn routed through another provider — skip the
resume non-destructively so the fresh thread is seeded with the updated history.
"""

from pathlib import Path

from agent.transports import codex_app_server_session as session_mod
from agent.transports.codex_app_server import CodexAppServerError
from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from hermes_state import SessionDB

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
                    quiet_mode=True, skip_context_files=True, skip_memory=True, session_db=db, session_id=SID, **kwargs)
    agent._spawn_background_review = lambda **kw: None
    return agent


def _run_turn(self, user_input, **kwargs):
    return TurnResult(final_text=f"echo {user_input}", thread_id=self._thread_id, turn_id="turn-1",
                      projected_messages=[{"role": "assistant", "content": f"echo {user_input}"}])


def test_rebuilt_agent_resumes_the_stored_codex_thread_and_an_unresumable_one_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(session_mod, "CodexAppServerClient", _WireClient)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", _run_turn)
    _WireClient.instances, _WireClient.dead, _WireClient.counter = [], set(), 0
    db = SessionDB(Path(tmp_path) / "state.db")
    try:
        # Process 1: first turn publishes the binding only after the transcript is durable.
        first = _agent(db)
        assert first.run_conversation("Remember the word amber.")["completed"]
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        assert [r["content"] for r in db.get_messages(SID)][-1] == "echo Remember the word amber."

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
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-2"
        assert notices == ["Codex thread could not be resumed; starting a new one."]
    finally:
        db.close()


def test_continuous_turns_resume_and_advance_the_watermark_without_wiping_on_read(monkeypatch, tmp_path):
    """#127103 — same continuous context across recreation resumes and the lookup never mutates the row."""
    from agent import codex_runtime as codex_runtime

    monkeypatch.setattr(session_mod, "CodexAppServerClient", _WireClient)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", _run_turn)
    _WireClient.instances, _WireClient.dead, _WireClient.counter = [], set(), 0
    db = SessionDB(Path(tmp_path) / "state.db")
    try:
        first = _agent(db)
        assert first.run_conversation("Remember the word amber.")["completed"]
        watermark = db.get_session_model_config_value(SID, "codex_thread_watermark")
        assert isinstance(watermark, dict)
        assert watermark == {"head_id": db.get_active_message_ids(SID)[-1],
                             "count": len(db.get_active_message_ids(SID))}

        # A pure lookup returns the binding and leaves the row alone (no destructive getter).
        probe = _agent(db)
        assert codex_runtime._stored_codex_thread_id(probe) == "thread-1"
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        assert db.get_session_model_config_value(SID, "codex_thread_watermark") == watermark
        # A second lookup agrees — the first did not consume the binding.
        assert codex_runtime._stored_codex_thread_id(probe) == "thread-1"

        # Recreation the way /api/sessions/{id}/chat does (new AIAgent, history reloaded from
        # the DB) resumes the same thread: the only row past the watermark is this turn's user.
        history = db.get_messages_as_conversation(SID)
        second = _agent(db)
        assert second.run_conversation("Which word?", conversation_history=history)["completed"]
        assert [m for m, _ in _WireClient.instances[1].requests] == ["thread/resume"]
        assert _WireClient.instances[1].requests[0][1]["threadId"] == "thread-1"
        # The resumed thread keeps the id; only the watermark advances to the new head.
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        advanced = db.get_session_model_config_value(SID, "codex_thread_watermark")
        assert isinstance(advanced, dict) and advanced["count"] > watermark["count"]
        assert advanced["head_id"] == db.get_active_message_ids(SID)[-1]
    finally:
        db.close()


def test_intervening_turn_skips_stale_resume_and_seeds_a_fresh_thread(monkeypatch, tmp_path):
    """#127103 — rows added past the watermark (another provider/route ran) must NOT resume.

    Goes through the real recreation path: first codex turn -> durable rows added outside the
    thread (as a non-codex route would persist them) -> new AIAgent with DB-loaded history.
    The fresh ``thread/start`` must carry the updated history as its seed, and the stale
    lookup must not have wiped the stored binding (the new turn overwrites it on commit).
    """
    from agent import codex_runtime as codex_runtime

    monkeypatch.setattr(session_mod, "CodexAppServerClient", _WireClient)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", _run_turn)
    _WireClient.instances, _WireClient.dead, _WireClient.counter = [], set(), 0
    db = SessionDB(Path(tmp_path) / "state.db")
    try:
        first = _agent(db)
        assert first.run_conversation("Remember the word amber.")["completed"]
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        watermark = db.get_session_model_config_value(SID, "codex_thread_watermark")
        assert isinstance(watermark, dict)

        # An intervening turn routed elsewhere lands durable rows past the watermark. Two rows
        # (user + assistant) is what any completed turn persists; the current turn's user row
        # has not been written yet at this point, so the gap is unambiguous.
        db.append_messages_batch(SID, [
            {"role": "user", "content": "other provider question about emerald?"},
            {"role": "assistant", "content": "other provider answer about emerald"},
        ])

        # The stored binding is now stale, but reading it must not delete it.
        probe = _agent(db)
        assert codex_runtime._stored_codex_thread_id(probe) is None
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        assert db.get_session_model_config_value(SID, "codex_thread_watermark") == watermark
        assert codex_runtime._stored_codex_thread_id(probe) is None

        # Recreated agent (API-server per-request shape): history reloaded from the DB.
        history = db.get_messages_as_conversation(SID)
        assert any("emerald" in str(m.get("content") or "") for m in history)
        second = _agent(db)
        assert second.run_conversation("Which word now?", conversation_history=history)["completed"]

        # No resume: a fresh thread seeded with the full updated history.
        assert [m for m, _ in _WireClient.instances[1].requests] == ["thread/start"]
        (_, start_params), = _WireClient.instances[1].requests
        seed = start_params.get("developerInstructions") or ""
        assert "Remember the word amber" in seed
        assert "emerald" in seed
        assert "Which word now?" not in seed  # the turn being submitted is never seeded

        # The fresh turn's commit rotates the binding (and its watermark) forward.
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-2"
        rotated = db.get_session_model_config_value(SID, "codex_thread_watermark")
        assert isinstance(rotated, dict) and rotated != watermark
        assert rotated["head_id"] == db.get_active_message_ids(SID)[-1]

        # And the rotated binding resumes again while the context stays continuous.
        follow_history = db.get_messages_as_conversation(SID)
        third = _agent(db)
        assert third.run_conversation("And then?", conversation_history=follow_history)["completed"]
        assert [m for m, _ in _WireClient.instances[2].requests] == ["thread/resume"]
        assert _WireClient.instances[2].requests[0][1]["threadId"] == "thread-2"
    finally:
        db.close()


def test_legacy_binding_without_watermark_still_resumes(monkeypatch, tmp_path):
    """Pre-#127103 rows carry only ``codex_thread_id``; they keep the historical resume behaviour."""
    monkeypatch.setattr(session_mod, "CodexAppServerClient", _WireClient)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", _run_turn)
    _WireClient.instances, _WireClient.dead, _WireClient.counter = [], set(), 0
    db = SessionDB(Path(tmp_path) / "state.db")
    try:
        first = _agent(db)
        assert first.run_conversation("Remember the word amber.")["completed"]
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        # Simulate an old row by dropping just the watermark.
        db.patch_session_model_config(SID, {"codex_thread_watermark": None})
        assert db.get_session_model_config_value(SID, "codex_thread_watermark") is None

        second = _agent(db)
        assert second.run_conversation("Which word?", conversation_history=[])["completed"]
        assert [m for m, _ in _WireClient.instances[1].requests] == ["thread/resume"]
        assert _WireClient.instances[1].requests[0][1]["threadId"] == "thread-1"
        # The new commit heals the row: id kept, watermark (re)captured.
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        assert isinstance(db.get_session_model_config_value(SID, "codex_thread_watermark"), dict)
    finally:
        db.close()


def test_single_intervening_user_row_skips_resume_before_new_row(monkeypatch, tmp_path):
    """kvnloo review control: a lone intervening durable user-only row must not resume.

    The watermark stores only ``{head_id, count}``, so one row past the head is ambiguous
    unless the allowance is bound to the actual current submit. A failed/aborted/
    intervening non-Codex turn can durably add exactly one user row; recreating Codex
    before adding a new row must skip the resume (fail closed), not mistake the stale
    row for this turn's already-persisted user input.
    """
    from agent import codex_runtime as codex_runtime

    monkeypatch.setattr(session_mod, "CodexAppServerClient", _WireClient)
    monkeypatch.setattr(CodexAppServerSession, "run_turn", _run_turn)
    _WireClient.instances, _WireClient.dead, _WireClient.counter = [], set(), 0
    db = SessionDB(Path(tmp_path) / "state.db")
    try:
        first = _agent(db)
        assert first.run_conversation("Remember the word amber.")["completed"]
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        watermark = db.get_session_model_config_value(SID, "codex_thread_watermark")
        assert isinstance(watermark, dict)

        # One intervening durable user-only row (e.g. a failed turn that persisted only
        # its input on another provider/route).
        db.append_messages_batch(SID, [
            {"role": "user", "content": "stale intervening user-only row"},
        ])

        # Recreate before adding a new row: the pure lookup carries no proof of the
        # current submit, so the ambiguous single row must fail closed.
        probe = _agent(db)
        assert codex_runtime._stored_codex_thread_id(probe) is None
        # Same for a pre-persist candidate without a durable row id: no match, no resume.
        assert codex_runtime._stored_codex_thread_id(
            probe, [{"role": "user", "content": "Which word now?"}]) is None
        # The stale lookup leaves the stored binding alone for the fresh turn to rotate.
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-1"
        assert db.get_session_model_config_value(SID, "codex_thread_watermark") == watermark

        # Full recreation path: persisting the new turn makes two rows past the
        # watermark, so the fresh thread is seeded with the updated history.
        history = db.get_messages_as_conversation(SID)
        assert any("stale intervening" in str(m.get("content") or "") for m in history)
        second = _agent(db)
        assert second.run_conversation("Which word now?", conversation_history=history)["completed"]
        assert [m for m, _ in _WireClient.instances[1].requests] == ["thread/start"]
        (_, start_params), = _WireClient.instances[1].requests
        seed = start_params.get("developerInstructions") or ""
        assert "Remember the word amber" in seed
        assert "stale intervening" in seed
        assert "Which word now?" not in seed
        assert db.get_session_model_config_value(SID, "codex_thread_id") == "thread-2"
    finally:
        db.close()
