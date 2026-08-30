"""codex_app_server thread resume across agent rebuilds (#82160).

The app-server path never replays Hermes' transcript into a new codex
thread — the live thread IS the conversation memory. Before resume
support, every AIAgent rebuild (agent-cache eviction, gateway restart)
called thread/start and silently dropped the whole conversation.

Verifies that:
  - ensure_started() tries thread/resume when a resume id is provided,
    and never calls thread/start on success
  - resume failure (RPC error, or a payload with no thread id) falls
    back to thread/start — a dead rollout must never fail the turn
  - the session->thread mapping helpers round-trip via
    HERMES_HOME/codex_threads.json, tolerate a corrupt store, prune,
    and refuse gateway-hygiene agents in both directions
  - run_conversation() wires the two together: the stored id reaches
    CodexAppServerSession, and the turn's thread id lands in the store
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import run_agent
from agent.codex_runtime import (
    _codex_thread_store_path,
    _load_codex_thread_id,
    _store_codex_thread_id,
)
from agent.transports.codex_app_server_session import (
    CodexAppServerError,
    CodexAppServerSession,
    TurnResult,
)


class StubClient:
    """Records JSON-RPC traffic; scripted thread/resume behavior."""

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

    def stderr_tail(self, n):
        return []

    def close(self):
        pass


def _session(client, resume_thread_id=None):
    return CodexAppServerSession(
        cwd="/tmp/resume-test-cwd",
        resume_thread_id=resume_thread_id,
        client_factory=lambda **kwargs: client,
    )


class TestEnsureStartedResume:
    def test_resume_happy_path_never_starts_a_fresh_thread(self):
        client = StubClient(resume_payload={"thread": {"id": "old-thread-9"}})
        tid = _session(client, resume_thread_id="old-thread-9").ensure_started()
        assert tid == "old-thread-9"
        methods = [m for m, _ in client.requests]
        assert methods == ["thread/resume"]
        _, params = client.requests[0]
        assert params["threadId"] == "old-thread-9"
        assert params["cwd"] == "/tmp/resume-test-cwd"

    def test_resume_rpc_error_falls_back_to_thread_start(self):
        client = StubClient(
            resume_error=CodexAppServerError(
                code=-32600, message="no rollout found for thread id x"
            )
        )
        tid = _session(client, resume_thread_id="gone-thread").ensure_started()
        assert tid == "fresh-thread-1"
        assert [m for m, _ in client.requests] == [
            "thread/resume", "thread/start",
        ]

    def test_resume_payload_without_id_falls_back_to_thread_start(self):
        client = StubClient(resume_payload={"unexpected": True})
        tid = _session(client, resume_thread_id="old-thread").ensure_started()
        assert tid == "fresh-thread-1"
        assert [m for m, _ in client.requests] == [
            "thread/resume", "thread/start",
        ]

    def test_no_resume_id_preserves_original_behavior(self):
        client = StubClient()
        tid = _session(client).ensure_started()
        assert tid == "fresh-thread-1"
        assert [m for m, _ in client.requests] == ["thread/start"]


class TestThreadIdStore:
    def _agent(self, session_id="sess-1", platform="telegram"):
        return SimpleNamespace(session_id=session_id, platform=platform)

    def test_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        agent = self._agent()
        assert _load_codex_thread_id(agent) is None
        _store_codex_thread_id(agent, "thread-abc")
        assert _load_codex_thread_id(agent) == "thread-abc"

    def test_corrupt_store_reads_as_empty_and_recovers_on_write(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _codex_thread_store_path().write_text("{not json", encoding="utf-8")
        agent = self._agent()
        assert _load_codex_thread_id(agent) is None
        _store_codex_thread_id(agent, "thread-abc")
        assert _load_codex_thread_id(agent) == "thread-abc"

    def test_hygiene_agents_neither_read_nor_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        real = self._agent()
        _store_codex_thread_id(real, "thread-abc")
        hygiene = self._agent(platform="gateway_hygiene")
        assert _load_codex_thread_id(hygiene) is None
        _store_codex_thread_id(hygiene, "clobber")
        assert _load_codex_thread_id(real) == "thread-abc"

    def test_store_prunes_oldest_beyond_cap(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for i in range(205):
            _store_codex_thread_id(self._agent(f"sess-{i:03d}"), f"t-{i}")
        store = json.loads(
            _codex_thread_store_path().read_text(encoding="utf-8")
        )
        assert len(store) <= 200
        assert "sess-204" in store  # newest survives
        assert "sess-000" not in store  # oldest pruned


class TestRunConversationWiring:
    def _make_codex_agent(self, **kwargs):
        return run_agent.AIAgent(
            api_key="stub",
            base_url="https://stub.invalid",
            provider="openai",
            api_mode="codex_app_server",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            **kwargs,
        )

    def test_stored_id_reaches_session_and_turn_id_lands_in_store(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        seen = {}
        original_init = CodexAppServerSession.__init__

        def spy_init(self, **kwargs):
            seen["resume_thread_id"] = kwargs.get("resume_thread_id")
            original_init(self, **kwargs)

        def fake_run_turn(self, user_input, **kwargs):
            return TurnResult(
                final_text="done",
                projected_messages=[{"role": "assistant", "content": "done"}],
                turn_id="turn-1",
                thread_id="thread-resumed-7",
            )

        monkeypatch.setattr(CodexAppServerSession, "__init__", spy_init)
        monkeypatch.setattr(CodexAppServerSession, "run_turn", fake_run_turn)
        monkeypatch.setattr(
            CodexAppServerSession,
            "ensure_started",
            lambda self: "thread-resumed-7",
        )

        agent = self._make_codex_agent(session_id="wiring-sess")
        _store_codex_thread_id(
            SimpleNamespace(session_id="wiring-sess", platform="cli"),
            "thread-resumed-7",
        )
        with patch.object(agent, "_spawn_background_review", return_value=None):
            result = agent.run_conversation("hello")

        assert result["codex_thread_id"] == "thread-resumed-7"
        assert seen["resume_thread_id"] == "thread-resumed-7"
        stored = _load_codex_thread_id(
            SimpleNamespace(session_id="wiring-sess", platform="cli")
        )
        assert stored == "thread-resumed-7"
