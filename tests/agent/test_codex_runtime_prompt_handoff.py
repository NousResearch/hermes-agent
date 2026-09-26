"""The codex_app_server runtime hands Hermes' composed system prompt to the codex thread (#74712, #26035).

The standard loop sends ``_cached_system_prompt + ephemeral_system_prompt`` as its system message; the
codex early-return used to send only cwd + raw user text, so SOUL.md / memory / channel_overrides were
composed and then silently dropped.
"""

from types import SimpleNamespace

from agent import codex_runtime
from agent.transports import codex_app_server_session as sess_mod


class _FakeClient:
    def __init__(self, **_kw):
        self.requests = []
        self.closed = 0

    def close(self):
        self.closed += 1

    def initialize(self, **_kw):
        return {}

    def request(self, method, params=None, timeout=None):
        self.requests.append((method, params))
        return {"thread": {"id": "t1"}}


def _agent(**overrides):
    base = dict(_codex_session=None, session_cwd="/tmp", tool_progress_callback=None,
                _cached_system_prompt="SOUL: you are Hermes", ephemeral_system_prompt="Always start with ZZZ")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_runtime_sends_composed_prompt_once_per_thread(monkeypatch):
    """Composition mirrors turn_context (prompt + blank line + ephemeral); sent on thread/start
    exactly once even though _ensure_codex_session runs on every turn."""
    client = _FakeClient()
    monkeypatch.setattr(sess_mod, "CodexAppServerClient", lambda **kw: client)
    agent = _agent()
    for _ in range(3):  # three turns reuse one session
        codex_runtime._ensure_codex_session(agent)
        agent._codex_session.ensure_started()
    starts = [p for (m, p) in client.requests if m == "thread/start"]
    assert len(starts) == 1
    assert starts[0]["developerInstructions"] == "SOUL: you are Hermes\n\nAlways start with ZZZ"


def test_runtime_omits_prompt_when_agent_has_none(monkeypatch):
    """No cached prompt and no ephemeral additions → no developerInstructions field at all."""
    client = _FakeClient()
    monkeypatch.setattr(sess_mod, "CodexAppServerClient", lambda **kw: client)
    agent = _agent(_cached_system_prompt=None, ephemeral_system_prompt=None)
    codex_runtime._ensure_codex_session(agent)
    agent._codex_session.ensure_started()
    (_, params), = [(m, p) for (m, p) in client.requests if m == "thread/start"]
    assert "developerInstructions" not in params


def test_runtime_retires_thread_when_prompt_composition_changes(monkeypatch):
    """TUI/Desktop ``/personality`` mutates the live agent's ephemeral prompt in place; the next turn must
    retire the thread started with the old composition and start one carrying the new developerInstructions."""
    client = _FakeClient()
    monkeypatch.setattr(sess_mod, "CodexAppServerClient", lambda **kw: client)
    agent = _agent()
    codex_runtime._ensure_codex_session(agent)
    agent._codex_session.ensure_started()
    agent.ephemeral_system_prompt = "Personality: pirate"
    codex_runtime._ensure_codex_session(agent)
    agent._codex_session.ensure_started()
    starts = [p["developerInstructions"] for (m, p) in client.requests if m == "thread/start"]
    assert starts == ["SOUL: you are Hermes\n\nAlways start with ZZZ", "SOUL: you are Hermes\n\nPersonality: pirate"]
    assert client.closed == 1  # the stale thread's client was closed, not leaked
