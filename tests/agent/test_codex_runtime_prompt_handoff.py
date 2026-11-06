"""The codex_app_server runtime hands Hermes' composed system prompt to the codex thread (#74712, #26035).

The standard loop sends ``_cached_system_prompt + ephemeral_system_prompt`` as its system message; the
codex early-return used to send only cwd + raw user text, so SOUL.md / memory / channel_overrides were
composed and then silently dropped.
"""

from types import SimpleNamespace

import pytest

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


def test_each_turn_sends_the_agent_s_current_wire_model(monkeypatch):
    """CLI/TUI ``/model`` mutates the live agent; every turn/start carries the model selected now (the
    ``-900k`` alias as its base slug), not the one the thread was started with."""
    turns = []

    class _Session:
        def __init__(self, **_kw):
            pass

        def ensure_started(self):
            return "t1"

        def run_turn(self, **kw):
            turns.append(kw["model"])
            raise RuntimeError("stop after turn/start")

        def close(self):
            pass

    monkeypatch.setattr(sess_mod, "CodexAppServerSession", _Session)
    agent = _agent(model="gpt-5.6-sol-900k", _interrupt_requested=False, _interrupt_message=None)
    for model in ("gpt-5.6-sol-900k", "gpt-5.5"):
        agent.model = model
        codex_runtime.run_codex_app_server_turn(agent, user_message="hi", original_user_message="hi",
                                                messages=[], effective_task_id="t")
    assert turns == ["gpt-5.6-sol", "gpt-5.5"]


def test_runtime_retires_thread_when_an_in_place_switch_changes_the_codex_provider(monkeypatch):
    """turn/start can change the model but not the provider: switching between codex's own provider and a
    named custom provider (``[model_providers.<id>]``) starts a thread carrying the new modelProvider."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "load_config", lambda: {"providers": {"my-gateway": {"api": "https://gw.example/v1"}}})
    client = _FakeClient()
    monkeypatch.setattr(sess_mod, "CodexAppServerClient", lambda **kw: client)
    agent = _agent(provider="openai-codex", requested_provider="openai-codex", model="gpt-5.5")
    for provider, requested in (("openai-codex", "openai-codex"), ("openai-codex", "openai-codex"),
                                ("custom", "custom:my-gateway")):
        agent.provider, agent.requested_provider = provider, requested
        codex_runtime._ensure_codex_session(agent)
        agent._codex_session.ensure_started()
    starts = [p.get("modelProvider") for (m, p) in client.requests if m == "thread/start"]
    assert starts == [None, "my-gateway"]
    assert client.closed == 1


@pytest.mark.parametrize("label", ["custom:my-gateway", "my-gateway"])
def test_runtime_retires_thread_when_model_switches_onto_a_named_provider(monkeypatch, label):
    """``/model`` (``_swap_switch_runtime``) sets ``provider`` and ``requested_provider`` to the configured name
    itself (``custom:<name>`` or the bare key), not startup's ``provider="custom"``. The switch still retires
    the thread (#130735) and starts one carrying the named provider's modelProvider."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "load_config", lambda: {"providers": {"my-gateway": {"api": "https://gw.example/v1"}}})
    client = _FakeClient()
    monkeypatch.setattr(sess_mod, "CodexAppServerClient", lambda **kw: client)
    agent = _agent(provider="openai-codex", requested_provider="openai-codex", model="gpt-5.5")
    for provider in ("openai-codex", label):
        agent.provider = agent.requested_provider = provider
        codex_runtime._ensure_codex_session(agent)
        agent._codex_session.ensure_started()
    starts = [p.get("modelProvider") for (m, p) in client.requests if m == "thread/start"]
    assert starts == [None, "my-gateway"]
    assert client.closed == 1


def test_runtime_gives_no_approval_callback_when_nobody_can_answer(monkeypatch):
    """`hermes chat -q` registers the CLI panel callback but nobody answers it: the codex session gets no
    callback, so exec/apply_patch requests fail closed at once instead of waiting the approval timeout."""
    from tools.terminal_tool import set_approval_callback

    monkeypatch.setattr(sess_mod, "CodexAppServerClient", lambda **kw: _FakeClient())
    panel = lambda *args, **kwargs: "once"
    set_approval_callback(panel)
    try:
        interactive = _agent()
        codex_runtime._ensure_codex_session(interactive)
        monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
        single_query = _agent()
        codex_runtime._ensure_codex_session(single_query)
    finally:
        set_approval_callback(None)
    assert interactive._codex_session._approval_callback is panel
    assert single_query._codex_session._approval_callback is None
