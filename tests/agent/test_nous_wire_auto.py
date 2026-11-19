"""``nous.anthropic_wire: auto`` decides a session's wire once, from its first response.

Contracts:
- the upstream classifier reads what Portal actually returns on both wires (OpenRouter stamps
  ``provider`` + ``gen-…`` ids; GMI/Vertex returns Anthropic-native ``msg_…`` ids, no provider);
- ``auto`` starts on chat and never promotes to native until GMI native is cleared;
- one decision per session, only on call 1, only for nous + anthropic/*, never for chat/native;
- through a real AIAgent and the real usage recorder, the hook fires exactly once and a switch
  failure never breaks the turn.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import nous_wire
from hermes_cli import providers as _providers


def _resp(**kw):
    return SimpleNamespace(**kw)


class TestClassifier:
    def test_openrouter_by_provider_field_or_gen_id(self):
        assert nous_wire.classify_upstream(_resp(provider="Anthropic", id="gen-1788708403-xXjDsFw")) == "openrouter"
        assert nous_wire.classify_upstream(_resp(provider="Claude Platform on AWS", id="x")) == "openrouter"
        # native wire: no provider field, but the id is still OpenRouter-minted
        assert nous_wire.classify_upstream(_resp(id="gen-1788680469-hyks5Nj9n7rmdAbMyQ5x")) == "openrouter"

    def test_gmi_by_anthropic_native_id_without_provider(self):
        assert nous_wire.classify_upstream(_resp(id="msg_01XrffvmxsRUWwCbVpBsADHo")) == "gmi"

    @pytest.mark.parametrize("r", [None, _resp(), _resp(id=None), _resp(id="chatcmpl-abc"), _resp(id="", provider="")])
    def test_unknown_never_classifies(self, r):
        assert nous_wire.classify_upstream(r) is None


class TestWireChoice:
    def test_chat_for_openrouter_and_unknown(self):
        assert nous_wire.wire_for_upstream("openrouter") == "chat_completions"
        assert nous_wire.wire_for_upstream(None) == "chat_completions"

    def test_gmi_is_chat_until_cleared_then_native(self, monkeypatch):
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", False)
        assert nous_wire.wire_for_upstream("gmi") == "chat_completions"
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", True)
        assert nous_wire.wire_for_upstream("gmi") == "anthropic_messages"


def _agent(**kw):
    a = SimpleNamespace(provider="nous", model="anthropic/claude-fable-5.1", api_mode="chat_completions",
                        api_key="k", base_url="https://inference-api.nousresearch.com/v1", session_id="s")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class TestHook:
    @pytest.fixture(autouse=True)
    def _auto(self, monkeypatch):
        monkeypatch.setattr(_providers, "_nous_anthropic_wire", lambda: "auto")
        self.switches = []
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model",
                            lambda agent, m, p, api_key="", base_url="", api_mode="", **k: self.switches.append(api_mode) or setattr(agent, "api_mode", api_mode))

    def test_gmi_cleared_schedules_once_and_applies_at_next_iteration(self, monkeypatch):
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", True)
        a = _agent()
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(id="msg_01abc"), 1) is True
        # decided but NOT switched yet: the response is still being consumed on the old wire
        assert a.api_mode == "chat_completions" and self.switches == []
        assert nous_wire.apply_pending_wire_switch(a) is True
        assert a.api_mode == "anthropic_messages" and self.switches == ["anthropic_messages"]
        assert nous_wire.apply_pending_wire_switch(a) is False  # nothing pending twice
        # a later call, even with a different-looking response, never flips again
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(provider="Anthropic", id="gen-1-x"), 2) is False
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(provider="Anthropic", id="gen-1-x"), 1) is False
        assert self.switches == ["anthropic_messages"]

    def test_openrouter_stays_on_chat(self):
        a = _agent()
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(provider="Anthropic", id="gen-1-x"), 1) is False
        assert a.api_mode == "chat_completions" and self.switches == []

    def test_gmi_uncleared_stays_on_chat(self, monkeypatch):
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", False)
        a = _agent()
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(id="msg_01abc"), 1) is False
        assert a.api_mode == "chat_completions" and self.switches == []

    @pytest.mark.parametrize("mode", ["chat", "native"])
    def test_explicit_modes_never_auto_switch(self, monkeypatch, mode):
        monkeypatch.setattr(_providers, "_nous_anthropic_wire", lambda: mode)
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", True)
        a = _agent(api_mode="chat_completions" if mode == "chat" else "anthropic_messages")
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(id="msg_01abc"), 1) is False
        assert self.switches == []

    def test_other_providers_and_models_untouched(self, monkeypatch):
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", True)
        assert nous_wire.maybe_switch_wire_after_first_response(_agent(provider="openrouter"), _resp(id="msg_01abc"), 1) is False
        assert nous_wire.maybe_switch_wire_after_first_response(_agent(model="openai/gpt-5.6-sol"), _resp(id="msg_01abc"), 1) is False
        assert self.switches == []

    def test_switch_failure_is_swallowed_and_decision_is_final(self, monkeypatch):
        monkeypatch.setattr(nous_wire, "GMI_NATIVE_WIRE_CLEARED", True)

        def boom(*a, **k):
            raise RuntimeError("no client")
        monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", boom)
        a = _agent()
        assert nous_wire.maybe_switch_wire_after_first_response(a, _resp(id="msg_01abc"), 1) is True
        assert nous_wire.apply_pending_wire_switch(a) is False
        assert a.api_mode == "chat_completions" and a._nous_wire_decided is True and a._nous_wire_pending is None


def test_real_agent_usage_recorder_calls_the_hook_once(tmp_path, monkeypatch):
    """The wiring: record_response_usage on a real AIAgent invokes the hook on call 1 only."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("nous:\n  anthropic_wire: auto\n", encoding="utf-8")
    from run_agent import AIAgent
    from agent import turn_usage
    calls = []
    monkeypatch.setattr(nous_wire, "maybe_switch_wire_after_first_response",
                        lambda agent, response, n: calls.append((n, nous_wire.classify_upstream(response))) or False)
    a = AIAgent(api_key="jwt", base_url="https://inference-api.nousresearch.com/v1", provider="nous",
                api_mode="chat_completions", model="anthropic/claude-fable-5.1", session_id="t", platform="cli",
                quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False, enabled_toolsets=["file"])
    try:
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105, prompt_tokens_details=None, completion_tokens_details=None)
        for n in (1, 2, 3):
            resp = SimpleNamespace(usage=usage, id="gen-1788708403-xXjDsFwabc", provider="Anthropic", model="anthropic/claude-fable-5.1")
            turn_usage.record_response_usage(a, resp, messages=[{"role": "user", "content": "hi"}], api_call_count=n,
                                             api_duration=0.1, compression_attempts=0, max_compression_attempts=3)
    finally:
        a.close()
    assert calls == [(1, "openrouter")]
