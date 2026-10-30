"""Final-answer extension seam: silent continuation, persistence exclusion, fail-open, load ownership."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from agent import final_answer_extensions as fae
from agent import session_persistence, turn_final_response, turn_stop_gates
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def registry(monkeypatch):
    import hermes_cli.plugins as plugins
    manager = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    def install(provider, name="example", key="example"):
        manifest = PluginManifest(name=name)
        registration = PluginContext(manifest, manager).register_final_answer_provider(key, provider)
        manager._plugins[name] = SimpleNamespace(enabled=True, error=None, manifest=manifest)
        return registration
    return manager, install


def _agent(**kw):
    base = dict(platform="cli", _user_id="u1", _chat_type="dm", session_id="s", _current_turn_id="t1",
                _interrupt_requested=False, _interim_content_was_streamed=lambda _t: False)
    base.update(kw)
    return SimpleNamespace(**base)


def _quiet_gates(monkeypatch):
    monkeypatch.setattr(turn_stop_gates, "_verify_on_stop_nudge", lambda agent: None)
    monkeypatch.setattr(turn_stop_gates, "_pre_verify_nudge", lambda *a: None)
    monkeypatch.setattr(turn_stop_gates, "_kanban_stop_nudge", lambda *a: None)
    monkeypatch.setattr(turn_stop_gates, "_late_lsp_verdicts", lambda agent: None)


def _gate(agent, text, messages):
    return turn_stop_gates.apply_stop_gates(
        agent, {"role": "assistant", "content": text}, final_response=text, messages=messages,
        conversation_history=[], pending_verification_response=None, pending_verification_response_previewed=None)


class OncePerTurn:
    """Example provider: nudge once per turn when the draft asks the user to do something."""

    def before_final(self, context, text):
        turn = context.identity["_current_turn_id"]
        if "please do it yourself" not in text or context.state.get("turn") == turn:
            return None
        context.state["turn"] = turn
        return fae.FinalNudge("Try the remaining steps before handing back.", "example_lint", "_example_synthetic")


def test_nudge_continues_once_and_rows_are_ephemeral(monkeypatch, registry):
    _, install = registry
    install(OncePerTurn())
    _quiet_gates(monkeypatch)
    agent, messages = _agent(), []
    verdict = _gate(agent, "please do it yourself", messages)
    assert verdict.continue_turn and verdict.final_response is None
    assert verdict.pending_verification_response == "please do it yourself"
    assert [m["role"] for m in messages] == ["assistant", "user"]
    for row in messages:
        assert row[fae.EXTENSION_SYNTHETIC_FLAG] and row["_example_synthetic"]
        assert session_persistence._is_ephemeral_scaffolding(row)
        assert any(row.get(f) for f in turn_final_response._EPHEMERAL_SCAFFOLDING_FLAGS)
    again = _gate(agent, "please do it yourself", messages)
    assert not again.continue_turn and again.final_response == "please do it yourself"


def test_no_provider_is_a_no_op(monkeypatch, registry):
    _quiet_gates(monkeypatch)
    messages = []
    verdict = _gate(_agent(), "done", messages)
    assert not verdict.continue_turn and verdict.final_response == "done" and messages == []


@pytest.mark.parametrize("mode", ["raises", "slow", "wrong_type", "bad_flag", "empty_text"])
def test_broken_provider_fails_open(monkeypatch, registry, mode):
    _, install = registry
    release = threading.Event()

    class Broken:
        def before_final(self, context, text):
            context.state["touched"] = True
            if mode == "raises":
                raise RuntimeError("boom")
            if mode == "slow":
                release.wait(5)
                return fae.FinalNudge("late", "late", "_late_synthetic")
            if mode == "wrong_type":
                return {"text": "x"}
            if mode == "bad_flag":
                return fae.FinalNudge("x", "x", "persisted_flag")
            return fae.FinalNudge("", "x", "_x_synthetic")
    install(Broken())
    monkeypatch.setattr(fae, "CALL_TIMEOUT_SECONDS", 0.05)
    _quiet_gates(monkeypatch)
    agent, messages = _agent(), []
    started = time.monotonic()
    verdict = _gate(agent, "done", messages)
    release.set()
    assert time.monotonic() - started < 2.0
    assert not verdict.continue_turn and verdict.final_response == "done" and messages == []
    assert getattr(agent, "_final_answer_extension_state", {}) == {}  # state committed only on a valid return


def test_provider_receives_copies_not_the_agent(monkeypatch, registry):
    _, install = registry
    seen = []

    class Spy:
        def before_final(self, context, text):
            seen.append(context)
            context.messages[0]["content"] if context.messages else None
            return None
    install(Spy())
    _quiet_gates(monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    _gate(_agent(), "done", messages)
    context = seen[0]
    assert not hasattr(context, "agent") and not hasattr(context, "send")
    with pytest.raises(TypeError):
        context.identity["platform"] = "forged"
    assert context.messages[0] is not messages[0]


def test_registration_is_load_owned_and_validated():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="example"), manager)
    with pytest.raises(ValueError):
        ctx.register_final_answer_provider("bad", object())
    registration = ctx.register_final_answer_provider("ok", OncePerTurn())
    with pytest.raises(ValueError):
        ctx.register_final_answer_provider("ok", OncePerTurn())
    registration.dispose()
    assert manager._final_answer_providers == {}
