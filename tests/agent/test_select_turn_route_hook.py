"""``select_turn_route``: a plugin picks the model for one turn; the next turn starts on the main one.

Real loopback providers (FakeLLMServer); assertions are on what each server actually received.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.plugins import VALID_HOOKS
from run_agent import AIAgent
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home


def _agent(main_url: str, **kw):
    return AIAgent(
        model="fake-model", provider="custom", base_url=main_url, api_key="sk-fake-e2e",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"], **kw,
    )


def _install(monkeypatch, decide):
    calls: list[dict] = []

    def invoke_hook(name, **kwargs):
        if name != "select_turn_route":
            return []
        calls.append(kwargs)
        result = decide(**kwargs)
        return [] if result is None else [result]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr("hermes_cli.plugins.has_hook", lambda name: name == "select_turn_route")
    return calls


def _home(tmp_path: Path, main_url: str, routed_url: str, context: int = 128000) -> Path:
    extra = ("custom_providers:\n  - name: routed\n"
             f"    base_url: {routed_url}\n    models:\n      routed-model:\n        context_length: {context}\n")
    return write_hermes_home(tmp_path / ".hermes", main_url, extra_config=extra)


def _route(url):
    return {"provider": "custom", "model": "routed-model", "base_url": url, "api_key": "sk-fake"}


def test_hook_is_registered():
    assert "select_turn_route" in VALID_HOOKS


def test_plugin_routes_one_turn_then_main_model_returns(tmp_path, monkeypatch):
    with FakeLLMServer([Text("main one"), Text("main three")]) as main, FakeLLMServer([Text("routed")]) as routed:
        monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path, main.base_url, routed.base_url)))
        calls = _install(monkeypatch, lambda user_message, **_: _route(routed.base_url) if "route me" in user_message else None)
        agent = _agent(main.base_url)
        first = agent.run_conversation("plain question")
        second = agent.run_conversation("please route me", conversation_history=first["messages"])
        agent.run_conversation("plain again", conversation_history=second["messages"])

        assert [r["model"] for r in routed.main_requests()] == ["routed-model"]
        assert [r["model"] for r in main.main_requests()] == ["fake-model", "fake-model"]
        assert second["model"] == "routed-model"
        assert agent.model == "fake-model" and agent._fallback_activated is False
        assert calls[1]["user_message"] == "please route me"
        assert calls[1]["session_model"] == "fake-model"


@pytest.mark.parametrize("answer", ["raises", "garbage", "no-model"])
def test_broken_or_empty_answer_runs_on_main_model(tmp_path, monkeypatch, answer):
    with FakeLLMServer([Text("main")]) as main, FakeLLMServer([]) as routed:
        monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path, main.base_url, routed.base_url)))

        def decide(**_):
            if answer == "raises":
                raise RuntimeError("plugin bug")
            return "not a dict" if answer == "garbage" else {"provider": "custom"}

        _install(monkeypatch, decide)
        result = _agent(main.base_url).run_conversation("hello")
        assert result["model"] == "fake-model"
        assert routed.main_requests() == []


def test_route_too_large_for_its_window_runs_on_main_model(tmp_path, monkeypatch):
    with FakeLLMServer([Text("main")]) as main, FakeLLMServer([]) as routed:
        monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path, main.base_url, routed.base_url, context=1000)))
        _install(monkeypatch, lambda **_: _route(routed.base_url))
        result = _agent(main.base_url).run_conversation("x " * 4000)
        assert result["model"] == "fake-model"
        assert routed.main_requests() == []


def test_subagents_are_not_routed(tmp_path, monkeypatch):
    with FakeLLMServer([Text("main")]) as main, FakeLLMServer([]) as routed:
        monkeypatch.setenv("HERMES_HOME", str(_home(tmp_path, main.base_url, routed.base_url)))
        calls = _install(monkeypatch, lambda **_: _route(routed.base_url))
        agent = _agent(main.base_url)
        agent.is_subagent = True
        assert agent.run_conversation("delegated work")["model"] == "fake-model"
        assert calls == [] and routed.main_requests() == []


@pytest.mark.parametrize("content,text,attached", [
    ("plain words", "plain words", False),
    ([{"type": "text", "text": "@tag look at this"}, {"type": "image_url", "image_url": {"url": "data:x"}}],
     "@tag look at this", True),
    ([{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}], "a\nb", False),
    ([{"type": "image_url", "image_url": {"url": "data:x"}}], "", True),
])
def test_hook_gets_the_text_of_a_message_with_attachments(monkeypatch, content, text, attached):
    """A message with an image still reaches the router with its words (and any routing tag)."""
    from types import SimpleNamespace
    from agent.voice_turn_route import _plugin_route
    calls = _install(monkeypatch, lambda **_: None)
    agent = SimpleNamespace(session_id="s", platform="desktop", provider="p", model="m")
    _plugin_route(agent, [{"role": "user", "content": content}])
    assert (calls[0]["user_message"], calls[0]["has_attachments"]) == (text, attached)
