"""``pre_auxiliary_call`` / ``post_auxiliary_call`` fire for auxiliary LLM calls (#79733).

Two invariants: (1) an auxiliary ``call_llm`` emits the pair with ``aux_task`` set and does NOT
fire the turn-scoped ``pre/post_api_request`` events; (2) a raising subscriber never breaks the
auxiliary call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.auxiliary_client import call_llm
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


@pytest.fixture
def manager(monkeypatch):
    mgr = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return PluginContext(PluginManifest(name="aux-observer", source="user"), mgr)


@pytest.fixture
def aux_client(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = SimpleNamespace(
        model="mock-model",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="A title", tool_calls=None),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model",
                        lambda *a, **k: ("openrouter", "mock-model", None, None, None))
    monkeypatch.setattr("agent.auxiliary_client._get_cached_client", lambda *a, **k: (client, "mock-model"))
    monkeypatch.setattr("agent.auxiliary_client._validate_llm_response", lambda resp, _task, **_kw: resp)
    return client


def test_call_llm_emits_auxiliary_events_not_api_request_events(manager, aux_client):  # manager: PluginContext
    fired = []
    for name in ("pre_auxiliary_call", "post_auxiliary_call", "pre_api_request", "post_api_request"):
        manager.register_hook(name, lambda _name=name, **kw: fired.append((_name, kw)))

    response = call_llm(task="title_generation", messages=[{"role": "user", "content": "hello"}])

    assert response is aux_client.chat.completions.create.return_value
    assert [name for name, _ in fired] == ["pre_auxiliary_call", "post_auxiliary_call"]
    pre, post = fired[0][1], fired[1][1]
    assert pre["aux_task"] == post["aux_task"] == "title_generation"
    assert pre["api_request_id"].startswith("aux-") and post["api_request_id"] == pre["api_request_id"]
    assert pre["provider"] == "openrouter" and pre["model"] == "mock-model"
    assert pre["request_messages"] == [{"role": "user", "content": "hello"}]
    assert pre["request"]["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert post["finish_reason"] == "stop" and post["error"] is None
    assert post["usage"]["input_tokens"] == 10 and post["usage"]["output_tokens"] == 5
    assert post["response"]["assistant_message"]["content"] == "A title"


def test_raising_subscriber_does_not_break_the_auxiliary_call(manager, aux_client):
    def boom(**_kw):
        raise RuntimeError("observer exploded")

    manager.register_hook("pre_auxiliary_call", boom)
    manager.register_hook("post_auxiliary_call", boom)

    response = call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])

    assert response is aux_client.chat.completions.create.return_value
