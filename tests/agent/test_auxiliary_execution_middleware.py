"""``auxiliary_execution`` middleware gates auxiliary LLM calls without touching ``llm_execution``.

Behaviour contracts: a refusal replaces the provider answer and the provider is never reached; a
pass-through changes nothing; with no plugin the funnel is untouched; a raising callback is
fail-open; ``llm_execution`` plugins never see auxiliary traffic.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.auxiliary_client import async_call_llm, call_llm
from agent.auxiliary_hooks import AuxiliaryCallRefused
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

MESSAGES = [{"role": "user", "content": "hello"}]


def _response(text, **extra):
    return SimpleNamespace(
        model="mock-model",
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=text, tool_calls=None),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        **extra,
    )


def _refusal(text="refused"):
    return _response(text, auxiliary_refused=True)


@pytest.fixture
def ctx(monkeypatch):
    mgr = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    monkeypatch.setattr(plugins_mod, "_plugin_managers_by_home", {})
    return PluginContext(PluginManifest(name="aux-gate", source="user"), mgr)


@pytest.fixture
def aux_client(monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    client.chat.completions.create.return_value = _response("A title")
    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model",
                        lambda *a, **k: ("openrouter", "mock-model", None, None, None))
    monkeypatch.setattr("agent.auxiliary_client._get_cached_client", lambda *a, **k: (client, "mock-model"))
    monkeypatch.setattr("agent.auxiliary_client._validate_llm_response", lambda resp, _task, **_kw: resp)
    return client


def test_refusal_raises_instead_of_returning_text_as_a_result(ctx, aux_client):
    ctx.register_middleware("auxiliary_execution", lambda **kw: _refusal("usage wall"))

    with pytest.raises(AuxiliaryCallRefused, match="usage wall"):
        call_llm(task="title_generation", messages=MESSAGES)
    aux_client.chat.completions.create.assert_not_called()


def test_replacement_without_the_marker_is_a_normal_result(ctx, aux_client):
    replacement = _response("cached answer")
    ctx.register_middleware("auxiliary_execution", lambda **kw: replacement)

    assert call_llm(task="title_generation", messages=MESSAGES) is replacement
    aux_client.chat.completions.create.assert_not_called()


def test_pass_through_returns_the_provider_answer(ctx, aux_client):
    ctx.register_middleware("auxiliary_execution", lambda **kw: kw["next_call"](kw["request"]))

    result = call_llm(task="title_generation", messages=MESSAGES)

    assert result is aux_client.chat.completions.create.return_value
    aux_client.chat.completions.create.assert_called_once()


def test_no_plugin_leaves_the_funnel_untouched(ctx, aux_client):
    assert call_llm(task="title_generation", messages=MESSAGES) is aux_client.chat.completions.create.return_value


def test_raising_middleware_is_fail_open(ctx, aux_client):
    def boom(**kw):
        raise RuntimeError("gate crashed")

    ctx.register_middleware("auxiliary_execution", boom)

    assert call_llm(task="title_generation", messages=MESSAGES) is aux_client.chat.completions.create.return_value
    aux_client.chat.completions.create.assert_called_once()


def test_payload_names_the_task_and_provider(ctx, aux_client):
    seen = {}

    def spy(**kw):
        seen.update(aux_task=kw["aux_task"], provider=kw["provider"], model=kw["model"])
        return kw["next_call"](kw["request"])

    ctx.register_middleware("auxiliary_execution", spy)
    call_llm(task="title_generation", messages=MESSAGES)

    assert seen["aux_task"] == "title_generation"
    assert seen["provider"] and seen["model"] == "mock-model"


def test_llm_execution_plugins_never_see_auxiliary_calls(ctx, aux_client):
    seen = []
    ctx.register_middleware("llm_execution", lambda **kw: seen.append(kw) or kw["next_call"](kw["request"]))

    call_llm(task="title_generation", messages=MESSAGES)

    assert seen == []


def test_post_hook_reports_the_response_the_chain_produced(ctx, aux_client):
    posted = []
    ctx.register_hook("post_auxiliary_call", lambda **kw: posted.append(kw))
    ctx.register_middleware("auxiliary_execution", lambda **kw: _response("cached answer"))

    call_llm(task="title_generation", messages=MESSAGES)

    assert len(posted) == 1 and posted[0]["error"] is None
    assert posted[0]["response"]["assistant_message"]["content"] == "cached answer"


def test_post_hook_reports_a_refusal_as_an_error(ctx, aux_client):
    posted = []
    ctx.register_hook("post_auxiliary_call", lambda **kw: posted.append(kw))
    ctx.register_middleware("auxiliary_execution", lambda **kw: _refusal())

    with pytest.raises(AuxiliaryCallRefused):
        call_llm(task="title_generation", messages=MESSAGES)

    assert len(posted) == 1 and posted[0]["error_type"] == "AuxiliaryCallRefused"


@pytest.mark.asyncio
async def test_async_twin_honours_a_refusal(ctx, monkeypatch):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    called = []

    async def create(**kw):
        called.append(kw)
        return _response("real")

    client.chat.completions.create = create
    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model",
                        lambda *a, **k: ("openrouter", "mock-model", None, None, None))
    monkeypatch.setattr("agent.auxiliary_client._get_cached_client", lambda *a, **k: (client, "mock-model"))
    monkeypatch.setattr("agent.auxiliary_client._validate_llm_response", lambda resp, _task, **_kw: resp)
    ctx.register_middleware("auxiliary_execution", lambda **kw: _refusal())

    with pytest.raises(AuxiliaryCallRefused):
        await async_call_llm(task="title_generation", messages=MESSAGES)
    assert called == []
