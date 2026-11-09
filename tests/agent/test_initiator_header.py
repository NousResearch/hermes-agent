"""Opt-in ``x-initiator`` for custom providers (``initiator_header: true``)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent import auxiliary_client as aux
from hermes_constants import get_hermes_home

_BASE = "http://127.0.0.1:4000/v1"
_BACKUP = "http://127.0.0.1:4001/v1"


def _reply(tool: bool) -> SimpleNamespace:
    calls = [SimpleNamespace(id="c1", type="function", function=SimpleNamespace(
        name="todo", arguments=json.dumps({"todos": []})))] if tool else None
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="" if tool else "done", tool_calls=calls),
        finish_reason="tool_calls" if tool else "stop")], model="router-model", usage=None)


def _agent(opted_in: bool):
    """An agent on the ``router`` provider; ``backup`` (always opted in) is there to fall back to."""
    from run_agent import AIAgent

    (get_hermes_home() / "config.yaml").write_text(
        f"providers:\n  router:\n    api: {_BASE}\n    api_key: k\n"
        + ("    initiator_header: true\n" if opted_in else "")
        + f"  backup:\n    api: {_BACKUP}\n    api_key: k\n    initiator_header: true\n", encoding="utf-8")
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(model="router-model", api_key="k", base_url=_BASE, enabled_toolsets=["todo"],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=5)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Use the tools."
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _sent(agent, replies, on_call=None) -> list:
    seen = []

    def create(**kwargs):
        seen.append((kwargs.get("extra_headers") or {}).get("x-initiator"))
        if on_call:
            on_call(len(seen))
        return replies.pop(0)

    agent.client.chat.completions.create.side_effect = create
    return seen


def _auxiliary_initiator(session_id):
    token = aux.set_runtime_main("router", "router-model", base_url=_BASE, session_id=session_id)
    try:
        kwargs = aux._build_call_kwargs("router", "router-model", [{"role": "user", "content": "title"}], base_url=_BASE)
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)
    return (kwargs.get("extra_headers") or {}).get("x-initiator")


def test_opted_in_provider_marks_a_persons_calls_user_and_the_agents_own_calls_agent():
    agent = _agent(True)
    seen = _sent(agent, [_reply(True), _reply(True), _reply(False), _reply(False)],
                 on_call=lambda n: n == 1 and agent.steer("use the staging config instead"))
    agent.run_conversation("tidy the todo list")
    agent.run_conversation("[System note: resume the interrupted task]", persist_user_display_kind="auto_continue")
    assert seen == ["user", "user", "agent", "agent"]
    assert _auxiliary_initiator(agent.session_id) == "agent"


def test_provider_without_opt_in_gets_no_initiator_header():
    agent = _agent(False)
    seen = _sent(agent, [_reply(True), _reply(False)])
    agent.run_conversation("tidy the todo list")
    assert seen == [None, None]
    assert _auxiliary_initiator(agent.session_id) is None


def test_a_host_whose_provider_answers_request_headers_ignores_the_opt_in(request):
    import providers
    from providers.base import ProviderProfile

    class Owner(ProviderProfile):
        def request_headers(self, *, base_url, model="", user_initiated=False):
            return {}

    providers.list_providers()
    providers.register_provider(Owner(name="hostowner", base_url=_BASE, hostnames=("127.0.0.1",)))

    def _cleanup():
        providers._REGISTRY.pop("hostowner", None)
        providers._SOURCES.pop("hostowner", None)
        providers._PROVIDER_LIST_CACHE = None

    request.addfinalizer(_cleanup)
    agent = _agent(True)
    seen = _sent(agent, [_reply(True), _reply(False)])
    agent.run_conversation("tidy the todo list")
    assert seen == [None, None]
    assert _auxiliary_initiator(agent.session_id) is None


def test_a_rejected_reply_is_resent_as_user():
    agent = _agent(True)
    cut_off = _reply(False)
    cut_off.choices[0].message.content = "<REASONING_SCRATCHPAD>still thinking"
    seen = _sent(agent, [cut_off, _reply(False)])
    agent.run_conversation("tidy the todo list")
    assert seen == ["user", "user"]


@pytest.mark.parametrize("opted_in, router_replies, backup_replies, on_router, on_backup", [
    pytest.param(True, [], [True, False], {"user"}, ["user", "agent"], id="first-call-fails-over"),
    pytest.param(False, [True], [False], {None}, ["agent"], id="follow-up-fails-over-to-opted-in"),
])
def test_retries_and_fallback_resend_a_call_with_its_initiator(opted_in, router_replies, backup_replies,
                                                                on_router, on_backup):
    """Retries, then fallback (which resets the retry counter), are attempts at the same call."""
    agent = _agent(opted_in)
    replies = {_BASE: [_reply(t) for t in router_replies], _BACKUP: [_reply(t) for t in backup_replies]}
    seen = {_BASE: [], _BACKUP: []}

    def create(**kwargs):
        seen[agent.base_url].append((kwargs.get("extra_headers") or {}).get("x-initiator"))
        if not replies[agent.base_url]:
            raise openai.InternalServerError("unavailable", body=None, response=httpx.Response(
                500, request=httpx.Request("POST", agent.base_url)))
        return replies[agent.base_url].pop(0)

    def fall_back(**_):
        agent.base_url = _BACKUP
        return True

    agent.client.chat.completions.create.side_effect = create
    agent._fallback_chain, agent._fallback_index = [{"provider": "backup", "model": "router-model"}], 0
    with patch.object(agent, "_try_activate_fallback", side_effect=fall_back), \
            patch("agent.retry_utils.jittered_backoff", return_value=0.0):
        agent.run_conversation("tidy the todo list")
    assert set(seen[_BASE]) == on_router and len(seen[_BASE]) > 1 + len(router_replies)
    assert seen[_BACKUP] == on_backup
