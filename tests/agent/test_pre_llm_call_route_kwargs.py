"""``pre_llm_call`` hooks receive the provider and base_url the turn will call.

``model`` alone is ambiguous: the same model id can be served by several providers
(direct, an aggregator, a local proxy), so a routing/cost/guard plugin needs both.
"""

from types import SimpleNamespace

from agent.turn_context import _collect_pre_llm_call_context


def test_pre_llm_call_gets_provider_and_base_url(monkeypatch):
    seen = {}

    def _hook(name, **kwargs):
        if name == "pre_llm_call":
            seen.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", _hook)
    agent = SimpleNamespace(session_id="s1", model="glm-5.3", provider="zai",
                            base_url="http://127.0.0.1:8793", platform="cli")
    _collect_pre_llm_call_context(agent, effective_task_id="t", turn_id="u", original_user_message="hi",
                                  messages=[], conversation_history=[])
    assert seen.get("provider") == "zai"
    assert seen.get("base_url") == "http://127.0.0.1:8793"
