"""A subscription primary at its limit: the switch notice names the per-token bill and the reset,
and opt-in ``fallback.on_subscription_limit`` (switch | wait | ask) decides whether to switch.
Real config.yaml in the sandboxed HERMES_HOME and the real fallback resolver (an OpenRouter entry
with its own key: building the client never touches the network)."""
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason
from hermes_constants import get_hermes_home
from run_agent import AIAgent

NOW = 1_700_000_000
RESET = NOW + 3600
METERED = {"provider": "openrouter", "model": "openai/gpt-5", "base_url": "https://openrouter.ai/api/v1",
           "api_key": "sk-or-v1-test"}
SAME_BACKEND = {"provider": "openai-codex", "model": "gpt-5.5"}
BROKEN = {"provider": "openrouter", "model": "openai/gpt-5-broken", "base_url": "https://openrouter.ai/api/v1",
          "api_key": "sk-or-v1-test"}


def _codex_agent(clarify=None, chain=(METERED,)):
    status = []
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="codex-oauth", provider="openai-codex", base_url="https://chatgpt.com/backend-api/codex",
            model="gpt-5.5", quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=list(chain), clarify_callback=clarify,
            status_callback=lambda kind, msg=None: status.append(str(msg)),
        )
    agent._test_status = status
    agent.client = None
    return agent


def _switch(agent, policy):
    (get_hermes_home() / "config.yaml").write_text(f"fallback:\n  on_subscription_limit: {policy}\n", encoding="utf-8")
    with patch("agent.fallback_cooldown.time.time", return_value=NOW):
        return agent._try_activate_fallback(reason=FailoverReason.rate_limit, reset_at=RESET)


def test_subscription_to_metered_switch_says_billed_per_token_and_reset_time():
    agent = _codex_agent()
    assert _switch(agent, "switch") is True
    assert agent.model == "openai/gpt-5"
    notice = " ".join(agent._test_status)
    assert "gpt-5.5 via openai-codex limit reached (resets " in notice
    assert "Continuing on openai/gpt-5 via openrouter, billed per token." in notice
    assert "Back on gpt-5.5 at the first turn after the reset." in notice


@pytest.mark.parametrize(("policy", "answer", "switched", "asked"), [
    ("wait", None, False, 0),
    ("ask", 0, True, 1),
    ("ask", 1, False, 1),
])
@pytest.mark.parametrize("chain", [(METERED,), (SAME_BACKEND, METERED), (BROKEN, METERED)],
                         ids=["metered-first", "skipped-entry-first", "broken-entry-first"])
def test_on_subscription_limit_policy_gates_the_metered_switch(policy, answer, switched, asked, chain):
    calls = []

    def clarify(questions):
        calls.append(questions)
        return {"answers": {"q0": questions[0]["choices_offered"][answer]}, "outcome": "submitted"}

    agent = _codex_agent(clarify, chain)
    from agent.route_binding import resolve_route_entry_client as real_resolve

    def resolve(entry, provider, model, **kw):  # an entry whose client cannot be built
        if model.endswith("-broken"):
            raise ValueError("malformed endpoint")
        return real_resolve(entry, provider, model, **kw)

    with patch("agent.fallback_billing.resolve_route_entry_client", resolve):
        assert _switch(agent, policy) is switched
    assert (agent.model == "openai/gpt-5") is switched
    assert len(calls) == asked
    if not switched:  # still on the primary, told why; the retry loop's next walk does not re-ask
        assert agent._fallback_index == 0
        assert any(f"on_subscription_limit is {policy}" in line for line in agent._test_status)
        with patch("agent.fallback_billing.resolve_route_entry_client", resolve):
            assert _switch(agent, policy) is False
        assert len(calls) == asked
