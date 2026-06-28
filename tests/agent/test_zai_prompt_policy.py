from types import SimpleNamespace

from agent.zai_prompt_policy import apply_zai_special_prompt, is_zai_request


def test_zai_request_detects_provider_model_and_endpoint():
    assert is_zai_request(SimpleNamespace(provider="zai", model="glm-5.2", base_url=""))
    assert is_zai_request(SimpleNamespace(provider="custom", model="glm-5.2", base_url=""))
    assert is_zai_request(SimpleNamespace(provider="custom", model="x", base_url="https://api.z.ai/api/coding/paas/v4"))
    assert not is_zai_request(SimpleNamespace(provider="openai-codex", model="gpt-5.5", base_url=""))


def test_zai_special_prompt_strips_hermes_branding_without_mutating_input():
    agent = SimpleNamespace(provider="zai", model="glm-5.2", base_url="https://api.z.ai/api/coding/paas/v4")
    messages = [
        {"role": "system", "content": "You are Hermes Agent, an intelligent AI assistant created by Nous Research.\nYou run on Hermes Agent (by Nous Research).\nUse tools."},
        {"role": "user", "content": "hi"},
    ]
    out = apply_zai_special_prompt(agent, messages)

    assert out is not messages
    assert messages[0]["content"].startswith("You are Hermes Agent")
    system = out[0]["content"]
    assert system.startswith("You are a precise local AI coding and operations assistant.")
    assert "Hermes" not in system
    assert "Nous Research" not in system
    assert "Use tools." in system
    assert out[1] == {"role": "user", "content": "hi"}


def test_zai_special_prompt_inserts_system_when_missing():
    agent = SimpleNamespace(provider="zai", model="glm-5.2", base_url="")
    out = apply_zai_special_prompt(agent, [{"role": "user", "content": "hi"}])
    assert out[0]["role"] == "system"
    assert "local AI" in out[0]["content"]
    assert out[1]["role"] == "user"


def test_zai_subagent_uses_compact_child_prompt_instead_of_parent_system():
    agent = SimpleNamespace(
        provider="zai",
        model="glm-5.2",
        base_url="",
        platform="subagent",
        ephemeral_system_prompt="You are a focused subagent. YOUR TASK:\nReview operator controls.",
    )
    messages = [
        {"role": "system", "content": "You are Hermes Agent. SOUL <available_skills> huge parent prompt"},
        {"role": "user", "content": "Begin"},
    ]
    out = apply_zai_special_prompt(agent, messages)
    system = out[0]["content"]
    assert system.startswith("You are a precise local AI coding and operations assistant.")
    assert "Review operator controls" in system
    assert "SOUL" not in system
    assert "available_skills" not in system
    assert "Hermes" not in system
    assert out[1:] == [{"role": "user", "content": "Begin"}]


def test_non_zai_request_is_unchanged():
    agent = SimpleNamespace(provider="openai-codex", model="gpt-5.5", base_url="")
    messages = [{"role": "system", "content": "You are Hermes Agent."}]
    assert apply_zai_special_prompt(agent, messages) is messages
