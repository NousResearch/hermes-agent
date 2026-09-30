"""Free-first routing ladder (#125289).

A ``free: true`` entry at the head of ``fallback_providers`` declares a free-tier
ladder: everyday (default-configured) agents promote the first resolvable free
entry to primary and demote the configured primary to the LAST rung of the
runtime fallback chain. Pinned routes (session ``/model`` overrides, explicit
provider/model choices) are never silently re-routed to a free model.
"""

import pytest

from hermes_cli.fallback_config import entry_identity, entry_is_free, get_fallback_chain

CHAIN = [
    {"provider": "gemini", "model": "gemini-3.8-flash", "free": True},
    {"provider": "openrouter", "model": "llama-4-9b:free", "free": True},
    {"provider": "anthropic", "model": "claude-sonnet-4-6"},
]


class _FakeRoutedClient:
    api_key = "sk-test-free"
    base_url = "https://gemini.example/v1"


class _FakeAgent:
    provider = "qwen"
    model = "qwen/qwen3.7-flash"
    base_url = ""
    api_key = ""
    quiet_mode = False
    _client_kwargs = {}


@pytest.fixture()
def gemini_first(monkeypatch):
    """The router resolves the gemini free entry, nothing else."""
    def _resolve(provider, model=None, raw_codex=False, explicit_base_url="", explicit_api_key=None):
        if provider == "gemini":
            client = _FakeRoutedClient()
            client._hermes_aux_effective_provider = "gemini"
            return client, model
        return None, None

    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "resolve_provider_client", _resolve)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"model": {"default": "qwen/qwen3.7-flash", "provider": "qwen"}},
    )
    return _resolve


def _promote(agent, chain=CHAIN):
    from agent.agent_init import _promote_free_first_entry
    return _promote_free_first_entry(agent, chain, 120)


def test_entry_is_free_flag_parsing():
    assert entry_is_free({"free": True})
    assert entry_is_free({"free": "true"})
    assert entry_is_free({"free": "1"})
    assert not entry_is_free({"free": False})
    assert not entry_is_free({"free": "false"})
    assert not entry_is_free({"free": "0"})
    assert not entry_is_free({"free": None})
    assert not entry_is_free({})


def test_get_fallback_chain_preserves_free_flag():
    chain = get_fallback_chain({"fallback_providers": CHAIN})
    assert [entry_is_free(e) for e in chain] == [True, True, False]
    assert entry_identity(chain[0]) == ("gemini", "gemini-3.8-flash", "")


def test_free_head_promoted_to_primary(gemini_first):
    agent = _FakeAgent()
    kwargs = _promote(agent)
    assert kwargs is not None
    assert kwargs["api_key"] == "sk-test-free"
    assert (agent.provider, agent.model) == ("gemini", "gemini-3.8-flash")
    # Promotion is NOT a fallback activation: the free route is the primary.
    assert agent._fallback_activated is False
    assert agent._free_first_promoted is True


def test_configured_primary_becomes_last_rung(gemini_first):
    agent = _FakeAgent()
    _promote(agent)
    ladder = [(e["provider"], e["model"]) for e in agent._fallback_chain]
    # Free head removed from the ladder (it IS the primary now), paid entry kept in
    # order, configured primary appended as the final rung.
    assert ladder == [
        ("anthropic", "claude-sonnet-4-6"),
        ("qwen", "qwen/qwen3.7-flash"),
    ]


def test_promoted_ladder_survives_init_fallback_chain(gemini_first):
    from agent.agent_init import _init_fallback_chain
    agent = _FakeAgent()
    _promote(agent)
    _init_fallback_chain(agent, CHAIN)
    ladder = [(e["provider"], e["model"]) for e in agent._fallback_chain]
    assert ladder[-1] == ("qwen", "qwen/qwen3.7-flash")


def test_second_free_entry_used_when_first_unresolvable(monkeypatch, gemini_first):
    def _resolve(provider, model=None, raw_codex=False, explicit_base_url="", explicit_api_key=None):
        if provider == "openrouter":
            client = _FakeRoutedClient()
            client.base_url = "https://openrouter.example/v1"
            client._hermes_aux_effective_provider = "openrouter"
            return client, model
        return None, None

    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "resolve_provider_client", _resolve)
    agent = _FakeAgent()
    kwargs = _promote(agent)
    assert kwargs is not None
    assert agent.provider == "openrouter"


def test_pinned_model_never_promoted(gemini_first):
    agent = _FakeAgent()
    agent.model = "gpt-5.4"
    assert _promote(agent) is None
    assert (agent.provider, agent.model) == ("qwen", "gpt-5.4")


def test_pinned_provider_never_promoted(gemini_first):
    agent = _FakeAgent()
    agent.provider = "openai"
    assert _promote(agent) is None
    assert agent.provider == "openai"


def test_unreadable_config_never_promotes_possible_pin(gemini_first, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: (_ for _ in ()).throw(RuntimeError("unreadable")))
    agent = _FakeAgent()
    assert _promote(agent) is None
    assert (agent.provider, agent.model) == ("qwen", "qwen/qwen3.7-flash")


def test_auto_provider_is_not_a_pin(gemini_first):
    agent = _FakeAgent()
    agent.provider = "auto"
    assert _promote(agent) is not None
    assert agent.provider == "gemini"


def test_no_free_entries_no_promotion(gemini_first):
    agent = _FakeAgent()
    assert _promote(agent, [{"provider": "anthropic", "model": "claude-sonnet-4-6"}]) is None


def test_quiet_mode_no_promotion(gemini_first):
    agent = _FakeAgent()
    agent.quiet_mode = True
    assert _promote(agent) is None


def test_all_free_chain_ladder_is_primary_only(gemini_first):
    agent = _FakeAgent()
    assert _promote(agent, [{"provider": "gemini", "model": "gemini-3.8-flash", "free": True}]) is not None
    ladder = [(e["provider"], e["model"]) for e in agent._fallback_chain]
    assert ladder == [("qwen", "qwen/qwen3.7-flash")]
