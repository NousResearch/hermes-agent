"""``api_mode_for_model`` answering None keeps the profile's declared ``api_mode`` at runtime resolution
(``providers/base.py`` contract), so a Responses-default provider with a few chat-only exceptions
never sends its other models through Chat Completions."""

import pytest


@pytest.fixture
def responses_default(monkeypatch):
    import hermes_cli.auth
    import providers
    from providers.base import ProviderProfile

    class ResponsesDefault(ProviderProfile):
        def api_mode_for_model(self, model, *, api_key=None):
            if model.startswith("broken"):
                raise RuntimeError("live catalog unavailable")
            return "chat_completions" if model.startswith("claude") else None

    providers.list_providers()  # discovery first, so the fake survives it
    providers.register_provider(ResponsesDefault(
        name="respdefault", base_url="https://api.respdefault.test/v1", api_mode="codex_responses",
        env_vars=("RESPDEFAULT_API_KEY",)))
    monkeypatch.setenv("RESPDEFAULT_API_KEY", "sk-fake")
    yield
    providers._REGISTRY.pop("respdefault", None)
    providers._SOURCES.pop("respdefault", None)
    providers._PROVIDER_LIST_CACHE = None
    hermes_cli.auth.PROVIDER_REGISTRY.pop("respdefault", None)


@pytest.mark.parametrize(("model", "api_mode"), [
    ("gpt-x", "codex_responses"),        # hook None -> the declared default
    ("claude-y", "chat_completions"),    # the hook's per-model exception
    ("broken-z", "codex_responses"),     # a raising hook degrades to the declared default
])
def test_hook_none_keeps_the_profiles_declared_api_mode(responses_default, model, api_mode):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    assert resolve_runtime_provider(requested="respdefault", target_model=model)["api_mode"] == api_mode


def test_a_model_pick_with_a_raising_hook_gets_the_wire_resolution_gave(responses_default):
    """``/model`` and session resume re-derive the wire through ``model_derived_api_mode``: a raising hook
    degrades there exactly as in runtime resolution instead of failing the switch."""
    from hermes_cli.model_switch import model_derived_api_mode
    from hermes_cli.runtime_provider import resolve_runtime_provider

    resolved = resolve_runtime_provider(requested="respdefault", target_model="broken-z")["api_mode"]
    assert model_derived_api_mode("respdefault", "broken-z", "sk-fake") == resolved == "codex_responses"
