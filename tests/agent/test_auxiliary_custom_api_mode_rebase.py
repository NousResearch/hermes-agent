"""Rebase integration: custom identity routing keeps upstream API-mode aliases."""
from unittest.mock import MagicMock


def test_explicit_custom_responses_alias_uses_responses_adapter(monkeypatch):
    from agent import auxiliary_client as aux
    import hermes_cli.runtime_provider as runtime

    entry = {"name": "relay", "provider_key": "relay", "base_url": "https://relay.example/v1", "api_key": "test-only", "model": "gpt-5.4-nano"}
    monkeypatch.setattr(runtime, "_get_named_custom_provider", lambda name: entry)
    monkeypatch.setattr(aux, "_named_custom_openai_wire_client", lambda *args: MagicMock())
    client, model = aux.resolve_provider_client("custom:relay", "gpt-5.4-nano", api_mode="responses")
    assert model == "gpt-5.4-nano"
    assert isinstance(client, aux.CodexAuxiliaryClient)
