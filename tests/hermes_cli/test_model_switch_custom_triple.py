"""``/model custom:<name>:<model>`` routes to the named custom provider (#9147).

``parse_model_input()`` decodes the triple, but PATH B of ``switch_model()`` never called it: step c
early-returns while the current provider is custom and step e is gated on ``not is_custom``, so the
whole string stayed the model on the OLD provider. Kept out of
``test_model_switch_custom_providers.py``, which is over the code-health line cap.
"""

import pytest

import hermes_yaml as yaml
from hermes_cli.model_switch import switch_model

_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


@pytest.fixture(autouse=True)
def _disable_live_custom_provider_model_probe(monkeypatch):
    """Keep the switch hermetic: no live custom-endpoint or catalog probes (as in the sibling module)."""
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_kw: None)
    monkeypatch.setattr("hermes_cli.models.cached_provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models.provider_model_ids", lambda *_a, **_kw: [])
    monkeypatch.setattr("hermes_cli.models_local.fetch_ollama_local_models", lambda *_a, **_kw: None)


@pytest.fixture
def two_custom_providers(monkeypatch, tmp_path):
    """A real temp HERMES_HOME declaring two named custom providers.

    ``parse_model_input()`` reads the configured custom-provider ids from config.yaml, so the
    triple syntax only decodes against a real file — passing ``custom_providers=`` to
    ``switch_model`` is not enough.
    """
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    providers = [
        {"name": "cch_openai", "base_url": "http://localhost:23000/v1", "api_mode": "chat_completions"},
        {"name": "cch_anthropic", "base_url": "http://localhost:23000", "api_mode": "anthropic_messages"},
    ]
    (home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"provider": "custom:cch_openai", "name": "foo"},
                        "custom_providers": providers}),
        encoding="utf-8")
    return providers


@pytest.fixture
def _stub_switch_tail(monkeypatch):
    """Credential resolution + validation + metadata stubbed; routing is what is under test."""
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {"api_key": "k", "base_url": "http://localhost:23000", "api_mode": "chat_completions"})
    monkeypatch.setattr(
        "hermes_cli.models_validate.validate_requested_model", lambda *a, **k: _MOCK_VALIDATION)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)


@pytest.mark.parametrize(
    "current_provider, current_base_url",
    [("custom:cch_openai", "http://localhost:23000/v1"), ("openai", "")],
    ids=["from-another-custom-provider", "from-a-native-provider"])
def test_custom_triple_syntax_routes_to_the_named_provider(
        two_custom_providers, _stub_switch_tail, current_provider, current_base_url):
    """#9147: ``/model custom:<name>:<model>`` must route to that custom provider.

    ``switch_model`` never consulted ``parse_model_input()``, which already decodes the triple.
    From another custom provider, step c (``_convert_vendor_colon_slug``) early-returns and step
    e is gated on ``not is_custom``, so the whole string stayed the model on the OLD provider;
    from a native provider step c mangled it into the slug ``custom/cch_anthropic:glm-5-turbo``.
    """
    result = switch_model(
        raw_input="custom:cch_anthropic:glm-5-turbo",
        current_provider=current_provider,
        current_model="foo",
        current_base_url=current_base_url,
        user_providers={},
        custom_providers=two_custom_providers)

    assert result.success is True
    assert result.target_provider == "custom:cch_anthropic"
    assert result.new_model == "glm-5-turbo"


def test_custom_triple_syntax_does_not_hijack_a_same_provider_switch(
        two_custom_providers, _stub_switch_tail):
    """The new step only fires when the input names a DIFFERENT provider: a plain model name on
    a custom provider still switches the model in place."""
    result = switch_model(
        raw_input="glm-5-turbo",
        current_provider="custom:cch_openai",
        current_model="foo",
        current_base_url="http://localhost:23000/v1",
        user_providers={},
        custom_providers=two_custom_providers)

    assert result.success is True
    assert result.target_provider == "custom:cch_openai"
    assert result.new_model == "glm-5-turbo"
