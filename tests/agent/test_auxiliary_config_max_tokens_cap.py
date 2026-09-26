"""Config-driven summary wire cap for local OpenAI-wire auxiliary routes.

``auxiliary.max_tokens_forward_providers`` + ``auxiliary.summary_max_tokens_ceiling``
opt a compression route into a hard ``max_tokens`` on the summary call. Backwards
compatible: without both keys the official no-cap policy is untouched.
"""

import pytest

from agent.auxiliary_client import _compression_route_max_tokens_ceiling, _configured_aux_string_list


@pytest.fixture
def aux_config(monkeypatch):
    def _patch(cfg: dict):
        def fake_load():
            return cfg

        import hermes_cli.config

        monkeypatch.setattr(hermes_cli.config, "load_config_readonly", fake_load)

    return _patch


def test_ceiling_applies_to_configured_provider(aux_config):
    aux_config({"auxiliary": {
        "max_tokens_forward_providers": ["my-local"],
        "summary_max_tokens_ceiling": 4000,
        "compression": {"provider": "my-local", "model": "local"},
    }})
    assert _compression_route_max_tokens_ceiling() == (4000, "my-local")


def test_provider_names_are_normalized(aux_config):
    aux_config({"auxiliary": {
        "max_tokens_forward_providers": ["MY-LOCAL"],
        "summary_max_tokens_ceiling": 4000,
        "compression": {"provider": "my-local"},
    }})
    assert _compression_route_max_tokens_ceiling() == (4000, "my-local")


def test_provider_not_listed_yields_none(aux_config):
    aux_config({"auxiliary": {
        "max_tokens_forward_providers": ["other-local"],
        "summary_max_tokens_ceiling": 4000,
        "compression": {"provider": "my-local"},
    }})
    assert _compression_route_max_tokens_ceiling() is None


def test_fallback_chain_disables_cap(aux_config):
    aux_config({"auxiliary": {
        "max_tokens_forward_providers": ["my-local"],
        "summary_max_tokens_ceiling": 4000,
        "compression": {"provider": "my-local", "fallback_chain": [{"provider": "openai"}]},
    }})
    assert _compression_route_max_tokens_ceiling() is None


def test_empty_provider_list_disables_cap(aux_config):
    aux_config({"auxiliary": {
        "summary_max_tokens_ceiling": 4000,
        "compression": {"provider": "my-local"},
    }})
    assert _compression_route_max_tokens_ceiling() is None


def test_nonpositive_ceiling_disables_cap(aux_config):
    aux_config({"auxiliary": {
        "max_tokens_forward_providers": ["my-local"],
        "summary_max_tokens_ceiling": 0,
        "compression": {"provider": "my-local"},
    }})
    assert _compression_route_max_tokens_ceiling() is None


def test_missing_config_disables_cap(aux_config):
    aux_config({})
    assert _compression_route_max_tokens_ceiling() is None


def test_configured_aux_string_list_normalizes(aux_config):
    aux_config({"auxiliary": {"max_tokens_forward_providers": [" MY-LOCAL ", "other-local"]}})
    assert _configured_aux_string_list("max_tokens_forward_providers") == ["my-local", "other-local"]


def test_forwards_max_tokens_extends_to_configured_provider(aux_config):
    aux_config({"auxiliary": {
        "max_tokens_forward_providers": ["my-local"],
        "compression": {"provider": "my-local"},
    }})
    from agent.auxiliary_client import _forwards_max_tokens

    assert _forwards_max_tokens("my-local", "my-local", "m", "", "compression") is True
    assert _forwards_max_tokens("other-local", "other-local", "m", "", "compression") is False