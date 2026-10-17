"""The CommandCode catalog collision must not discard its registered API contract."""

from types import SimpleNamespace

import pytest

from hermes_cli import providers as catalog
from providers import get_provider_profile


@pytest.mark.parametrize("name", ["commandcode", "commandcode-chat"])
def test_registered_commandcode_contract_survives_catalog_collision(name, monkeypatch):
    """Resolve the real profile, not a mock dialect or the unrelated CMD_API_KEY entry."""
    profile = get_provider_profile(name)
    assert profile is not None
    collision = SimpleNamespace(name="Catalog CommandCode", env=("CMD_API_KEY",),
                                api="https://catalog.example.test/v1", doc="")
    monkeypatch.setattr(catalog, "_models_dev_info", lambda *_args, **_kwargs: collision)

    resolved = catalog.resolve_provider_full(name, {}, [])

    assert resolved is not None
    assert resolved.id == profile.name == "commandcode"
    assert resolved.source == "plugin-profile"
    assert resolved.transport == "openai_chat"
    assert resolved.base_url == profile.base_url
    assert resolved.api_key_env_vars == ("COMMANDCODE_API_KEY",)
    assert resolved.base_url_env_var == "COMMANDCODE_BASE_URL"
    assert catalog.determine_api_mode(name) == profile.api_mode == "chat_completions"


def test_commandcode_explicit_user_endpoint_still_wins(monkeypatch):
    """The compatibility exception must not outrank an explicitly configured endpoint."""
    monkeypatch.setattr(catalog, "_models_dev_info", lambda *_args, **_kwargs: None)
    configured = {"commandcode": {"base_url": "https://configured.example.test/v1", "key_env": "USER_API_KEY"}}

    resolved = catalog.resolve_provider_full("commandcode-chat", configured, [])

    assert resolved is not None
    assert resolved.source == "user-config"
    assert resolved.id == "commandcode"
    assert resolved.base_url == configured["commandcode"]["base_url"]
    assert resolved.api_key_env_vars == ("USER_API_KEY",)
