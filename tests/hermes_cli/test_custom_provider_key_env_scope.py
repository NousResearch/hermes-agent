"""Custom-provider ``key_env`` reads in the CLI picker/catalog helpers go through the profile
secret scope (#67935): a key that lives only in the profile's ``.env`` must be found, and
another profile's process-env value must never be picked up under multiplexing."""

from unittest.mock import patch

import pytest

from agent import secret_scope


@pytest.fixture
def scoped_profile(tmp_path, monkeypatch):
    """Gateway shape: multiplex on, this profile's ``.env`` installed as the secret scope,
    the variable absent from (or different in) the process environment."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: old-model\ncustom_providers: []\n")
    (home / ".env").write_text("EXAMPLE_PROVIDER_API_KEY=sk-from-profile-dotenv\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("EXAMPLE_PROVIDER_API_KEY", "sk-other-profile-process-env")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        yield home
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_provider_config_key_env_resolves_through_secret_scope(scoped_profile):
    from hermes_cli.models_local import _api_key_from_provider_config

    entry = {"base_url": "https://ollama.internal/v1", "key_env": "EXAMPLE_PROVIDER_API_KEY"}
    assert _api_key_from_provider_config(entry, "key_env", "api_key_env") == "sk-from-profile-dotenv"


def test_named_custom_flow_probes_with_scoped_key_env(scoped_profile):
    from hermes_cli.model_setup_flows import _model_flow_named_custom

    provider_info = {
        "name": "Example Provider",
        "base_url": "https://api.example-provider.test/v1",
        "api_key": "",
        "key_env": "EXAMPLE_PROVIDER_API_KEY",
        "model": "qwen3.6-35b-fast",
    }
    with patch("hermes_cli.models.fetch_api_models", return_value=["qwen3.6-35b-fast"]) as mock_fetch, \
         patch("hermes_cli.curses_ui.curses_radiolist", side_effect=ImportError), \
         patch("builtins.input", return_value="1"), \
         patch("builtins.print"):
        _model_flow_named_custom({}, provider_info)

    assert mock_fetch.call_args.args[0] == "sk-from-profile-dotenv"
