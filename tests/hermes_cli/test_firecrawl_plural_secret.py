"""Plural Firecrawl credentials participate in the existing secret configuration boundary."""
import pytest


@pytest.mark.parametrize("nested_value", ['["opaque-one", "opaque-two"]', ["opaque-one", "opaque-two"]])
def test_plural_config_is_secret_env_not_yaml(tmp_path, monkeypatch, capsys, nested_value):
    from hermes_cli import config
    from tools.environments.local_env_policy import _build_provider_env_blocklist
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    value = '["opaque-one", "opaque-two"]'
    assert config._is_env_config_key("FIRECRAWL_API_KEYS")
    assert config._is_secret_config_key("FIRECRAWL_API_KEYS")
    assert config.OPTIONAL_ENV_VARS["FIRECRAWL_API_KEYS"]["password"]
    assert "FIRECRAWL_API_KEYS" in _build_provider_env_blocklist()
    config.set_config_value("FIRECRAWL_API_KEYS", value)
    from hermes_cli.nous_subscription import _has_managed_default_direct, _get_gateway_direct_credentials
    assert _has_managed_default_direct("web")
    assert _get_gateway_direct_credentials()["web"]
    assert config.get_env_value("FIRECRAWL_API_KEYS") == value
    assert "opaque-one" not in str(config.redact_config_value({"FIRECRAWL_API_KEYS": value}))
    assert "opaque-two" not in str(config.redact_config_value({"env": {"FIRECRAWL_API_KEYS": nested_value}}))
    assert "opaque-two" not in str(config.redact_config_value({"mcp.env.FIRECRAWL_API_KEYS": nested_value}))
    assert "opaque-two" not in capsys.readouterr().out
    assert not (tmp_path / "config.yaml").exists()
    config.get_config_value("FIRECRAWL_API_KEYS")
    output = capsys.readouterr().out
    assert "opaque-one" not in output and "opaque-two" not in output
    config.unset_config_value("FIRECRAWL_API_KEYS")
    assert config.get_env_value("FIRECRAWL_API_KEYS") is None
    from tools.environments.local import _sanitize_subprocess_env
    from tools.env_passthrough import register_env_passthrough, is_env_passthrough
    register_env_passthrough(["FIRECRAWL_API_KEYS"])
    assert not is_env_passthrough("FIRECRAWL_API_KEYS")
    assert "FIRECRAWL_API_KEYS" not in _sanitize_subprocess_env({"FIRECRAWL_API_KEYS": value})


@pytest.mark.parametrize("text", [
    '{"FIRECRAWL_API_KEYS": ["opaque-one", "opaque-two"]}',
    "{'FIRECRAWL_API_KEYS': ['opaque-one', 'opaque-two']}",
    'FIRECRAWL_API_KEYS=[\n  "opaque-one",\n  "opaque-two"\n]',
    'FIRECRAWL_API_KEYS=\'["opaque-one", "opaque-two"]\'',
    'FIRECRAWL_API_KEYS=["opaque-one", "opaque-two"]',
    'FIRECRAWL_API_KEYS="[\\"opaque-one\\", \\"opaque-two\\"]"',
    '{"FIRECRAWL_API_KEYS": "[\\"opaque-one\\", \\"opaque-two\\"]"}',
])
def test_plural_array_redaction_covers_every_element(text):
    from agent.redact import redact_sensitive_text
    result = redact_sensitive_text(text, force=True)
    assert "opaque-one" not in result
    assert "opaque-two" not in result
