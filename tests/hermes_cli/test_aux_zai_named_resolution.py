"""Named ZAI endpoints must survive auxiliary resolution without a per-task URL."""

import pytest


@pytest.mark.parametrize("style", ["legacy", "keyed"])
@pytest.mark.parametrize("credential", [{}, {"key_env": "ZAI_API_KEY"}, {"api_key": "test-zai-coding-key"}])
def test_named_zai_endpoint_uses_builtin_credentials(tmp_path, monkeypatch, style, credential):
    import hermes_yaml as yaml
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-coding-key")
    endpoint = "https://api.z.ai/api/coding/paas/v4"
    entry = {"base_url": endpoint, "api_key": "", "api_mode": "chat_completions", **credential}
    config = ({"custom_providers": [{"name": "zai", **entry}]} if style == "legacy"
              else {"providers": {"zai": entry}})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    client, model = resolve_provider_client("zai", "glm-4.5v")

    assert client is not None
    assert str(client.base_url).rstrip("/") == endpoint
    assert client.api_key == "test-zai-coding-key"
    assert model == "glm-4.5v"

    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider(requested="zai")
    assert runtime["base_url"] == endpoint
    assert runtime["api_key"] == "test-zai-coding-key"


@pytest.mark.parametrize("style", ["legacy", "keyed"])
def test_named_zai_explicit_overrides_and_tuning_only(tmp_path, monkeypatch, style):
    import hermes_yaml as yaml
    from agent.auxiliary_client import resolve_provider_client
    from hermes_cli.runtime_provider import _get_named_custom_provider

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-coding-key")
    entry = {"name": "zai", "api_key": "entry-key", "base_url": "https://api.z.ai/api/coding/paas/v4"}
    config = ({"custom_providers": [entry]} if style == "legacy" else {"providers": {"zai": entry}})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    client, _ = resolve_provider_client("zai", "glm-4.5v", explicit_base_url="https://override.test/v1",
                                        explicit_api_key="override-key")
    assert str(client.base_url).rstrip("/") == "https://override.test/v1"
    assert client.api_key == "override-key"

    del entry["base_url"]
    assert _get_named_custom_provider("zai", config=config) is None
