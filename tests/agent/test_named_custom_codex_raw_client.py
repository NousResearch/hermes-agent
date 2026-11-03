"""A named custom ``codex_responses`` entry hands raw_codex callers the SDK client, not the aux wrapper."""

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Write a minimal config so load_config doesn't fail
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import hermes_yaml as yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict))


class TestNamedCustomCodexRawClient:
    """A ``transport: codex_responses`` entry with ``key_cmd`` must hand raw_codex callers the SDK
    client itself: the aux wrapper copies ``real_client.api_key`` — empty for a rotating provider —
    and hides ``_api_key_provider``, so the fallback swap installed an unauthenticated client."""

    _CFG = {
        "model": {"default": "test-model"},
        "providers": {
            "qlb-codex": {
                "api": "http://127.0.0.1:47391/v1",
                "transport": "codex_responses",
                "key_cmd": "printf minted-token",
                "default_model": "gpt-5.6-sol",
            },
        },
    }

    def test_raw_codex_returns_sdk_client_with_key_provider(self, tmp_path):
        _write_config(tmp_path, self._CFG)
        from agent import auxiliary_client as ac
        from openai import OpenAI

        client, model = ac.resolve_provider_client(
            "custom:qlb-codex", model="gpt-5.6-sol", raw_codex=True, api_mode="codex_responses")
        assert isinstance(client, OpenAI)
        assert model == "gpt-5.6-sol"
        assert callable(vars(client).get("_api_key_provider"))

    def test_aux_callers_still_get_the_wrapper(self, tmp_path):
        _write_config(tmp_path, self._CFG)
        from agent import auxiliary_client as ac

        client, _ = ac.resolve_provider_client(
            "custom:qlb-codex", model="gpt-5.6-sol", api_mode="codex_responses")
        assert isinstance(client, ac.CodexAuxiliaryClient)
