"""Gateway provider-override credential resolution keys off the override's MODEL (#112600).

Channel overrides, persisted ``/model`` switches and API-server provider refreshes all resolve
credentials through ``gateway.run._resolve_runtime_agent_kwargs_for_provider``; without the model
the ladder keys off config's ``default`` and a ``*-free`` default decides the api_mode/base_url
for a Go-only model ("Model ... is not supported")."""

import pytest


@pytest.fixture()
def _zen_free_default_home(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: mimo-v2.5-free\n  provider: opencode\n  base_url: https://opencode.ai/zen/v1\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test-go")


def test_provider_override_runtime_uses_the_override_model(_zen_free_default_home):
    from gateway.run import _resolve_runtime_agent_kwargs_for_provider

    runtime = _resolve_runtime_agent_kwargs_for_provider("opencode-go", target_model="mimo-v2.5")
    assert runtime["provider"] == "opencode-go"
    assert runtime["base_url"] == "https://opencode.ai/zen/go/v1"


def test_fallback_chain_runtime_uses_the_entry_model(_zen_free_default_home, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"fallback_model": [{"provider": "opencode-go", "model": "mimo-v2.5"}]})
    fb = gateway_run._try_resolve_fallback_provider()
    assert fb is not None
    assert fb["model"] == "mimo-v2.5"
    assert fb["base_url"] == "https://opencode.ai/zen/go/v1"
