"""Custom OpenCode-family provider runtime resolution (#85589).

A user-defined provider whose name extends an OpenCode family slug
(``opencode-go-bridge``) — or whose base_url is hosted on opencode.ai —
must get the same per-model api_mode derivation and /v1 base-url
normalization as the built-in ``opencode-go`` / ``opencode-zen``
providers. Before the fix, such providers resolved with a static
api_mode (chat_completions), so /v1/responses-only models like
``grok-4.5`` failed with HTTP 503 "Endpoint unavailable".
"""

import os
import tempfile

import pytest


@pytest.fixture()
def _bridge_env(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("OPENCODE_GO_BRIDGE_API_KEY", "sk-test-bridge")

    def write_config(provider_name: str, extra: str = "", default: str = "grok-4.5"):
        (hermes_home / "config.yaml").write_text(
            f"""
model:
  provider: {provider_name}
  default: {default}
providers:
  {provider_name}:
    base_url: https://opencode.ai/zen/go/v1
    key_env: OPENCODE_GO_BRIDGE_API_KEY
{extra}"""
        )

    return write_config


def _resolve(provider, model):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(requested=provider, target_model=model)


class TestCustomOpencodeFamilyRuntime:
    def test_family_provider_routes_grok_to_responses(self, _bridge_env):
        """The #85589 repro: grok-4.5 on opencode-go-bridge → codex_responses."""
        _bridge_env("opencode-go-bridge")
        r = _resolve("opencode-go-bridge", "grok-4.5")
        assert r["api_mode"] == "codex_responses"
        assert r["base_url"].rstrip("/").endswith("/zen/go/v1")

    def test_family_provider_routes_minimax_to_messages_and_strips_v1(self, _bridge_env):
        _bridge_env("opencode-go-bridge")
        r = _resolve("opencode-go-bridge", "minimax-m2.7")
        assert r["api_mode"] == "anthropic_messages"
        # /v1 stripped so the Anthropic SDK's own /v1/messages doesn't double up
        assert r["base_url"].rstrip("/").endswith("/zen/go")

    def test_family_provider_keeps_chat_models_on_v1(self, _bridge_env):
        _bridge_env("opencode-go-bridge")
        r = _resolve("opencode-go-bridge", "deepseek-v4-flash")
        assert r["api_mode"] == "chat_completions"
        assert r["base_url"].rstrip("/").endswith("/zen/go/v1")

    def test_explicit_transport_in_config_wins(self, _bridge_env):
        """A user-declared transport is authoritative — no family override."""
        _bridge_env("opencode-go-bridge", extra="    transport: chat_completions\n")
        r = _resolve("opencode-go-bridge", "grok-4.5")
        assert r["api_mode"] == "chat_completions"

    def test_arbitrary_name_on_opencode_host_derives_family(self, _bridge_env):
        """A provider with a non-family name pointing at opencode.ai still
        gets per-model routing (host match)."""
        _bridge_env("my-oc-proxy", default="gpt-5.6-luna")
        r = _resolve("my-oc-proxy", "gpt-5.6-luna")
        assert r["api_mode"] == "codex_responses"
