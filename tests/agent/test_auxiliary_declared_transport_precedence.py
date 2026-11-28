"""Regression: named provider's declared transport must win over a stale task api_mode.

Scenario (a real deployment): the main model runs on codex_responses, and an
auxiliary task is pinned to a named custom provider declaring
``api_mode: chat_completions``. A stale ``api_mode=codex_responses`` left over from
the main session must not wrap that client in the Responses adapter — doing so
sends Chat Completions requests to a /responses wire and fails.

Before the fix ``entry_api_mode = req.api_mode or entry.api_mode`` let the stale
task mode win, and there was no explicit chat_completions arm (the else fell into
``_wrap_transport``, which re-read the stale mode). Both are fixed here.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import hermes_yaml as yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.safe_dump(config_dict))


class _SentinelClient:
    """Stands in for the raw OpenAI-wire client so tests never build a real SDK client."""
    base_url = "http://sentinel.local/v1"
    api_key = "sentinel"


@pytest.fixture(autouse=True)
def _no_real_client(monkeypatch):
    # Avoid constructing a real OpenAI client (keeps tests hermetic and fast).
    import agent.auxiliary_client as ac
    monkeypatch.setattr(ac, "_named_custom_openai_wire_client",
                        lambda *a, **k: _SentinelClient())


class TestDeclaredTransportBeatsStaleTaskMode:
    def test_chat_completions_entry_not_wrapped_as_responses(self, tmp_path):
        from agent.auxiliary_client import resolve_provider_client, CodexAuxiliaryClient
        _write_config(tmp_path, {
            "model": {"provider": "custom:chatgw", "default": "aux-model"},
            "providers": {"chatgw": {"name": "chatgw", "base_url": "http://chatgw.local/v1",
                                     "api_key": "sk-test", "api_mode": "chat_completions",
                                     "default_model": "aux-model"}},
        })
        # Stale codex_responses inherited from the main session must not win.
        client, model = resolve_provider_client("chatgw", "aux-model", api_mode="codex_responses")
        assert client is not None
        assert isinstance(client, _SentinelClient), (
            "declared chat_completions transport was wrongly wrapped as "
            + type(client).__name__
        )
        assert not isinstance(client, CodexAuxiliaryClient)
        assert model == "aux-model"

    def test_declared_codex_responses_still_wrapped(self, tmp_path):
        # A provider that genuinely declares codex_responses must still get the adapter,
        # even when the task carries a stale chat_completions mode.
        from agent.auxiliary_client import resolve_provider_client, CodexAuxiliaryClient
        _write_config(tmp_path, {
            "model": {"provider": "custom:codexgw", "default": "aux-model"},
            "providers": {"codexgw": {"name": "codexgw", "base_url": "https://codexgw.local/v1",
                                      "api_key": "sk-test", "api_mode": "codex_responses",
                                      "default_model": "aux-model"}},
        })
        client, _model = resolve_provider_client("codexgw", "aux-model", api_mode="chat_completions")
        assert isinstance(client, CodexAuxiliaryClient), (
            "declared codex_responses transport must still produce the Responses adapter"
        )
