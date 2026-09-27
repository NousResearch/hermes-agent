"""A ``key_cmd`` Claude Code OAuth token keeps its OAuth identity on native api.anthropic.com routes (#114967).

A named custom provider pointed at ``api.anthropic.com`` with ``api_mode: anthropic_messages`` and a
``key_cmd`` callable used to be built with ``is_oauth=False`` / string-only OAuth detection, so the
request carried a bare Bearer without the Claude Code identity and Anthropic answered
``429 rate_limit_error: Error``. Third-party Anthropic-protocol hosts must stay non-OAuth.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.anthropic_credentials import anthropic_route_is_oauth

OAUTH = "sk-ant-oat01-keycmd-token"
CONSOLE = "sk-ant-api03-console-key"


def test_route_identity_requires_native_host_and_oauth_shape():
    calls = []

    def key_cmd():
        calls.append(1)
        return OAUTH

    assert anthropic_route_is_oauth("https://api.anthropic.com", key_cmd) is True
    assert anthropic_route_is_oauth("https://api.anthropic.com/v1", OAUTH) is True
    assert anthropic_route_is_oauth("", OAUTH) is True  # empty = native default route
    # provider=anthropic behind a proxy keeps today's behaviour (identity travels with the provider).
    assert anthropic_route_is_oauth("https://proxy.example.com", OAUTH, provider="anthropic") is True
    # Third-party hosts and Console keys never qualify; a callable that cannot mint is non-OAuth.
    assert anthropic_route_is_oauth("https://api.minimax.io/anthropic", key_cmd) is False
    assert anthropic_route_is_oauth("https://api.anthropic.com", CONSOLE) is False
    assert anthropic_route_is_oauth("https://notapi.anthropic.com.evil.example", key_cmd) is False
    assert anthropic_route_is_oauth("https://api.anthropic.com", lambda: (_ for _ in ()).throw(RuntimeError("no token"))) is False
    assert len(calls) == 1  # the third-party route never materialised the callable


def test_key_cmd_oauth_route_builds_claude_code_identity_on_main_and_aux_clients(monkeypatch):
    pytest.importorskip("anthropic")
    from agent.agent_init import _init_anthropic_client
    from agent.anthropic_adapter import _OAUTH_ONLY_BETAS, build_anthropic_client
    from agent.auxiliary_client import AnthropicAuxiliaryClient

    def key_cmd():
        return OAUTH

    # Main runtime: named custom provider -> api.anthropic.com with a callable token source.
    agent = SimpleNamespace(provider="enterprise-anthropic", model="claude-opus-5", quiet_mode=True)
    _init_anthropic_client(agent, key_cmd, "https://api.anthropic.com", None)
    assert agent._is_anthropic_oauth is True
    headers = agent._anthropic_client._custom_headers
    assert headers["user-agent"].startswith("claude-code/") and headers["x-app"] == "cli"
    assert all(beta in headers["anthropic-beta"] for beta in _OAUTH_ONLY_BETAS)

    # Same token source at a third-party Anthropic-protocol host: no identity, no OAuth betas.
    third_party = build_anthropic_client(key_cmd, "https://api.minimax.io/anthropic")
    tp_headers = third_party._custom_headers
    assert "user-agent" not in tp_headers and not any(beta in tp_headers.get("anthropic-beta", "") for beta in _OAUTH_ONLY_BETAS)

    # Aux: the custom anthropic_messages route (production entry) carries the flag from the same rule.
    import agent.auxiliary_client as aux
    monkeypatch.setattr(aux, "_resolve_custom_runtime", lambda: ("https://api.anthropic.com", key_cmd, "anthropic_messages"))
    monkeypatch.setattr(aux, "_read_main_model_for_aux", lambda: "claude-opus-5")
    native_aux, _model = aux._try_custom_endpoint()
    assert isinstance(native_aux, AnthropicAuxiliaryClient) and native_aux.chat.completions._is_oauth is True
    monkeypatch.setattr(aux, "_resolve_custom_runtime", lambda: ("https://api.minimax.io/anthropic", key_cmd, "anthropic_messages"))
    third_party_aux, _model = aux._try_custom_endpoint()
    assert isinstance(third_party_aux, AnthropicAuxiliaryClient) and third_party_aux.chat.completions._is_oauth is False
