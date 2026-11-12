"""Unreadable YAML must not remove native Anthropic's explicit redirect protection."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.client_lifecycle import resolve_primary_follow_redirects
from hermes_cli.config import atomic_config_replace, load_config_readonly
from hermes_cli.config_read_errors import FailedConfigRead


@pytest.mark.parametrize("policy", [True, False])
def test_static_anthropic_reconstruction_keeps_policy_after_broken_yaml(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    atomic_config_replace(config_path, {"providers": {"anthropic": {"follow_redirects": policy}}})
    agent = SimpleNamespace(provider="anthropic", model="claude-test", requested_provider="anthropic")
    base_url = "https://api.anthropic.com"
    assert resolve_primary_follow_redirects(agent, base_url) is policy
    config_path.write_text("providers: [\n", encoding="utf-8")
    assert isinstance(load_config_readonly(), FailedConfigRead)

    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": "https://foreign.test/v1/messages"})

    # Keep the real SDK, builders, request hooks and redirect handling; replace only
    # the wire transport so even an unguarded replay cannot reach the network.
    monkeypatch.setattr(httpx.Client, "_init_transport", lambda *args, **kwargs: httpx.MockTransport(respond))
    monkeypatch.setattr(httpx.Client, "_init_proxy_transport", lambda *args, **kwargs: httpx.MockTransport(respond))
    monkeypatch.setitem(anthropic.DefaultHttpxClient.__init__.__globals__, "HTTPTransport",
                        lambda **kwargs: httpx.MockTransport(respond))
    client = build_anthropic_client(
        "sk-ant-api03-private", base_url=base_url,
        follow_redirects=resolve_primary_follow_redirects(agent, base_url),
    )
    client.max_retries = 0
    try:
        assert client._client.follow_redirects is policy
        with pytest.raises(anthropic.APIConnectionError if policy else anthropic.APIStatusError) as error:
            client.messages.create(model=agent.model, max_tokens=1,
                                   messages=[{"role": "user", "content": "private prompt"}])
        if policy:
            assert isinstance(error.value.__cause__, httpx.LocalProtocolError)
            assert "Cross-origin" in str(error.value.__cause__)
        else:
            assert error.value.status_code == 307
    finally:
        client.close()
    assert len(seen) == 1
    assert str(seen[0].url) == base_url + "/v1/messages"
    assert seen[0].headers["x-api-key"] == "sk-ant-api03-private"
    assert b"private prompt" in seen[0].content
