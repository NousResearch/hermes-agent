from types import SimpleNamespace

import pytest


def test_native_anthropic_wire_checks_final_route_before_opening_stream(tmp_path, monkeypatch):
    """A request-mutated denied model never reaches native ``messages.stream``."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "routing_policy:\n  enabled: true\n  deny:\n    models: [blocked-*]\n",
        encoding="utf-8",
    )

    from agent import chat_completion_helpers as helpers
    from agent import relay_llm
    from hermes_cli.routing_policy import RoutingPolicyError

    class Messages:
        def __init__(self):
            self.calls = []

        def stream(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("native Anthropic stream must not open")

    messages = Messages()
    client = SimpleNamespace(base_url="https://api.anthropic.example/v1", messages=messages)
    agent = SimpleNamespace(
        provider="anthropic",
        model="allowed-model",
        base_url="https://api.anthropic.example/v1",
        log_prefix="",
    )
    call = helpers._StreamingCall(agent, {"model": "blocked-wire-model"}, None)
    call._new_diag = lambda: {}
    monkeypatch.setattr(helpers, "_relay_stream_identity", lambda *_args: {})
    monkeypatch.setattr(helpers, "_relay_stream_metadata", lambda *_args: {})

    def invoke_opener(_request, opener, **_kwargs):
        return opener({"model": "blocked-wire-model"})

    monkeypatch.setattr(relay_llm, "stream", invoke_opener)

    with pytest.raises(RoutingPolicyError, match="selected model"):
        call._call_anthropic(client)
    assert messages.calls == []
