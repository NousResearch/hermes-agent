from types import SimpleNamespace

import pytest


def test_native_anthropic_final_wire_uses_session_owner_policy_before_opening_stream(tmp_path, monkeypatch):
    """Ambient A cannot permit B's final mutated native-Anthropic wire send."""
    home = tmp_path / "hermes"
    restricted = home / "profiles" / "restricted"
    for config, policy in (
        (home / "config.yaml", "routing_policy:\n  enabled: true\n"),
        (restricted / "config.yaml", "routing_policy:\n  enabled: true\n  deny:\n    models: [blocked-*]\n"),
    ):
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))  # A is ambient while B owns the session.

    from hermes_state import SessionDB
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
    default_db = SessionDB(db_path=home / "state.db")
    restricted_db = SessionDB(db_path=restricted / "state.db")
    try:
        default_db.create_session("a-before", "cli", model="allowed-model")
        agent = SimpleNamespace(
            provider="anthropic",
            model="allowed-model",
            base_url="https://api.anthropic.example/v1",
            log_prefix="",
            _session_db=restricted_db,
        )
        call = helpers._StreamingCall(agent, {"model": "allowed-model"}, None)
        call._new_diag = lambda: {}
        monkeypatch.setattr(helpers, "_relay_stream_identity", lambda *_args: {})
        monkeypatch.setattr(helpers, "_relay_stream_metadata", lambda *_args: {})

        def invoke_opener(_request, opener, **_kwargs):
            return opener({"model": "blocked-wire-model"})

        monkeypatch.setattr(relay_llm, "stream", invoke_opener)

        with pytest.raises(RoutingPolicyError, match="selected model"):
            call._call_anthropic(client)
    finally:
        restricted_db.close()
        default_db.close()
    assert messages.calls == []
