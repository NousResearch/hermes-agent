"""CLI relay uses a real control socket and the selected profile's live adapter."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.run_bootstrap import _start_gateway_start_control_socket
from gateway.run_send_relay import send_message_verb
from hermes_cli.send_relay import relay_matrix_send


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
async def test_cli_relay_keeps_profile_and_gateway_loop(tmp_path, monkeypatch):
    from gateway.config import Platform
    import gateway.run as gateway
    from hermes_cli.profiles import get_active_profile_name

    home = tmp_path / "profiles" / "alice"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        'platforms:\n  matrix:\n    enabled: true\n    token: test-token\n'
        '    extra:\n      homeserver: https://matrix.example.org\n      user_id: "@alice:example.org"\n'
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    observed = []

    async def send(chat_id, message, metadata=None):
        observed.append((get_active_profile_name(), asyncio.get_running_loop(), chat_id, message))
        return SimpleNamespace(success=True, message_id="$encrypted")

    adapter = SimpleNamespace(send=send)
    runner = SimpleNamespace(
        _gateway_loop=asyncio.get_running_loop(),
        session_runtime_descriptor={},
        session_ticket_store=None,
        _served_profile_for_home=lambda requested: "alice" if requested == home else None,
        _authorization_adapter=lambda platform, profile: adapter if platform == Platform.MATRIX and profile == "alice" else None,
    )
    monkeypatch.setattr(gateway, "_gateway_runner_ref", lambda: runner)
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        args = {"action": "send", "target": "matrix:!room:example.org", "message": "hello"}
        result = await asyncio.to_thread(relay_matrix_send, args["target"], args)
        assert result["success"] is True
        assert observed == [("alice", asyncio.get_running_loop(), "!room:example.org", "hello")]
        refused = await asyncio.to_thread(send_message_verb(runner), {**args, "home": str(tmp_path / "other")})
        assert "not served" in refused["error"]
        assert len(observed) == 1
    finally:
        await server.stop()
        server.cleanup_files()


def test_uncertain_relay_delivery_is_not_retried(monkeypatch):
    import gateway.control_socket as control
    calls = []

    def query(home, verb, **kwargs):
        calls.append(verb)
        return {"supported_verbs": ["send-message"]} if verb == "send-capabilities" else None

    monkeypatch.setattr(control, "query_gateway_control", query)
    result = relay_matrix_send("matrix:!room:example.org", {"message": "hello"})
    assert "outcome unknown" in result["error"]
    assert calls == ["send-capabilities", "send-message"]
