"""Lower-level Matrix senders relay native media through the selected gateway owner."""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.control_socket import query_gateway_control
from gateway.run_bootstrap import _start_gateway_start_control_socket
from plugins.platforms.matrix.adapter import _standalone_send
from tools.send_message_senders import _send_matrix_via_adapter


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [_send_matrix_via_adapter, _standalone_send])
async def test_standalone_entrypoints_relay_text_media_thread_and_profile(tmp_path, monkeypatch, sender):
    from gateway.config import Platform
    from hermes_cli.profiles import get_active_profile_name
    from gateway.run import _profile_runtime_scope

    home = tmp_path / "profiles" / "alice"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    media = tmp_path / "picture.png"
    media.write_bytes(b"fixture")
    calls = []

    class Adapter:
        def __init__(self, name):
            self.name = name

        async def send(self, chat, message, metadata=None):
            calls.append(("text", get_active_profile_name(), asyncio.get_running_loop(), chat, message, metadata))
            return SimpleNamespace(success=True, message_id="$text-" + self.name)

        async def send_image_file(self, chat, path, metadata=None):
            calls.append(("media", get_active_profile_name(), asyncio.get_running_loop(), chat, path, metadata))
            return SimpleNamespace(success=True, message_id="$media-" + self.name)

    adapter, bob_adapter = Adapter("alice"), Adapter("bob")
    bob = tmp_path / "profiles" / "bob"
    bob.mkdir(parents=True)
    (bob / "config.yaml").write_text("{}\n")
    profiles = {home: "alice", bob: "bob"}
    adapters = {"alice": adapter, "bob": bob_adapter}
    runner = SimpleNamespace(
        _gateway_loop=asyncio.get_running_loop(), session_runtime_descriptor={}, session_ticket_store=None,
        _served_profile_for_home=lambda requested: profiles.get(requested),
        _authorization_adapter=lambda platform, profile: adapters.get(profile) if platform == Platform.MATRIX else None,
    )
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    (bob / "gateway.sock.path").write_text(str(server._bind_path))
    pconfig = SimpleNamespace(extra={"e2ee_mode": "required"})
    try:
        with patch("tools.send_message_senders._live_adapter", return_value=(None, None)), patch(
            "plugins.platforms.matrix.adapter.MatrixAdapter", side_effect=AssertionError("ephemeral fallback")):
            with _profile_runtime_scope(home):
                result = await sender(pconfig, "!room:example.org", "hello", media_files=[(str(media), False)], thread_id="$thread")
        assert result["success"]
        assert result["message_id"] == "$media-alice"
        loop = asyncio.get_running_loop()
        assert calls == [
            ("text", "alice", loop, "!room:example.org", "hello", {"thread_id": "$thread"}),
            ("media", "alice", loop, "!room:example.org", str(media), {"thread_id": "$thread"}),
        ]
        with patch("tools.send_message_senders._live_adapter", return_value=(None, None)), patch(
            "plugins.platforms.matrix.adapter.MatrixAdapter", side_effect=AssertionError("ephemeral fallback")):
            with _profile_runtime_scope(bob):
                bob_result = await sender(pconfig, "!bob:example.org", "bob message")
        assert bob_result["success"]
        assert bob_result["message_id"] == "$text-bob"
        assert calls[-1] == ("text", "bob", loop, "!bob:example.org", "bob message", None)
        refused = await asyncio.to_thread(query_gateway_control, home, "matrix-send", params={
            "home": str(tmp_path / "profiles" / "unserved"), "chat_id": "!room:example.org", "message": "hello"})
        assert "not served" in refused["error"]
        assert len(calls) == 3
    finally:
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [_send_matrix_via_adapter, _standalone_send])
async def test_uncertain_standalone_relay_never_falls_back(sender, monkeypatch):
    calls = []

    def query(home, verb, **kwargs):
        calls.append(verb)
        return {"supported_verbs": ["matrix-send"]} if verb == "send-capabilities" else None

    monkeypatch.setattr("gateway.control_socket.query_gateway_control", query)
    with patch("tools.send_message_senders._live_adapter", return_value=(None, None)), patch(
        "plugins.platforms.matrix.adapter.MatrixAdapter", side_effect=AssertionError("ephemeral fallback")):
        result = await sender(SimpleNamespace(extra={"e2ee_mode": "required"}), "!room:example.org", "hello")
    assert "outcome unknown" in result["error"]
    assert calls == ["send-capabilities", "matrix-send"]
