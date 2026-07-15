from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import pytest

from gateway import discord_guild_history_client as client
from gateway.discord_connector_protocol import DiscordConnectorHistoryAuthorityKind
from gateway.session_context import clear_session_vars, set_session_vars


class _CredentialGuard(dict):
    def get(self, key, default=None):
        assert "TOKEN" not in str(key).upper()
        assert "SECRET" not in str(key).upper()
        return super().get(key, default)


def test_service_gate_requires_exact_pinned_socket_without_reading_credentials(
    monkeypatch,
) -> None:
    with tempfile.TemporaryDirectory(prefix="mdch-", dir="/tmp") as directory:
        socket_path = Path(directory) / "connector.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(1)
        pinned_url = f"unix://{socket_path}"
        monkeypatch.setattr(client, "DEFAULT_DISCORD_CONNECTOR_SOCKET", socket_path)
        monkeypatch.setattr(client, "PINNED_DISCORD_CONNECTOR_URL", pinned_url)
        monkeypatch.setattr(
            client.os,
            "environ",
            _CredentialGuard({"GATEWAY_RELAY_URL": pinned_url}),
        )
        try:
            assert client.discord_guild_history_configured() is True
            client.os.environ["GATEWAY_RELAY_URL"] = (
                f"unix://{Path(directory) / 'other.sock'}"
            )
            assert client.discord_guild_history_configured() is False
        finally:
            listener.close()


def test_service_gate_rejects_regular_file_and_symlink(tmp_path, monkeypatch) -> None:
    target = tmp_path / "not-a-socket"
    target.write_text("no credential", encoding="utf-8")
    link = tmp_path / "connector.sock"
    link.symlink_to(target)
    pinned_url = f"unix://{link}"
    monkeypatch.setattr(client, "DEFAULT_DISCORD_CONNECTOR_SOCKET", link)
    monkeypatch.setattr(client, "PINNED_DISCORD_CONNECTOR_URL", pinned_url)
    monkeypatch.setenv("GATEWAY_RELAY_URL", pinned_url)
    assert client.discord_guild_history_configured() is False


def test_client_derives_authenticated_requester_and_never_accepts_model_authority() -> None:
    class _Transport:
        observed = None

        def read_guild_history(self, channel_id, **kwargs):
            self.observed = (channel_id, kwargs)
            return {"messages": []}

    transport = _Transport()
    history_client = object.__new__(client.DiscordGuildHistoryClient)
    history_client._transport = transport
    tokens = set_session_vars(platform="discord", user_id="1279454038731264061")
    try:
        assert history_client.read(channel_id="1504852355588423801", limit=1) == {
            "messages": []
        }
    finally:
        clear_session_vars(tokens)
    authority = transport.observed[1]["authority"]
    assert authority.kind is DiscordConnectorHistoryAuthorityKind.AUTHENTICATED_USER
    assert authority.requester_user_id == "1279454038731264061"

    with pytest.raises(TypeError):
        history_client.read(
            channel_id="1504852355588423801",
            limit=1,
            authority=authority,
        )

    with pytest.raises(
        client.DiscordGuildHistoryClientError,
        match="discord_history_requester_context_missing",
    ):
        history_client.read(channel_id="1504852355588423801", limit=1)
