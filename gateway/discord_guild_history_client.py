"""Credential-free client factory for authorized Discord guild history reads.

The token remains exclusively in the privileged connector service. The model
can select only the exact target/cursors; this client derives the requester or
reviewed cron authority from task-local runtime context before crossing IPC.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from gateway.canonical_writer_client import (
    ExactServerMainPidAuthorizer,
    SystemctlServerMainPidProvider,
)
from gateway.discord_connector_service import (
    DEFAULT_DISCORD_CONNECTOR_SOCKET,
    DEFAULT_DISCORD_CONNECTOR_UNIT,
    DEFAULT_DISCORD_CONNECTOR_USER,
)
from gateway.discord_history_authority import (
    DiscordHistoryAuthorityError,
    resolve_discord_history_authority,
)
from gateway.relay.discord_connector_transport import (
    DiscordConnectorRelayTransport,
    DiscordConnectorTransportError,
)

PINNED_DISCORD_CONNECTOR_URL = f"unix://{DEFAULT_DISCORD_CONNECTOR_SOCKET}"


class DiscordGuildHistoryClientError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def discord_guild_history_configured() -> bool:
    """Return true only for the live pinned connector socket, never a token."""

    if os.environ.get("GATEWAY_RELAY_URL", "").strip() != PINNED_DISCORD_CONNECTOR_URL:
        return False
    try:
        current = DEFAULT_DISCORD_CONNECTOR_SOCKET.lstat()
    except (FileNotFoundError, OSError):
        return False
    return not DEFAULT_DISCORD_CONNECTOR_SOCKET.is_symlink() and stat.S_ISSOCK(
        current.st_mode
    )


class DiscordGuildHistoryClient:
    """Narrow exact operation wrapper; no raw connector dispatcher is exposed."""

    def __init__(self, transport: DiscordConnectorRelayTransport) -> None:
        if not isinstance(transport, DiscordConnectorRelayTransport):
            raise TypeError("Discord guild history transport is invalid")
        self._transport = transport

    def read(
        self,
        *,
        channel_id: str,
        limit: int,
        before_message_id: str | None = None,
        after_message_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            authority = resolve_discord_history_authority(channel_id)
            return self._transport.read_guild_history(
                channel_id,
                limit=limit,
                before_message_id=before_message_id,
                after_message_id=after_message_id,
                authority=authority,
            )
        except DiscordHistoryAuthorityError as exc:
            raise DiscordGuildHistoryClientError(exc.code) from exc
        except DiscordConnectorTransportError as exc:
            raise DiscordGuildHistoryClientError(exc.code) from exc


def privileged_discord_guild_history_client() -> DiscordGuildHistoryClient:
    """Build the exact local client without resolving any Discord credential."""

    if not discord_guild_history_configured():
        raise DiscordGuildHistoryClientError("discord_guild_history_unavailable")
    try:
        import pwd

        connector_uid = pwd.getpwnam(DEFAULT_DISCORD_CONNECTOR_USER).pw_uid
    except (ImportError, KeyError, OSError) as exc:
        raise DiscordGuildHistoryClientError(
            "discord_connector_identity_unavailable"
        ) from exc
    authorizer = ExactServerMainPidAuthorizer(
        server_unit=DEFAULT_DISCORD_CONNECTOR_UNIT,
        expected_server_uid=connector_uid,
        main_pid_provider=SystemctlServerMainPidProvider(),
    )
    return DiscordGuildHistoryClient(
        DiscordConnectorRelayTransport(
            Path(DEFAULT_DISCORD_CONNECTOR_SOCKET),
            server_authorizer=authorizer,
        )
    )


__all__ = [
    "DiscordGuildHistoryClient",
    "DiscordGuildHistoryClientError",
    "PINNED_DISCORD_CONNECTOR_URL",
    "discord_guild_history_configured",
    "privileged_discord_guild_history_client",
]
