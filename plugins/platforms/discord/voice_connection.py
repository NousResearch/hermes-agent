"""Temporary discord.py connection-state compatibility for room moves.

Import only from the live join path: Discord is an optional dependency.
Remove when native discord.py passes the reconnect/move and disconnect tests
in tests/test_discord_voice_connection.py without this correction.
"""
from discord import VoiceClient
from discord.voice_state import VoiceConnectionState


class _VoiceConnectionState(VoiceConnectionState):
    async def _connect(self, *args, **kwargs):
        await super()._connect(*args, **kwargs)
        if self.is_connected():
            # discord.py 2.7.1 leaves an expected disconnect acknowledgment set
            # after automatic recovery. A later move's 4014 then looks like a
            # kick (upstream PRs 9683/9772). Only retire it after a new connection;
            # leave failed connects and genuine current disconnects untouched.
            self._disconnected.clear()


class DiscordVoiceClient(VoiceClient):
    def create_connection_state(self):
        return _VoiceConnectionState(self)
