"""Plugin-loaded Matrix adapters register watches in their own profile store."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.matrix.followup_mixin import _MatrixFollowupChoice


@pytest.mark.asyncio
async def test_plugin_imported_adapter_registers_final_watch_across_profiles(tmp_path):
    plugin_path = Path(__file__).resolve().parents[4] / "plugins/platforms/matrix"
    manifest = PluginManifest(name="matrix", key="platforms/matrix", source="bundled", path=str(plugin_path))
    results = []
    for label in ("a", "b", "a"):
        home = tmp_path / label
        home.mkdir(exist_ok=True)
        token = set_hermes_home_override(home)
        try:
            module = PluginManager(scope_key=str(home))._load_directory_module(manifest)
            loaded_adapter = module.register.__globals__["MatrixAdapter"]
            adapter = loaded_adapter(PlatformConfig(enabled=True, extra={"user_id": "@bot:test"}))
            adapter._store_dir = home / "matrix"
            source = SessionSource(platform=Platform.MATRIX, chat_id="!room:test", user_id="@alice:test", profile=label)
            adapter._reaction_followup_actions["session"] = _MatrixFollowupChoice(
                "turn", (), source.chat_id, source.user_id, "", label, "sid",
            )
            adapter._client = AsyncMock()
            adapter._client.send_message_event.return_value = "$reply"
            await adapter._send_room_message(source.chat_id, {"msgtype": "m.text", "body": "Final answer"})
            adapter.on_streamed_final_delivery(source, "session", ("$reply",), "Final answer")
            candidate = adapter._followup_store().candidate(source.chat_id, "$reply")
            results.append((loaded_adapter is MatrixAdapter, candidate["profile"] if candidate else None))
            if adapter._watch_purge_handle is not None:
                adapter._watch_purge_handle.cancel()
        finally:
            reset_hermes_home_override(token)
    assert results == [(False, "a"), (False, "b"), (False, "a")]
