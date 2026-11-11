"""A changed profile's rescan drops status verdicts about credentials the user removed.

Removing a duplicated bot token from a secondary profile's .env must clear its
``<profile>:<platform>`` fatal entry (Desktop: "Backend telegram down"), while a
platform that is still live keeps its entry.
"""
import json
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.status import flush_runtime_status, publish_runtime_status
from tests.gateway.test_multiplex_hot_serve import _mkprofile, _runner


def _platforms(home):
    flush_runtime_status()
    return json.loads((home / "gateway_state.json").read_text(encoding="utf-8")).get("platforms") or {}


@pytest.mark.asyncio
async def test_rescan_drops_fatal_for_removed_credential_keeps_live(tmp_path, monkeypatch):
    runner, home = _runner(tmp_path, monkeypatch)
    alpha = _mkprofile(home, "alpha", "DISCORD_BOT_TOKEN=alpha-token\n")
    with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
        await runner._start_secondary_profile_adapters()
        assert Platform.DISCORD in runner._profile_adapters["alpha"]
        publish_runtime_status(platform="alpha:discord", platform_state="connected")
        publish_runtime_status(platform="alpha:telegram", platform_state="fatal",
                               error_code="duplicate_credential", error_message="dup")
        publish_runtime_status(platform="telegram", platform_state="connected")
        assert _platforms(home)["alpha:telegram"]["state"] == "fatal"

        # User deletes the copied TELEGRAM_BOT_TOKEN line: the .env change triggers a rescan.
        (alpha / ".env").write_text("DISCORD_BOT_TOKEN=alpha-token\n# edited\n", encoding="utf-8")
        assert (await runner.reconcile_served_profiles())["rescanned"] == ["alpha"]

        platforms = _platforms(home)
        assert "alpha:telegram" not in platforms
        assert platforms["alpha:discord"]["state"] == "connected"
        assert platforms["telegram"]["state"] == "connected"
