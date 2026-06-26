import inspect
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="order-dependent discord mock leakage in full suite on Windows")

from plugins.platforms.discord.adapter import DiscordAdapter


def test_discord_media_methods_accept_metadata_kwarg():
    for method_name in ("send_voice", "send_image_file", "send_image"):
        signature = inspect.signature(getattr(DiscordAdapter, method_name))
        assert "metadata" in signature.parameters, method_name
