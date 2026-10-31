"""Exercise real PTB in an isolated process, outside gateway telegram mocks."""

import asyncio
import sys
from urllib.parse import unquote
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

bale = load_plugin_adapter("bale")


def main():
    import json
    from telegram import Bot
    from telegram.request import BaseRequest

    urls = []

    class Transport(BaseRequest):
        @property
        def read_timeout(self):
            return 5

        async def initialize(self):
            pass

        async def shutdown(self):
            pass

        async def do_request(self, url, method, **kwargs):
            urls.append(url)
            if url.endswith("/getMe"):
                data = {"id": 123, "is_bot": True, "first_name": "Test"}
            elif url.endswith("/getFile"):
                data = {
                    "file_id": "fixture",
                    "file_unique_id": "unique",
                    "file_path": "docs/test.txt",
                }
            else:
                return 200, b"fixture-content"
            return 200, json.dumps({"ok": True, "result": data}).encode()

    async def download():
        config = PlatformConfig(enabled=True, token="123:test")
        bale.BaleAdapter(config)
        async with Bot(
            token=config.token,
            base_url=config.extra["base_url"],
            base_file_url=config.extra["base_file_url"],
            request=Transport(),
            get_updates_request=Transport(),
        ) as bot:
            file = await bot.get_file("fixture")
            assert await file.download_as_bytearray() == b"fixture-content"

    asyncio.run(download())
    assert unquote(urls[-1]) == "https://tapi.bale.ai/file/bot123:test/docs/test.txt"


if __name__ == "__main__":
    main()
