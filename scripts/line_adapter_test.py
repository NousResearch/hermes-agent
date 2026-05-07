"""
Standalone test harness for the LINE platform plugin.

Boots only the LineAdapter — no full Hermes gateway required — and prints
every inbound event to stdout so you can verify webhook signature, body
parsing, and chat-id resolution against the live LINE Messaging API.

Usage:
    export LINE_CHANNEL_ACCESS_TOKEN=...
    export LINE_CHANNEL_SECRET=...
    # optional: export LINE_PORT=3979
    python scripts/line_adapter_test.py

Then expose the port with a tunnel (cloudflared / ngrok / devtunnel) and
set the channel's webhook URL in the LINE developers console to:
    https://<tunnel-host>/line/webhook
Click "Verify" in the console — you should see a successful signature
verification logged here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Make the repo importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("line-test")


async def main() -> None:
    if not os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or not os.getenv("LINE_CHANNEL_SECRET"):
        sys.exit(
            "Set LINE_CHANNEL_ACCESS_TOKEN and LINE_CHANNEL_SECRET before running."
        )

    # Lazy-import so the missing-deps message above fires before any heavy
    # gateway import work.
    from gateway.config import PlatformConfig
    from plugins.platforms.line.adapter import LineAdapter

    cfg = PlatformConfig(enabled=True, extra={})
    adapter = LineAdapter(cfg)

    # Replace handle_message with a printer so events surface without the
    # full gateway pipeline.
    async def _print_event(event):  # type: ignore[no-untyped-def]
        src = event.source
        log.info(
            "MSG  chat=%s type=%s user=%s text=%r media=%s",
            src.chat_id,
            src.chat_type,
            src.user_id,
            event.text,
            event.media_urls,
        )
        # Echo the text back so you can test the outbound path too.
        if event.text:
            result = await adapter.send(src.chat_id, f"echo: {event.text}")
            log.info("SEND ok=%s err=%s", result.success, result.error)

    adapter.handle_message = _print_event  # type: ignore[assignment]

    ok = await adapter.connect()
    if not ok:
        sys.exit(f"connect failed: {adapter._fatal_error}")

    log.info(
        "LINE adapter listening on 0.0.0.0:%d%s — Ctrl-C to stop",
        adapter._port,
        adapter._webhook_path,
    )
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await adapter.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
