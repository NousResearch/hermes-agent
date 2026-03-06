#!/usr/bin/env python3
"""Run the fake HA server standalone for manual testing."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fakes.fake_ha_server import FakeHAServer


async def main():
    server = FakeHAServer(token="test-token-123")
    await server.start()
    print(f"\nFake HA Server running at: {server.url}", flush=True)
    print(f"Token: test-token-123", flush=True)
    print("Press Ctrl+C to stop\n", flush=True)
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
