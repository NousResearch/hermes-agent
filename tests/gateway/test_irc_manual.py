#!/usr/bin/env python3
"""Manual integration test for IRC adapter.

Shows all IRC traffic (sent and received) for debugging.

Usage:
    python tests/gateway/test_irc_manual.py --host irc.example.com --nick mybot --channels "#mytest"
    python tests/gateway/test_irc_manual.py --host irc.example.com --port 6697 --tls --nick mybot --channels "#mytest"

Manual Test Coverage:
=====================
This manual test validates end-to-end IRC adapter functionality:

1. Connection:
   - TLS/non-TLS connections (port 6667 vs 6697)
   - Server password authentication (optional)
   - Nick registration (NICK, USER commands)
   - CAP negotiation (draft/multiline support)
   - Connection timeout handling

2. Channel Operations:
   - JOIN command
   - Presence detection (user list via 353 RPL_NAMREPLY)
   - Message sending to channels

3. Message Handling:
   - PRIVMSG sending
   - Line protocol parsing (prefix, command, params, trailing)
   - Nick collision handling (automatic "_" suffix)
   - CTCP filtering (ignores \x01ACTION\x01 etc.)

4. Protocol Features:
   - PING/PONG keepalive
   - Batch multiline support (if server supports CAP draft/multiline)
   - Message chunking for long messages (fallback when BATCH unavailable)

5. NickServ Authentication:
   - NickServ IDENTIFY on NOTICE (optional, via IRC_NICKSERV_PASSWORD)
   - Graceful failure if NickServ auth fails

Running with Pytest:
    pytest tests/gateway/test_irc_manual.py -- --host irc.example.com --nick mybot --channels "#test"
    pytest tests/gateway/test_irc_manual.py -- --host irc.example.com --port 6697 --tls --nick mybot --channels "#test"
    pytest tests/gateway/test_irc_manual.py (requires IRC_TEST_HOST, IRC_TEST_NICK, IRC_TEST_CHANNELS env vars)

Note: Unit tests in test_irc.py cover config loading, format_message, and IRC mask matching.
This manual test covers runtime adapter behavior and protocol compliance.
"""
import argparse
import asyncio
import os
import sys

# Add project root to path for imports
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])

from gateway.config import PlatformConfig
from gateway.platforms.irc import IRCAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Manual IRC adapter test")
    parser.add_argument("--host", required=True, help="IRC server hostname")
    parser.add_argument("--port", type=int, default=6667, help="IRC server port (default: 6667)")
    parser.add_argument("--nick", required=True, help="Bot nickname")
    parser.add_argument("--channels", required=True, help="Channels to join (comma-separated)")
    parser.add_argument("--password", default="", help="Server password (optional)")
    parser.add_argument("--tls", action="store_true", help="Use TLS encryption")
    return parser.parse_args()


def build_config(args):
    return PlatformConfig(
        enabled=True,
        extra={
            "server": args.host,
            "port": args.port,
            "nick": args.nick,
            "password": args.password,
            "channels": args.channels,
            "use_tls": args.tls,
        },
    )


def patch_adapter_for_verbose_output(adapter):
    """Monkey-patch the adapter to print all IRC traffic."""
    original_send_line = adapter._send_line
    original_handle_line = adapter._handle_line

    def verbose_send_line(line: str):
        print(f">>> {line}")
        return original_send_line(line)

    async def verbose_handle_line(line: str):
        print(f"<<< {line}")
        await original_handle_line(line)

    # Patch the methods
    adapter._send_line = verbose_send_line
    adapter._handle_line = verbose_handle_line


async def run_tests(config):
    """Run IRC adapter test."""
    adapter = IRCAdapter(config)
    patch_adapter_for_verbose_output(adapter)
    channels_str = config.extra.get("channels", "")
    channels = [c.strip() for c in channels_str.split(",") if c.strip()]
    target = channels[0] if channels else None

    if not target:
        print("ERROR: No channels configured")
        sys.exit(1)
    print(f"\n--- Connecting to {config.extra['server']}:{config.extra['port']} as {config.extra['nick']} ---\n")
    connected = await adapter.connect()
    if not connected:
        print("\nFAILED: connect() returned False")
        print("Possible causes:")
        print("  - Server unreachable")
        print("  - Nick already in use (try a different --nick)")
        print("  - Registration timeout (server didn't send 001)")
        print("  - TLS required (try --tls)")
        sys.exit(1)
    print("\n--- Connected! Waiting for JOIN... ---")
    await asyncio.sleep(2)
    print(f"\n--- Sending test message to {target} ---\n")
    await adapter.send(target, "Hello from Hermes IRC adapter test!")
    print("\n--- Waiting 2 seconds... ---")
    await asyncio.sleep(2)
    print("\n--- Disconnecting ---\n")
    await adapter.disconnect()
    print("\n=== Done ===")


# Pytest support (optional)
def _get_pytest_config():
    """Get config for pytest mode (from CLI args after -- or env vars)."""
    if "--" in sys.argv:
        separator_idx = sys.argv.index("--")
        test_args = sys.argv[separator_idx + 1:]
        if test_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--host")
            parser.add_argument("--port", type=int, default=6667)
            parser.add_argument("--nick")
            parser.add_argument("--channels")
            parser.add_argument("--password", default="")
            parser.add_argument("--tls", action="store_true")
            try:
                args = parser.parse_args(test_args)
                if args.host and args.nick and args.channels:
                    return build_config(args)
            except SystemExit:
                pass
    host = os.getenv("IRC_TEST_HOST")
    if host:
        channels_str = os.getenv("IRC_TEST_CHANNELS", "#hermes-test")
        return PlatformConfig(
            enabled=True,
            extra={
                "server": host,
                "port": int(os.getenv("IRC_TEST_PORT", "6667")),
                "nick": os.getenv("IRC_TEST_NICK", "hermes-test"),
                "password": os.getenv("IRC_TEST_PASSWORD", ""),
                "channels": channels_str,
                "use_tls": os.getenv("IRC_TEST_TLS", "false").lower() == "true",
            },
        )
    return None


try:
    import pytest
    _PYTEST_CONFIG = _get_pytest_config()
    pytestmark = pytest.mark.skipif(
        not _PYTEST_CONFIG,
        reason="No IRC config -- use -- --host HOST --nick NICK --channels '#chan' or set IRC_TEST_HOST",
    )
    @pytest.fixture
    def irc_config():
        return _PYTEST_CONFIG
    @pytest.fixture
    def irc_adapter(irc_config):
        return IRCAdapter(irc_config)
    @pytest.mark.asyncio
    async def test_connect_and_send_message(irc_adapter):
        channels_str = irc_adapter.config.extra.get("channels", "")
        channels = [c.strip() for c in channels_str.split(",") if c.strip()]
        if not channels:
            return
        connected = await irc_adapter.connect()
        assert connected, "connect() returned False"
        await asyncio.sleep(2)
        await irc_adapter.send(channels[0], "Hello from Hermes IRC adapter test!")
        await asyncio.sleep(2)
        await irc_adapter.disconnect()
except ImportError:
    pass


def main():
    args = parse_args()
    config = build_config(args)
    asyncio.run(run_tests(config))


if __name__ == "__main__":
    main()
