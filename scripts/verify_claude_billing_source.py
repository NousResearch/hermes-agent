#!/usr/bin/env python3
"""Manual, opt-in check of which account a Claude subscription turn would bill.

Nothing calls this. It exists so a maintainer can verify the billing source on
a real machine, deliberately, and see exactly what each step costs.

Two modes:

``--probe`` (default, free)
    Connects the Claude Agent SDK to the CLI through Hermes' sanitized-launch
    transport, reads the CLI's own initialize response, and disconnects. No
    user message is ever written to the child, so no model request is made:
    it consumes no tokens and no plan quota. Prints the environment Hermes
    would hand the child, which credentials were stripped, and how the CLI
    classified the account.

``--live-turn`` (COSTS QUOTA — never run this in CI or a test)
    Sends one tiny real prompt so the turn appears in the account's usage.
    This is the only way to prove the billing *bucket* end to end; a maintainer
    then confirms it landed on the plan, not on the Console org, by checking
    claude.ai usage and console.anthropic.com usage after the run.

Usage::

    python scripts/verify_claude_billing_source.py            # free probe
    python scripts/verify_claude_billing_source.py --live-turn --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.claude_billing import (  # noqa: E402
    BLOCKED_CHILD_ENV_VARS,
    blocking_credentials,
    classify_account,
    sanitized_child_env,
    static_billing_refusal,
)

LIVE_BANNER = """
================================================================================
  --live-turn sends ONE REAL PROMPT to Claude.

  It consumes quota on whichever account the CLI resolves — your Claude plan
  if everything below is configured correctly, your Anthropic Console org if
  it is not. That is the thing being tested, so it cannot be avoided.

  Re-run with --yes to proceed.
================================================================================
"""


def report_environment() -> int:
    print("== environment Hermes would hand the Claude CLI ==")
    stripped = sorted(k for k in BLOCKED_CHILD_ENV_VARS if k in os.environ)
    child = sanitized_child_env()
    print(f"  parent variables:      {len(os.environ)}")
    print(f"  child variables:       {len(child)}")
    print(f"  stripped:              {', '.join(stripped) or '(none present)'}")
    print(f"  CLAUDE_CONFIG_DIR:     {'passed through' if 'CLAUDE_CONFIG_DIR' in os.environ else 'not set'}")
    print(f"  HOME:                  {'unchanged' if child.get('HOME') == os.environ.get('HOME') else 'CHANGED — bug'}")

    print("\n== static precedence check ==")
    slots = blocking_credentials()
    if slots:
        for slot in slots:
            print(f"  BLOCKING  {slot.name} (precedence rank {slot.rank}) — {slot.fix}")
    else:
        print("  nothing outranks the subscription")

    refusal = static_billing_refusal()
    if refusal:
        print("\n" + refusal)
        return 1
    return 0


def run_probe() -> int:
    from agent.transports.claude_sanitized_transport import build_sanitized_transport

    try:
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    except ImportError:
        print("claude-agent-sdk is not installed: pip install 'hermes-agent[claude-code]'")
        return 2

    options = ClaudeAgentOptions(tools=[], allowed_tools=[], setting_sources=[])

    async def _go() -> dict:
        client = ClaudeSDKClient(
            options=options, transport=build_sanitized_transport(options)
        )
        await client.connect()
        try:
            info = await client.get_server_info()
        finally:
            await client.disconnect()
        return (info or {}).get("account") or {}

    print("\n== zero-cost init probe (no model request) ==")
    account = asyncio.run(_go())
    print("  CLI account payload:", json.dumps(account, default=str))
    source = classify_account(account)
    print(f"  classified as:       {source.kind}")
    if source.plan:
        print(f"  plan:                {source.plan}")
    if source.account:
        print(f"  account:             {source.account}")
    if source.detail:
        print(f"  decided by:          {source.detail}")
    return 0 if source.is_subscription else 1


def run_live_turn() -> int:
    from agent.transports.claude_sanitized_transport import build_sanitized_transport

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    options = ClaudeAgentOptions(tools=[], allowed_tools=[], setting_sources=[])

    async def _go() -> None:
        client = ClaudeSDKClient(
            options=options, transport=build_sanitized_transport(options)
        )
        await client.connect()
        try:
            await client.query("Reply with the single word: ok")
            async for message in client.receive_response():
                print("  ", type(message).__name__, getattr(message, "result", ""))
        finally:
            await client.disconnect()

    print("\n== live turn (BILLED) ==")
    asyncio.run(_go())
    print(
        "\nNow confirm the bucket by hand:\n"
        "  • claude.ai → Settings → Usage should show this turn\n"
        "  • console.anthropic.com → Usage should NOT show it"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="free init probe (default)")
    parser.add_argument(
        "--live-turn", action="store_true", help="send one real, billed prompt"
    )
    parser.add_argument("--yes", action="store_true", help="confirm --live-turn")
    args = parser.parse_args(argv)

    status = report_environment()
    if status:
        return status

    if args.live_turn:
        if not args.yes:
            print(LIVE_BANNER)
            return 2
        if run_probe():
            print("Refusing --live-turn: the probe says this would not bill the plan.")
            return 1
        return run_live_turn()

    return run_probe()


if __name__ == "__main__":
    sys.exit(main())
