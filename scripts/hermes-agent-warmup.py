#!/usr/bin/env python3
"""Warm-up health check for hermes-agent — validates that the running service
can actually make a successful Anthropic API call after restart.

Detects the regression we observed on 2026-05-06: auto-update + restart
sometimes left the service in a state where the first real request would
fail with HTTP 400 'out of extra usage'. We exercise the full OAuth path
(env vars + patched mcp_/beta gates) by making a tiny direct API call.

Deployment example — sys.path insert below assumes the repo lives at
/home/leos/hermes-agent. Companion: scripts/hermes-agent-updater.sh.

Exit codes:
  0 — OK (request succeeded)
  1 — failed; caller should restart hermes-agent again
"""
import os
import sys
import time

# Mirror what hermes_cli/main.py does at startup so the env vars from
# ~/.hermes/.env take effect for the OAuth gates.
sys.path.insert(0, "/home/leos/hermes-agent")
try:
    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv()
except Exception as e:
    print(f"WARN: load_hermes_dotenv failed: {e}", file=sys.stderr)

from agent.anthropic_adapter import build_anthropic_client

try:
    from agent.anthropic_adapter import _oauth_mcp_prefix_enabled
except ImportError:
    _oauth_mcp_prefix_enabled = None

from agent.credential_pool import load_pool


def main() -> int:
    env_summary = {
        k: os.environ.get(k, "<unset>")
        for k in (
            "HERMES_OAUTH_NO_MCP_PREFIX",
            "HERMES_OAUTH_COMPACT_GUIDANCE",
            "HERMES_OAUTH_FORCE_DROP_1M_BETA",
        )
    }
    print(f"warmup env: {env_summary}")
    prefix_state = _oauth_mcp_prefix_enabled() if _oauth_mcp_prefix_enabled else "<symbol unavailable>"
    print(f"warmup mcp_prefix_enabled={prefix_state} (expect False)")

    pool = load_pool("anthropic")
    entries = pool.entries()
    if not entries:
        print("FAIL: no anthropic credentials in pool", file=sys.stderr)
        return 1
    entry = entries[0]
    token = entry.access_token
    if not token:
        print(f"FAIL: pool entry {entry.source} has no access_token", file=sys.stderr)
        return 1

    client = build_anthropic_client(token, base_url="https://api.anthropic.com")

    t0 = time.time()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=20,
            system="You are Claude Code, Anthropic's official CLI for Claude.",
            messages=[{"role": "user", "content": "Reply with exactly: WARMUP-OK"}],
        )
    except Exception as e:
        msg = str(e)[:200]
        print(f"FAIL: messages.create raised {type(e).__name__}: {msg}", file=sys.stderr)
        return 1
    dt = time.time() - t0

    text = resp.content[0].text if resp.content else ""
    print(f"warmup OK in {dt:.2f}s, in={resp.usage.input_tokens} out={resp.usage.output_tokens}: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
