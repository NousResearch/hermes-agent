"""Live A/B probe: Standard Webhooks (webhook-id/-timestamp/-signature) against the REAL webhook adapter.

Starts a real ``WebhookAdapter`` (aiohttp, loopback, dedicated port) with one secret-configured
route and one ``whsec_`` route, then sends real HTTP POSTs signed exactly as the Standard
Webhooks spec (https://github.com/standard-webhooks/standard-webhooks) and GitLab's signing-token
docs describe: HMAC-SHA256 over ``{id}.{timestamp}.{body}``, ``v1,<base64>``. Cases cover the
positive path plus negative paths (wrong secret, tampered body, stale timestamp, partial headers,
wrong scheme) and the regressions that must not move (svix-*, GitLab X-Gitlab-Token, dual-token).

    HERMES_HOME=$(mktemp -d) python evals/gateway_status_render/standard_webhooks_ab.py [--port 18644]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from unittest.mock import AsyncMock

import aiohttp


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL).stdout.strip()


def _sign(secret: str, msg_id: str, ts: str, body: bytes) -> str:
    key = base64.b64decode(secret.removeprefix("whsec_")) if secret.startswith("whsec_") else secret.encode()
    digest = hmac.new(key, f"{msg_id}.{ts}.".encode() + body, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode()


def _std_headers(secret: str, body: bytes, *, msg_id: str = "msg_1", ts: str | None = None, sign_with: str | None = None):
    ts = ts or str(int(time.time()))
    return {"webhook-id": msg_id, "webhook-timestamp": ts, "webhook-signature": _sign(sign_with or secret, msg_id, ts, body)}


async def main(port: int) -> None:
    sys.path.insert(0, os.getcwd())
    from gateway.config import PlatformConfig
    from gateway.platforms.webhook import WebhookAdapter

    raw_secret = "gitlab-legacy-token"
    whsec = "whsec_" + base64.b64encode(b"0123456789abcdef0123456789abcdef").decode()
    routes = {
        "raw": {"secret": raw_secret, "prompt": "raw route"},
        "std": {"secret": whsec, "prompt": "std route"},
    }
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": port, "routes": routes}))
    adapter.handle_message = AsyncMock()
    assert await adapter.connect(), "adapter did not bind"
    body = b'{"event_type":"invoice.paid","n":1}'
    stale = str(int(time.time()) - 600)
    cases = [
        ("std_whsec_valid", "std", _std_headers(whsec, body), 202),
        ("std_raw_secret_valid", "raw", _std_headers(raw_secret, body, msg_id="msg_2"), 202),
        ("std_wrong_secret", "std", _std_headers(whsec, body, msg_id="msg_3", sign_with="whsec_" + base64.b64encode(b"x" * 32).decode()), 401),
        ("std_tampered_body", "std", _std_headers(whsec, b'{"event_type":"invoice.paid","n":2}', msg_id="msg_4"), 401),
        ("std_stale_timestamp_replay", "std", _std_headers(whsec, body, msg_id="msg_5", ts=stale), 401),
        ("std_partial_headers_fail_closed", "std", {"webhook-id": "msg_6", "webhook-timestamp": str(int(time.time()))}, 401),
        ("std_wrong_scheme_v1a", "std", {**_std_headers(whsec, body, msg_id="msg_7"), "webhook-signature": "v1a," + base64.b64encode(b"\x00" * 64).decode()}, 401),
        ("std_replay_same_id", "std", _std_headers(whsec, body, msg_id="msg_1"), 200),  # dedupe => 200 duplicate
        ("svix_still_valid", "std", {"svix-id": "svx_1", "svix-timestamp": str(int(time.time())), "svix-signature": _sign(whsec, "svx_1", str(int(time.time())), body)}, 202),
        ("gitlab_token_only_still_valid", "raw", {"X-Gitlab-Token": raw_secret}, 202),
        # GitLab >= 19 sends webhook-id/webhook-timestamp on EVERY delivery and webhook-signature only when a
        # signing token is configured (lib/gitlab/web_hooks.rb, app/services/web_hook_service.rb): a legacy
        # secret-token install must keep working.
        ("gitlab_token_with_unsigned_webhook_id_ts", "raw", {"X-Gitlab-Token": raw_secret, "webhook-id": "gl_9", "webhook-timestamp": str(int(time.time()))}, 202),
        ("gitlab_dual_token_signed_with_other_key", "raw", {**_std_headers(raw_secret, body, msg_id="msg_8", sign_with="some-other-signing-token"), "X-Gitlab-Token": raw_secret}, None),
        ("no_auth", "std", {}, 401),
    ]
    results: dict = {"head": _git_head(), "cases": []}
    async with aiohttp.ClientSession() as http:
        for name, route, headers, expect in cases:
            # svix case: recompute with one shared ts so header and signature agree
            if name == "svix_still_valid":
                ts = str(int(time.time()))
                headers = {"svix-id": "svx_1", "svix-timestamp": ts, "svix-signature": _sign(whsec, "svx_1", ts, body)}
            async with http.post(f"http://127.0.0.1:{port}/webhooks/{route}", data=body, headers={**headers, "Content-Type": "application/json"}) as resp:
                text = await resp.text()
                results["cases"].append({"case": name, "route": route, "status": resp.status, "expected": expect,
                                         "ok": (expect is None or resp.status == expect), "body": text[:120]})
    await adapter.disconnect()
    results["mismatches"] = [c["case"] for c in results["cases"] if not c["ok"]]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18644)
    asyncio.run(main(ap.parse_args().port))
