"""Wire-contract A/B for Slack native task-card streams (#87743).

Runs the REAL SlackAdapter.send_native_task_card_progress / stop_native_task_card_progress and
send_draft/_seal_stream paths against a REAL slack_sdk AsyncWebClient whose ``base_url`` points at a
local aiohttp receiver that records every request body. The receiver enforces Slack's documented
mutual-exclusion rule for chat.startStream/appendStream/stopStream (``markdown_text`` and
``chunks`` cannot both be present → ``cannot_provide_both_markdown_text_and_chunks``).

THIS IS WIRE-CONTRACT PROOF, NOT SLACK LIVE: no Slack workspace or token is involved. It proves what
bytes the adapter puts on the wire and that they satisfy the documented contract.

Usage:
  <python-with-slack_sdk> evals/slack_stream_wire_contract.py --repo <worktree> --out <json>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

EXCLUSIVE_METHODS = {"chat.startStream", "chat.appendStream", "chat.stopStream"}


def _make_app(log):
    from aiohttp import web

    async def handler(request):
        method = request.match_info["method"]
        ctype = request.headers.get("Content-Type", "")
        if "json" in ctype:
            body = await request.json()
        else:
            body = dict(await request.post())
        entry = {"method": method, "content_type": ctype, "body": body}
        log.append(entry)
        if method in EXCLUSIVE_METHODS and "markdown_text" in body and "chunks" in body:
            entry["response"] = {"ok": False, "error": "cannot_provide_both_markdown_text_and_chunks"}
        else:
            entry["response"] = {"ok": True, "channel": body.get("channel"), "ts": body.get("ts") or "1700000000.000100"}
        return web.json_response(entry["response"])

    app = web.Application()
    app.router.add_post("/api/{method}", handler)
    return app


async def run(repo: str, out: str):
    from aiohttp import web

    sys.path.insert(0, repo)
    for m in list(sys.modules):
        if m.startswith(("gateway", "plugins", "tools", "hermes", "agent")):
            del sys.modules[m]
    from slack_sdk.web.async_client import AsyncWebClient
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter, SLACK_AVAILABLE

    assert SLACK_AVAILABLE, "slack_sdk must be importable for a real-client wire probe"

    log: list = []
    runner = web.AppRunner(_make_app(log))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}/api/"

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-wire-probe"))

    class _App:  # minimal stand-in for slack_bolt AsyncApp: only .client is read by these paths
        client = AsyncWebClient(token="xoxb-wire-probe", base_url=base_url)

    adapter._app = _App()
    metadata = {"thread_id": "1700000000.000001", "user_id": "U1", "recipient_team_id": "T1", "recipient_user_id": "U1"}
    results = {}

    # --- Task-card rail (the bug): start + append(chunks) + stop
    r1 = await adapter.send_native_task_card_progress(
        "C1", [{"id": "call-1", "title": "terminal - ls", "status": "in_progress"}],
        metadata=metadata, fallback_text="Hermes is working\n- terminal - ls - running")
    r2 = await adapter.send_native_task_card_progress(
        "C1", [{"id": "call-1", "title": "terminal - ls", "status": "complete"}],
        metadata=metadata, fallback_text="Hermes is working\n- terminal - ls - complete")
    await adapter.stop_native_task_card_progress("C1", metadata=metadata)
    results["task_card"] = {"first": {"success": r1.success, "error": r1.error},
                            "second": {"success": r2.success, "error": r2.error}}

    # --- Text stream rail (regression): start(markdown_text) + append(markdown_text) + stop(markdown_text)
    d1 = await adapter.send_draft("C2", 1, "Hello", metadata=metadata)
    d2 = await adapter.send_draft("C2", 1, "Hello world", metadata=metadata)
    stream = adapter._active_streams.get("C2")
    sealed = await adapter._seal_stream("C2", stream, final_text="Hello world!") if stream else None
    results["text_stream"] = {"first": {"success": d1.success, "error": d1.error},
                              "second": {"success": d2.success, "error": d2.error}, "sealed": sealed}

    await runner.cleanup()

    violations = [e for e in log if e["method"] in EXCLUSIVE_METHODS and "markdown_text" in e["body"] and "chunks" in e["body"]]
    summary = {
        "repo": repo, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": "WIRE-CONTRACT PROOF (local receiver, slack_sdk real client) — NOT Slack live",
        "slack_sdk_version": __import__("slack_sdk.version", fromlist=["__version__"]).__version__,
        "requests": [{"method": e["method"], "content_type": e["content_type"],
                      "body_keys": sorted(e["body"].keys()), "body": e["body"], "response": e["response"]} for e in log],
        "violations": [{"method": v["method"], "body_keys": sorted(v["body"].keys())} for v in violations],
        "results": results,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    seq = [(e["method"], sorted(e["body"].keys()), e["response"].get("error")) for e in log]
    for s in seq:
        print(*s)
    print("violations:", len(violations), "| task_card:", results["task_card"], "| text_stream:", results["text_stream"])
    return 1 if violations else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("HERMES_HOME", os.path.join(os.path.dirname(a.out), "hermes-home-probe"))
    os.makedirs(os.environ["HERMES_HOME"], exist_ok=True)
    sys.exit(asyncio.run(run(os.path.abspath(a.repo), a.out)))


if __name__ == "__main__":
    main()
