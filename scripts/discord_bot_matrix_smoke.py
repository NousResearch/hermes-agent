#!/usr/bin/env python3
"""Discord Hermes bot-to-bot routing smoke test.

Runs a pairwise explicit-mention matrix across Hermes Discord bot profiles and a
negative no-mention matrix. It never prints tokens.

Usage examples:
  python3 scripts/discord_bot_matrix_smoke.py \
    --profile default:~/.hermes \
    --profile case:~/.hermes/profiles/nj-case-law-ftd \
    --profile statutes:~/.hermes/profiles/nj-statutes-ftd

Requirements:
  - Each profile home has .env with DISCORD_BOT_TOKEN and DISCORD_HOME_CHANNEL.
  - The gateway processes should already be running with DISCORD_ALLOW_BOTS=mentions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "HermesBotMatrixSmoke/1.0"


def read_env(home: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    env = home.expanduser() / ".env"
    for line in env.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def req(token: str, method: str, path: str, data: dict | None = None):
    body = None
    headers = {"Authorization": "Bot " + token, "User-Agent": UA}
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(
        "https://discord.com/api/v10" + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            txt = resp.read().decode()
            return resp.status, json.loads(txt) if txt else None
    except urllib.error.HTTPError as e:
        txt = e.read().decode(errors="replace")
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = txt[:500]
        return e.code, {"error": parsed}


def channel_messages(token: str, channel_id: str, after: str | None = None, limit: int = 50):
    qs = {"limit": str(limit)}
    if after:
        qs["after"] = str(after)
    status, data = req(token, "GET", f"/channels/{channel_id}/messages?" + urllib.parse.urlencode(qs))
    if status >= 300:
        raise RuntimeError(f"fetch {channel_id} failed {status}: {data}")
    return data


def send(token: str, channel_id: str, content: str, parse_users: bool = True):
    allowed = {"parse": ["users"] if parse_users else []}
    status, data = req(token, "POST", f"/channels/{channel_id}/messages", {"content": content, "allowed_mentions": allowed})
    if status >= 300:
        raise RuntimeError(f"send {channel_id} failed {status}: {data}")
    return data


def load_bots(profile_args: list[str]) -> dict[str, dict]:
    bots = {}
    for item in profile_args:
        if ":" not in item:
            raise SystemExit(f"bad --profile {item!r}; expected name:/path/to/home")
        name, raw_home = item.split(":", 1)
        home = Path(raw_home).expanduser()
        vals = read_env(home)
        token = vals.get("DISCORD_BOT_TOKEN")
        channel = vals.get("DISCORD_HOME_CHANNEL")
        if not token or not channel:
            raise SystemExit(f"{name}: missing DISCORD_BOT_TOKEN or DISCORD_HOME_CHANNEL in {home}/.env")
        status, me = req(token, "GET", "/users/@me")
        if status >= 300:
            raise SystemExit(f"{name}: /users/@me failed {status}: {me}")
        bots[name] = {"token": token, "channel": channel, "id": me["id"], "username": me["username"]}
    if len(bots) < 2:
        raise SystemExit("need at least two --profile entries")
    return bots


def explicit_matrix(bots: dict[str, dict], timeout: int) -> list[dict]:
    nonce = "btb-" + time.strftime("%Y%m%d%H%M%S") + "-" + str(random.randint(1000, 9999))
    results: list[dict] = []
    for sender in bots:
        for receiver in bots:
            if sender == receiver:
                continue
            s, r = bots[sender], bots[receiver]
            test_id = f"{nonce}-{sender}-to-{receiver}"
            # Current Hermes bot-to-bot admission requires the structured
            # BOT_MSG v1 envelope; a raw mention plus free-form body is
            # intentionally rejected as malformed.
            content = "\n".join(
                [
                    f"<@{r['id']}>",
                    "BOT_MSG v1",
                    "reply_expected: true",
                    "kind: status",
                    f"correlation_id: {test_id}",
                    "---",
                    (
                        f"BOT-ROUTING-SMOKE {test_id}. Reply once with exactly: "
                        f"ACK {test_id}. Do not mention any bot. Do not ask questions."
                    ),
                ]
            )
            sent = send(s["token"], r["channel"], content, parse_users=True)
            results.append({"sender": sender, "receiver": receiver, "test_id": test_id, "channel": r["channel"], "sent_id": sent["id"]})
            time.sleep(1.5)

    # Hermes Discord replies are commonly in the auto-created per-message thread.
    deadline = time.time() + timeout
    pending = {x["test_id"]: x for x in results}
    while pending and time.time() < deadline:
        for tid, x in list(pending.items()):
            r = bots[x["receiver"]]
            status, data = req(r["token"], "GET", f"/channels/{x['sent_id']}/messages?limit=20")
            if status == 404:
                continue
            if status >= 300:
                x["error"] = {"status": status, "data": data}
                pending.pop(tid, None)
                continue
            matches = []
            for m in data:
                if m.get("author", {}).get("id") == r["id"] and tid in (m.get("content") or ""):
                    matches.append({
                        "id": m["id"],
                        "content": (m.get("content") or "")[:300],
                        "mentions": [u.get("id") for u in m.get("mentions", [])],
                    })
            if matches:
                x["responses"] = matches
                pending.pop(tid, None)
        if pending:
            time.sleep(5)
    return results


def negative_matrix(bots: dict[str, dict], wait: int) -> list[dict]:
    nonce = "neg-" + time.strftime("%Y%m%d%H%M%S") + "-" + str(random.randint(1000, 9999))
    results = []
    for sender in bots:
        for receiver in bots:
            if sender == receiver:
                continue
            s, r = bots[sender], bots[receiver]
            test_id = f"{nonce}-{sender}-to-{receiver}"
            content = f"BOT-ROUTING-NEGATIVE {test_id}. This intentionally mentions no bot. No Hermes bot should respond."
            sent = send(s["token"], r["channel"], content, parse_users=False)
            results.append({"sender": sender, "receiver": receiver, "test_id": test_id, "channel": r["channel"], "sent_id": sent["id"]})
            time.sleep(1)
    time.sleep(wait)
    for x in results:
        r = bots[x["receiver"]]
        status, data = req(r["token"], "GET", f"/channels/{x['sent_id']}/messages?limit=20")
        x["thread_status"] = status
        if status < 300:
            x["thread_messages"] = [
                {"author_id": m.get("author", {}).get("id"), "author": m.get("author", {}).get("username"), "content": (m.get("content") or "")[:160]}
                for m in data
            ]
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="append", required=True, help="name:/path/to/hermes-home")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--negative-wait", type=int, default=45)
    args = ap.parse_args()
    bots = load_bots(args.profile)
    explicit = explicit_matrix(bots, args.timeout)
    negative = negative_matrix(bots, args.negative_wait)
    safe_bots = {k: {"id": v["id"], "username": v["username"], "channel": v["channel"]} for k, v in bots.items()}
    out = {"bots": safe_bots, "explicit": explicit, "negative": negative}
    print(json.dumps(out, indent=2))
    explicit_failed = [x for x in explicit if not x.get("responses")]
    negative_failed = [x for x in negative if x.get("thread_status") != 404]
    return 1 if explicit_failed or negative_failed else 0


if __name__ == "__main__":
    sys.exit(main())
