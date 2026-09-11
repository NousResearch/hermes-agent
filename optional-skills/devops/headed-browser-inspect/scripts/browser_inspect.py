#!/usr/bin/env python3
"""Inspect a headed Playwright page over a unix socket.

A second Playwright connect() cannot see pages owned by another client.
Keep one hold process attached to the visible tab and RPC evaluate/goto
through this socket. Skip navigation when the URL is already loaded.

Never print cookies. Never copy profile cookies into a temp page.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

WS = os.environ.get("HERMES_BROWSER_WS", "ws://127.0.0.1:9377/camoufox")
SOCK = Path(os.environ.get("HERMES_INSPECT_SOCK", str(Path.home() / ".hermes/takeover/inspect.sock")))


def norm_url(url: str) -> str:
    return (url or "").split("?")[0].rstrip("/")


def serve_loop(page) -> None:
    SOCK.parent.mkdir(parents=True, exist_ok=True)
    if SOCK.exists():
        SOCK.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    srv.listen(1)
    srv.settimeout(0.5)
    print(f"inspect serve {SOCK} {page.url}", flush=True)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            try:
                line = conn.makefile("r", encoding="utf-8").readline()
                req = json.loads(line) if line else {}
                op = req.get("op")
                out = {"ok": True}
                if op == "ping":
                    out["url"] = page.url
                    out["title"] = page.title()
                elif op == "goto":
                    url = req["url"]
                    if norm_url(page.url) != norm_url(url):
                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        time.sleep(float(req.get("wait") or 2))
                        out["navigated"] = True
                    else:
                        out["navigated"] = False
                    out["url"] = page.url
                elif op == "eval":
                    out["result"] = page.evaluate(req["js"])
                    out["url"] = page.url
                else:
                    out = {"error": f"unknown op {op}"}
                conn.sendall((json.dumps(out, default=str) + "\n").encode())
            except Exception as e:
                try:
                    conn.sendall((json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n").encode())
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        srv.close()
        try:
            SOCK.unlink()
        except FileNotFoundError:
            pass


def rpc(msg: dict, timeout: float = 90) -> dict:
    if not SOCK.exists():
        raise SystemExit("inspect server not running; start hold_page.py")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(str(SOCK))
    sock.sendall((json.dumps(msg) + "\n").encode())
    line = sock.makefile("r", encoding="utf-8").readline()
    sock.close()
    if not line:
        raise SystemExit("inspect server closed")
    data = json.loads(line)
    if data.get("error"):
        raise SystemExit(data["error"])
    return data


def cmd_eval(args) -> None:
    if args.url:
        nav = rpc({"op": "goto", "url": args.url, "wait": args.wait})
        data = rpc({"op": "eval", "js": args.js})
        print(json.dumps({"navigated": nav.get("navigated"), "url": data.get("url"), "result": data.get("result")}, indent=2, default=str))
        return
    data = rpc({"op": "eval", "js": args.js})
    print(json.dumps(data.get("result"), indent=2, default=str))


def cmd_ping(_args) -> None:
    print(json.dumps(rpc({"op": "ping"}), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="RPC client for a held headed browser tab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ping")
    p.set_defaults(func=cmd_ping)
    e = sub.add_parser("eval")
    e.add_argument("url", nargs="?")
    e.add_argument("--js", default="() => ({url: location.href, title: document.title, text: (document.body.innerText||'').slice(0,2000)})")
    e.add_argument("--wait", type=float, default=2.0)
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
