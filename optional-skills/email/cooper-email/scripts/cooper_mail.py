#!/usr/bin/env python3
"""Small stdlib-only client for the Cooper Email HTTP API.

Subcommands print one JSON object to stdout. The API key comes from the
COOPER_API_KEY environment variable and is never printed, except once by
`onboard`, which is the only moment the service returns it.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = os.environ.get("COOPER_BASE_URL", "https://cooperemail.com/api/v1").rstrip("/")
USER_AGENT = "hermes-cooper-email-skill/1.0"


def _key(required=True):
    key = os.environ.get("COOPER_API_KEY", "").strip()
    if required and not key:
        raise SystemExit(json.dumps({"ok": False, "error": "COOPER_API_KEY is not set"}))
    return key


def _inbox(args):
    inbox = getattr(args, "inbox", None) or os.environ.get("COOPER_INBOX_ID", "").strip()
    if not inbox:
        raise SystemExit(json.dumps({"ok": False, "error": "pass --inbox or set COOPER_INBOX_ID"}))
    return inbox


def build_request(method, path, body=None, auth=True, idempotency_key=None, query=None):
    url = BASE + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v not in (None, "")})
    headers = {"user-agent": USER_AGENT, "accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["content-type"] = "application/json"
    if auth:
        headers["authorization"] = "Bearer " + _key()
    if idempotency_key:
        headers["idempotency-key"] = idempotency_key
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def call(req, timeout=70):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8") or "{}"
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": raw[:500]}
        return err.code, payload


def cmd_onboard(args):
    body = {"username": args.username}
    if args.display_name:
        body["display_name"] = args.display_name
    if args.owner_email:
        body["owner_email"] = args.owner_email
    return call(build_request("POST", "/onboard", body, auth=False))


def cmd_me(args):
    return call(build_request("GET", "/me"))


def cmd_send(args):
    body = {"to": args.to, "subject": args.subject, "text": args.text}
    if args.cc:
        body["cc"] = args.cc
    return call(build_request("POST", f"/inboxes/{_inbox(args)}/messages", body,
                              idempotency_key=args.idempotency_key))


def cmd_list(args):
    query = {"unread": "true" if args.unread else None, "limit": args.limit}
    return call(build_request("GET", f"/inboxes/{_inbox(args)}/messages", query=query))


def cmd_read(args):
    return call(build_request("GET", f"/inboxes/{_inbox(args)}/messages/{args.message_id}"))


def cmd_reply(args):
    body = {"text": args.text}
    return call(build_request("POST", f"/inboxes/{_inbox(args)}/messages/{args.message_id}/reply", body,
                              idempotency_key=args.idempotency_key))


def cmd_wait(args):
    return call(build_request("GET", f"/inboxes/{_inbox(args)}/wait", query={"wait": args.wait}))


def cmd_wait_code(args):
    query = {"wait": args.wait, "from": args.sender}
    return call(build_request("GET", f"/inboxes/{_inbox(args)}/wait-for-code", query=query))


def parser():
    p = argparse.ArgumentParser(description="Cooper Email helper")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("onboard", help="create an account, inbox, and API key (no auth)")
    s.add_argument("username")
    s.add_argument("--display-name")
    s.add_argument("--owner-email")
    s.set_defaults(fn=cmd_onboard)

    sub.add_parser("me", help="check the API key").set_defaults(fn=cmd_me)

    s = sub.add_parser("send", help="send a plain-text email")
    s.add_argument("--inbox")
    s.add_argument("--to", action="append", required=True)
    s.add_argument("--cc", action="append")
    s.add_argument("--subject", required=True)
    s.add_argument("--text", required=True)
    s.add_argument("--idempotency-key")
    s.set_defaults(fn=cmd_send)

    s = sub.add_parser("list", help="list messages")
    s.add_argument("--inbox")
    s.add_argument("--unread", action="store_true")
    s.add_argument("--limit", type=int)
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("read", help="read one message")
    s.add_argument("message_id")
    s.add_argument("--inbox")
    s.set_defaults(fn=cmd_read)

    s = sub.add_parser("reply", help="reply on the same thread")
    s.add_argument("message_id")
    s.add_argument("--inbox")
    s.add_argument("--text", required=True)
    s.add_argument("--idempotency-key")
    s.set_defaults(fn=cmd_reply)

    s = sub.add_parser("wait", help="hold until new mail arrives (max 55 s)")
    s.add_argument("--inbox")
    s.add_argument("--wait", type=int, default=55)
    s.set_defaults(fn=cmd_wait)

    s = sub.add_parser("wait-code", help="hold until a one-time code arrives (max 55 s)")
    s.add_argument("--inbox")
    s.add_argument("--wait", type=int, default=55)
    s.add_argument("--from", dest="sender")
    s.set_defaults(fn=cmd_wait_code)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if getattr(args, "wait", None) is not None:
        args.wait = max(0, min(55, args.wait))
    status, payload = args.fn(args)
    print(json.dumps({"ok": 200 <= status < 300, "status": status, "data": payload}))
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
