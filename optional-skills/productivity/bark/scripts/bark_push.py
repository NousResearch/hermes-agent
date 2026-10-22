#!/usr/bin/env python3
"""Send a Bark push notification.

Standard library only. The device key is read from the ``BARK_TOKEN``
environment variable only — there is no command-line flag for it, so the key
never lands in shell history, a process list or a tool transcript. The server
comes from ``--server`` and defaults to ``https://api.day.app``; a self-hosted
address is the skill's ``skills.config.bark.server`` setting, passed in by the
caller. No server environment variable is read.

The device key is part of the request URL, so every message this script prints
is redacted before it reaches stdout or stderr.

Examples:
    python3 bark_push.py --title "Hermes" --body "Build finished"
    python3 bark_push.py --body "Disk almost full" --level timeSensitive
    python3 bark_push.py --body "done" --id build-42
    python3 bark_push.py --id build-42 --delete
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_SERVER = "https://api.day.app"
LEVELS = ("active", "timeSensitive", "passive", "critical")
REDACTED = "***"
TOKEN_HELP = (
    "BARK_TOKEN is not set — open Bark on the iPhone, copy the key from the "
    "test URL on its home page, and add BARK_TOKEN=<key> to ~/.hermes/.env"
)


class BarkError(RuntimeError):
    """The Bark server could not be reached or answered with an error."""


def redact(text: str, token: str) -> str:
    """Replace the device key with ``***`` so it never reaches a transcript."""
    if not token:
        return text
    return text.replace(token, REDACTED)


def resolve_token() -> str:
    token = (os.environ.get("BARK_TOKEN") or "").strip()
    if not token:
        raise SystemExit(TOKEN_HELP)
    return token


def resolve_server(explicit: str | None = None) -> str:
    """Return the server base URL; ``BARK_SERVER`` is deliberately not read."""
    server = (explicit or DEFAULT_SERVER).strip()
    return server.rstrip("/")


def build_payload(
    *,
    body: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    markdown: str | None = None,
    level: str | None = None,
    sound: str | None = None,
    group: str | None = None,
    url: str | None = None,
    copy: str | None = None,
    badge: int | None = None,
    call: bool = False,
    push_id: str | None = None,
    delete: bool = False,
) -> dict:
    """Build the JSON body Bark accepts, dropping unset fields."""
    if delete:
        if not push_id:
            raise ValueError("--delete needs --id")
        # Bark deletes by id alone; body, title and the rest are irrelevant.
        return {"id": push_id, "delete": "1"}
    if markdown:
        body = None
    if not body and not markdown:
        raise ValueError("a push needs --body or --markdown")
    if level is not None and level not in LEVELS:
        raise ValueError(f"unsupported level {level!r}; use one of {', '.join(LEVELS)}")

    payload: dict = {
        "title": title,
        "subtitle": subtitle,
        "body": body,
        "markdown": markdown,
        "level": level,
        "sound": sound,
        "group": group,
        "url": url,
        "copy": copy,
        "badge": badge,
        "id": push_id,
    }
    payload = {key: value for key, value in payload.items() if value not in (None, "")}
    if call:
        payload["call"] = "1"
    if delete:
        payload["delete"] = "1"
    return payload


def send_push(
    payload: dict,
    token: str,
    server: str = DEFAULT_SERVER,
    timeout: float = 10.0,
    opener=None,
) -> dict:
    """POST the payload to ``<server>/<token>`` and return the parsed response."""
    endpoint = f"{server.rstrip('/')}/{token}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise BarkError(
            redact(f"Bark returned HTTP {exc.code}: {detail}", token)
        ) from None
    except urllib.error.URLError as exc:
        raise BarkError(redact(f"cannot reach {server}: {exc.reason}", token)) from None
    except (TimeoutError, OSError) as exc:
        # A read timeout arrives as TimeoutError, not URLError; report it without a traceback.
        raise BarkError(
            redact(f"cannot reach {server}: {exc or type(exc).__name__}", token)
        ) from None
    except http.client.HTTPException as exc:
        raise BarkError(
            redact(f"invalid response from {server}: {type(exc).__name__}: {exc}", token)
        ) from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise BarkError(
            redact(f"unexpected response from {server}: {raw[:200]}", token)
        ) from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a Bark push notification to an iPhone.",
        epilog=TOKEN_HELP,
    )
    parser.add_argument("--body", help="notification body")
    parser.add_argument("--title", help="notification title")
    parser.add_argument("--subtitle", help="notification subtitle")
    parser.add_argument("--markdown", help="markdown body (replaces --body in the app)")
    parser.add_argument("--level", choices=LEVELS, help="interruption level")
    parser.add_argument("--sound", help="ringtone name, e.g. minuet")
    parser.add_argument("--group", help="notification group")
    parser.add_argument("--url", help="URL opened when the push is tapped")
    parser.add_argument("--copy", help="text copied with the push")
    parser.add_argument("--badge", type=int, help="app badge number; 0 clears it")
    parser.add_argument("--call", action="store_true", help="repeat the ringtone for 30 seconds")
    parser.add_argument("--id", dest="push_id", help="notification id; reusing it replaces the push")
    parser.add_argument("--delete", action="store_true", help="withdraw the push with this --id")
    parser.add_argument(
        "--server",
        help="Bark server base URL (default https://api.day.app; self-hosted: skills.config.bark.server)",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="request timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="print the payload without sending")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = build_payload(
            body=args.body,
            title=args.title,
            subtitle=args.subtitle,
            markdown=args.markdown,
            level=args.level,
            sound=args.sound,
            group=args.group,
            url=args.url,
            copy=args.copy,
            badge=args.badge,
            call=args.call,
            push_id=args.push_id,
            delete=args.delete,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    token = resolve_token()
    try:
        result = send_push(payload, token, resolve_server(args.server), args.timeout)
    except BarkError as exc:
        print(f"error: {redact(str(exc), token)}", file=sys.stderr)
        return 1
    print(redact(json.dumps(result, ensure_ascii=False), token))
    return 0 if result.get("code") == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
