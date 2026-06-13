from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hermes_home import get_hermes_home

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents.readonly",
]
TOKEN_PATH = get_hermes_home() / "google_token.json"


def _gws_binary() -> str | None:
    return shutil.which("gws")


def _ensure_authenticated() -> None:
    if not TOKEN_PATH.exists():
        print(f"Missing Google token file: {TOKEN_PATH}", file=sys.stderr)
        raise SystemExit(1)


def _run_gws(
    parts: list[str],
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    _ensure_authenticated()
    binary = _gws_binary()
    if not binary:
        raise SystemExit("gws CLI is not installed")

    cmd = [binary, *parts]
    if params is not None:
        cmd.extend(["--params", json.dumps(params, separators=(",", ":"))])
    if body is not None:
        cmd.extend(["--body", json.dumps(body, separators=(",", ":"))])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.returncode)
    if not proc.stdout.strip():
        return {}
    return json.loads(proc.stdout)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_later(days: int) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(days=days))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _headers_by_lower(message: dict[str, Any]) -> dict[str, str]:
    headers = message.get("payload", {}).get("headers", [])
    return {
        str(item.get("name", "")).lower(): str(item.get("value", ""))
        for item in headers
        if isinstance(item, dict)
    }


def _message_summary(message: dict[str, Any]) -> dict[str, Any]:
    headers = _headers_by_lower(message)
    return {
        "id": message.get("id"),
        "threadId": message.get("threadId"),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "snippet": message.get("snippet", ""),
        "labels": message.get("labelIds", []),
    }


def _raw_message(msg: EmailMessage) -> str:
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


def calendar_list(args: argparse.Namespace) -> None:
    params = {
        "calendarId": args.calendar,
        "timeMin": args.start or _iso_now(),
        "timeMax": args.end or _iso_later(7),
        "maxResults": args.max,
    }
    result = _run_gws(["calendar", "events", "list"], params=params)
    print(json.dumps(result))


def gmail_get(args: argparse.Namespace) -> None:
    message = _run_gws(
        ["gmail", "users", "messages", "get"],
        params={"userId": "me", "id": args.message_id, "format": "full"},
    )
    print(json.dumps(_message_summary(message)))


def gmail_search(args: argparse.Namespace) -> None:
    listing = _run_gws(
        ["gmail", "users", "messages", "list"],
        params={"userId": "me", "q": args.query, "maxResults": args.max},
    )
    results = []
    for item in listing.get("messages", []):
        message = _run_gws(
            ["gmail", "users", "messages", "get"],
            params={
                "userId": "me",
                "id": item.get("id"),
                "format": "metadata",
                "metadataHeaders": ["From", "To", "Subject", "Date"],
            },
        )
        results.append(_message_summary(message))
    print(json.dumps(results))


def gmail_send(args: argparse.Namespace) -> None:
    msg = EmailMessage()
    msg["To"] = args.to
    msg["Subject"] = args.subject
    if getattr(args, "cc", ""):
        msg["Cc"] = args.cc
    if getattr(args, "from_header", ""):
        msg["From"] = args.from_header
    if getattr(args, "html", False):
        msg.add_alternative(args.body, subtype="html")
    else:
        msg.set_content(args.body)

    body: dict[str, Any] = {"raw": _raw_message(msg)}
    if getattr(args, "thread_id", ""):
        body["threadId"] = args.thread_id
    _run_gws(["gmail", "users", "messages", "send"], params={"userId": "me"}, body=body)


def gmail_reply(args: argparse.Namespace) -> None:
    original = _run_gws(
        ["gmail", "users", "messages", "get"],
        params={
            "userId": "me",
            "id": args.message_id,
            "format": "metadata",
            "metadataHeaders": ["From", "Subject", "Message-ID"],
        },
    )
    headers = _headers_by_lower(original)
    subject = headers.get("subject", "")
    reply_subject = subject if subject.lower().startswith("re:") else f"Re: {subject}"

    msg = EmailMessage()
    msg["To"] = headers.get("from", "")
    msg["Subject"] = reply_subject
    if getattr(args, "from_header", ""):
        msg["From"] = args.from_header
    message_id = headers.get("message-id", "")
    if message_id:
        msg["In-Reply-To"] = message_id
        msg["References"] = message_id
    msg.set_content(args.body)

    body = {"raw": _raw_message(msg), "threadId": original.get("threadId")}
    _run_gws(["gmail", "users", "messages", "send"], params={"userId": "me"}, body=body)


def get_credentials() -> Any:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
        creds.refresh(Request())
        payload = json.loads(creds.to_json())
        payload["type"] = "authorized_user"
        TOKEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return creds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Google Workspace API helper")
    sub = parser.add_subparsers(dest="command")

    cal = sub.add_parser("calendar-list")
    cal.add_argument("--start", default="")
    cal.add_argument("--end", default="")
    cal.add_argument("--max", type=int, default=25)
    cal.add_argument("--calendar", default="primary")
    cal.set_defaults(func=calendar_list)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
