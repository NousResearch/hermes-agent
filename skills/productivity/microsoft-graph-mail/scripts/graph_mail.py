#!/usr/bin/env python3
"""Read-only Microsoft Graph mail CLI for Hermes Agent."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _log_sanitizer import _sanitize_for_log
from microsoft_auth import get_valid_access_token

GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
MAX_RESULTS = 50
DEFAULT_RESULTS = 10
MAX_BODY_LENGTH = 50_000

try:
    httpx = importlib.import_module("httpx")
except ModuleNotFoundError:
    httpx = None


def _strip_html(value: str) -> str:
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _json_dump(value: Any) -> str:
    serialized = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        serialized
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _bounded_text(value: str, limit: int = MAX_BODY_LENGTH) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n\n[truncated]"


def _email_address(entry: dict | None) -> str:
    if not entry:
        return ""
    email_address = entry.get("emailAddress") or {}
    name = email_address.get("name") or ""
    address = email_address.get("address") or ""
    if name and address:
        return f"{name} <{address}>"
    return address or name


def _recipient_list(values: list[dict] | None) -> list[str]:
    return [_email_address(item) for item in values or [] if _email_address(item)]


def _summarize_message(message: dict) -> dict:
    return {
        "id": message.get("id", ""),
        "conversationId": message.get("conversationId", ""),
        "from": _email_address(message.get("from")),
        "to": _recipient_list(message.get("toRecipients")),
        "subject": message.get("subject", ""),
        "date": message.get("receivedDateTime", ""),
        "snippet": message.get("bodyPreview", ""),
        "isRead": message.get("isRead"),
        "importance": message.get("importance", ""),
    }


def _full_message(message: dict) -> dict:
    result = _summarize_message(message)
    result["cc"] = _recipient_list(message.get("ccRecipients"))
    result["webLink"] = message.get("webLink", "")
    body = message.get("body") or {}
    content = body.get("content") or ""
    if (body.get("contentType") or "").lower() == "html":
        content = _strip_html(content)
    result["body"] = _bounded_text(content)
    return result


def _require_httpx():
    if httpx is None:
        print("ERROR: Missing dependency 'httpx'. Install Hermes with its core dependencies.", file=sys.stderr)
        sys.exit(1)
    return httpx


def _graph_get(client: Any, token: str, path_or_url: str, params: dict | None = None) -> dict:
    url = path_or_url if path_or_url.startswith("http") else f"{GRAPH_ROOT}{path_or_url}"
    try:
        response = client.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
    except Exception as exc:
        print(f"ERROR: Microsoft Graph request failed: {_sanitize_for_log(exc)}", file=sys.stderr)
        sys.exit(1)

    if response.status_code >= 400:
        message = response.text
        try:
            payload = response.json()
            error = payload.get("error") or {}
            message = error.get("message") or error.get("code") or message
        except Exception:
            pass
        print(f"ERROR: Microsoft Graph request failed ({response.status_code}): {_sanitize_for_log(message)}", file=sys.stderr)
        sys.exit(1)

    try:
        return response.json()
    except Exception as exc:
        print(f"ERROR: Microsoft Graph response was not JSON: {_sanitize_for_log(exc)}", file=sys.stderr)
        sys.exit(1)


def _limit(value: int) -> int:
    return max(1, min(value, MAX_RESULTS))


def _collect_messages(client: Any, token: str, params: dict, limit: int) -> list[dict]:
    messages: list[dict] = []
    next_url: str | None = "/me/messages"
    next_params: dict | None = params
    while next_url and len(messages) < limit:
        payload = _graph_get(client, token, next_url, params=next_params)
        messages.extend(payload.get("value", []))
        next_url = payload.get("@odata.nextLink")
        next_params = None
    return messages[:limit]


def list_messages(args) -> None:
    token = get_valid_access_token()
    httpx_module = _require_httpx()
    limit = _limit(args.max)
    params = {
        "$top": min(limit, 25),
        "$select": "id,conversationId,from,toRecipients,subject,receivedDateTime,bodyPreview,isRead,importance",
        "$orderby": "receivedDateTime desc",
    }
    with httpx_module.Client(timeout=30) as client:
        messages = _collect_messages(client, token, params, limit)
    print(_json_dump([_summarize_message(message) for message in messages]))


def search_messages(args) -> None:
    token = get_valid_access_token()
    httpx_module = _require_httpx()
    limit = _limit(args.max)
    params = {
        "$top": min(limit, 25),
        "$select": "id,conversationId,from,toRecipients,subject,receivedDateTime,bodyPreview,isRead,importance",
        "$search": f'"{args.query}"',
    }
    with httpx_module.Client(timeout=30) as client:
        messages = _collect_messages(client, token, params, limit)
    print(_json_dump([_summarize_message(message) for message in messages]))


def get_message(args) -> None:
    token = get_valid_access_token()
    httpx_module = _require_httpx()
    message_id = quote(args.message_id, safe="")
    params = {
        "$select": "id,conversationId,from,toRecipients,ccRecipients,subject,receivedDateTime,body,bodyPreview,isRead,importance,webLink",
    }
    with httpx_module.Client(timeout=30) as client:
        message = _graph_get(client, token, f"/me/messages/{message_id}", params=params)
    print(_json_dump(_full_message(message)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Microsoft Graph mail for the authenticated Hermes user")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="List latest messages")
    list_parser.add_argument("--max", type=int, default=DEFAULT_RESULTS, help=f"Maximum messages (1-{MAX_RESULTS})")
    list_parser.set_defaults(func=list_messages)

    search_parser = sub.add_parser("search", help="Search messages using Microsoft Graph mail search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max", type=int, default=DEFAULT_RESULTS, help=f"Maximum messages (1-{MAX_RESULTS})")
    search_parser.set_defaults(func=search_messages)

    get_parser = sub.add_parser("get", help="Read a message by ID")
    get_parser.add_argument("message_id", help="Message ID returned by search/list")
    get_parser.set_defaults(func=get_message)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()