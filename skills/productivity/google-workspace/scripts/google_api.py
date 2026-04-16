#!/usr/bin/env python3
"""Google Workspace API CLI for Hermes Agent.

Uses the Google Workspace CLI (`gws`) when available, but preserves the
existing Hermes-facing JSON contract and falls back to the Python client
libraries if `gws` is not installed.

Usage:
  python google_api.py gmail search "is:unread" [--max 10]
  python google_api.py gmail get MESSAGE_ID
  python google_api.py gmail send --to user@example.com --subject "Hi" --body "Hello"
  python google_api.py gmail reply MESSAGE_ID --body "Thanks"
  python google_api.py calendar list [--from DATE] [--to DATE] [--calendar primary]
  python google_api.py calendar create --summary "Meeting" --start DATETIME --end DATETIME
  python google_api.py drive search "budget report" [--max 10]
  python google_api.py contacts list [--max 20]
  python google_api.py sheets get SHEET_ID RANGE
  python google_api.py sheets update SHEET_ID RANGE --values '[[...]]'
  python google_api.py sheets append SHEET_ID RANGE --values '[[...]]'
  python google_api.py docs get DOC_ID
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"

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


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _ensure_authenticated():
    if not TOKEN_PATH.exists():
        print("Not authenticated. Run the setup script first:", file=sys.stderr)
        print(f"  python {Path(__file__).parent / 'setup.py'}", file=sys.stderr)
        sys.exit(1)


def _stored_token_scopes() -> list[str]:
    try:
        data = json.loads(TOKEN_PATH.read_text())
    except Exception:
        return list(SCOPES)
    scopes = data.get("scopes")
    if isinstance(scopes, list) and scopes:
        return scopes
    return list(SCOPES)


def _gws_binary() -> str | None:
    override = os.getenv("HERMES_GWS_BIN")
    if override:
        return override
    return shutil.which("gws")


def _gws_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE"] = str(TOKEN_PATH)
    return env


def _run_gws(parts: list[str], *, params: dict | None = None, body: dict | None = None):
    binary = _gws_binary()
    if not binary:
        raise RuntimeError("gws not installed")

    _ensure_authenticated()

    cmd = [binary, *parts]
    if params is not None:
        cmd.extend(["--params", json.dumps(params)])
    if body is not None:
        cmd.extend(["--json", json.dumps(body)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_gws_env(),
    )
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip() or "Unknown gws error"
        print(err, file=sys.stderr)
        sys.exit(result.returncode or 1)

    stdout = result.stdout.strip()
    if not stdout:
        return {}

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        print("ERROR: Unexpected non-JSON output from gws:", file=sys.stderr)
        print(stdout, file=sys.stderr)
        sys.exit(1)


def _headers_dict(msg: dict) -> dict[str, str]:
    return {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}


def _extract_message_body(msg: dict) -> str:
    body = ""
    payload = msg.get("payload", {})
    if payload.get("body", {}).get("data"):
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    elif payload.get("parts"):
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                break
        if not body:
            for part in payload["parts"]:
                if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                    break
    return body


def _extract_doc_text(doc: dict) -> str:
    text_parts = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph", {})
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun", {})
            if text_run.get("content"):
                text_parts.append(text_run["content"])
    return "".join(text_parts)


def _datetime_with_timezone(value: str) -> str:
    if not value:
        return value
    if "T" not in value:
        return value
    if value.endswith("Z"):
        return value
    tail = value[10:]
    if "+" in tail or "-" in tail:
        return value
    return value + "Z"


def get_credentials():
    """Load and refresh credentials from token file."""
    _ensure_authenticated()

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), _stored_token_scopes())
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(
            json.dumps(
                _normalize_authorized_user_payload(json.loads(creds.to_json())),
                indent=2,
            )
        )
    if not creds.valid:
        print("Token is invalid. Re-run setup.", file=sys.stderr)
        sys.exit(1)
    return creds


def build_service(api, version):
    from googleapiclient.discovery import build

    return build(api, version, credentials=get_credentials())


# =========================================================================
# Gmail
# =========================================================================


def gmail_search(args):
    if _gws_binary():
        results = _run_gws(
            ["gmail", "users", "messages", "list"],
            params={"userId": "me", "q": args.query, "maxResults": args.max},
        )
        messages = results.get("messages", [])
        output = []
        for msg_meta in messages:
            msg = _run_gws(
                ["gmail", "users", "messages", "get"],
                params={
                    "userId": "me",
                    "id": msg_meta["id"],
                    "format": "metadata",
                    "metadataHeaders": ["From", "To", "Subject", "Date"],
                },
            )
            headers = _headers_dict(msg)
            output.append(
                {
                    "id": msg["id"],
                    "threadId": msg["threadId"],
                    "from": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": msg.get("snippet", ""),
                    "labels": msg.get("labelIds", []),
                }
            )
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    service = build_service("gmail", "v1")
    results = service.users().messages().list(
        userId="me", q=args.query, maxResults=args.max
    ).execute()
    messages = results.get("messages", [])
    if not messages:
        print("No messages found.")
        return

    output = []
    for msg_meta in messages:
        msg = service.users().messages().get(
            userId="me", id=msg_meta["id"], format="metadata",
            metadataHeaders=["From", "To", "Subject", "Date"],
        ).execute()
        headers = _headers_dict(msg)
        output.append({
            "id": msg["id"],
            "threadId": msg["threadId"],
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "labels": msg.get("labelIds", []),
        })
    print(json.dumps(output, indent=2, ensure_ascii=False))



def gmail_get(args):
    if _gws_binary():
        msg = _run_gws(
            ["gmail", "users", "messages", "get"],
            params={"userId": "me", "id": args.message_id, "format": "full"},
        )
        headers = _headers_dict(msg)
        result = {
            "id": msg["id"],
            "threadId": msg["threadId"],
            "from": headers.get("From", ""),
            "to": headers.get("To", ""),
            "subject": headers.get("Subject", ""),
            "date": headers.get("Date", ""),
            "labels": msg.get("labelIds", []),
            "body": _extract_message_body(msg),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    service = build_service("gmail", "v1")
    msg = service.users().messages().get(
        userId="me", id=args.message_id, format="full"
    ).execute()

    headers = _headers_dict(msg)
    result = {
        "id": msg["id"],
        "threadId": msg["threadId"],
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "subject": headers.get("Subject", ""),
        "date": headers.get("Date", ""),
        "labels": msg.get("labelIds", []),
        "body": _extract_message_body(msg),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))



def gmail_send(args):
    if _gws_binary():
        message = MIMEText(args.body, "html" if args.html else "plain")
        message["to"] = args.to
        message["subject"] = args.subject
        if args.cc:
            message["cc"] = args.cc
        if args.from_header:
            message["from"] = args.from_header

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        body = {"raw": raw}
        if args.thread_id:
            body["threadId"] = args.thread_id

        result = _run_gws(
            ["gmail", "users", "messages", "send"],
            params={"userId": "me"},
            body=body,
        )
        print(json.dumps({"status": "sent", "id": result["id"], "threadId": result.get("threadId", "")}, indent=2))
        return

    service = build_service("gmail", "v1")
    message = MIMEText(args.body, "html" if args.html else "plain")
    message["to"] = args.to
    message["subject"] = args.subject
    if args.cc:
        message["cc"] = args.cc
    if args.from_header:
        message["from"] = args.from_header

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"raw": raw}

    if args.thread_id:
        body["threadId"] = args.thread_id

    result = service.users().messages().send(userId="me", body=body).execute()
    print(json.dumps({"status": "sent", "id": result["id"], "threadId": result.get("threadId", "")}, indent=2))



def gmail_reply(args):
    if _gws_binary():
        original = _run_gws(
            ["gmail", "users", "messages", "get"],
            params={
                "userId": "me",
                "id": args.message_id,
                "format": "metadata",
                "metadataHeaders": ["From", "Subject", "Message-ID"],
            },
        )
        headers = _headers_dict(original)

        subject = headers.get("Subject", "")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"

        message = MIMEText(args.body)
        message["to"] = headers.get("From", "")
        message["subject"] = subject
        if args.from_header:
            message["from"] = args.from_header
        if headers.get("Message-ID"):
            message["In-Reply-To"] = headers["Message-ID"]
            message["References"] = headers["Message-ID"]

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        result = _run_gws(
            ["gmail", "users", "messages", "send"],
            params={"userId": "me"},
            body={"raw": raw, "threadId": original["threadId"]},
        )
        print(json.dumps({"status": "sent", "id": result["id"], "threadId": result.get("threadId", "")}, indent=2))
        return

    service = build_service("gmail", "v1")
    original = service.users().messages().get(
        userId="me", id=args.message_id, format="metadata",
        metadataHeaders=["From", "Subject", "Message-ID"],
    ).execute()
    headers = _headers_dict(original)

    subject = headers.get("Subject", "")
    if not subject.startswith("Re:"):
        subject = f"Re: {subject}"

    message = MIMEText(args.body)
    message["to"] = headers.get("From", "")
    message["subject"] = subject
    if args.from_header:
        message["from"] = args.from_header
    if headers.get("Message-ID"):
        message["In-Reply-To"] = headers["Message-ID"]
        message["References"] = headers["Message-ID"]

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    body = {"raw": raw, "threadId": original["threadId"]}

    result = service.users().messages().send(userId="me", body=body).execute()
    print(json.dumps({"status": "sent", "id": result["id"], "threadId": result.get("threadId", "")}, indent=2))



def gmail_labels(args):
    if _gws_binary():
        results = _run_gws(["gmail", "users", "labels", "list"], params={"userId": "me"})
        labels = [{"id": l["id"], "name": l["name"], "type": l.get("type", "")} for l in results.get("labels", [])]
        print(json.dumps(labels, indent=2))
        return

    service = build_service("gmail", "v1")
    results = service.users().labels().list(userId="me").execute()
    labels = [{"id": l["id"], "name": l["name"], "type": l.get("type", "")} for l in results.get("labels", [])]
    print(json.dumps(labels, indent=2))



def gmail_modify(args):
    body = {}
    if args.add_labels:
        body["addLabelIds"] = args.add_labels.split(",")
    if args.remove_labels:
        body["removeLabelIds"] = args.remove_labels.split(",")

    if _gws_binary():
        result = _run_gws(
            ["gmail", "users", "messages", "modify"],
            params={"userId": "me", "id": args.message_id},
            body=body,
        )
        print(json.dumps({"id": result["id"], "labels": result.get("labelIds", [])}, indent=2))
        return

    service = build_service("gmail", "v1")
    result = service.users().messages().modify(userId="me", id=args.message_id, body=body).execute()
    print(json.dumps({"id": result["id"], "labels": result.get("labelIds", [])}, indent=2))


# =========================================================================
# Calendar
# =========================================================================


def calendar_list(args):
    now = datetime.now(timezone.utc)
    time_min = _datetime_with_timezone(args.start or now.isoformat())
    time_max = _datetime_with_timezone(args.end or (now + timedelta(days=7)).isoformat())

    if _gws_binary():
        results = _run_gws(
            ["calendar", "events", "list"],
            params={
                "calendarId": args.calendar,
                "timeMin": time_min,
                "timeMax": time_max,
                "maxResults": args.max,
                "singleEvents": True,
                "orderBy": "startTime",
            },
        )
        events = []
        for e in results.get("items", []):
            events.append({
                "id": e["id"],
                "summary": e.get("summary", "(no title)"),
                "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
                "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date", "")),
                "location": e.get("location", ""),
                "description": e.get("description", ""),
                "status": e.get("status", ""),
                "htmlLink": e.get("htmlLink", ""),
            })
        print(json.dumps(events, indent=2, ensure_ascii=False))
        return

    service = build_service("calendar", "v3")
    results = service.events().list(
        calendarId=args.calendar, timeMin=time_min, timeMax=time_max,
        maxResults=args.max, singleEvents=True, orderBy="startTime",
    ).execute()

    events = []
    for e in results.get("items", []):
        events.append({
            "id": e["id"],
            "summary": e.get("summary", "(no title)"),
            "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")),
            "end": e.get("end", {}).get("dateTime", e.get("end", {}).get("date", "")),
            "location": e.get("location", ""),
            "description": e.get("description", ""),
            "status": e.get("status", ""),
            "htmlLink": e.get("htmlLink", ""),
        })
    print(json.dumps(events, indent=2, ensure_ascii=False))



def calendar_create(args):
    event = {
        "summary": args.summary,
        "start": {"dateTime": args.start},
        "end": {"dateTime": args.end},
    }
    if args.location:
        event["location"] = args.location
    if args.description:
        event["description"] = args.description
    if args.attendees:
        event["attendees"] = [{"email": e.strip()} for e in args.attendees.split(",") if e.strip()]

    if _gws_binary():
        result = _run_gws(
            ["calendar", "events", "insert"],
            params={"calendarId": args.calendar},
            body=event,
        )
        print(json.dumps({
            "status": "created",
            "id": result["id"],
            "summary": result.get("summary", ""),
            "htmlLink": result.get("htmlLink", ""),
        }, indent=2))
        return

    service = build_service("calendar", "v3")
    result = service.events().insert(calendarId=args.calendar, body=event).execute()
    print(json.dumps({
        "status": "created",
        "id": result["id"],
        "summary": result.get("summary", ""),
        "htmlLink": result.get("htmlLink", ""),
    }, indent=2))



def calendar_delete(args):
    if _gws_binary():
        _run_gws(["calendar", "events", "delete"], params={"calendarId": args.calendar, "eventId": args.event_id})
        print(json.dumps({"status": "deleted", "eventId": args.event_id}))
        return

    service = build_service("calendar", "v3")
    service.events().delete(calendarId=args.calendar, eventId=args.event_id).execute()
    print(json.dumps({"status": "deleted", "eventId": args.event_id}))


# =========================================================================
# Drive
# =========================================================================


def drive_search(args):
    query = args.query if args.raw_query else f"fullText contains '{args.query}'"
    if _gws_binary():
        results = _run_gws(
            ["drive", "files", "list"],
            params={
                "q": query,
                "pageSize": args.max,
                "fields": "files(id, name, mimeType, modifiedTime, webViewLink)",
            },
        )
        print(json.dumps(results.get("files", []), indent=2, ensure_ascii=False))
        return

    service = build_service("drive", "v3")
    results = service.files().list(
        q=query, pageSize=args.max, fields="files(id, name, mimeType, modifiedTime, webViewLink)",
    ).execute()
    files = results.get("files", [])
    print(json.dumps(files, indent=2, ensure_ascii=False))


# =========================================================================
# Contacts
# =========================================================================


def contacts_list(args):
    if _gws_binary():
        results = _run_gws(
            ["people", "people", "connections", "list"],
            params={
                "resourceName": "people/me",
                "pageSize": args.max,
                "personFields": "names,emailAddresses,phoneNumbers",
            },
        )
        contacts = []
        for person in results.get("connections", []):
            names = person.get("names", [{}])
            emails = person.get("emailAddresses", [])
            phones = person.get("phoneNumbers", [])
            contacts.append({
                "name": names[0].get("displayName", "") if names else "",
                "emails": [e.get("value", "") for e in emails],
                "phones": [p.get("value", "") for p in phones],
            })
        print(json.dumps(contacts, indent=2, ensure_ascii=False))
        return

    service = build_service("people", "v1")
    results = service.people().connections().list(
        resourceName="people/me",
        pageSize=args.max,
        personFields="names,emailAddresses,phoneNumbers",
    ).execute()
    contacts = []
    for person in results.get("connections", []):
        names = person.get("names", [{}])
        emails = person.get("emailAddresses", [])
        phones = person.get("phoneNumbers", [])
        contacts.append({
            "name": names[0].get("displayName", "") if names else "",
            "emails": [e.get("value", "") for e in emails],
            "phones": [p.get("value", "") for p in phones],
        })
    print(json.dumps(contacts, indent=2, ensure_ascii=False))


# =========================================================================
# Sheets
# =========================================================================


def sheets_get(args):
    if _gws_binary():
        result = _run_gws(
            ["sheets", "spreadsheets", "values", "get"],
            params={"spreadsheetId": args.sheet_id, "range": args.range},
        )
        print(json.dumps(result.get("values", []), indent=2, ensure_ascii=False))
        return

    service = build_service("sheets", "v4")
    result = service.spreadsheets().values().get(
        spreadsheetId=args.sheet_id, range=args.range,
    ).execute()
    print(json.dumps(result.get("values", []), indent=2, ensure_ascii=False))



def sheets_update(args):
    values = json.loads(args.values)
    body = {"values": values}

    if _gws_binary():
        result = _run_gws(
            ["sheets", "spreadsheets", "values", "update"],
            params={
                "spreadsheetId": args.sheet_id,
                "range": args.range,
                "valueInputOption": "USER_ENTERED",
            },
            body=body,
        )
        print(json.dumps({"updatedCells": result.get("updatedCells", 0), "updatedRange": result.get("updatedRange", "")}, indent=2))
        return

    service = build_service("sheets", "v4")
    result = service.spreadsheets().values().update(
        spreadsheetId=args.sheet_id, range=args.range,
        valueInputOption="USER_ENTERED", body=body,
    ).execute()
    print(json.dumps({"updatedCells": result.get("updatedCells", 0), "updatedRange": result.get("updatedRange", "")}, indent=2))



def sheets_append(args):
    values = json.loads(args.values)
    body = {"values": values}

    if _gws_binary():
        result = _run_gws(
            ["sheets", "spreadsheets", "values", "append"],
            params={
                "spreadsheetId": args.sheet_id,
                "range": args.range,
                "valueInputOption": "USER_ENTERED",
                "insertDataOption": "INSERT_ROWS",
            },
            body=body,
        )
        print(json.dumps({"updatedCells": result.get("updates", {}).get("updatedCells", 0)}, indent=2))
        return

    service = build_service("sheets", "v4")
    result = service.spreadsheets().values().append(
        spreadsheetId=args.sheet_id, range=args.range,
        valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS", body=body,
    ).execute()
    print(json.dumps({"updatedCells": result.get("updates", {}).get("updatedCells", 0)}, indent=2))


# =========================================================================
# Docs
# =========================================================================


def docs_get(args):
    if _gws_binary():
        doc = _run_gws(["docs", "documents", "get"], params={"documentId": args.doc_id})
        result = {
            "title": doc.get("title", ""),
            "documentId": doc.get("documentId", ""),
            "body": _extract_doc_text(doc),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    service = build_service("docs", "v1")
    doc = service.documents().get(documentId=args.doc_id).execute()
    result = {
        "title": doc.get("title", ""),
        "documentId": doc.get("documentId", ""),
        "body": _extract_doc_text(doc),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


# =========================================================================
# CLI parser
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Google Workspace API for Hermes Agent")
    sub = parser.add_subparsers(dest="service", required=True)

    # --- Gmail ---
    gmail = sub.add_parser("gmail")
    gmail_sub = gmail.add_subparsers(dest="action", required=True)

    p = gmail_sub.add_parser("search")
    p.add_argument("query", help="Gmail search query (e.g. 'is:unread')")
    p.add_argument("--max", type=int, default=10)
    p.set_defaults(func=gmail_search)

    p = gmail_sub.add_parser("get")
    p.add_argument("message_id")
    p.set_defaults(func=gmail_get)

    p = gmail_sub.add_parser("send")
    p.add_argument("--to", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--cc", default="")
    p.add_argument("--from", dest="from_header", default="", help="Custom From header (e.g. '\"Agent Name\" <user@example.com>')")
    p.add_argument("--html", action="store_true", help="Send body as HTML")
    p.add_argument("--thread-id", default="", help="Thread ID for threading")
    p.set_defaults(func=gmail_send)

    p = gmail_sub.add_parser("reply")
    p.add_argument("message_id", help="Message ID to reply to")
    p.add_argument("--body", required=True)
    p.add_argument("--from", dest="from_header", default="", help="Custom From header (e.g. '\"Agent Name\" <user@example.com>')")
    p.set_defaults(func=gmail_reply)

    p = gmail_sub.add_parser("labels")
    p.set_defaults(func=gmail_labels)

    p = gmail_sub.add_parser("modify")
    p.add_argument("message_id")
    p.add_argument("--add-labels", default="", help="Comma-separated label IDs to add")
    p.add_argument("--remove-labels", default="", help="Comma-separated label IDs to remove")
    p.set_defaults(func=gmail_modify)

    # --- Calendar ---
    cal = sub.add_parser("calendar")
    cal_sub = cal.add_subparsers(dest="action", required=True)

    p = cal_sub.add_parser("list")
    p.add_argument("--start", default="", help="Start time (ISO 8601)")
    p.add_argument("--end", default="", help="End time (ISO 8601)")
    p.add_argument("--max", type=int, default=25)
    p.add_argument("--calendar", default="primary")
    p.set_defaults(func=calendar_list)

    p = cal_sub.add_parser("create")
    p.add_argument("--summary", required=True)
    p.add_argument("--start", required=True, help="Start (ISO 8601 with timezone)")
    p.add_argument("--end", required=True, help="End (ISO 8601 with timezone)")
    p.add_argument("--location", default="")
    p.add_argument("--description", default="")
    p.add_argument("--attendees", default="", help="Comma-separated email addresses")
    p.add_argument("--calendar", default="primary")
    p.set_defaults(func=calendar_create)

    p = cal_sub.add_parser("delete")
    p.add_argument("event_id")
    p.add_argument("--calendar", default="primary")
    p.set_defaults(func=calendar_delete)

    # --- Drive ---
    drv = sub.add_parser("drive")
    drv_sub = drv.add_subparsers(dest="action", required=True)

    p = drv_sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--raw-query", action="store_true", help="Use query as raw Drive API query")
    p.set_defaults(func=drive_search)

    # --- Contacts ---
    con = sub.add_parser("contacts")
    con_sub = con.add_subparsers(dest="action", required=True)

    p = con_sub.add_parser("list")
    p.add_argument("--max", type=int, default=50)
    p.set_defaults(func=contacts_list)

    # --- Sheets ---
    sh = sub.add_parser("sheets")
    sh_sub = sh.add_subparsers(dest="action", required=True)

    p = sh_sub.add_parser("get")
    p.add_argument("sheet_id")
    p.add_argument("range")
    p.set_defaults(func=sheets_get)

    p = sh_sub.add_parser("update")
    p.add_argument("sheet_id")
    p.add_argument("range")
    p.add_argument("--values", required=True, help="JSON array of arrays")
    p.set_defaults(func=sheets_update)

    p = sh_sub.add_parser("append")
    p.add_argument("sheet_id")
    p.add_argument("range")
    p.add_argument("--values", required=True, help="JSON array of arrays")
    p.set_defaults(func=sheets_append)

    # --- Docs ---
    docs = sub.add_parser("docs")
    docs_sub = docs.add_subparsers(dest="action", required=True)

    p = docs_sub.add_parser("get")
    p.add_argument("doc_id")
    p.set_defaults(func=docs_get)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
