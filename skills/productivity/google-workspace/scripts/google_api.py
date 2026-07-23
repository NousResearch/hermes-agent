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

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/contacts",
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
    return {
        h["name"].lower(): h["value"]
        for h in msg.get("payload", {}).get("headers", [])
        if h.get("name")
    }


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
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "subject": headers.get("subject", ""),
                    "date": headers.get("date", ""),
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
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
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
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
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
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "labels": msg.get("labelIds", []),
        "body": _extract_message_body(msg),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))



def gmail_send(args):
    if _gws_binary():
        message = MIMEText(args.body, "html" if args.html else "plain")
        message["To"] = args.to
        message["Subject"] = args.subject
        if args.cc:
            message["Cc"] = args.cc
        if args.from_header:
            message["From"] = args.from_header

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
    message["To"] = args.to
    message["Subject"] = args.subject
    if args.cc:
        message["Cc"] = args.cc
    if args.from_header:
        message["From"] = args.from_header

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

        subject = headers.get("subject", "")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"

        message = MIMEText(args.body)
        message["To"] = headers.get("from", "")
        message["Subject"] = subject
        if args.from_header:
            message["From"] = args.from_header
        if headers.get("message-id"):
            message["In-Reply-To"] = headers["message-id"]
            message["References"] = headers["message-id"]

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

    subject = headers.get("subject", "")
    if not subject.startswith("Re:"):
        subject = f"Re: {subject}"

    message = MIMEText(args.body)
    message["To"] = headers.get("from", "")
    message["Subject"] = subject
    if args.from_header:
        message["From"] = args.from_header
    if headers.get("message-id"):
        message["In-Reply-To"] = headers["message-id"]
        message["References"] = headers["message-id"]

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


def drive_get(args):
    """Get metadata for a single Drive file by ID."""
    fields = "id, name, mimeType, modifiedTime, size, webViewLink, parents, owners(emailAddress)"
    if _gws_binary():
        result = _run_gws(
            ["drive", "files", "get"],
            params={"fileId": args.file_id, "fields": fields},
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    service = build_service("drive", "v3")
    result = service.files().get(fileId=args.file_id, fields=fields).execute()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def drive_upload(args):
    """Upload a local file to Drive. Falls through to Python client even when gws
    is installed, because gws doesn't do multipart uploads."""
    import mimetypes
    from googleapiclient.http import MediaFileUpload

    local_path = Path(args.path).expanduser()
    if not local_path.exists():
        print(f"ERROR: file not found: {local_path}", file=sys.stderr)
        sys.exit(1)

    mime = args.mime_type or mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    metadata = {"name": args.name or local_path.name}
    if args.parent:
        metadata["parents"] = [args.parent]

    service = build_service("drive", "v3")
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    result = service.files().create(
        body=metadata,
        media_body=media,
        fields="id, name, mimeType, webViewLink",
    ).execute()
    print(json.dumps({
        "status": "uploaded",
        "id": result["id"],
        "name": result.get("name", ""),
        "mimeType": result.get("mimeType", ""),
        "webViewLink": result.get("webViewLink", ""),
    }, indent=2, ensure_ascii=False))


def drive_download(args):
    """Download a Drive file to a local path. Google-native files (Docs/Sheets/Slides)
    must be exported; binary files are downloaded as-is."""
    import io
    from googleapiclient.http import MediaIoBaseDownload

    service = build_service("drive", "v3")

    # Look up the file to decide download vs export.
    meta = service.files().get(fileId=args.file_id, fields="id, name, mimeType").execute()
    mime = meta.get("mimeType", "")
    name = meta.get("name", args.file_id)

    # Map Google-native MIME types to a sensible export default.
    native_export_map = {
        "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
        "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
        "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
        "application/vnd.google-apps.drawing": ("image/png", ".png"),
    }

    out_path = Path(args.output).expanduser() if args.output else Path.cwd() / name

    if mime in native_export_map:
        export_mime = args.export_mime or native_export_map[mime][0]
        default_ext = native_export_map[mime][1]
        if not args.output and not out_path.suffix:
            out_path = out_path.with_suffix(default_ext)
        request = service.files().export_media(fileId=args.file_id, mimeType=export_mime)
    else:
        request = service.files().get_media(fileId=args.file_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = io.FileIO(str(out_path), "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()

    print(json.dumps({
        "status": "downloaded",
        "id": args.file_id,
        "name": name,
        "path": str(out_path),
        "mimeType": mime,
    }, indent=2, ensure_ascii=False))


def drive_create_folder(args):
    body = {
        "name": args.name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if args.parent:
        body["parents"] = [args.parent]

    if _gws_binary():
        result = _run_gws(
            ["drive", "files", "create"],
            params={"fields": "id, name, webViewLink"},
            body=body,
        )
        print(json.dumps({
            "status": "created",
            "id": result["id"],
            "name": result.get("name", ""),
            "webViewLink": result.get("webViewLink", ""),
        }, indent=2, ensure_ascii=False))
        return

    service = build_service("drive", "v3")
    result = service.files().create(body=body, fields="id, name, webViewLink").execute()
    print(json.dumps({
        "status": "created",
        "id": result["id"],
        "name": result.get("name", ""),
        "webViewLink": result.get("webViewLink", ""),
    }, indent=2, ensure_ascii=False))


def drive_share(args):
    permission = {
        "type": args.type,
        "role": args.role,
    }
    if args.type in {"user", "group"}:
        if not args.email:
            print("ERROR: --email is required for type=user or type=group", file=sys.stderr)
            sys.exit(1)
        permission["emailAddress"] = args.email
    elif args.type == "domain":
        if not args.domain:
            print("ERROR: --domain is required for type=domain", file=sys.stderr)
            sys.exit(1)
        permission["domain"] = args.domain

    if _gws_binary():
        result = _run_gws(
            ["drive", "permissions", "create"],
            params={
                "fileId": args.file_id,
                "sendNotificationEmail": args.notify,
            },
            body=permission,
        )
        print(json.dumps({
            "status": "shared",
            "permissionId": result.get("id", ""),
            "fileId": args.file_id,
            "role": permission["role"],
            "type": permission["type"],
        }, indent=2, ensure_ascii=False))
        return

    service = build_service("drive", "v3")
    result = service.permissions().create(
        fileId=args.file_id,
        body=permission,
        sendNotificationEmail=args.notify,
        fields="id",
    ).execute()
    print(json.dumps({
        "status": "shared",
        "permissionId": result.get("id", ""),
        "fileId": args.file_id,
        "role": permission["role"],
        "type": permission["type"],
    }, indent=2, ensure_ascii=False))


def drive_delete(args):
    """Trash or permanently delete a Drive file. Defaults to trash (reversible)."""
    if args.permanent:
        if _gws_binary():
            _run_gws(["drive", "files", "delete"], params={"fileId": args.file_id})
            print(json.dumps({"status": "deleted", "fileId": args.file_id, "permanent": True}))
            return
        service = build_service("drive", "v3")
        service.files().delete(fileId=args.file_id).execute()
        print(json.dumps({"status": "deleted", "fileId": args.file_id, "permanent": True}))
        return

    # Trash (reversible). Use files.update with trashed=True.
    body = {"trashed": True}
    if _gws_binary():
        _run_gws(
            ["drive", "files", "update"],
            params={"fileId": args.file_id},
            body=body,
        )
        print(json.dumps({"status": "trashed", "fileId": args.file_id, "permanent": False}))
        return

    service = build_service("drive", "v3")
    service.files().update(fileId=args.file_id, body=body).execute()
    print(json.dumps({"status": "trashed", "fileId": args.file_id, "permanent": False}))


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


def sheets_create(args):
    """Create a new spreadsheet. Returns the new spreadsheet ID and URL."""
    body = {"properties": {"title": args.title}}
    if args.sheet_name:
        body["sheets"] = [{"properties": {"title": args.sheet_name}}]

    if _gws_binary():
        result = _run_gws(["sheets", "spreadsheets", "create"], body=body)
        print(json.dumps({
            "status": "created",
            "spreadsheetId": result.get("spreadsheetId", ""),
            "title": result.get("properties", {}).get("title", ""),
            "spreadsheetUrl": result.get("spreadsheetUrl", ""),
        }, indent=2, ensure_ascii=False))
        return

    service = build_service("sheets", "v4")
    result = service.spreadsheets().create(
        body=body, fields="spreadsheetId,properties,spreadsheetUrl",
    ).execute()
    print(json.dumps({
        "status": "created",
        "spreadsheetId": result.get("spreadsheetId", ""),
        "title": result.get("properties", {}).get("title", ""),
        "spreadsheetUrl": result.get("spreadsheetUrl", ""),
    }, indent=2, ensure_ascii=False))


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


def docs_create(args):
    """Create a new Doc. Optionally seed it with initial body text."""
    body = {"title": args.title}

    if _gws_binary():
        doc = _run_gws(["docs", "documents", "create"], body=body)
    else:
        service = build_service("docs", "v1")
        doc = service.documents().create(body=body).execute()

    doc_id = doc.get("documentId", "")

    if args.body and doc_id:
        _docs_insert_text(doc_id, args.body, index=1)

    print(json.dumps({
        "status": "created",
        "documentId": doc_id,
        "title": doc.get("title", ""),
        "url": f"https://docs.google.com/document/d/{doc_id}/edit" if doc_id else "",
    }, indent=2, ensure_ascii=False))


def docs_append(args):
    """Append text to the end of an existing Doc."""
    if _gws_binary():
        doc = _run_gws(["docs", "documents", "get"], params={"documentId": args.doc_id})
    else:
        service = build_service("docs", "v1")
        doc = service.documents().get(documentId=args.doc_id).execute()

    # The end-of-body index is one less than the segment endIndex of the body
    # (trailing newline is always at length-1). Docs indexes are 1-based; use
    # endIndex - 1 to insert before the final newline.
    content = doc.get("body", {}).get("content", [])
    end_index = 1
    for element in content:
        ei = element.get("endIndex")
        if isinstance(ei, int) and ei > end_index:
            end_index = ei
    insert_index = max(end_index - 1, 1)

    text = args.text if args.text.endswith("\n") else args.text + "\n"
    _docs_insert_text(args.doc_id, text, index=insert_index)

    print(json.dumps({
        "status": "appended",
        "documentId": args.doc_id,
        "inserted_at": insert_index,
        "characters": len(text),
    }, indent=2, ensure_ascii=False))


def _docs_insert_text(doc_id: str, text: str, index: int) -> None:
    """Send a batchUpdate with a single insertText request."""
    requests = [{
        "insertText": {
            "location": {"index": index},
            "text": text,
        }
    }]
    if _gws_binary():
        _run_gws(
            ["docs", "documents", "batchUpdate"],
            params={"documentId": doc_id},
            body={"requests": requests},
        )
        return

    service = build_service("docs", "v1")
    service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()


# =========================================================================
# Tasks
# =========================================================================


def _tasks_default_list(service) -> str:
    """Return the id of the user's default task list (@default)."""
    return "@default"


def tasks_lists(args):
    """List the user's task lists (each has an id used by other commands)."""
    service = build_service("tasks", "v1")
    result = service.tasklists().list(maxResults=100).execute()
    lists = [{"id": tl["id"], "title": tl.get("title", "")} for tl in result.get("items", [])]
    print(json.dumps(lists, indent=2, ensure_ascii=False))


def tasks_list(args):
    """List tasks in a task list (defaults to the primary list)."""
    service = build_service("tasks", "v1")
    tasklist = args.tasklist or _tasks_default_list(service)
    result = service.tasks().list(
        tasklist=tasklist,
        maxResults=args.max,
        showCompleted=args.show_completed,
        showHidden=args.show_completed,
    ).execute()
    tasks = [{
        "id": t["id"],
        "title": t.get("title", ""),
        "status": t.get("status", ""),
        "due": t.get("due", ""),
        "notes": t.get("notes", ""),
    } for t in result.get("items", [])]
    print(json.dumps(tasks, indent=2, ensure_ascii=False))


def tasks_add(args):
    """Create a task. --due must be an RFC 3339 timestamp (date portion is used)."""
    service = build_service("tasks", "v1")
    tasklist = args.tasklist or _tasks_default_list(service)
    body = {"title": args.title}
    if args.notes:
        body["notes"] = args.notes
    if args.due:
        # Tasks API only honors the date; time-of-day is ignored. Accept a
        # bare date (YYYY-MM-DD) or a full RFC 3339 string.
        due = args.due if "T" in args.due else f"{args.due}T00:00:00.000Z"
        body["due"] = due
    result = service.tasks().insert(tasklist=tasklist, body=body).execute()
    print(json.dumps({
        "status": "created",
        "id": result["id"],
        "title": result.get("title", ""),
        "due": result.get("due", ""),
    }, indent=2, ensure_ascii=False))


def tasks_done(args):
    """Mark a task completed."""
    service = build_service("tasks", "v1")
    tasklist = args.tasklist or _tasks_default_list(service)
    result = service.tasks().patch(
        tasklist=tasklist, task=args.task_id, body={"status": "completed"},
    ).execute()
    print(json.dumps({
        "status": "completed", "id": result["id"], "title": result.get("title", ""),
    }, indent=2, ensure_ascii=False))


def tasks_delete(args):
    """Permanently delete a task (no trash for Tasks)."""
    service = build_service("tasks", "v1")
    tasklist = args.tasklist or _tasks_default_list(service)
    service.tasks().delete(tasklist=tasklist, task=args.task_id).execute()
    print(json.dumps({"status": "deleted", "id": args.task_id}, indent=2, ensure_ascii=False))


# =========================================================================
# Slides
# =========================================================================


def slides_create(args):
    """Create a new presentation (optionally with a title slide heading)."""
    service = build_service("slides", "v1")
    pres = service.presentations().create(body={"title": args.title}).execute()
    pres_id = pres.get("presentationId", "")
    print(json.dumps({
        "status": "created",
        "presentationId": pres_id,
        "title": pres.get("title", ""),
        "url": f"https://docs.google.com/presentation/d/{pres_id}/edit" if pres_id else "",
        "slides": len(pres.get("slides", [])),
    }, indent=2, ensure_ascii=False))


def slides_get(args):
    """Summarize a presentation: slide count and text found on each slide."""
    service = build_service("slides", "v1")
    pres = service.presentations().get(presentationId=args.presentation_id).execute()
    out = {
        "presentationId": pres.get("presentationId", ""),
        "title": pres.get("title", ""),
        "slideCount": len(pres.get("slides", [])),
        "slides": [],
    }
    for i, slide in enumerate(pres.get("slides", [])):
        texts = []
        for el in slide.get("pageElements", []):
            shape = el.get("shape", {})
            for te in shape.get("text", {}).get("textElements", []):
                content = te.get("textRun", {}).get("content", "")
                if content.strip():
                    texts.append(content.strip())
        out["slides"].append({"index": i, "objectId": slide.get("objectId", ""), "text": texts})
    print(json.dumps(out, indent=2, ensure_ascii=False))


def slides_add(args):
    """Append a new blank slide, then place a text box with the given text on it."""
    service = build_service("slides", "v1")
    import uuid
    slide_id = "slide_" + uuid.uuid4().hex[:12]
    box_id = "box_" + uuid.uuid4().hex[:12]
    requests = [{
        "createSlide": {
            "objectId": slide_id,
            "slideLayoutReference": {"predefinedLayout": "BLANK"},
        }
    }, {
        "createShape": {
            "objectId": box_id,
            "shapeType": "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": slide_id,
                "size": {
                    "width": {"magnitude": 6000000, "unit": "EMU"},
                    "height": {"magnitude": 3000000, "unit": "EMU"},
                },
                "transform": {
                    "scaleX": 1, "scaleY": 1,
                    "translateX": 1000000, "translateY": 1000000, "unit": "EMU",
                },
            },
        }
    }, {
        "insertText": {"objectId": box_id, "text": args.text, "insertionIndex": 0}
    }]
    service.presentations().batchUpdate(
        presentationId=args.presentation_id, body={"requests": requests},
    ).execute()
    print(json.dumps({
        "status": "slide_added",
        "presentationId": args.presentation_id,
        "slideObjectId": slide_id,
        "characters": len(args.text),
    }, indent=2, ensure_ascii=False))


# =========================================================================
# Forms
# =========================================================================


def forms_create(args):
    """Create a new Form. Only the title can be set at creation time (API rule);
    the document title defaults to the form title."""
    service = build_service("forms", "v1")
    form = service.forms().create(body={"info": {"title": args.title}}).execute()
    form_id = form.get("formId", "")
    print(json.dumps({
        "status": "created",
        "formId": form_id,
        "title": form.get("info", {}).get("title", ""),
        "responderUri": form.get("responderUri", ""),
        "editUrl": f"https://docs.google.com/forms/d/{form_id}/edit" if form_id else "",
    }, indent=2, ensure_ascii=False))


def forms_get(args):
    """Fetch a form's metadata and its list of questions."""
    service = build_service("forms", "v1")
    form = service.forms().get(formId=args.form_id).execute()
    items = []
    for it in form.get("items", []):
        q = it.get("questionItem", {}).get("question", {})
        qtype = "text"
        if "choiceQuestion" in q:
            qtype = q["choiceQuestion"].get("type", "RADIO")
        elif "scaleQuestion" in q:
            qtype = "scale"
        items.append({
            "itemId": it.get("itemId", ""),
            "title": it.get("title", ""),
            "type": qtype,
            "required": q.get("required", False),
        })
    print(json.dumps({
        "formId": form.get("formId", ""),
        "title": form.get("info", {}).get("title", ""),
        "responderUri": form.get("responderUri", ""),
        "items": items,
    }, indent=2, ensure_ascii=False))


def forms_add_question(args):
    """Append a question. --type one of: text, paragraph, radio, checkbox, dropdown.
    For choice types, pass --options as a comma-separated list."""
    service = build_service("forms", "v1")
    question = {"required": args.required}
    qtype = args.type.lower()
    if qtype in ("text", "paragraph"):
        question["textQuestion"] = {"paragraph": qtype == "paragraph"}
    elif qtype in ("radio", "checkbox", "dropdown"):
        choice_map = {"radio": "RADIO", "checkbox": "CHECKBOX", "dropdown": "DROP_DOWN"}
        opts = [{"value": o.strip()} for o in (args.options or "").split(",") if o.strip()]
        if not opts:
            print(json.dumps({"error": "choice question requires --options"}))
            return
        question["choiceQuestion"] = {"type": choice_map[qtype], "options": opts}
    else:
        print(json.dumps({"error": f"unknown --type '{args.type}'"}))
        return

    request = {
        "createItem": {
            "item": {
                "title": args.title,
                "questionItem": {"question": question},
            },
            "location": {"index": args.index},
        }
    }
    service.forms().batchUpdate(
        formId=args.form_id, body={"requests": [request]},
    ).execute()
    print(json.dumps({
        "status": "question_added",
        "formId": args.form_id,
        "title": args.title,
        "type": qtype,
    }, indent=2, ensure_ascii=False))


# =========================================================================
# Contacts (write)
# =========================================================================


def contacts_create(args):
    """Create a new contact in the user's account."""
    service = build_service("people", "v1")
    body = {}
    if args.name:
        body["names"] = [{"givenName": args.name}]
    if args.email:
        body["emailAddresses"] = [{"value": args.email}]
    if args.phone:
        body["phoneNumbers"] = [{"value": args.phone}]
    result = service.people().createContact(body=body).execute()
    print(json.dumps({
        "status": "created",
        "resourceName": result.get("resourceName", ""),
        "name": args.name,
        "email": args.email or "",
        "phone": args.phone or "",
    }, indent=2, ensure_ascii=False))


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

    p = drv_sub.add_parser("get")
    p.add_argument("file_id")
    p.set_defaults(func=drive_get)

    p = drv_sub.add_parser("upload")
    p.add_argument("path", help="Local file path to upload")
    p.add_argument("--name", default="", help="Override file name in Drive (defaults to local filename)")
    p.add_argument("--parent", default="", help="Parent folder ID")
    p.add_argument("--mime-type", default="", help="Override MIME type (auto-detected if omitted)")
    p.set_defaults(func=drive_upload)

    p = drv_sub.add_parser("download")
    p.add_argument("file_id")
    p.add_argument("--output", default="", help="Local output path (defaults to ./<name> in cwd)")
    p.add_argument("--export-mime", default="", help="Export MIME for Google-native files (overrides defaults: pdf for Docs/Slides, csv for Sheets, png for Drawings)")
    p.set_defaults(func=drive_download)

    p = drv_sub.add_parser("create-folder")
    p.add_argument("name")
    p.add_argument("--parent", default="", help="Parent folder ID (defaults to root)")
    p.set_defaults(func=drive_create_folder)

    p = drv_sub.add_parser("share")
    p.add_argument("file_id")
    p.add_argument("--role", default="reader", choices=["reader", "commenter", "writer", "fileOrganizer", "organizer", "owner"])
    p.add_argument("--type", default="user", choices=["user", "group", "domain", "anyone"])
    p.add_argument("--email", default="", help="Email address (required for type=user or type=group)")
    p.add_argument("--domain", default="", help="Domain (required for type=domain)")
    p.add_argument("--notify", action="store_true", help="Send notification email")
    p.set_defaults(func=drive_share)

    p = drv_sub.add_parser("delete")
    p.add_argument("file_id")
    p.add_argument("--permanent", action="store_true", help="Permanently delete (default is trash, which is reversible)")
    p.set_defaults(func=drive_delete)

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

    p = sh_sub.add_parser("create")
    p.add_argument("--title", required=True, help="Spreadsheet title")
    p.add_argument("--sheet-name", default="", help="Name of the first tab (defaults to 'Sheet1')")
    p.set_defaults(func=sheets_create)

    # --- Docs ---
    docs = sub.add_parser("docs")
    docs_sub = docs.add_subparsers(dest="action", required=True)

    p = docs_sub.add_parser("get")
    p.add_argument("doc_id")
    p.set_defaults(func=docs_get)

    p = docs_sub.add_parser("create")
    p.add_argument("--title", required=True, help="Document title")
    p.add_argument("--body", default="", help="Initial body text (optional)")
    p.set_defaults(func=docs_create)

    p = docs_sub.add_parser("append")
    p.add_argument("doc_id")
    p.add_argument("--text", required=True, help="Text to append to the end of the document")
    p.set_defaults(func=docs_append)

    # --- Tasks ---
    tasks = sub.add_parser("tasks")
    tasks_sub = tasks.add_subparsers(dest="action", required=True)

    p = tasks_sub.add_parser("lists", help="List task lists")
    p.set_defaults(func=tasks_lists)

    p = tasks_sub.add_parser("list", help="List tasks in a list")
    p.add_argument("--tasklist", default="", help="Task list id (defaults to @default)")
    p.add_argument("--max", type=int, default=100)
    p.add_argument("--show-completed", action="store_true", help="Include completed tasks")
    p.set_defaults(func=tasks_list)

    p = tasks_sub.add_parser("add", help="Create a task")
    p.add_argument("title")
    p.add_argument("--notes", default="", help="Task notes/description")
    p.add_argument("--due", default="", help="Due date (YYYY-MM-DD or RFC 3339)")
    p.add_argument("--tasklist", default="", help="Task list id (defaults to @default)")
    p.set_defaults(func=tasks_add)

    p = tasks_sub.add_parser("done", help="Mark a task completed")
    p.add_argument("task_id")
    p.add_argument("--tasklist", default="", help="Task list id (defaults to @default)")
    p.set_defaults(func=tasks_done)

    p = tasks_sub.add_parser("delete", help="Delete a task")
    p.add_argument("task_id")
    p.add_argument("--tasklist", default="", help="Task list id (defaults to @default)")
    p.set_defaults(func=tasks_delete)

    # --- Slides ---
    slides = sub.add_parser("slides")
    slides_sub = slides.add_subparsers(dest="action", required=True)

    p = slides_sub.add_parser("create", help="Create a presentation")
    p.add_argument("--title", required=True, help="Presentation title")
    p.set_defaults(func=slides_create)

    p = slides_sub.add_parser("get", help="Summarize a presentation's slides/text")
    p.add_argument("presentation_id")
    p.set_defaults(func=slides_get)

    p = slides_sub.add_parser("add", help="Append a slide with a text box")
    p.add_argument("presentation_id")
    p.add_argument("--text", required=True, help="Text to place on the new slide")
    p.set_defaults(func=slides_add)

    # --- Forms ---
    forms = sub.add_parser("forms")
    forms_sub = forms.add_subparsers(dest="action", required=True)

    p = forms_sub.add_parser("create", help="Create a form")
    p.add_argument("--title", required=True, help="Form title")
    p.set_defaults(func=forms_create)

    p = forms_sub.add_parser("get", help="Show a form's questions")
    p.add_argument("form_id")
    p.set_defaults(func=forms_get)

    p = forms_sub.add_parser("add-question", help="Append a question to a form")
    p.add_argument("form_id")
    p.add_argument("--title", required=True, help="Question text")
    p.add_argument("--type", default="text", help="text, paragraph, radio, checkbox, or dropdown")
    p.add_argument("--options", default="", help="Comma-separated options (choice types)")
    p.add_argument("--required", action="store_true", help="Mark the question required")
    p.add_argument("--index", type=int, default=0, help="Insertion index (0 = first)")
    p.set_defaults(func=forms_add_question)

    # --- Contacts: write (create) appended to existing contacts parser ---
    p = con_sub.add_parser("create", help="Create a new contact")
    p.add_argument("--name", required=True, help="Contact given name")
    p.add_argument("--email", default="", help="Email address")
    p.add_argument("--phone", default="", help="Phone number")
    p.set_defaults(func=contacts_create)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
