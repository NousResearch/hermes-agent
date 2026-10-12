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
from datetime import datetime, timedelta, timezone, UTC
import email
from email.header import decode_header
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imaplib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import smtplib
import subprocess
import sys

# Ensure sibling modules (_hermes_home) are importable when run standalone.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from _hermes_home import get_hermes_home

HERMES_HOME = get_hermes_home()
TOKEN_PATH = HERMES_HOME / "google_token.json"
CLIENT_SECRET_PATH = HERMES_HOME / "google_client_secret.json"
APP_PASSWORD_PATH = HERMES_HOME / "gmail_app_password.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
]


def _normalize_authorized_user_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if not normalized.get("type"):
        normalized["type"] = "authorized_user"
    return normalized


def _get_app_password_config() -> dict | None:
    """Find and load gmail_app_password.json from standard locations."""
    candidates = [
        APP_PASSWORD_PATH,
        HERMES_HOME / "gmail_app_password.json",
        Path.home() / ".hermes" / "gmail_app_password.json",
        Path.home() / ".hermes" / "profiles" / "jobhunter" / "gmail_app_password.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("email") and data.get("app_password"):
                    return data
            except Exception:
                pass
    return None


def _is_oauth_valid() -> bool:
    """Check if OAuth token exists and is valid (or refreshable)."""
    if not TOKEN_PATH.exists():
        return False
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), _stored_token_scopes())
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_PATH.write_text(
                json.dumps(
                    _normalize_authorized_user_payload(json.loads(creds.to_json())),
                    indent=2,
                ),
                encoding="utf-8",
            )
        return bool(creds.valid)
    except Exception:
        return False


def _ensure_authenticated(service: str = "all"):
    if service == "gmail" and _get_app_password_config():
        return
    if not TOKEN_PATH.exists():
        if service in ("all", "gmail") and _get_app_password_config():
            return
        print("Not authenticated. Run the setup script first:", file=sys.stderr)
        print(f"  python {Path(__file__).parent / 'setup.py'}", file=sys.stderr)
        sys.exit(1)


def _stored_token_scopes() -> list[str]:
    try:
        data = json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
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
        text=True, encoding='utf-8', errors='replace',
        env=_gws_env(),
        check=False,
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


def _extract_body_text(body: dict) -> str:
    text_parts = []
    for element in body.get("content", []):
        paragraph = element.get("paragraph", {})
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun", {})
            if text_run.get("content"):
                text_parts.append(text_run["content"])
    return "".join(text_parts)


def _extract_doc_text(doc: dict) -> str:
    return _extract_body_text(doc.get("body", {}))


def _flatten_doc_tabs(doc: dict) -> list[dict]:
    """Flatten Google's recursive ``tabs``/``childTabs`` tree (preorder).

    A tabbed Doc keeps each tab's content in its own body with an independent
    index space; the legacy top-level ``body`` only carries the first tab, so
    reads and writes that ignore ``tabs`` silently drop or mistarget content.
    Returns [] for the legacy single-body response shape (no ``tabs`` field).
    """
    flat: list[dict] = []

    def visit(tabs, level):
        for tab in tabs or []:
            props = tab.get("tabProperties") or {}
            doc_tab = tab.get("documentTab") or {}
            flat.append({
                "tabId": props.get("tabId", ""),
                "title": props.get("title", ""),
                "level": level,
                "body": doc_tab.get("body") or {},
            })
            visit(tab.get("childTabs"), level + 1)

    visit(doc.get("tabs"), 0)
    return flat


def _resolve_write_tab(doc: dict, tab_arg: str | None) -> tuple[str | None, dict]:
    """Pick exactly one tab body for a write; never merge index spaces.

    Legacy docs (no ``tabs``) return (None, body) — the write carries no tabId.
    A multi-tab doc requires an explicit --tab; an unknown ID errors instead of
    quietly falling back to the first tab.
    """
    tabs = _flatten_doc_tabs(doc)
    if not tabs:
        return None, doc.get("body", {})
    if tab_arg:
        for tab in tabs:
            if tab["tabId"] == tab_arg:
                return tab["tabId"], tab["body"]
        print(
            json.dumps({
                "error": f"unknown tab ID {tab_arg!r}",
                "tabs": [{"tabId": t["tabId"], "title": t["title"]} for t in tabs],
            }, indent=2, ensure_ascii=False),
            file=sys.stderr,
        )
        sys.exit(1)
    if len(tabs) == 1:
        return tabs[0]["tabId"], tabs[0]["body"]
    print(
        json.dumps({
            "error": f"document has {len(tabs)} tabs; pass --tab <tabId> to pick one",
            "tabs": [{"tabId": t["tabId"], "title": t["title"]} for t in tabs],
        }, indent=2, ensure_ascii=False),
        file=sys.stderr,
    )
    sys.exit(1)


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
    _ensure_authenticated("google_workspace")

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), _stored_token_scopes())
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(
                json.dumps(
                    _normalize_authorized_user_payload(json.loads(creds.to_json())),
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[google_api] OAuth token refresh failed: {exc}", file=sys.stderr)
    if not creds.valid:
        if _get_app_password_config():
            print("[google_api] OAuth token is expired/invalid, but gmail_app_password.json is configured.", file=sys.stderr)
        else:
            print("Token is invalid. Re-run setup.", file=sys.stderr)
            sys.exit(1)
    return creds


def build_service(api, version):
    from googleapiclient.discovery import build

    return build(api, version, credentials=get_credentials())


# =========================================================================
# Gmail
# =========================================================================


def _decode_mime_words(s: str | None) -> str:
    if not s:
        return ""
    decoded = []
    for text, enc in decode_header(s):
        if isinstance(text, bytes):
            decoded.append(text.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(str(text))
    return "".join(decoded)


def _send_via_smtp(args_or_dict, app_cfg: dict) -> dict:
    email_addr = app_cfg["email"]
    app_pwd = app_cfg["app_password"]
    smtp_host = app_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(app_cfg.get("smtp_port", 587))

    if hasattr(args_or_dict, "to"):
        to_addr = args_or_dict.to
        subject = args_or_dict.subject
        body = args_or_dict.body
        cc_addr = getattr(args_or_dict, "cc", "")
        from_hdr = getattr(args_or_dict, "from_header", "")
        is_html = getattr(args_or_dict, "html", False)
        attachments = (getattr(args_or_dict, "attachment", []) or []) + (getattr(args_or_dict, "attachments", []) or [])
        in_reply_to = getattr(args_or_dict, "in_reply_to", "")
        references = getattr(args_or_dict, "references", "")
    else:
        to_addr = args_or_dict.get("to")
        subject = args_or_dict.get("subject", "")
        body = args_or_dict.get("body", "")
        cc_addr = args_or_dict.get("cc", "")
        from_hdr = args_or_dict.get("from_header", "")
        is_html = args_or_dict.get("html", False)
        attachments = (args_or_dict.get("attachment", []) or []) + (args_or_dict.get("attachments", []) or [])
        in_reply_to = args_or_dict.get("in_reply_to", "")
        references = args_or_dict.get("references", "")

    if attachments:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "html" if is_html else "plain", "utf-8"))
        for att in attachments:
            att_path = Path(att)
            if not att_path.exists():
                print(f"Warning: Attachment {att} not found, skipping.", file=sys.stderr)
                continue
            with open(att_path, "rb") as f:
                content = f.read()
            mime_type, _ = mimetypes.guess_type(str(att_path))
            if mime_type and "/" in mime_type:
                subtype = mime_type.split("/", 1)[1]
            elif att_path.suffix == ".pdf":
                subtype = "pdf"
            else:
                subtype = "octet-stream"
            part = MIMEApplication(content, _subtype=subtype)
            part.add_header("Content-Disposition", "attachment", filename=att_path.name)
            msg.attach(part)
    else:
        msg = MIMEText(body, "html" if is_html else "plain", "utf-8")

    msg["From"] = from_hdr or email_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    if cc_addr:
        msg["Cc"] = cc_addr
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    recipients = [addr.strip() for addr in to_addr.split(",") if addr.strip()]
    if cc_addr:
        recipients.extend([addr.strip() for addr in cc_addr.split(",") if addr.strip()])

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(email_addr, app_pwd)
        server.sendmail(from_hdr or email_addr, recipients, msg.as_string())

    return {
        "status": "sent",
        "backend": "smtp_fallback",
        "sender": email_addr,
        "to": to_addr,
        "subject": subject,
    }


def _search_via_imap(args, app_cfg: dict) -> list[dict]:
    email_addr = app_cfg["email"]
    app_pwd = app_cfg["app_password"]
    imap_host = app_cfg.get("imap_host", "imap.gmail.com")
    imap_port = int(app_cfg.get("imap_port", 993))

    with imaplib.IMAP4_SSL(imap_host, imap_port, timeout=30) as mail:
        mail.login(email_addr, app_pwd)
        mail.select("INBOX", readonly=True)

        q = (getattr(args, "query", "") or "").strip()
        criteria = "ALL"
        if "is:unread" in q.lower() or "label:unread" in q.lower():
            criteria = "UNSEEN"
        elif q.lower().startswith("from:"):
            sender = q[5:].strip().strip('"').strip("'")
            criteria = f'(FROM "{sender}")'
        elif q.lower().startswith("to:"):
            recipient = q[3:].strip().strip('"').strip("'")
            criteria = f'(TO "{recipient}")'
        elif q.lower().startswith("subject:"):
            subj = q[8:].strip().strip('"').strip("'")
            criteria = f'(SUBJECT "{subj}")'

        status, msg_nums = mail.search(None, criteria)
        if status != "OK" or not msg_nums or not msg_nums[0]:
            status, msg_nums = mail.search(None, "ALL")

        if status != "OK" or not msg_nums or not msg_nums[0]:
            return []

        nums = msg_nums[0].split()
        max_results = getattr(args, "max", 10)
        nums = nums[-max_results:]
        nums.reverse()

        output = []
        for num in nums:
            res, data = mail.fetch(num, "(RFC822.HEADER)")
            if res != "OK" or not data or not data[0]:
                continue
            raw_email = data[0][1]
            parsed = email.message_from_bytes(raw_email)
            subject = _decode_mime_words(parsed.get("Subject", ""))
            from_hdr = _decode_mime_words(parsed.get("From", ""))
            to_hdr = _decode_mime_words(parsed.get("To", ""))
            date_hdr = parsed.get("Date", "")

            output.append({
                "id": num.decode("utf-8", errors="replace"),
                "threadId": "",
                "from": from_hdr,
                "to": to_hdr,
                "subject": subject,
                "date": date_hdr,
                "snippet": subject,
                "labels": ["INBOX"],
                "backend": "imap_fallback",
            })
        return output


def _get_via_imap(args, app_cfg: dict) -> dict:
    email_addr = app_cfg["email"]
    app_pwd = app_cfg["app_password"]
    imap_host = app_cfg.get("imap_host", "imap.gmail.com")
    imap_port = int(app_cfg.get("imap_port", 993))

    msg_id = getattr(args, "message_id", str(args))

    with imaplib.IMAP4_SSL(imap_host, imap_port, timeout=30) as mail:
        mail.login(email_addr, app_pwd)
        mail.select("INBOX", readonly=True)

        res, data = mail.fetch(msg_id, "(RFC822)")
        if res != "OK" or not data or not data[0]:
            status, nums = mail.search(None, f'HEADER Message-ID "{msg_id}"')
            if status == "OK" and nums and nums[0]:
                res, data = mail.fetch(nums[0].split()[-1], "(RFC822)")

        if res != "OK" or not data or not data[0]:
            return {"error": f"Message {msg_id} not found via IMAP"}

        raw_email = data[0][1]
        parsed = email.message_from_bytes(raw_email)

        body = ""
        if parsed.is_multipart():
            for part in parsed.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get("Content-Disposition"))
                if ctype == "text/plain" and "attachment" not in cdispo:
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
            if not body:
                for part in parsed.walk():
                    ctype = part.get_content_type()
                    cdispo = str(part.get("Content-Disposition"))
                    if ctype == "text/html" and "attachment" not in cdispo:
                        body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                        break
        else:
            body = parsed.get_payload(decode=True).decode("utf-8", errors="replace")

        return {
            "id": msg_id,
            "threadId": "",
            "from": _decode_mime_words(parsed.get("From", "")),
            "to": _decode_mime_words(parsed.get("To", "")),
            "subject": _decode_mime_words(parsed.get("Subject", "")),
            "date": parsed.get("Date", ""),
            "labels": ["INBOX"],
            "body": body,
            "backend": "imap_fallback",
        }


def send_email_with_attachments(
    to: str,
    subject: str,
    body: str,
    attachments: list[str | Path] | None = None,
    cc: str = "",
    from_header: str = "",
    html: bool = False,
    thread_id: str = "",
) -> dict:
    """Send an email with optional attachments, trying Gmail API first, falling back to SMTP App Password."""
    class Args:
        pass

    a = Args()
    a.to = to
    a.subject = subject
    a.body = body
    a.attachment = [str(x) for x in (attachments or [])]
    a.attachments = []
    a.cc = cc
    a.from_header = from_header
    a.html = html
    a.thread_id = thread_id

    app_cfg = _get_app_password_config()
    oauth_ok = _is_oauth_valid()

    if not oauth_ok and app_cfg:
        return _send_via_smtp(a, app_cfg)

    try:
        service = build_service("gmail", "v1")
        all_att = [str(x) for x in (attachments or [])]
        msg = MIMEMultipart() if all_att else MIMEText(body, "html" if html else "plain", "utf-8")
        if all_att:
            msg.attach(MIMEText(body, "html" if html else "plain", "utf-8"))
            for att in all_att:
                att_path = Path(att)
                if not att_path.exists():
                    continue
                with open(att_path, "rb") as f:
                    part = MIMEApplication(f.read(), _subtype="pdf" if att_path.suffix == ".pdf" else "octet-stream")
                part.add_header("Content-Disposition", "attachment", filename=att_path.name)
                msg.attach(part)
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc
        if from_header:
            msg["From"] = from_header

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        b = {"raw": raw}
        if thread_id:
            b["threadId"] = thread_id

        res = service.users().messages().send(userId="me", body=b).execute()
        return {"status": "sent", "id": res["id"], "threadId": res.get("threadId", ""), "backend": "gmail_api"}
    except Exception as exc:
        if app_cfg:
            print(f"[google_api] Gmail API failed ({exc}), falling back to SMTP App Password...", file=sys.stderr)
            return _send_via_smtp(a, app_cfg)
        raise


def gmail_search(args: argparse.Namespace) -> None:
    app_cfg = _get_app_password_config()
    oauth_ok = _is_oauth_valid()

    if not oauth_ok and app_cfg:
        results = _search_via_imap(args, app_cfg)
        if not results:
            print("No messages found.")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    try:
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
    except Exception as exc:
        if app_cfg:
            print(f"[google_api] OAuth search failed ({exc}), falling back to IMAP App Password...", file=sys.stderr)
            results = _search_via_imap(args, app_cfg)
            if not results:
                print("No messages found.")
            else:
                print(json.dumps(results, indent=2, ensure_ascii=False))
            return
        raise


def gmail_get(args):
    app_cfg = _get_app_password_config()
    oauth_ok = _is_oauth_valid()

    if not oauth_ok and app_cfg:
        res = _get_via_imap(args, app_cfg)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    try:
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
    except Exception as exc:
        if app_cfg:
            print(f"[google_api] OAuth get failed ({exc}), falling back to IMAP App Password...", file=sys.stderr)
            res = _get_via_imap(args, app_cfg)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        raise


def gmail_send(args):
    app_cfg = _get_app_password_config()
    attachments = (getattr(args, "attachment", []) or []) + (getattr(args, "attachments", []) or [])

    oauth_ok = _is_oauth_valid()
    if not oauth_ok and app_cfg:
        try:
            res = _send_via_smtp(args, app_cfg)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        except Exception as e:
            print(f"Error sending via SMTP fallback: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        if _gws_binary():
            msg = MIMEMultipart() if attachments else MIMEText(args.body, "html" if args.html else "plain", "utf-8")
            if attachments:
                msg.attach(MIMEText(args.body, "html" if args.html else "plain", "utf-8"))
                for att in attachments:
                    att_path = Path(att)
                    if not att_path.exists():
                        continue
                    with open(att_path, "rb") as f:
                        part = MIMEApplication(f.read(), _subtype="pdf" if att_path.suffix == ".pdf" else "octet-stream")
                    part.add_header("Content-Disposition", "attachment", filename=att_path.name)
                    msg.attach(part)
            msg["To"] = args.to
            msg["Subject"] = args.subject
            if args.cc:
                msg["Cc"] = args.cc
            if args.from_header:
                msg["From"] = args.from_header

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
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
        msg = MIMEMultipart() if attachments else MIMEText(args.body, "html" if args.html else "plain", "utf-8")
        if attachments:
            msg.attach(MIMEText(args.body, "html" if args.html else "plain", "utf-8"))
            for att in attachments:
                att_path = Path(att)
                if not att_path.exists():
                    continue
                with open(att_path, "rb") as f:
                    part = MIMEApplication(f.read(), _subtype="pdf" if att_path.suffix == ".pdf" else "octet-stream")
                part.add_header("Content-Disposition", "attachment", filename=att_path.name)
                msg.attach(part)
        msg["To"] = args.to
        msg["Subject"] = args.subject
        if args.cc:
            msg["Cc"] = args.cc
        if args.from_header:
            msg["From"] = args.from_header

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        body = {"raw": raw}
        if args.thread_id:
            body["threadId"] = args.thread_id

        result = service.users().messages().send(userId="me", body=body).execute()
        print(json.dumps({"status": "sent", "id": result["id"], "threadId": result.get("threadId", "")}, indent=2))
    except Exception as exc:
        if app_cfg:
            print(f"[google_api] OAuth send failed ({exc}), falling back to SMTP App Password...", file=sys.stderr)
            try:
                res = _send_via_smtp(args, app_cfg)
                print(json.dumps(res, indent=2, ensure_ascii=False))
                return
            except Exception as smtp_err:
                print(f"Error sending via SMTP fallback after OAuth failure: {smtp_err}", file=sys.stderr)
                sys.exit(1)
        raise


def gmail_reply(args):
    app_cfg = _get_app_password_config()
    oauth_ok = _is_oauth_valid()

    if not oauth_ok and app_cfg:
        try:
            orig = _get_via_imap(args, app_cfg)
            to_addr = orig.get("from", "")
            subject = orig.get("subject", "")
            if not subject.startswith("Re:"):
                subject = f"Re: {subject}" if subject else "Re:"

            class ReplyArgs:
                to = to_addr
                subject = subject
                body = args.body
                cc = ""
                from_header = getattr(args, "from_header", "")
                html = False
                attachment = []
                attachments = []
                in_reply_to = args.message_id
                references = args.message_id

            res = _send_via_smtp(ReplyArgs(), app_cfg)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        except Exception as e:
            print(f"Error replying via SMTP fallback: {e}", file=sys.stderr)
            sys.exit(1)

    try:
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
    except Exception as exc:
        if app_cfg:
            print(f"[google_api] OAuth reply failed ({exc}), falling back to SMTP App Password...", file=sys.stderr)
            try:
                orig = _get_via_imap(args, app_cfg)
                to_addr = orig.get("from", "")
                subject = orig.get("subject", "")
                if not subject.startswith("Re:"):
                    subject = f"Re: {subject}" if subject else "Re:"

                class ReplyArgs:
                    to = to_addr
                    subject = subject
                    body = args.body
                    cc = ""
                    from_header = getattr(args, "from_header", "")
                    html = False
                    attachment = []
                    attachments = []
                    in_reply_to = args.message_id
                    references = args.message_id

                res = _send_via_smtp(ReplyArgs(), app_cfg)
                print(json.dumps(res, indent=2, ensure_ascii=False))
                return
            except Exception as smtp_err:
                print(f"Error replying via SMTP fallback after OAuth failure: {smtp_err}", file=sys.stderr)
                sys.exit(1)
        raise



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
    now = datetime.now(UTC)
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
    tab_arg = getattr(args, "tab", None)
    params = {"documentId": args.doc_id, "includeTabsContent": True}
    if _gws_binary():
        doc = _run_gws(["docs", "documents", "get"], params=params)
    else:
        service = build_service("docs", "v1")
        doc = service.documents().get(
            documentId=args.doc_id, includeTabsContent=True,
        ).execute()

    result = {
        "title": doc.get("title", ""),
        "documentId": doc.get("documentId", ""),
    }
    tabs = _flatten_doc_tabs(doc)
    if not tabs:
        # Legacy single-body response shape.
        result["body"] = _extract_doc_text(doc)
    elif tab_arg:
        _, body = _resolve_write_tab(doc, tab_arg)
        result["tab"] = tab_arg
        result["body"] = _extract_body_text(body)
    else:
        result["tabs"] = [
            {
                "tabId": t["tabId"],
                "title": t["title"],
                "level": t["level"],
                "body": _extract_body_text(t["body"]),
            }
            for t in tabs
        ]
        # Keep "body" populated for single-tab docs so existing callers work.
        if len(tabs) == 1:
            result["body"] = result["tabs"][0]["body"]
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
    """Append text to the end of an existing Doc (one tab of it, if tabbed)."""
    if _gws_binary():
        doc = _run_gws(
            ["docs", "documents", "get"],
            params={"documentId": args.doc_id, "includeTabsContent": True},
        )
    else:
        service = build_service("docs", "v1")
        doc = service.documents().get(
            documentId=args.doc_id, includeTabsContent=True,
        ).execute()

    tab_id, body = _resolve_write_tab(doc, getattr(args, "tab", None))

    # The end-of-body index is one less than the segment endIndex of the body
    # (trailing newline is always at length-1). Docs indexes are 1-based; use
    # endIndex - 1 to insert before the final newline. Each tab has its own
    # index space, so the write location must carry the tab ID.
    content = body.get("content", [])
    end_index = 1
    for element in content:
        ei = element.get("endIndex")
        if isinstance(ei, int) and ei > end_index:
            end_index = ei
    insert_index = max(end_index - 1, 1)

    text = args.text if args.text.endswith("\n") else args.text + "\n"
    _docs_insert_text(args.doc_id, text, index=insert_index, tab_id=tab_id)

    result = {
        "status": "appended",
        "documentId": args.doc_id,
        "inserted_at": insert_index,
        "characters": len(text),
    }
    if tab_id:
        result["tab"] = tab_id
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _docs_insert_text(doc_id: str, text: str, index: int, tab_id: str | None = None) -> None:
    """Send a batchUpdate with a single insertText request."""
    location: dict = {"index": index}
    if tab_id:
        location["tabId"] = tab_id
    requests = [{
        "insertText": {
            "location": location,
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
    p.add_argument("--attachment", "-a", action="append", default=[], help="Path to file to attach (can be repeated)")
    p.add_argument("--attachments", nargs="+", default=[], help="Paths to files to attach (space-separated)")
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
    p.add_argument("--tab", default=None, help="Tab ID to read (tabbed Docs)")
    p.set_defaults(func=docs_get)

    p = docs_sub.add_parser("create")
    p.add_argument("--title", required=True, help="Document title")
    p.add_argument("--body", default="", help="Initial body text (optional)")
    p.set_defaults(func=docs_create)

    p = docs_sub.add_parser("append")
    p.add_argument("doc_id")
    p.add_argument("--text", required=True, help="Text to append to the end of the document")
    p.add_argument("--tab", default=None, help="Tab ID to append to (required for multi-tab Docs)")
    p.set_defaults(func=docs_append)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
