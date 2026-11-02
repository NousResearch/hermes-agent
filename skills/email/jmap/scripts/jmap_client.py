#!/usr/bin/env python3
"""JMAP Email Client (RFC 8620 / RFC 8621) for Hermes Agent.

Provides zero-dependency email management (list, read, search, send, submit)
over standard JMAP. Compatible with Atomic Mail, Fastmail, Stalwart,
and custom self-hosted JMAP servers.
"""

from __future__ import annotations

import argparse
import html.parser
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JMAP_CORE_CAPABILITY = "urn:ietf:params:jmap:core"
JMAP_MAIL_CAPABILITY = "urn:ietf:params:jmap:mail"
JMAP_SUBMISSION_CAPABILITY = "urn:ietf:params:jmap:submission"


class HTMLTextExtractor(html.parser.HTMLParser):
    """Simple parser to extract visible plain text from HTML emails."""

    def __init__(self) -> None:
        super().__init__()
        self.text_parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in ("script", "style", "head"):
            self._skip = True
        elif tag.lower() in ("p", "br", "div", "tr", "li"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html_content: str) -> str:
    """Convert HTML string to clean plain text."""
    if not html_content:
        return ""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html_content)
        extractor.close()
        return extractor.get_text()
    except Exception:
        return html_content


class JMAPClient:
    """Zero-dependency JMAP client supporting RFC 8620/8621."""

    def __init__(
        self,
        session_url: str,
        token: str,
        account_id: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.session_url = session_url.rstrip("/")
        self.token = token
        self.account_id = account_id
        self.timeout = timeout
        self._session_data: Optional[Dict[str, Any]] = None
        self._api_url: Optional[str] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent-JMAP/1.0",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def load_session(self) -> Dict[str, Any]:
        """Fetch JMAP session discovery document."""
        if self._session_data:
            return self._session_data

        req = urllib.request.Request(
            self.session_url,
            headers=self._get_headers(),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"JMAP session request failed (HTTP {e.code}): {err_body}")
        except Exception as e:
            raise RuntimeError(f"Could not connect to JMAP session at {self.session_url}: {e}")

        self._session_data = data
        self._api_url = data.get("apiUrl")
        if not self._api_url:
            raise RuntimeError("JMAP session did not return an 'apiUrl'")

        # Auto-discover primary mail account if not specified
        if not self.account_id:
            primary_accounts = data.get("primaryAccounts", {})
            if JMAP_MAIL_CAPABILITY in primary_accounts:
                self.account_id = primary_accounts[JMAP_MAIL_CAPABILITY]
            else:
                accounts = data.get("accounts", {})
                for acc_id, acc_info in accounts.items():
                    if JMAP_MAIL_CAPABILITY in acc_info.get("accountCapabilities", {}):
                        self.account_id = acc_id
                        break

        if not self.account_id:
            raise RuntimeError("No mail-capable JMAP account found in session")

        return data

    def call(self, method_calls: List[List[Any]]) -> List[List[Any]]:
        """Execute a JMAP batch call."""
        self.load_session()
        assert self._api_url is not None

        payload = {
            "using": [
                JMAP_CORE_CAPABILITY,
                JMAP_MAIL_CAPABILITY,
                JMAP_SUBMISSION_CAPABILITY,
            ],
            "methodCalls": method_calls,
        }

        req = urllib.request.Request(
            self._api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._get_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"JMAP API call failed (HTTP {e.code}): {err_body}")
        except Exception as e:
            raise RuntimeError(f"JMAP request error: {e}")

        responses = res_data.get("methodResponses", [])
        for resp_item in responses:
            if len(resp_item) >= 2 and resp_item[0] == "error":
                err_type = resp_item[1].get("type", "unknownError")
                description = resp_item[1].get("description", "")
                raise RuntimeError(f"JMAP error ({err_type}): {description}")

        return responses

    def get_mailboxes(self) -> Dict[str, Dict[str, Any]]:
        """Fetch mailboxes mapped by name (case-insensitive)."""
        responses = self.call([
            ["Mailbox/get", {"accountId": self.account_id}, "call-mbx"],
        ])
        mailboxes: Dict[str, Dict[str, Any]] = {}
        for name, args, _ in responses:
            if name == "Mailbox/get":
                for mbx in args.get("list", []):
                    mbx_name = mbx.get("name", "")
                    mailboxes[mbx_name.lower()] = mbx
                    mailboxes[mbx.get("id")] = mbx
        return mailboxes

    def list_emails(
        self,
        mailbox_name: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List summary info for emails matching criteria."""
        mailboxes = self.get_mailboxes()
        target_mbx = mailboxes.get(mailbox_name.lower())
        mbx_id = target_mbx.get("id") if target_mbx else None

        filter_dict: Dict[str, Any] = {}
        if mbx_id:
            filter_dict["inMailbox"] = mbx_id
        if unread_only:
            filter_dict["notKeyword"] = "$seen"
        if query_text:
            filter_dict["text"] = query_text

        query_args: Dict[str, Any] = {
            "accountId": self.account_id,
            "filter": filter_dict,
            "sort": [{"property": "receivedAt", "isAscending": False}],
            "limit": limit,
        }

        responses = self.call([
            ["Email/query", query_args, "call-q"],
            [
                "Email/get",
                {
                    "accountId": self.account_id,
                    "#ids": {
                        "resultOf": "call-q",
                        "name": "Email/query",
                        "path": "/ids",
                    },
                    "properties": [
                        "id",
                        "subject",
                        "from",
                        "to",
                        "receivedAt",
                        "preview",
                        "keywords",
                        "hasAttachment",
                    ],
                },
                "call-get",
            ],
        ])

        for name, args, _ in responses:
            if name == "Email/get":
                return args.get("list", [])
        return []

    def get_email(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Fetch full email content by ID."""
        responses = self.call([
            [
                "Email/get",
                {
                    "accountId": self.account_id,
                    "ids": [email_id],
                    "properties": [
                        "id",
                        "blobId",
                        "threadId",
                        "mailboxIds",
                        "keywords",
                        "hasAttachment",
                        "headers",
                        "sender",
                        "from",
                        "to",
                        "cc",
                        "bcc",
                        "replyTo",
                        "subject",
                        "sentAt",
                        "receivedAt",
                        "bodyValues",
                        "textBody",
                        "htmlBody",
                        "attachments",
                    ],
                    "bodyProperties": ["partId", "blobId", "size", "name", "type", "charset"],
                    "fetchTextBodyValues": True,
                    "fetchHTMLBodyValues": True,
                },
                "call-get-full",
            ],
        ])

        for name, args, _ in responses:
            if name == "Email/get":
                email_list = args.get("list", [])
                if email_list:
                    item = email_list[0]
                    # Format body text cleanly
                    body_values = item.get("bodyValues", {})
                    plain_parts = []
                    for part in item.get("textBody", []):
                        pid = part.get("partId")
                        if pid in body_values:
                            plain_parts.append(body_values[pid].get("value", ""))

                    if not plain_parts:
                        for part in item.get("htmlBody", []):
                            pid = part.get("partId")
                            if pid in body_values:
                                html_val = body_values[pid].get("value", "")
                                plain_parts.append(html_to_text(html_val))

                    item["extractedBody"] = "\n".join(plain_parts).strip()
                    return item
        return None

    def send_email(
        self,
        to: List[str],
        subject: str,
        body: str,
        from_addr: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        draft_only: bool = False,
    ) -> Dict[str, Any]:
        """Compose and send/submit an email message via JMAP."""
        mailboxes = self.get_mailboxes()
        drafts_mbx = mailboxes.get("drafts") or mailboxes.get("draft")
        drafts_id = drafts_mbx.get("id") if drafts_mbx else None

        mailbox_ids = {}
        if drafts_id:
            mailbox_ids[drafts_id] = True

        email_obj: Dict[str, Any] = {
            "subject": subject,
            "to": [{"email": addr.strip()} for addr in to if addr.strip()],
            "bodyValues": {"1": {"value": body, "isTruncated": False}},
            "textBody": [{"partId": "1", "type": "text/plain"}],
        }

        if from_addr:
            email_obj["from"] = [{"email": from_addr.strip()}]
        if cc:
            email_obj["cc"] = [{"email": addr.strip()} for addr in cc if addr.strip()]
        if bcc:
            email_obj["bcc"] = [{"email": addr.strip()} for addr in bcc if addr.strip()]
        if mailbox_ids:
            email_obj["mailboxIds"] = mailbox_ids

        method_calls: List[List[Any]] = [
            [
                "Email/set",
                {
                    "accountId": self.account_id,
                    "create": {"draft-1": email_obj},
                },
                "call-create-email",
            ],
        ]

        if not draft_only:
            # Enqueue submission
            submission_obj = {
                "emailId": "#draft-1",
                "identityId": None,
            }
            method_calls.append([
                "EmailSubmission/set",
                {
                    "accountId": self.account_id,
                    "#onSuccessCreateEmail": {
                        "resultOf": "call-create-email",
                        "name": "Email/set",
                        "path": "/created/draft-1/id",
                    },
                    "create": {"sub-1": submission_obj},
                },
                "call-submit",
            ])

        responses = self.call(method_calls)
        created_email_id = None
        submission_id = None

        for name, args, _ in responses:
            if name == "Email/set":
                created = args.get("created", {})
                if "draft-1" in created:
                    created_email_id = created["draft-1"].get("id")
                not_created = args.get("notCreated", {})
                if "draft-1" in not_created:
                    err = not_created["draft-1"]
                    raise RuntimeError(f"Failed to create email draft: {err}")
            elif name == "EmailSubmission/set":
                created = args.get("created", {})
                if "sub-1" in created:
                    submission_id = created["sub-1"].get("id")
                not_created = args.get("notCreated", {})
                if "sub-1" in not_created:
                    err = not_created["sub-1"]
                    raise RuntimeError(f"Email created (id={created_email_id}) but submission failed: {err}")

        return {
            "status": "draft_created" if draft_only else "sent",
            "emailId": created_email_id,
            "submissionId": submission_id,
            "to": to,
            "subject": subject,
        }


def load_credentials_from_file(custom_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve JMAP credentials from filesystem."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path).expanduser())

    # Hermes home or standard config paths
    try:
        from hermes_constants import get_hermes_home
        candidates.append(get_hermes_home() / "jmap-credentials.json")
    except Exception:
        pass

    candidates.append(Path.home() / ".hermes" / "jmap-credentials.json")
    candidates.append(Path.home() / ".config" / "jmap" / "credentials.json")

    for path in candidates:
        if path and path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session_url = data.get("session_url") or data.get("sessionUrl") or data.get("url")
                    token = data.get("token") or data.get("api_token") or data.get("bearer_token")
                    account_id = data.get("account_id") or data.get("accountId")
                    return session_url, token, account_id
            except Exception:
                pass
    return None, None, None


def resolve_client(args: argparse.Namespace) -> JMAPClient:
    """Build configured JMAPClient from CLI args, env, or credentials file."""
    f_url, f_tok, f_acc = load_credentials_from_file(getattr(args, "config", None))

    session_url = (
        getattr(args, "session_url", None)
        or os.environ.get("JMAP_SESSION_URL")
        or os.environ.get("JMAP_URL")
        or f_url
    )
    token = (
        getattr(args, "token", None)
        or os.environ.get("JMAP_API_TOKEN")
        or os.environ.get("JMAP_TOKEN")
        or f_tok
        or ""
    )
    account_id = (
        getattr(args, "account_id", None)
        or os.environ.get("JMAP_ACCOUNT_ID")
        or f_acc
    )

    if not session_url:
        raise ValueError(
            "JMAP session URL is required. Set JMAP_SESSION_URL env var, "
            "provide --session-url, or configure ~/.hermes/jmap-credentials.json"
        )

    return JMAPClient(session_url=session_url, token=token, account_id=account_id)


def format_addresses(addr_list: Any) -> str:
    """Format email address list to string."""
    if not isinstance(addr_list, list):
        return str(addr_list or "")
    formatted = []
    for item in addr_list:
        if isinstance(item, dict):
            name = item.get("name")
            email = item.get("email", "")
            if name:
                formatted.append(f"{name} <{email}>")
            else:
                formatted.append(email)
        else:
            formatted.append(str(item))
    return ", ".join(formatted)


def main() -> int:
    parser = argparse.ArgumentParser(description="JMAP Email Client for Hermes Agent")
    parser.add_argument("--session-url", help="JMAP session discovery URL")
    parser.add_argument("--token", help="Bearer API token")
    parser.add_argument("--account-id", help="JMAP account ID (auto-discovered if omitted)")
    parser.add_argument("--config", help="Path to credentials JSON file")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON output")

    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List emails from mailbox")
    list_parser.add_argument("--mailbox", default="INBOX", help="Mailbox name (default: INBOX)")
    list_parser.add_argument("--limit", type=int, default=10, help="Max emails to return (default: 10)")
    list_parser.add_argument("--unread", action="store_true", help="Only list unread emails")
    list_parser.add_argument("--query", help="Search query string")

    # get
    get_parser = subparsers.add_parser("get", help="Fetch complete email content")
    get_parser.add_argument("id", help="Email ID")

    # send
    send_parser = subparsers.add_parser("send", help="Send or draft an email")
    send_parser.add_argument("--to", required=True, nargs="+", help="Recipient email addresses")
    send_parser.add_argument("--subject", required=True, help="Email subject")
    send_parser.add_argument("--body", required=True, help="Plain text message body")
    send_parser.add_argument("--from-addr", help="Sender email address")
    send_parser.add_argument("--cc", nargs="+", help="CC recipients")
    send_parser.add_argument("--bcc", nargs="+", help="BCC recipients")
    send_parser.add_argument("--draft", action="store_true", help="Save as draft instead of sending")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    try:
        client = resolve_client(args)
    except Exception as e:
        sys.stderr.write(f"Configuration error: {e}\n")
        return 1

    try:
        if args.command == "list":
            emails = client.list_emails(
                mailbox_name=args.mailbox,
                limit=args.limit,
                unread_only=args.unread,
                query_text=args.query,
            )
            if args.json:
                print(json.dumps(emails, indent=2))
            else:
                if not emails:
                    print(f"No emails found in {args.mailbox}.")
                    return 0
                print(f"--- Emails in {args.mailbox} (Showing {len(emails)}) ---")
                for item in emails:
                    eid = item.get("id", "")
                    sender = format_addresses(item.get("from", []))
                    subj = item.get("subject", "(no subject)")
                    date = item.get("receivedAt", "")[:16].replace("T", " ")
                    is_unread = "$seen" not in (item.get("keywords") or {})
                    unread_tag = "[UNREAD] " if is_unread else ""
                    print(f"• ID: {eid} | {date} | {unread_tag}From: {sender}")
                    print(f"  Subject: {subj}")
                    preview = item.get("preview", "").strip()
                    if preview:
                        print(f"  Preview: {preview[:100]}...")
                    print()

        elif args.command == "get":
            email_data = client.get_email(args.id)
            if not email_data:
                sys.stderr.write(f"Email with ID '{args.id}' not found.\n")
                return 1
            if args.json:
                print(json.dumps(email_data, indent=2))
            else:
                print(f"ID: {email_data.get('id')}")
                print(f"Date: {email_data.get('receivedAt', '')}")
                print(f"From: {format_addresses(email_data.get('from'))}")
                print(f"To: {format_addresses(email_data.get('to'))}")
                if email_data.get("cc"):
                    print(f"CC: {format_addresses(email_data.get('cc'))}")
                print(f"Subject: {email_data.get('subject', '(no subject)')}")
                print("-" * 50)
                body = email_data.get("extractedBody", "")
                print(body or "(empty body)")

        elif args.command == "send":
            res = client.send_email(
                to=args.to,
                subject=args.subject,
                body=args.body,
                from_addr=args.from_addr,
                cc=args.cc,
                bcc=args.bcc,
                draft_only=args.draft,
            )
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                action = "Draft saved" if args.draft else "Email sent successfully"
                print(f"✓ {action} (ID: {res.get('emailId')}) to {', '.join(args.to)}")

        return 0

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
