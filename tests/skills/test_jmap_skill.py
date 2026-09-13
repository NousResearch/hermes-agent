"""Unit tests for the JMAP email integration skill."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from skills.email.jmap.scripts.jmap_client import (
    HTMLTextExtractor,
    JMAPClient,
    format_addresses,
    html_to_text,
    load_credentials_from_file,
)
from skills.email.jmap.scripts.email_triage import triage_inbox


MOCK_SESSION_DATA = {
    "capabilities": {
        "urn:ietf:params:jmap:core": {},
        "urn:ietf:params:jmap:mail": {},
        "urn:ietf:params:jmap:submission": {},
    },
    "apiUrl": "https://api.example.com/jmap/api",
    "downloadUrl": "https://api.example.com/jmap/download/{blobId}",
    "uploadUrl": "https://api.example.com/jmap/upload/{accountId}",
    "primaryAccounts": {
        "urn:ietf:params:jmap:mail": "acc-mail-123",
        "urn:ietf:params:jmap:submission": "acc-mail-123",
    },
    "accounts": {
        "acc-mail-123": {
            "name": "user@example.com",
            "accountCapabilities": {
                "urn:ietf:params:jmap:mail": {},
                "urn:ietf:params:jmap:submission": {},
            },
        }
    },
}


class TestHTMLTextExtractor:
    def test_extracts_text_and_ignores_scripts(self):
        html_input = """
        <html>
            <head><style>body { color: red; }</style></head>
            <body>
                <script>alert('test');</script>
                <p>Hello <b>World</b>!</p>
                <div>Here is a <a href="https://example.com">link</a>.</div>
            </body>
        </html>
        """
        text = html_to_text(html_input)
        assert "Hello World!" in text
        assert "Here is a link." in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_empty_html_returns_empty(self):
        assert html_to_text("") == ""
        assert html_to_text(None) == ""


class TestJMAPClient:
    @patch("urllib.request.urlopen")
    def test_load_session_auto_discovers_account_and_api_url(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_SESSION_DATA).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = JMAPClient(session_url="https://api.example.com/jmap/session", token="secret-tok")
        session = client.load_session()

        assert session["apiUrl"] == "https://api.example.com/jmap/api"
        assert client.account_id == "acc-mail-123"
        assert client._api_url == "https://api.example.com/jmap/api"

    @patch("urllib.request.urlopen")
    def test_list_emails_executes_mailbox_and_query_calls(self, mock_urlopen):
        session_resp = MagicMock()
        session_resp.read.return_value = json.dumps(MOCK_SESSION_DATA).encode("utf-8")
        session_resp.__enter__.return_value = session_resp

        # Response 1: Mailbox/get
        mbx_resp_data = {
            "methodResponses": [
                ["Mailbox/get", {"list": [{"id": "mbx-inbox", "name": "INBOX"}]}, "call-mbx"],
            ]
        }
        mbx_resp = MagicMock()
        mbx_resp.read.return_value = json.dumps(mbx_resp_data).encode("utf-8")
        mbx_resp.__enter__.return_value = mbx_resp

        # Response 2: Email/query + Email/get
        email_resp_data = {
            "methodResponses": [
                ["Email/query", {"ids": ["msg-1", "msg-2"]}, "call-q"],
                [
                    "Email/get",
                    {
                        "list": [
                            {
                                "id": "msg-1",
                                "subject": "Project Status",
                                "from": [{"name": "Lead", "email": "lead@example.com"}],
                                "receivedAt": "2026-08-30T12:00:00Z",
                                "preview": "Everything looks good.",
                                "keywords": {"$seen": True},
                            },
                            {
                                "id": "msg-2",
                                "subject": "Urgent Review",
                                "from": [{"name": "Security", "email": "sec@example.com"}],
                                "receivedAt": "2026-08-30T13:00:00Z",
                                "preview": "Please verify your access.",
                                "keywords": {},
                            },
                        ]
                    },
                    "call-get",
                ],
            ]
        }
        query_resp = MagicMock()
        query_resp.read.return_value = json.dumps(email_resp_data).encode("utf-8")
        query_resp.__enter__.return_value = query_resp

        mock_urlopen.side_effect = [session_resp, mbx_resp, query_resp]

        client = JMAPClient(session_url="https://api.example.com/jmap/session", token="token")
        emails = client.list_emails(mailbox_name="INBOX", limit=5)

        assert len(emails) == 2
        assert emails[0]["id"] == "msg-1"
        assert emails[0]["subject"] == "Project Status"
        assert emails[1]["id"] == "msg-2"
        assert emails[1]["subject"] == "Urgent Review"

    @patch("urllib.request.urlopen")
    def test_get_email_full_content_and_body_extraction(self, mock_urlopen):
        session_resp = MagicMock()
        session_resp.read.return_value = json.dumps(MOCK_SESSION_DATA).encode("utf-8")
        session_resp.__enter__.return_value = session_resp

        full_email_resp_data = {
            "methodResponses": [
                [
                    "Email/get",
                    {
                        "list": [
                            {
                                "id": "msg-1",
                                "subject": "Quarterly Report",
                                "from": [{"email": "finance@example.com"}],
                                "to": [{"email": "user@example.com"}],
                                "receivedAt": "2026-08-30T10:00:00Z",
                                "bodyValues": {
                                    "p1": {"value": "Attached is the quarterly financial report."}
                                },
                                "textBody": [{"partId": "p1", "type": "text/plain"}],
                            }
                        ]
                    },
                    "call-get-full",
                ]
            ]
        }
        api_resp = MagicMock()
        api_resp.read.return_value = json.dumps(full_email_resp_data).encode("utf-8")
        api_resp.__enter__.return_value = api_resp

        mock_urlopen.side_effect = [session_resp, api_resp]

        client = JMAPClient(session_url="https://api.example.com/jmap/session", token="token")
        item = client.get_email("msg-1")

        assert item is not None
        assert item["id"] == "msg-1"
        assert item["extractedBody"] == "Attached is the quarterly financial report."

    @patch("urllib.request.urlopen")
    def test_send_email_creates_draft_and_submits(self, mock_urlopen):
        session_resp = MagicMock()
        session_resp.read.return_value = json.dumps(MOCK_SESSION_DATA).encode("utf-8")
        session_resp.__enter__.return_value = session_resp

        # Mailbox/get response
        mbx_resp = MagicMock()
        mbx_resp.read.return_value = json.dumps({
            "methodResponses": [
                ["Mailbox/get", {"list": [{"id": "mbx-drafts", "name": "Drafts"}]}, "call-mbx"],
            ]
        }).encode("utf-8")
        mbx_resp.__enter__.return_value = mbx_resp

        # Email/set + EmailSubmission/set response
        send_resp_data = {
            "methodResponses": [
                ["Email/set", {"created": {"draft-1": {"id": "email-new-123"}}}, "call-create-email"],
                ["EmailSubmission/set", {"created": {"sub-1": {"id": "sub-new-456"}}}, "call-submit"],
            ]
        }
        send_resp = MagicMock()
        send_resp.read.return_value = json.dumps(send_resp_data).encode("utf-8")
        send_resp.__enter__.return_value = send_resp

        mock_urlopen.side_effect = [session_resp, mbx_resp, send_resp]

        client = JMAPClient(session_url="https://api.example.com/jmap/session", token="token")
        result = client.send_email(
            to=["recipient@example.com"],
            subject="Hello JMAP",
            body="Test body",
            bcc=["archive@example.com"],
            draft_only=False,
        )

        assert result["status"] == "sent"
        assert result["emailId"] == "email-new-123"
        assert result["submissionId"] == "sub-new-456"

    @patch("urllib.request.urlopen")
    def test_send_email_draft_only_skips_submission(self, mock_urlopen):
        session_resp = MagicMock()
        session_resp.read.return_value = json.dumps(MOCK_SESSION_DATA).encode("utf-8")
        session_resp.__enter__.return_value = session_resp

        mbx_resp = MagicMock()
        mbx_resp.read.return_value = json.dumps({
            "methodResponses": [
                ["Mailbox/get", {"list": [{"id": "mbx-drafts", "name": "Drafts"}]}, "call-mbx"],
            ]
        }).encode("utf-8")
        mbx_resp.__enter__.return_value = mbx_resp

        send_resp_data = {
            "methodResponses": [
                ["Email/set", {"created": {"draft-1": {"id": "draft-only-789"}}}, "call-create-email"],
            ]
        }
        send_resp = MagicMock()
        send_resp.read.return_value = json.dumps(send_resp_data).encode("utf-8")
        send_resp.__enter__.return_value = send_resp

        mock_urlopen.side_effect = [session_resp, mbx_resp, send_resp]

        client = JMAPClient(session_url="https://api.example.com/jmap/session", token="token")
        result = client.send_email(
            to=["recipient@example.com"],
            subject="Draft Email",
            body="Draft content",
            draft_only=True,
        )

        assert result["status"] == "draft_created"
        assert result["emailId"] == "draft-only-789"
        assert result["submissionId"] is None


class TestEmailTriage:
    def test_triage_inbox_categorizes_correctly(self):
        mock_client = MagicMock()
        mock_client.list_emails.return_value = [
            {
                "id": "e1",
                "subject": "Urgent: Invoice approval needed ASAP",
                "from": [{"name": "Vendor", "email": "billing@vendor.com"}],
                "preview": "Please approve payment.",
                "keywords": {},
            },
            {
                "id": "e2",
                "subject": "Weekly Newsletter",
                "from": [{"email": "newsletter@news.com"}],
                "preview": "Top stories this week...",
                "keywords": {"$seen": True},
            },
            {
                "id": "e3",
                "subject": "Catch up for lunch?",
                "from": [{"name": "Alice", "email": "alice@friend.com"}],
                "preview": "Are you free tomorrow?",
                "keywords": {},
            },
        ]

        triage = triage_inbox(mock_client, limit=10, unread_only=False)

        assert triage["total_analyzed"] == 3
        assert len(triage["categories"]["action_required"]) == 1
        assert triage["categories"]["action_required"][0]["id"] == "e1"
        assert len(triage["categories"]["notifications"]) == 1
        assert triage["categories"]["notifications"][0]["id"] == "e2"
        assert len(triage["categories"]["general"]) == 1
        assert triage["categories"]["general"][0]["id"] == "e3"


class TestAddressFormattingAndCredentials:
    def test_format_addresses(self):
        addrs = [
            {"name": "Alice", "email": "alice@example.com"},
            {"email": "bob@example.com"},
        ]
        assert format_addresses(addrs) == "Alice <alice@example.com>, bob@example.com"

    def test_load_credentials_from_file(self, tmp_path):
        cred_file = tmp_path / "creds.json"
        cred_file.write_text(json.dumps({
            "session_url": "https://api.atomicmail.io/jmap/session",
            "token": "tok-12345",
            "account_id": "acc-999"
        }), encoding="utf-8")

        url, tok, acc = load_credentials_from_file(str(cred_file))
        assert url == "https://api.atomicmail.io/jmap/session"
        assert tok == "tok-12345"
        assert acc == "acc-999"
