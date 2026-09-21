from __future__ import annotations

import json

import pytest

from scripts.mcp.gmail_metadata_mcp import GmailMetadataReader, GmailMetadataPolicyError


class FakeGmailMessages:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, object]] = []

    def list(self, **kwargs):
        assert kwargs == {"userId": "me", "q": "is:unread", "maxResults": 2}
        return _Execute({"messages": [{"id": "m1"}, {"id": "m2"}]})

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        message_id = kwargs["id"]
        return _Execute(
            {
                "id": message_id,
                "snippet": f"preview-{message_id}",
                "internalDate": "1720000000000",
                "payload": {
                    "headers": [
                        {"name": "From", "value": f"sender-{message_id}@example.test"},
                        {"name": "Subject", "value": f"subject-{message_id}"},
                    ]
                },
            }
        )


class _Execute:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value

    def execute(self) -> dict[str, object]:
        return self.value


class FakeGmailService:
    def __init__(self) -> None:
        self.messages_client = FakeGmailMessages()

    def users(self):
        return self

    def messages(self):
        return self.messages_client


def test_returns_only_normalised_metadata_from_gmail_metadata_calls() -> None:
    service = FakeGmailService()
    reader = GmailMetadataReader(
        allowed_accounts={"primary@example.test"},
        service_factory=lambda account: service,
    )

    payload = json.loads(
        reader.search_gmail_messages(
            query="is:unread", user_google_email="primary@example.test", page_size=2
        )
    )

    assert payload == {
        "messages": [
            {
                "id": "m1",
                "from": "sender-m1@example.test",
                "subject": "subject-m1",
                "received_at": "2024-07-03T09:46:40+00:00",
                "preview": "preview-m1",
            },
            {
                "id": "m2",
                "from": "sender-m2@example.test",
                "subject": "subject-m2",
                "received_at": "2024-07-03T09:46:40+00:00",
                "preview": "preview-m2",
            },
        ]
    }
    assert service.messages_client.get_calls == [
        {
            "userId": "me",
            "id": "m1",
            "format": "metadata",
            "metadataHeaders": ["From", "Subject"],
        },
        {
            "userId": "me",
            "id": "m2",
            "format": "metadata",
            "metadataHeaders": ["From", "Subject"],
        },
    ]


def test_rejects_unlisted_account_before_calling_google() -> None:
    calls: list[str] = []
    reader = GmailMetadataReader(
        allowed_accounts={"primary@example.test"},
        service_factory=lambda account: calls.append(account),
    )

    with pytest.raises(GmailMetadataPolicyError, match="not allowlisted"):
        reader.search_gmail_messages(
            query="is:unread", user_google_email="other@example.test", page_size=1
        )

    assert calls == []


def test_rejects_any_query_other_than_unread_metadata() -> None:
    reader = GmailMetadataReader(
        allowed_accounts={"primary@example.test"},
        service_factory=lambda account: pytest.fail("provider must not be called"),
    )

    with pytest.raises(GmailMetadataPolicyError, match="only is:unread"):
        reader.search_gmail_messages(
            query="in:inbox", user_google_email="primary@example.test", page_size=1
        )
