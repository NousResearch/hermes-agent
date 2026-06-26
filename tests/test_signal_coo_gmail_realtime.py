from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_cli.signal_coo.google_auth import GoogleAccount
from hermes_cli.signal_coo import gmail_realtime


def _encoded_notification(email: str = "eric@example.com", history_id: str = "200") -> str:
    payload = json.dumps({"emailAddress": email, "historyId": history_id}).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _account(tmp_path: Path, *, alias: str = "personal", email: str = "eric@example.com") -> GoogleAccount:
    return GoogleAccount(
        alias=alias,
        email=email,
        role="personal",
        enabled=True,
        token_path=tmp_path / f"{alias}-token.json",
        client_secret_path=tmp_path / "client.json",
        scopes=("https://www.googleapis.com/auth/gmail.modify",),
    )


def test_decode_pubsub_data_handles_urlsafe_padding() -> None:
    assert gmail_realtime.decode_pubsub_data(_encoded_notification()) == {
        "emailAddress": "eric@example.com",
        "historyId": "200",
    }


def test_history_message_ids_includes_inbox_label_added() -> None:
    assert gmail_realtime._history_message_ids(
        [
            {"messages": [{"id": "m-fallback", "threadId": "t1"}]},
            {"labelsAdded": [{"message": {"id": "m-label"}, "labelIds": ["INBOX"]}]},
            {"labelsAdded": [{"message": {"id": "m-skip"}, "labelIds": ["CATEGORY_PROMOTIONS"]}]},
            {"messagesAdded": [{"message": {"id": "m-message", "labelIds": ["INBOX"]}}]},
        ]
    ) == ["m-fallback", "m-label", "m-message"]


def test_process_pubsub_pull_uses_history_messages_fallback_and_filters_sent(tmp_path, monkeypatch) -> None:
    account = _account(tmp_path)
    state_path = tmp_path / "watch-state.json"
    gmail_realtime.write_json(
        state_path,
        {
            "version": 1,
            "accounts": {"personal": {"email": "eric@example.com", "history_id": "100"}},
            "processed_message_keys": [],
        },
    )
    received = [
        {
            "ackId": "ack-1",
            "message": {
                "messageId": "pubsub-1",
                "publishTime": "2026-06-25T10:00:00Z",
                "data": _encoded_notification(history_id="200"),
            },
        }
    ]
    inbox_record = {
        "account_alias": "personal",
        "message_id": "m-inbox",
        "thread_id": "t1",
        "sender": "Max Shapiro",
        "sender_email": "max@example.com",
        "sender_domain": "example.com",
        "subject": "Re: Max <> Eric",
        "date": "Thu, 25 Jun 2026 10:00:00 +0000",
        "category": "calendar_scheduling",
        "juno_bucket": "reply",
        "priority": "high",
        "snippet": "Can you send availability?",
        "body_excerpt": "",
        "labels": ["IMPORTANT", "INBOX"],
        "links": [],
        "evidence_ids": ["gmail:personal:m-inbox"],
    }
    sent_record = {
        **inbox_record,
        "message_id": "m-sent",
        "sender": "eric@example.com",
        "labels": ["SENT"],
        "evidence_ids": ["gmail:personal:m-sent"],
    }
    acked: list[str] = []

    monkeypatch.setattr(gmail_realtime, "_enabled_gmail_accounts", lambda config_path: [account])
    monkeypatch.setattr(gmail_realtime, "pull_pubsub_messages", lambda **kwargs: received)
    monkeypatch.setattr(gmail_realtime, "_read_token", lambda account: "access-token")
    monkeypatch.setattr(
        gmail_realtime,
        "_list_history",
        lambda **kwargs: ([{"messages": [{"id": "m-sent"}, {"id": "m-inbox"}]}], 1, []),
    )

    def fake_metadata(account, token, message_id):
        return (sent_record if message_id == "m-sent" else inbox_record), 1

    monkeypatch.setattr(gmail_realtime, "_gmail_message_metadata", fake_metadata)
    monkeypatch.setattr(gmail_realtime, "load_relationship_context", lambda path: {"people": [], "source_rules": {}, "principles": []})
    monkeypatch.setattr(
        gmail_realtime,
        "build_morning_briefing_candidates",
        lambda records, relationship_context=None: {"critical_emails": [], "learn_contact_candidates": [], "llm_decision_contract": {}},
    )
    monkeypatch.setattr(gmail_realtime, "ack_pubsub_messages", lambda subscription_name, ack_ids: acked.extend(ack_ids))

    result = gmail_realtime.process_pubsub_pull(
        config_path=tmp_path / "config.yaml",
        relationship_context_path=tmp_path / "relationship.yaml",
        state_path=state_path,
        subscription_name="projects/test/subscriptions/torben",
    )

    assert result["wakeAgent"] is True
    assert [candidate["message_key"] for candidate in result["candidates"]] == ["personal:m-inbox"]
    assert acked == ["ack-1"]


def test_register_gmail_watches_writes_cursor_without_mailbox_mutation(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "google_accounts.yaml"
    token_path = tmp_path / "token.json"
    client_path = tmp_path / "client.json"
    token_path.write_text("{}", encoding="utf-8")
    client_path.write_text("{}", encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(
            {
                "accounts": {
                    "personal": {
                        "email": "eric@example.com",
                        "role": "personal",
                        "enabled": True,
                        "token_path": str(token_path),
                        "client_secret_path": str(client_path),
                        "scopes": ["https://www.googleapis.com/auth/gmail.modify"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    posted: dict[str, Any] = {}

    monkeypatch.setattr(gmail_realtime, "_read_token", lambda account: "access-token")

    def fake_post(url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        posted.update({"url": url, "token": token, "payload": payload})
        return {"historyId": "12345", "expiration": "1790000000000"}

    monkeypatch.setattr(gmail_realtime, "_google_post", fake_post)

    state_path = tmp_path / "watch-state.json"
    result = gmail_realtime.register_gmail_watches(
        config_path=config_path,
        state_path=state_path,
        topic_name="projects/test/topics/torben-gmail-watch",
    )

    assert result["wakeAgent"] is False
    assert result["status"] == "pass"
    assert result["diagnostics"]["gmail_mailbox_mutations"] == 0
    assert posted["payload"] == {
        "topicName": "projects/test/topics/torben-gmail-watch",
        "labelIds": ["INBOX"],
        "labelFilterBehavior": "INCLUDE",
    }
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["accounts"]["personal"]["history_id"] == "12345"


def test_process_pubsub_pull_no_messages_is_silent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(gmail_realtime, "pull_pubsub_messages", lambda **kwargs: [])

    result = gmail_realtime.process_pubsub_pull(
        config_path=tmp_path / "missing.yaml",
        relationship_context_path=tmp_path / "relationship.yaml",
        state_path=tmp_path / "watch-state.json",
        subscription_name="projects/test/subscriptions/torben",
    )

    assert result["wakeAgent"] is False
    assert result["reason"] == "no pubsub notifications"
    assert result["diagnostics"]["external_mutations"] == 0


def test_process_pubsub_pull_no_messages_runs_history_fallback(tmp_path, monkeypatch) -> None:
    account = _account(tmp_path)
    state_path = tmp_path / "watch-state.json"
    gmail_realtime.write_json(
        state_path,
        {
            "version": 1,
            "accounts": {"personal": {"email": "eric@example.com", "history_id": "100"}},
            "processed_message_keys": [],
        },
    )
    record = {
        "account_alias": "personal",
        "message_id": "m-inbox",
        "thread_id": "t1",
        "sender": "Max Shapiro",
        "sender_email": "max@example.com",
        "sender_domain": "example.com",
        "subject": "Re: Max <> Eric",
        "date": "Thu, 25 Jun 2026 10:00:00 +0000",
        "category": "calendar_scheduling",
        "juno_bucket": "reply",
        "priority": "high",
        "snippet": "Can you send availability?",
        "body_excerpt": "",
        "labels": ["IMPORTANT", "INBOX"],
        "links": [],
        "evidence_ids": ["gmail:personal:m-inbox"],
    }

    monkeypatch.setattr(gmail_realtime, "_enabled_gmail_accounts", lambda config_path: [account])
    monkeypatch.setattr(gmail_realtime, "pull_pubsub_messages", lambda **kwargs: [])
    monkeypatch.setattr(gmail_realtime, "_read_token", lambda account: "access-token")
    monkeypatch.setattr(
        gmail_realtime,
        "_list_history",
        lambda **kwargs: ([{"id": "150", "messages": [{"id": "m-inbox"}]}], 1, []),
    )
    monkeypatch.setattr(gmail_realtime, "_gmail_message_metadata", lambda account, token, message_id: (record, 1))
    monkeypatch.setattr(gmail_realtime, "load_relationship_context", lambda path: {"people": [], "source_rules": {}, "principles": []})
    monkeypatch.setattr(
        gmail_realtime,
        "build_morning_briefing_candidates",
        lambda records, relationship_context=None: {"critical_emails": [], "learn_contact_candidates": [], "llm_decision_contract": {}},
    )

    result = gmail_realtime.process_pubsub_pull(
        config_path=tmp_path / "config.yaml",
        relationship_context_path=tmp_path / "relationship.yaml",
        state_path=state_path,
        subscription_name="projects/test/subscriptions/torben",
    )

    assert result["wakeAgent"] is True
    assert result["diagnostics"]["history_fallback"] is True
    assert result["candidates"][0]["message_key"] == "personal:m-inbox"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["accounts"]["personal"]["history_id"] == "150"


def test_process_pubsub_pull_fetches_history_stages_candidate_and_acks(tmp_path, monkeypatch) -> None:
    account = _account(tmp_path)
    state_path = tmp_path / "watch-state.json"
    gmail_realtime.write_json(
        state_path,
        {
            "version": 1,
            "accounts": {"personal": {"email": "eric@example.com", "history_id": "100"}},
            "processed_message_keys": [],
        },
    )
    received = [
        {
            "ackId": "ack-1",
            "message": {
                "messageId": "pubsub-1",
                "publishTime": "2026-06-25T10:00:00Z",
                "data": _encoded_notification(history_id="200"),
            },
        }
    ]
    record = {
        "account_alias": "personal",
        "message_id": "m1",
        "thread_id": "t1",
        "sender": "Kim Moore",
        "sender_email": "kim@example.com",
        "sender_domain": "example.com",
        "subject": "Follow up on funding",
        "date": "Thu, 25 Jun 2026 10:00:00 +0000",
        "category": "founder_funding_customer",
        "juno_bucket": "reply",
        "priority": "high",
        "snippet": "Can you send availability for a follow-up?",
        "body_excerpt": "",
        "labels": ["INBOX"],
        "links": [],
        "evidence_ids": ["gmail:personal:m1"],
    }
    acked: list[str] = []

    monkeypatch.setattr(gmail_realtime, "_enabled_gmail_accounts", lambda config_path: [account])
    monkeypatch.setattr(gmail_realtime, "pull_pubsub_messages", lambda **kwargs: received)
    monkeypatch.setattr(gmail_realtime, "_read_token", lambda account: "access-token")
    monkeypatch.setattr(
        gmail_realtime,
        "_list_history",
        lambda **kwargs: ([{"messagesAdded": [{"message": {"id": "m1", "labelIds": ["INBOX"]}}]}], 1, []),
    )
    monkeypatch.setattr(gmail_realtime, "_fetch_records", lambda **kwargs: ([record], 1, []))
    monkeypatch.setattr(gmail_realtime, "load_relationship_context", lambda path: {"people": [], "source_rules": {}, "principles": []})
    monkeypatch.setattr(
        gmail_realtime,
        "build_morning_briefing_candidates",
        lambda records, relationship_context=None: {"critical_emails": [], "learn_contact_candidates": [], "llm_decision_contract": {}},
    )
    monkeypatch.setattr(gmail_realtime, "ack_pubsub_messages", lambda subscription_name, ack_ids: acked.extend(ack_ids))

    result = gmail_realtime.process_pubsub_pull(
        config_path=tmp_path / "config.yaml",
        relationship_context_path=tmp_path / "relationship.yaml",
        state_path=state_path,
        subscription_name="projects/test/subscriptions/torben",
    )

    assert result["wakeAgent"] is True
    assert result["candidates"][0]["message_key"] == "personal:m1"
    assert result["candidates"][0]["handle"].startswith("EA-")
    assert acked == ["ack-1"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["accounts"]["personal"]["history_id"] == "200"
    assert "personal:m1" in state["processed_message_keys"]


def test_process_pubsub_pull_suppresses_duplicate_message_keys(tmp_path, monkeypatch) -> None:
    account = _account(tmp_path)
    state_path = tmp_path / "watch-state.json"
    gmail_realtime.write_json(
        state_path,
        {
            "version": 1,
            "accounts": {"personal": {"email": "eric@example.com", "history_id": "100"}},
            "processed_message_keys": ["personal:m1"],
        },
    )
    received = [
        {
            "ackId": "ack-1",
            "message": {
                "messageId": "pubsub-1",
                "publishTime": "2026-06-25T10:00:00Z",
                "data": _encoded_notification(history_id="200"),
            },
        }
    ]
    record = {
        "account_alias": "personal",
        "message_id": "m1",
        "thread_id": "t1",
        "sender": "Kim Moore",
        "subject": "Follow up on funding",
        "category": "founder_funding_customer",
        "juno_bucket": "reply",
        "labels": ["INBOX"],
        "evidence_ids": ["gmail:personal:m1"],
    }
    acked: list[str] = []

    monkeypatch.setattr(gmail_realtime, "_enabled_gmail_accounts", lambda config_path: [account])
    monkeypatch.setattr(gmail_realtime, "pull_pubsub_messages", lambda **kwargs: received)
    monkeypatch.setattr(gmail_realtime, "_read_token", lambda account: "access-token")
    monkeypatch.setattr(
        gmail_realtime,
        "_list_history",
        lambda **kwargs: ([{"messagesAdded": [{"message": {"id": "m1", "labelIds": ["INBOX"]}}]}], 1, []),
    )
    monkeypatch.setattr(gmail_realtime, "_fetch_records", lambda **kwargs: ([record], 1, []))
    monkeypatch.setattr(gmail_realtime, "load_relationship_context", lambda path: {"people": [], "source_rules": {}, "principles": []})
    monkeypatch.setattr(
        gmail_realtime,
        "build_morning_briefing_candidates",
        lambda records, relationship_context=None: {"critical_emails": [], "learn_contact_candidates": [], "llm_decision_contract": {}},
    )
    monkeypatch.setattr(gmail_realtime, "ack_pubsub_messages", lambda subscription_name, ack_ids: acked.extend(ack_ids))

    result = gmail_realtime.process_pubsub_pull(
        config_path=tmp_path / "config.yaml",
        relationship_context_path=tmp_path / "relationship.yaml",
        state_path=state_path,
        subscription_name="projects/test/subscriptions/torben",
    )

    assert result["wakeAgent"] is False
    assert result["reason"] == "pubsub notifications processed with no realtime candidates"
    assert acked == ["ack-1"]


def test_fresh_realtime_records_suppresses_stale_history_messages() -> None:
    now = datetime.now(timezone.utc)
    old_record = {
        "message_id": "old-1",
        "internal_date_ms": str(int((now - timedelta(hours=3)).timestamp() * 1000)),
    }
    fresh_record = {
        "message_id": "fresh-1",
        "internal_date_ms": str(int((now - timedelta(minutes=5)).timestamp() * 1000)),
    }
    warnings: list[str] = []

    kept = gmail_realtime._fresh_realtime_records(
        [old_record, fresh_record],
        max_age_seconds=3600,
        warnings=warnings,
    )

    assert kept == [fresh_record]
    assert warnings == ["suppressed 1 stale Gmail history message(s) older than realtime max age"]


def test_run_gmail_realtime_canary_processes_and_cleans_up(tmp_path, monkeypatch) -> None:
    account = _account(tmp_path)
    state_path = tmp_path / "watch-state.json"
    gmail_realtime.write_json(
        state_path,
        {
            "version": 1,
            "accounts": {"personal": {"email": "eric@example.com", "history_id": "100"}},
            "processed_message_keys": [],
        },
    )
    cleanup: list[str] = []

    monkeypatch.setattr(gmail_realtime, "_enabled_gmail_accounts", lambda config_path: [account])
    monkeypatch.setattr(gmail_realtime, "_read_token", lambda account: "access-token")
    monkeypatch.setattr(
        gmail_realtime,
        "_gmail_import_message",
        lambda account, token, subject, body, label_ids=None: ("m-canary", 1),
    )
    monkeypatch.setattr(gmail_realtime, "_gmail_modify_message_labels", lambda token, message_id, add, remove=None: 1)

    def fake_process(**kwargs):
        state = gmail_realtime.load_json(state_path, {})
        state["last_pubsub_message_ids_by_account"] = {"personal": ["m-canary"]}
        gmail_realtime.write_json(state_path, state)
        return {
            "task": "torben_gmail_pubsub_pull",
            "wakeAgent": False,
            "reason": "pubsub notifications processed with no realtime candidates",
            "diagnostics": {"new_message_count": 1, "gmail_writes": 0, "external_mutations": 0},
        }

    monkeypatch.setattr(gmail_realtime, "process_pubsub_pull", fake_process)
    monkeypatch.setattr(gmail_realtime, "_gmail_trash_message", lambda token, message_id: cleanup.append(message_id) or 1)
    monkeypatch.setattr(gmail_realtime.time, "sleep", lambda seconds: None)

    result = gmail_realtime.run_gmail_realtime_canary(
        config_path=tmp_path / "config.yaml",
        relationship_context_path=tmp_path / "relationship.yaml",
        state_path=state_path,
        poll_interval_seconds=1,
        timeout_seconds=5,
    )

    assert result["status"] == "pass"
    assert result["wakeAgent"] is False
    assert result["canary_message"]["processed_by_pubsub_history"] is True
    assert result["canary_message"]["cleanup_status"] == "trashed_canary_message"
    assert result["diagnostics"]["gmail_mailbox_mutations"] == 3
    assert cleanup == ["m-canary"]
