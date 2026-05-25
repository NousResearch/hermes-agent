import hashlib
import hmac
import json

import pytest

from hermes_cli import linear_aig


SECRET = "linear-signing-secret"
NOW_MS = 1_770_000_000_000


def _body(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _signature(raw_body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def _payload(**overrides):
    payload = {
        "action": "created",
        "webhookTimestamp": NOW_MS,
        "agentSession": {
            "id": "ags_123",
            "promptContext": "Implement WHO-192",
            "issue": {
                "identifier": "WHO-192",
                "title": "Implement Hermes Linear AIG webhook receiver",
            },
        },
    }
    payload.update(overrides)
    return payload


def test_verify_linear_signature_accepts_valid_signature():
    payload = _payload()
    raw_body = _body(payload)

    assert linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"Linear-Signature": _signature(raw_body)},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_accepts_sha256_prefix():
    payload = _payload()
    raw_body = _body(payload)

    assert linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"linear-signature": f"sha256={_signature(raw_body)}"},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_rejects_bad_signature():
    payload = _payload()

    assert not linear_aig.verify_linear_signature(
        raw_body=_body(payload),
        headers={"Linear-Signature": "0" * 64},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_rejects_stale_timestamp():
    payload = _payload(webhookTimestamp=NOW_MS - 90_000)
    raw_body = _body(payload)

    assert not linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"Linear-Signature": _signature(raw_body)},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_verify_linear_signature_rejects_missing_timestamp():
    payload = _payload()
    payload.pop("webhookTimestamp")
    raw_body = _body(payload)

    assert not linear_aig.verify_linear_signature(
        raw_body=raw_body,
        headers={"Linear-Signature": _signature(raw_body)},
        signing_secret=SECRET,
        payload=payload,
        now_ms=NOW_MS,
    )


def test_parse_agent_session_event_extracts_issue_and_prompt_context():
    event = linear_aig.parse_agent_session_event(_payload())

    assert event.action == "created"
    assert event.agent_session_id == "ags_123"
    assert event.prompt == "Implement WHO-192"
    assert event.issue_identifier == "WHO-192"
    assert event.issue_title == "Implement Hermes Linear AIG webhook receiver"


def test_parse_agent_session_event_accepts_nested_data_shape():
    event = linear_aig.parse_agent_session_event(
        {
            "action": "created",
            "data": {
                "promptContext": "Nested prompt",
                "agentSession": {"id": "ags_nested"},
            },
        }
    )

    assert event.agent_session_id == "ags_nested"
    assert event.prompt_context == "Nested prompt"


def test_parse_agent_session_event_requires_action_and_session_id():
    with pytest.raises(ValueError, match="missing action"):
        linear_aig.parse_agent_session_event({"agentSession": {"id": "ags_123"}})

    with pytest.raises(ValueError, match="missing agentSession.id"):
        linear_aig.parse_agent_session_event({"action": "created"})


def test_build_activity_content_supports_linear_content_types():
    assert linear_aig.build_activity_content("thought", "Starting") == {
        "type": "thought",
        "body": "Starting",
    }
    assert linear_aig.build_activity_content("action", "checkout", url=None) == {
        "type": "action",
        "action": "checkout",
    }

    with pytest.raises(ValueError, match="Unsupported"):
        linear_aig.build_activity_content("note", "Nope")


@pytest.mark.asyncio
async def test_process_event_emits_start_and_response_activity():
    sent = []

    async def sender(agent_session_id, content):
        sent.append((agent_session_id, content))

    async def dispatcher(event):
        assert event.issue_identifier == "WHO-192"
        return "Ready for review."

    receiver = linear_aig.LinearAIGReceiver(
        signing_secret=SECRET,
        activity_sender=sender,
        task_dispatcher=dispatcher,
    )

    await receiver.process_event(linear_aig.parse_agent_session_event(_payload()))

    assert sent == [
        (
            "ags_123",
            {
                "type": "thought",
                "body": "Hermes received Linear Agent Session `created` and is starting.",
            },
        ),
        ("ags_123", {"type": "response", "body": "Ready for review."}),
    ]


@pytest.mark.asyncio
async def test_process_event_emits_error_activity_on_dispatch_failure():
    sent = []

    async def sender(agent_session_id, content):
        sent.append((agent_session_id, content))

    async def dispatcher(_event):
        raise RuntimeError("boom")

    receiver = linear_aig.LinearAIGReceiver(
        signing_secret=SECRET,
        activity_sender=sender,
        task_dispatcher=dispatcher,
    )

    await receiver.process_event(linear_aig.parse_agent_session_event(_payload()))

    assert sent[-1] == ("ags_123", {"type": "error", "body": "Hermes failed: boom"})


@pytest.mark.asyncio
async def test_send_agent_activity_posts_graphql_payload(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"agentActivityCreate": {"success": True}}}

    def post(url, *, headers, json, timeout):
        calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return Response()

    monkeypatch.setattr(linear_aig.httpx, "post", post)

    await linear_aig.send_agent_activity(
        api_key="lin_api_key",
        agent_session_id="ags_123",
        content={"type": "response", "body": "Done"},
        graphql_url="https://linear.test/graphql",
        timeout_seconds=3,
    )

    assert calls[0]["url"] == "https://linear.test/graphql"
    assert calls[0]["headers"] == {
        "Authorization": "Bearer lin_api_key",
        "Content-Type": "application/json",
    }
    assert "agentActivityCreate" in calls[0]["json"]["query"]
    assert calls[0]["json"]["variables"] == {
        "input": {
            "agentSessionId": "ags_123",
            "content": {"type": "response", "body": "Done"},
        },
    }
    assert calls[0]["timeout"] == 3


@pytest.mark.asyncio
async def test_send_agent_activity_raises_on_graphql_error(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"errors": [{"message": "bad input"}]}

    monkeypatch.setattr(linear_aig.httpx, "post", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="Linear GraphQL error"):
        await linear_aig.send_agent_activity(
            api_key="lin_api_key",
            agent_session_id="ags_123",
            content={"type": "response", "body": "Done"},
        )
