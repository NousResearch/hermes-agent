from __future__ import annotations

import io
import json
import socket
from urllib.error import HTTPError

import pytest

from cron import api_origin_delivery as delivery


def job(**changes):
    value = {
        "id": "nightly-job-1",
        "name": "Inventory nightly sync",
        "execution_id": "execution-123",
        "deliver": "origin",
        "origin": {"platform": "api_server", "chat_id": "api-session"},
    }
    value.update(changes)
    return value


def config(**changes):
    settings = {
        "url": "http://forge.test/api/cron/deliveries",
        "timeout_seconds": 3,
    }
    settings.update(changes)
    return {"cron": {"api_origin_delivery": settings}}


class Response:
    status = 200

    def __init__(self, body):
        self.body = json.dumps(body).encode() if not isinstance(body, bytes) else body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, limit):
        return self.body[:limit]


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("CRON_API_ORIGIN_DELIVERY_TOKEN", "dedicated-token")
    monkeypatch.setattr(
        delivery,
        "_execution_timestamp",
        lambda _job: "2026-09-22T22:00:00+00:00",
    )


def test_success_posts_bounded_redacted_contract_and_validates_ack(monkeypatch):
    captured = {}

    def send(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response(
            {"ok": True, "entry_id": 7, "run_id": "execution-123", "duplicate": False}
        )

    monkeypatch.setattr(delivery, "urlopen", send)
    handled, error = delivery.deliver_api_origin(
        job(),
        "Authorization: Bearer secret-value\n" + "x" * 13_000,
        execution_success=True,
        for_failure=False,
        config=config(),
    )

    assert handled is True and error is None
    assert captured["timeout"] == 3
    assert captured["request"].get_header("Authorization") == "Bearer dedicated-token"
    body = json.loads(captured["request"].data)
    assert body == {
        "job_id": "nightly-job-1",
        "job_name": "Inventory nightly sync",
        "run_id": "execution-123",
        "executed_at": "2026-09-22T22:00:00+00:00",
        "delivered_at": body["delivered_at"],
        "execution_status": "success",
        "summary": body["summary"],
        "delivery_status": "success",
    }
    assert "secret-value" not in body["summary"]
    assert "[REDACTED]" in body["summary"]
    assert len(body["summary"]) == 12_000


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (b"not-json", "invalid JSON"),
        ({"ok": True}, "run_id mismatch"),
        (
            {"ok": True, "run_id": "execution-123", "entry_id": "bad", "duplicate": False},
            "malformed acknowledgement",
        ),
    ],
)
def test_malformed_responses_record_delivery_failure(monkeypatch, response, expected):
    monkeypatch.setattr(delivery, "urlopen", lambda *_args, **_kwargs: Response(response))
    handled, error = delivery.deliver_api_origin(
        job(), "summary", execution_success=True, for_failure=False, config=config()
    )
    assert handled is True
    assert expected in error


def test_non_2xx_auth_failure_is_useful_and_does_not_expose_body(monkeypatch):
    def rejected(*_args, **_kwargs):
        raise HTTPError(
            "http://forge.test", 401, "Unauthorized", {}, io.BytesIO(b"secret response")
        )

    monkeypatch.setattr(delivery, "urlopen", rejected)
    handled, error = delivery.deliver_api_origin(
        job(), "summary", execution_success=True, for_failure=False, config=config()
    )
    assert handled is True
    assert error == "API-origin Cron delivery endpoint rejected the request (HTTP 401)"
    assert "secret response" not in error


def test_timeout_records_delivery_failure(monkeypatch):
    monkeypatch.setattr(
        delivery, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(socket.timeout())
    )
    handled, error = delivery.deliver_api_origin(
        job(), "summary", execution_success=False, for_failure=True, config=config()
    )
    assert handled is True
    assert error == "API-origin Cron delivery timed out after 3s"


def test_missing_configuration_or_credential_fails_closed(monkeypatch):
    monkeypatch.delenv("CRON_API_ORIGIN_DELIVERY_TOKEN")
    assert delivery.deliver_api_origin(
        job(), "summary", execution_success=True, for_failure=False, config=config()
    ) == (
        True,
        "API-origin Cron delivery credential CRON_API_ORIGIN_DELIVERY_TOKEN is unavailable",
    )
    assert delivery.deliver_api_origin(
        job(), "summary", execution_success=True, for_failure=False, config={}
    ) == (True, "API-origin Cron delivery URL is not configured")


def test_non_api_and_failure_lane_override_keep_existing_delivery_behaviour(monkeypatch):
    called = False

    def send(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(delivery, "urlopen", send)
    assert delivery.deliver_api_origin(
        job(origin={"platform": "telegram", "chat_id": "1"}),
        "summary",
        execution_success=True,
        for_failure=False,
        config=config(),
    ) == (False, None)
    assert delivery.deliver_api_origin(
        job(failure_deliver="local"),
        "summary",
        execution_success=False,
        for_failure=True,
        config=config(),
    ) == (False, None)
    assert called is False
