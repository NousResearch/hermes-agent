import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.dev_control.clarifications import DevClarificationStore
from gateway.dev_control.product_events import DevProductEventStore, product_event_signature
from gateway.dev_control.production_signals import DevProductionSignalStore, generate_signal_report, transition_backlog_proposal
from gateway.dev_control.signal_source import ProductSignalSource, SignalWindow
from gateway.dev_execution import DevExecutionStore
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server import cors_middleware, security_headers_middleware
from gateway.config import PlatformConfig
from gateway.subagent_events import SubagentEventStore


def _event(**overrides):
    payload = {
        "event_id": f"event-{time.time_ns()}",
        "client_ts": time.time(),
        "type": "product.api_failure",
        "app_version": "1.2.3",
        "session_id": "session-1",
        "message_redacted": "Gateway returned HTTP 500 for /v1/models",
        "context": {
            "error_type": "http_status",
            "endpoint": "/v1/models",
            "status": "500",
            "user_text": "do not store this",
        },
    }
    payload.update(overrides)
    return payload


def test_product_event_ingest_aggregates_duplicate_signatures_and_redacts(tmp_path):
    store = DevProductEventStore(tmp_path / "state.db")
    first = _event(event_id="event-1", message_redacted="Bearer sk-secret1234567890 at /Users/felipe/private/file.txt")
    second = _event(event_id="event-2", message_redacted="different sk-secret1234567890 message should not fragment signature")

    result = store.ingest_batch({"events": [first, second]})
    events = store.list_events()

    assert result["accepted"] == 2
    assert len(events) == 1
    assert events[0]["event_id"] == "event-1"
    assert events[0]["count"] == 2
    assert "[REDACTED]" in events[0]["message_redacted"]
    assert "user_text" not in events[0]["context"]
    assert events[0]["signature"] == product_event_signature("product.api_failure", events[0]["context"])


def test_product_signal_source_clusters_with_evidence_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PRODUCT_SIGNAL_MIN_COUNT", "2")
    store = DevProductEventStore(tmp_path / "state.db")
    store.ingest_batch({"events": [_event(event_id="event-1"), _event(event_id="event-2")]})

    result = ProductSignalSource(store).fetch_clusters(SignalWindow.last_days(7))

    assert result["source"] == "product"
    assert result["analyzed_event_count"] == 2
    assert len(result["clusters"]) == 1
    cluster = result["clusters"][0]
    assert cluster["key"].startswith("product:product.api_failure:")
    assert cluster["count"] == 2
    assert cluster["evidence_refs"][0]["kind"] == "product_event"
    assert cluster["query_descriptor"]["source"] == "product"


def test_product_signal_report_generates_proposal_and_promotes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PRODUCT_SIGNAL_MIN_COUNT", "1")
    db_path = tmp_path / "state.db"
    product_store = DevProductEventStore(db_path)
    signal_store = DevProductionSignalStore(db_path)
    event_store = SubagentEventStore(db_path)
    clarification_store = DevClarificationStore(db_path)
    product_store.ingest_batch({"events": [_event(event_id="event-1")]})

    with patch("gateway.dev_control.clarifications.call_llm", return_value=_fake_clarification_response()):
        report = generate_signal_report(
            signal_store=signal_store,
            event_store=event_store,
            product_event_store=product_store,
            source="product",
            window_days=7,
        )
        proposal = report["proposals"][0]
        promoted = transition_backlog_proposal(
            signal_store=signal_store,
            clarification_store=clarification_store,
            proposal_id=proposal["proposal_id"],
            action="promote",
            project_id="OrynWorkspace",
        )

    assert report["status"] == "completed_with_clusters"
    assert proposal["payload"]["source"] == "production_signal"
    assert proposal["evidence_refs"][0]["kind"] == "product_event"
    assert promoted["seeded_clarification_id"]


@pytest.mark.asyncio
async def test_product_events_api_ingests_and_lists(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    adapter._dev_execution_store = DevExecutionStore(tmp_path / "state.db")
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_get("/v1/dev/product-events", adapter._handle_dev_product_events)
    app.router.add_post("/v1/dev/product-events", adapter._handle_dev_product_events)

    async with TestClient(TestServer(app)) as cli:
        ingest = await cli.post(
            "/v1/dev/product-events",
            json={"events": [_event(event_id="event-1"), _event(event_id="event-2")]},
            headers={"Authorization": "Bearer sk-secret"},
        )
        listed = await cli.get(
            "/v1/dev/product-events",
            headers={"Authorization": "Bearer sk-secret"},
        )
        assert ingest.status == 200
        ingest_data = await ingest.json()
        assert listed.status == 200
        list_data = await listed.json()

    assert ingest_data["accepted"] == 2
    assert len(list_data["data"]) == 1
    assert list_data["data"][0]["count"] == 2


def _fake_clarification_response():
    payload = {
        "questions": [
            {
                "question_id": "q1",
                "prompt": "What product failure should be fixed first?",
                "recommended_option_id": "a",
                "allow_freeform": True,
                "reason": "Scope controls implementation risk.",
                "options": [
                    {"option_id": "a", "label": "API failure", "description": "Fix the repeated API failure."},
                    {"option_id": "b", "label": "Defer", "description": "Leave it for later."},
                ],
            }
        ]
    }

    class Message:
        content = __import__("json").dumps(payload)

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    return Response()
