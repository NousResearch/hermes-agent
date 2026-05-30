import inspect
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.dev_control.clarifications import DevClarificationStore
from gateway.dev_control.incidents import (
    DevIncidentStore,
    detect_incidents,
    rollback_recommendation,
    resolve_incident,
)
from gateway.dev_control.product_events import DevProductEventStore
from gateway.dev_control.production_signals import DevProductionSignalStore, transition_backlog_proposal
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


def _stores(tmp_path):
    db_path = tmp_path / "state.db"
    return (
        DevProductEventStore(db_path=db_path),
        DevIncidentStore(db_path=db_path),
        DevProductionSignalStore(db_path=db_path),
        DevClarificationStore(db_path=db_path),
    )


def _ingest_crash(store, *, version="2.0.0", count=3, endpoint="/launch"):
    events = [
        {
            "event_id": f"evt-{version}-{index}",
            "type": "product.crash",
            "app_version": version,
            "message": "Crash in startup coordinator",
            "context": {"endpoint": endpoint, "error_type": "StartupCrash"},
        }
        for index in range(count)
    ]
    result = store.ingest_batch({"events": events})
    assert result["accepted"] == count


def _ci_fetcher(states):
    def fetch(*, repo, ref):
        return {"state": states.get(ref, "failure"), "warnings": [], "failing": []}

    return fetch


def test_severe_correlated_product_crash_creates_incident(tmp_path):
    product_store, incident_store, signal_store, _ = _stores(tmp_path)
    _ingest_crash(product_store, version="2.0.0", count=3)
    now = time.time()

    result = detect_incidents(
        incident_store=incident_store,
        product_event_store=product_store,
        signal_store=signal_store,
        current_release={"version": "2.0.0", "commit": "current", "released_at": now - 120},
        releases=[{"version": "1.9.0", "commit": "good", "released_at": now - 86400}],
        repo="Felippen/Oryn",
        ci_status_fetcher=_ci_fetcher({"good": "success"}),
        now=now,
    )

    assert result["counts"]["incident_count"] == 1
    incident = result["incidents"][0]
    assert incident["severity"] == "urgent"
    assert incident["evidence_refs"]
    assert incident["correlated_release"]["version"] == "2.0.0"
    assert incident["recommendation"]["target_commit"] == "good"
    assert incident_store.get_incident(incident["incident_id"]) is not None


def test_severe_uncorrelated_product_crash_does_not_create_incident(tmp_path):
    product_store, incident_store, signal_store, _ = _stores(tmp_path)
    _ingest_crash(product_store, version="1.8.0", count=3)
    now = time.time()

    result = detect_incidents(
        incident_store=incident_store,
        product_event_store=product_store,
        signal_store=signal_store,
        current_release={"version": "2.0.0", "commit": "current", "released_at": now - 7 * 86400},
        releases=[],
        repo="Felippen/Oryn",
        ci_status_fetcher=_ci_fetcher({}),
        now=now,
    )

    assert result["counts"]["incident_count"] == 0
    assert result["counts"]["uncorrelated_severe_count"] == 1
    assert incident_store.list_incidents() == []


def test_rollback_recommendation_uses_most_recent_prior_ci_green_release():
    now = time.time()
    recommendation = rollback_recommendation(
        incident_cluster={"key": "product:product.crash:abc", "count": 4},
        current_release={"version": "2.0.0", "commit": "current", "released_at": now},
        releases=[
            {"version": "1.9.0", "commit": "red", "released_at": now - 3600, "tag": "v1.9.0"},
            {"version": "1.8.0", "commit": "green", "released_at": now - 7200, "tag": "v1.8.0"},
        ],
        repo="Felippen/Oryn",
        ci_status_fetcher=_ci_fetcher({"red": "failure", "green": "success"}),
    )

    assert recommendation["available"] is True
    assert recommendation["target_commit"] == "green"
    assert recommendation["target_tag"] == "v1.8.0"
    assert [item["commit"] for item in recommendation["checked_releases"]] == ["red", "green"]
    assert recommendation["manual_only"] is True


def test_rollback_recommendation_states_when_no_green_target_exists():
    now = time.time()
    recommendation = rollback_recommendation(
        incident_cluster={"key": "product:product.crash:abc", "count": 4},
        current_release={"version": "2.0.0", "commit": "current", "released_at": now},
        releases=[{"version": "1.9.0", "commit": "red", "released_at": now - 3600}],
        repo="Felippen/Oryn",
        ci_status_fetcher=_ci_fetcher({"red": "failure"}),
    )

    assert recommendation["available"] is False
    assert recommendation["target_commit"] is None
    assert "No prior stable release" in recommendation["rationale"]


def test_incident_lifecycle_resolve_creates_postmortem_proposal_that_can_promote(tmp_path, monkeypatch):
    product_store, incident_store, signal_store, clarification_store = _stores(tmp_path)
    _ingest_crash(product_store, version="2.0.0", count=2)
    now = time.time()
    detected = detect_incidents(
        incident_store=incident_store,
        product_event_store=product_store,
        signal_store=signal_store,
        current_release={"version": "2.0.0", "commit": "current", "released_at": now - 60},
        releases=[{"version": "1.9.0", "commit": "good", "released_at": now - 86400}],
        repo="Felippen/Oryn",
        ci_status_fetcher=_ci_fetcher({"good": "success"}),
        now=now,
    )
    incident = detected["incidents"][0]

    resolved = resolve_incident(
        incident_store=incident_store,
        signal_store=signal_store,
        incident_id=incident["incident_id"],
        postmortem={
            "timeline": ["Crash spike detected", "Manual rollback recommended"],
            "action_taken": "Operator republished the previous stable manifest.",
            "root_cause_hypothesis": "Startup migration regressed.",
            "preventive_action": "Add startup migration smoke tests.",
        },
    )

    assert resolved["status"] == "resolved"
    assert resolved["proposal_id"]
    proposal = signal_store.get_proposal(resolved["proposal_id"])
    assert proposal["payload"]["source"] == "incident_postmortem"
    promoted = transition_backlog_proposal(
        signal_store=signal_store,
        clarification_store=clarification_store,
        proposal_id=resolved["proposal_id"],
        action="promote",
        project_id="OrynWorkspace",
    )
    assert promoted["status"] == "promoted"
    assert promoted["seeded_clarification_id"]


def test_incident_module_does_not_execute_rollback_actions():
    import gateway.dev_control.incidents as incidents

    source = inspect.getsource(incidents)
    assert "subprocess" not in source
    assert "os.system" not in source
    assert "WorkspacePublishRunner" not in source
    assert "manifest.write" not in source


@pytest.mark.asyncio
async def test_incident_api_detect_list_detail_and_acknowledge(tmp_path, monkeypatch):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    db_path = tmp_path / "state.db"
    product_store = DevProductEventStore(db_path=db_path)
    _ingest_crash(product_store, version="2.0.0", count=2)
    monkeypatch.setattr(adapter, "_ensure_dev_execution_store", lambda: type("Store", (), {"db_path": db_path})())
    monkeypatch.setattr("gateway.dev_control.incidents.fetch_ci_status", lambda repo, ref: {"state": "success", "warnings": []})

    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_post("/v1/dev/incidents/detect", adapter._handle_dev_incident_detect)
    app.router.add_get("/v1/dev/incidents", adapter._handle_dev_incidents)
    app.router.add_get("/v1/dev/incidents/{incident_id}", adapter._handle_dev_incident_detail)
    app.router.add_post("/v1/dev/incidents/{incident_id}/{action}", adapter._handle_dev_incident_action)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/v1/dev/incidents/detect",
            headers={"Authorization": "Bearer sk-secret"},
            json={
                "repo": "Felippen/Oryn",
                "current_release": {"version": "2.0.0", "commit": "current", "released_at": time.time() - 60},
                "releases": [{"version": "1.9.0", "commit": "good", "released_at": time.time() - 86400}],
            },
        )
        detected = await response.json()
        incident_id = detected["incidents"][0]["incident_id"]
        listed_response = await cli.get("/v1/dev/incidents", headers={"Authorization": "Bearer sk-secret"})
        detail_response = await cli.get(f"/v1/dev/incidents/{incident_id}", headers={"Authorization": "Bearer sk-secret"})
        ack_response = await cli.post(f"/v1/dev/incidents/{incident_id}/acknowledge", headers={"Authorization": "Bearer sk-secret"})
        listed = await listed_response.json()
        detail = await detail_response.json()
        ack = await ack_response.json()

    assert response.status == 200
    assert listed_response.status == 200
    assert detail_response.status == 200
    assert ack_response.status == 200
    assert listed["total"] == 1
    assert detail["incident_id"] == incident_id
    assert ack["status"] == "acknowledged"
