from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture()
def dashboard_client(_isolate_hermes_home, monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"dashboard": {"evidence_cards_enabled": True}},
    )
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def _payload(**overrides):
    payload = {
        "kind": "validation",
        "title": "Focused EV1 validation",
        "summary": "pytest tests/hermes_cli/test_mission_control_evidence_cards.py passed",
        "details": "Supplied validation evidence only.",
        "structured_payload": {"command": "pytest focused", "exit_code": 0},
        "limitations": ["No broad tests run."],
        "redaction_notes": ["No raw secrets retained."],
        "source": "discord://thread/ev1",
        "created_by": "dashboard-test",
        "created_from": "operator_supplied",
    }
    payload.update(overrides)
    return payload


def test_evidence_card_routes_require_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)
    assert client.get("/api/mission-control/evidence-cards").status_code == 401
    assert client.post("/api/mission-control/evidence-cards", json={"title": "x"}).status_code == 401
    assert client.get("/api/mission-control/evidence-cards/card_demo").status_code == 401


def test_evidence_card_routes_are_not_public():
    from hermes_cli.web_server import _PUBLIC_API_PATHS

    assert "/api/mission-control/evidence-cards" not in _PUBLIC_API_PATHS


def test_evidence_cards_feature_flag_default_disabled(_isolate_hermes_home, monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli import web_server
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    assert DEFAULT_CONFIG["dashboard"]["evidence_cards_enabled"] is False
    monkeypatch.setattr(web_server, "load_config", lambda: {})
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    resp = client.get("/api/mission-control/evidence-cards")

    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"]


def test_create_list_get_and_audit_store_file_per_record(dashboard_client):
    from hermes_constants import get_hermes_home

    create = dashboard_client.post("/api/mission-control/evidence-cards", json=_payload())

    assert create.status_code == 200
    card = create.json()["card"]
    assert card["schema"] == "mission-control.evidence-card.v1"
    assert card["kind"] == "validation"
    assert card["trusted_for_execution"] is False
    assert card["inert_context_only"] is True
    assert card["authorizing"] is False

    list_resp = dashboard_client.get("/api/mission-control/evidence-cards")
    assert list_resp.status_code == 200
    assert [item["id"] for item in list_resp.json()["items"]] == [card["id"]]

    get_resp = dashboard_client.get(f"/api/mission-control/evidence-cards/{card['id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["card"]["id"] == card["id"]

    state_dir = get_hermes_home() / "state" / "mission-control" / "evidence-cards"
    audit_path = get_hermes_home() / "state" / "mission-control" / "evidence-cards-audit.jsonl"
    assert (state_dir / f"{card['id']}.json").is_file()
    assert "evidence_card_created" in audit_path.read_text(encoding="utf-8")


def test_all_required_evidence_kinds_are_accepted(dashboard_client):
    kinds = [
        "repo_state",
        "diff_summary",
        "validation",
        "secret_scan",
        "safety_scan",
        "dirty_worktree",
        "file_locality",
        "capability_state",
        "reviewer_verdict",
        "limitation",
        "commit_report",
    ]

    for kind in kinds:
        resp = dashboard_client.post("/api/mission-control/evidence-cards", json=_payload(kind=kind))
        assert resp.status_code == 200
        assert resp.json()["card"]["kind"] == kind


def test_unknown_kind_is_rejected(dashboard_client):
    resp = dashboard_client.post("/api/mission-control/evidence-cards", json=_payload(kind="approval"))

    assert resp.status_code == 400
    assert "kind" in resp.json()["detail"]


def test_inert_non_authorizing_flags_are_forced(dashboard_client):
    resp = dashboard_client.post(
        "/api/mission-control/evidence-cards",
        json=_payload(
            trusted_for_execution=True,
            inert_context_only=False,
            authorizing=True,
        ),
    )

    assert resp.status_code == 200
    card = resp.json()["card"]
    assert card["trusted_for_execution"] is False
    assert card["inert_context_only"] is True
    assert card["authorizing"] is False


def test_bounded_fields_are_safely_truncated(dashboard_client):
    from hermes_cli import mission_control_evidence_cards as cards

    long_text = "x" * (cards.MAX_TEXT_CHARS + 50)
    long_item = "y" * (cards.MAX_LIST_ITEM_CHARS + 50)
    structured = {
        f"k{i}": "z" * 200 for i in range(cards.MAX_STRUCTURED_KEYS + 25)
    }

    resp = dashboard_client.post(
        "/api/mission-control/evidence-cards",
        json=_payload(
            summary=long_text,
            details=long_text,
            structured_payload=structured,
            limitations=[long_item] * (cards.MAX_LIST_ITEMS + 25),
            redaction_notes=[long_item] * (cards.MAX_LIST_ITEMS + 25),
        ),
    )

    assert resp.status_code == 200
    card = resp.json()["card"]
    assert len(card["summary"]) == cards.MAX_TEXT_CHARS
    assert len(card["details"]) == cards.MAX_TEXT_CHARS
    assert len(card["structured_payload"]) == cards.MAX_STRUCTURED_KEYS
    assert len(card["limitations"]) == cards.MAX_LIST_ITEMS
    assert len(card["limitations"][0]) == cards.MAX_LIST_ITEM_CHARS
    assert len(card["redaction_notes"]) == cards.MAX_LIST_ITEMS
    assert len(card["redaction_notes"][0]) == cards.MAX_LIST_ITEM_CHARS


def test_secret_like_values_are_redacted_from_response_store_and_audit(dashboard_client):
    from hermes_constants import get_hermes_home

    resp = dashboard_client.post(
        "/api/mission-control/evidence-cards",
        json=_payload(
            summary="Authorization: Bearer CARDSECRET",
            details="api_key=DETAILSECRET",
            structured_payload={
                "nested": {"token": "STRUCTSECRET"},
                "note": "Bearer STRUCTTEXTSECRET",
            },
            limitations=["password=LIMITSECRET"],
            redaction_notes=["client_secret=REDACTSECRET"],
        ),
    )

    assert resp.status_code == 200
    rendered_response = json.dumps(resp.json())
    for secret in [
        "CARDSECRET",
        "DETAILSECRET",
        "STRUCTSECRET",
        "STRUCTTEXTSECRET",
        "LIMITSECRET",
        "REDACTSECRET",
    ]:
        assert secret not in rendered_response

    card = resp.json()["card"]
    stored = (
        get_hermes_home()
        / "state"
        / "mission-control"
        / "evidence-cards"
        / f"{card['id']}.json"
    ).read_text(encoding="utf-8")
    audit = (
        get_hermes_home() / "state" / "mission-control" / "evidence-cards-audit.jsonl"
    ).read_text(encoding="utf-8")
    for secret in [
        "CARDSECRET",
        "DETAILSECRET",
        "STRUCTSECRET",
        "STRUCTTEXTSECRET",
        "LIMITSECRET",
        "REDACTSECRET",
    ]:
        assert secret not in stored
        assert secret not in audit


def test_card_creation_does_not_call_runtime_or_probing_paths(dashboard_client, monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.goals as runtime_goals
    import hermes_cli.mission_control as packets

    def fail(*args, **kwargs):
        raise AssertionError("Evidence Card creation must remain inert")

    monkeypatch.setattr(runtime_goals, "load_goal", fail)
    monkeypatch.setattr(runtime_goals, "save_goal", fail)
    monkeypatch.setattr(runtime_goals, "GoalManager", fail)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_handle_goal_command", fail)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_dispatch_command", fail, raising=False)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_persist_safe_goal_task_contract", fail)
    monkeypatch.setattr(packets, "parse_worker_result_metadata", fail)
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(urllib.request, "urlopen", fail)

    original_read_text = Path.read_text
    original_stat = Path.stat
    forbidden_values = {
        "/tmp/not-read",
        "~/not-expanded",
        "../not-normalized",
        "https://example.invalid/not-fetched",
    }

    def guarded_read_text(self, *args, **kwargs):
        if str(self) in forbidden_values:
            raise AssertionError("Evidence Card creation must not read supplied refs")
        return original_read_text(self, *args, **kwargs)

    def guarded_stat(self, *args, **kwargs):
        if str(self) in forbidden_values:
            raise AssertionError("Evidence Card creation must not stat supplied refs")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "stat", guarded_stat)

    resp = dashboard_client.post(
        "/api/mission-control/evidence-cards",
        json=_payload(
            kind="file_locality",
            structured_payload={
                "refs": sorted(forbidden_values),
                "secret_scan": "supplied only, do not run",
                "tests": "supplied only, do not run",
            },
        ),
    )

    assert resp.status_code == 200
    assert resp.json()["card"]["structured_payload"]["refs"] == sorted(forbidden_values)


def test_reviewer_verdict_can_store_not_available(dashboard_client):
    resp = dashboard_client.post(
        "/api/mission-control/evidence-cards",
        json=_payload(
            kind="reviewer_verdict",
            structured_payload={"verdict": "not_available", "reason": "No reviewer in EV1."},
        ),
    )

    assert resp.status_code == 200
    assert resp.json()["card"]["structured_payload"]["verdict"] == "not_available"


def test_commit_report_can_store_supplied_pre_commit_validation(dashboard_client):
    resp = dashboard_client.post(
        "/api/mission-control/evidence-cards",
        json=_payload(
            kind="commit_report",
            structured_payload={
                "commit": "not_created",
                "pre_commit_validation": {
                    "pytest": "focused passed",
                    "py_compile": "changed files passed",
                },
            },
        ),
    )

    assert resp.status_code == 200
    card = resp.json()["card"]
    assert card["structured_payload"]["commit"] == "not_created"
    assert card["structured_payload"]["pre_commit_validation"]["pytest"] == "focused passed"
