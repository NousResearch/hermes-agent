"""A running update's receipt summary carries its last stage boundary (#137848).

The running record is rewritten at every stage boundary; the action status surfaces that stage so a
remote client bounds a long update by the ABSENCE of progress, never by wall time. A log-quiet but
advancing update (a long dependency install, a silent build) must look alive to the client.
"""

import json
from pathlib import Path

import pytest

import hermes_cli.web_server_gateway as _web_server_gateway

_ACTION_ID = "e" * 32


@pytest.fixture()
def client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _running_receipt(receipt_dir: Path, stages: list[dict]) -> None:
    receipt = {
        "schema": 1, "started_at": "2026-10-25T07:00:00+00:00", "finished_at": None,
        "outcome": "running", "action_id": _ACTION_ID, "pre_update": {"sha": "a" * 40},
        "post_update": {}, "steps": [], "skips": [], "fleet": [], "stages": stages,
    }
    (receipt_dir / "latest.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_running_status_names_the_last_stage_boundary_and_advances(client, tmp_path, monkeypatch):
    import hermes_cli.update_receipt as ur

    receipt_dir = tmp_path / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    monkeypatch.setattr(ur, "_receipt_dir", lambda: receipt_dir)
    monkeypatch.setattr(_web_server_gateway, "_ACTION_LOG_DIR", tmp_path / "actions")
    (tmp_path / "actions").mkdir()
    (tmp_path / "actions" / "hermes-update.log").write_text(
        f"=== hermes-update started 2026-10-25 07:00:00 {_ACTION_ID} ===\n", encoding="utf-8")
    for registry in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
        monkeypatch.setattr(_web_server_gateway, registry, {})

    _running_receipt(receipt_dir, [{"name": "fetch", "outcome": "success", "at": "2026-10-25T07:01:00+00:00"}])
    first = client.get("/api/actions/hermes-update/status").json()

    _running_receipt(receipt_dir, [{"name": "fetch", "outcome": "success", "at": "2026-10-25T07:01:00+00:00"},
                                   {"name": "dependencies", "outcome": "success", "at": "2026-10-25T07:19:00+00:00"}])
    second = client.get("/api/actions/hermes-update/status").json()

    assert first["exit_code"] is None and second["exit_code"] is None
    assert first["lines"] == second["lines"], "the log tail did not move: only the receipt shows progress"
    assert (first["receipt"]["stage"], first["receipt"]["stage_at"]) == ("fetch", "2026-10-25T07:01:00+00:00")
    assert (second["receipt"]["stage"], second["receipt"]["stage_at"]) == ("dependencies", "2026-10-25T07:19:00+00:00")
