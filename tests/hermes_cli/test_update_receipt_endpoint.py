"""Phase-1 bullet 3 (#91277): the dashboard/Desktop READ the update receipt.

The update receipt (written by every `hermes update` run since #91283,
`latest.json` pointer) is the durable outcome record. These tests pin:

- GET /api/hermes/update/receipt returns the full receipt + summary; 404
  when none exists.
- GET /api/actions/hermes-update/status attaches the receipt summary, and
  uses a finished receipt as the outcome when both in-memory registries AND
  the log marker are gone (dashboard restarted + log rotated).
"""

import json
from pathlib import Path

import pytest

import hermes_cli.web_server as web_server


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


def _write_receipt(tmp_path: Path, monkeypatch, *, outcome="success") -> dict:
    receipt = {
        "schema": 1,
        "started_at": "2026-08-23T07:00:00+00:00",
        "finished_at": "2026-08-23T07:03:20+00:00",
        "argv": ["hermes", "update"],
        "pid": 12345,
        "outcome": outcome,
        "pre_update": {"sha": "a" * 40, "version": "0.20.4"},
        "post_update": {"sha": "b" * 40, "version": "0.20.5"},
        "steps": [{"name": "pre_update_backup", "ok": True, "detail": "", "at": ""}],
        "skips": [],
        "gateway_restart": {},
        "fleet": [
            {"profile": "default", "pid": 1, "code_sha": "b" * 40, "state": "current"}
        ],
    }
    receipt_dir = tmp_path / "logs" / "update_receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "latest.json").write_text(json.dumps(receipt), encoding="utf-8")
    import hermes_cli.update_receipt as ur

    monkeypatch.setattr(ur, "_receipt_dir", lambda: receipt_dir)
    return receipt


class TestUpdateReceiptEndpoint:

    def test_receipt_endpoint_returns_full_receipt_and_summary(self, client, tmp_path, monkeypatch):
        receipt = _write_receipt(tmp_path, monkeypatch)

        resp = client.get("/api/hermes/update/receipt")

        assert resp.status_code == 200
        data = resp.json()
        assert data["receipt"]["outcome"] == "success"
        assert data["receipt"]["steps"] == receipt["steps"]
        summary = data["summary"]
        assert summary["outcome"] == "success"
        assert summary["pre_sha"] == "a" * 40
        assert summary["post_sha"] == "b" * 40
        assert summary["post_version"] == "0.20.5"
        assert summary["fleet_states"] == ["current"]

    def test_receipt_endpoint_404_when_no_receipt(self, client, tmp_path, monkeypatch):
        import hermes_cli.update_receipt as ur

        monkeypatch.setattr(ur, "_receipt_dir", lambda: tmp_path / "none")

        resp = client.get("/api/hermes/update/receipt")

        assert resp.status_code == 404


class TestUpdateStatusReadsReceipt:

    def _clear_registries(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_server, "_ACTION_LOG_DIR", tmp_path / "actions")
        (tmp_path / "actions").mkdir(exist_ok=True)
        monkeypatch.setattr(web_server, "_ACTION_PROCS", {})
        monkeypatch.setattr(web_server, "_ACTION_RESULTS", {})
        monkeypatch.setattr(web_server, "_ACTION_COMMANDS", {})
        monkeypatch.setattr(web_server, "_ACTION_IDS", {})

    def test_status_attaches_receipt_summary(self, client, tmp_path, monkeypatch):
        _write_receipt(tmp_path, monkeypatch)
        self._clear_registries(monkeypatch, tmp_path)

        resp = client.get("/api/actions/hermes-update/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["receipt"]["outcome"] == "success"

    def test_finished_receipt_is_the_outcome_when_marker_and_memory_gone(self, client, tmp_path, monkeypatch):
        """Dashboard restarted (registries lost) AND update.log has no
        completion marker (rotated): the receipt alone reports success."""
        _write_receipt(tmp_path, monkeypatch, outcome="success")
        self._clear_registries(monkeypatch, tmp_path)

        resp = client.get("/api/actions/hermes-update/status")

        data = resp.json()
        assert data["running"] is False
        assert data["exit_code"] == 0

    def test_partial_receipt_maps_to_exit_1(self, client, tmp_path, monkeypatch):
        _write_receipt(tmp_path, monkeypatch, outcome="partial")
        self._clear_registries(monkeypatch, tmp_path)

        resp = client.get("/api/actions/hermes-update/status")

        assert resp.json()["exit_code"] == 1

    def test_running_receipt_proves_nothing(self, client, tmp_path, monkeypatch):
        """An unfinished receipt (crashed run / mid-update) must not report
        an outcome — clients keep polling."""
        receipt = _write_receipt(tmp_path, monkeypatch, outcome="running")
        self._clear_registries(monkeypatch, tmp_path)

        resp = client.get("/api/actions/hermes-update/status")

        assert resp.json()["exit_code"] is None

    def test_non_update_actions_untouched(self, client, tmp_path, monkeypatch):
        _write_receipt(tmp_path, monkeypatch)
        self._clear_registries(monkeypatch, tmp_path)

        resp = client.get("/api/actions/gateway-restart/status")

        data = resp.json()
        assert "receipt" not in data
