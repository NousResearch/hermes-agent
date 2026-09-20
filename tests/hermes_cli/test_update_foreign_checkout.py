"""Separate-checkout gateways are not stale relative to this updater (#117449)."""

from pathlib import Path

from hermes_cli import update_cmd_fleet, update_receipt


def test_fleet_row_classifies_different_checkout_as_external():
    row = update_receipt._fleet_row(
        "pinned", 42, "foreign-sha", "old", "local-sha",
        code_root=Path("/foreign/hermes"), expected_root=Path("/local/hermes"),
    )
    assert row["state"] == "external"
    assert update_receipt.row_is_external(row)


def test_same_checkout_old_sha_remains_stale():
    row = update_receipt._fleet_row(
        "local", 42, "old", "old", "local-sha",
        code_root=Path("/local/hermes"), expected_root=Path("/local/hermes"),
    )
    assert row["state"] == "stale"


def test_relative_paths_cannot_establish_code_root():
    assert update_receipt._code_root_for_path("venv/bin/python") is None


def test_external_row_does_not_report_stale_runtime(monkeypatch):
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: "local-sha")
    receipt = {"fleet": [{"state": "external", "code_sha": "foreign-sha"}]}
    assert not update_cmd_fleet._receipt_reports_stale_runtime(receipt)


def test_external_row_does_not_block_live_coverage(monkeypatch):
    monkeypatch.setattr(update_cmd_fleet, "_current_checkout_sha", lambda: "local-sha")
    monkeypatch.setattr(
        update_receipt, "collect_fleet_versions",
        lambda: [
            {"profile": "local", "state": "current", "code_sha": "local-sha"},
            {"profile": "pinned", "state": "external", "code_sha": "foreign-sha"},
        ],
    )
    receipt = {"plan": {"runtimes": [{"kind": "gateway", "profile": "local"}]}}
    assert update_cmd_fleet._live_fleet_covers_receipt(
        "local-sha", receipt, {("gateway", "local")}
    )
