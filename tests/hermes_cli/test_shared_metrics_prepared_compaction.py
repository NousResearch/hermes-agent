"""Opt-in shared metrics for prepared compaction (``compression.prepare_ahead``).

* Adoption: the daily snapshot says whether the profile turned the feature on, read as the agent reads it,
  and snapshots counted before the field existed still package against the v4 schema.
* ``hermes.prepared_compaction.count``: every pass and splice outcome lands as one closed-enum row in the
  profile that owns the compressor, including the rows the background pass thread records, and every
  waste mode (a summary nothing uses, a wait that timed out, a failed pass, a spliced compaction that did
  not commit) says why.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_snapshot import collect_install_snapshot

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "hermes_cli" / "observability" / "schemas"
    / "hermes.shared_metrics.v4.schema.json"
)
RESOURCE = {"architecture": "x86_64", "hermes_version": "test", "install_method": "git", "os_family": "linux"}


def test_snapshot_reports_prepare_ahead_and_older_snapshots_still_package(tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    states = {
        value: collect_install_snapshot({"compression": {"prepare_ahead": value}})["compression_prepare_ahead"]
        for value in (True, "yes", "on", False, "false", None)
    }
    assert states == {True: "on", "yes": "on", "on": "on", False: "off", "false": "off", None: "off"}
    current = collect_install_snapshot({})
    assert current["compression_prepare_ahead"] == "off", "absent reads off, as the agent's default"

    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    before_upgrade = {k: v for k, v in current.items() if k != "compression_prepare_ahead"}
    store.record_counter(contract.INSTALL_SNAPSHOT_METRIC, before_upgrade, RESOURCE)
    store.record_counter(contract.INSTALL_SNAPSHOT_METRIC, {**current, "compression_prepare_ahead": "on"}, RESOURCE)
    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(package)
    snapshots = [m["dimensions"] for m in package["metrics"] if m["name"] == contract.INSTALL_SNAPSHOT_METRIC]
    assert sorted(d.get("compression_prepare_ahead", "absent") for d in snapshots) == ["absent", "on"]
