"""The daily install snapshot says who hosts the install (``hosting``: ``self_hosted`` or ``nous_cloud``).

* ``nous_cloud`` only when the environment says so (``HERMES_HOSTING=nous_cloud``, set by the hosting
  platform); anything else, or nothing, reads ``self_hosted``. The variable's raw value never leaves.
* A snapshot whose group chat counts could not be read still carries ``hosting``.
* Snapshots counted before the field existed still package against the v4 schema.
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


@pytest.fixture
def root_home(tmp_path, monkeypatch):
    home = tmp_path / "root"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_HOSTING", raising=False)
    return home


@pytest.mark.parametrize(("value", "expected"), [
    (None, "self_hosted"),
    ("nous_cloud", "nous_cloud"),
    (" NOUS_CLOUD ", "nous_cloud"),
    ("", "self_hosted"),
    ("acme-private-host", "self_hosted"),
])
def test_snapshot_reads_hosting_from_the_environment(root_home, monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("HERMES_HOSTING", value)

    snapshot = collect_install_snapshot({})

    assert snapshot["hosting"] == expected
    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, snapshot), snapshot


def test_snapshot_without_readable_group_chats_keeps_hosting(root_home, monkeypatch):
    (root_home / "state.db").write_bytes(b"not a sqlite database" * 64)
    monkeypatch.setenv("HERMES_HOSTING", "nous_cloud")

    snapshot = collect_install_snapshot({})

    assert snapshot["hosting"] == "nous_cloud"
    assert not any(k.startswith("group_chat_") for k in snapshot)
    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, snapshot), snapshot


def test_a_row_with_hosting_but_no_other_new_field_is_not_a_valid_shape(root_home):
    # hosting and the group chat counts ship together, so an older row never has hosting alone.
    snapshot = collect_install_snapshot({})
    hosting_only = {k: v for k, v in snapshot.items() if not k.startswith("group_chat_")}
    hosting_only.pop("compression_prepare_ahead")
    assert not contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, hosting_only)


def test_snapshots_with_and_without_hosting_package_against_the_v4_schema(root_home, tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    monkeypatch.setenv("HERMES_HOSTING", "nous_cloud")
    current = collect_install_snapshot({})
    before_upgrade = {k: v for k, v in current.items() if k != "hosting" and not k.startswith("group_chat_")}

    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    store.record_counter(contract.INSTALL_SNAPSHOT_METRIC, current, RESOURCE)
    store.record_counter(contract.INSTALL_SNAPSHOT_METRIC, before_upgrade, RESOURCE)
    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8-sig"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8-sig"))
    jsonschema.Draft202012Validator(schema).validate(package)
    snapshots = [m["dimensions"] for m in package["metrics"] if m["name"] == contract.INSTALL_SNAPSHOT_METRIC]
    assert sorted(d.get("hosting", "absent") for d in snapshots) == ["absent", "nous_cloud"]

    dimensions = schema["$defs"]["install_snapshot_counter"]["properties"]["dimensions"]
    assert "hosting" not in dimensions["required"]
    assert set(dimensions["properties"]["hosting"]["enum"]) == contract.HOSTINGS == {"nous_cloud", "self_hosted"}
