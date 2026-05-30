import os
import sys
from pathlib import Path

from gateway.dev_control.lab_environment import validate_lab_environment
from gateway.dev_control.product_events import DevProductEventStore
from gateway.dev_control.production_signals import DevProductionSignalStore, run_signal_digest_sources
from gateway.dev_control.reliability import DevReliabilityStore
from gateway.subagent_events import SubagentEventStore

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from seed_dev_lab_data import seed_lab_data  # noqa: E402


def test_lab_safety_rejects_production_home_and_ports():
    result = validate_lab_environment(
        hermes_home=Path("~/.hermes/profiles/dev").expanduser(),
        gateway_port=8642,
        repo_roots=[Path("~/Projects/Oryn").expanduser()],
        env={"HERMES_DEV_MERGE_EXECUTOR_ENABLED": "true"},
    )

    assert not result["ok"]
    assert any("~/.hermes" in item for item in result["errors"])
    assert any("reserved" in item for item in result["errors"])
    assert any("production path" in item for item in result["errors"])
    assert any("MERGE_EXECUTOR" in item for item in result["errors"])


def test_lab_safety_accepts_lab_home_and_roots(tmp_path, monkeypatch):
    lab_home = tmp_path / ".oryn-lab"
    monkeypatch.setenv("ORYN_LAB_HOME", str(lab_home))
    result = validate_lab_environment(
        hermes_home=lab_home / "hermes-home",
        gateway_port=8662,
        repo_roots=[lab_home / "repos/Oryn", lab_home / "repos/hermes-agent"],
        env={},
    )

    assert result["ok"]


def test_seed_data_and_digest_write_to_lab_db(tmp_path, monkeypatch):
    lab_home = tmp_path / ".oryn-lab"
    db_path = lab_home / "hermes-home/state.db"
    monkeypatch.setenv("ORYN_LAB_HOME", str(lab_home))
    monkeypatch.setenv("HERMES_HOME", str(lab_home / "hermes-home"))

    seeded = seed_lab_data(db_path)
    assert seeded["outcome_count"] == 3
    assert seeded["product_events_accepted"] == 2

    signal_store = DevProductionSignalStore(db_path)
    event_store = SubagentEventStore(db_path)
    product_store = DevProductEventStore(db_path)
    reliability_store = DevReliabilityStore(db_path)
    digest = run_signal_digest_sources(
        signal_store=signal_store,
        event_store=event_store,
        product_event_store=product_store,
        reliability_store=reliability_store,
        sources=["product", "reliability"],
        window_days=7,
    )

    assert digest["ok"]
    assert digest["summary"]["proposal_count"] >= 1
    assert signal_store.list_proposals(limit=10)
    assert not (Path(os.path.expanduser("~/.hermes/profiles/dev/state.db")) == db_path)
