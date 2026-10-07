"""v4 reliability metrics: hermes.update.run / hermes.update.stage from the final update receipt,
hermes.process.exit from per-process markers and turn watchdogs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hermes_cli import update_receipt
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_process as process_metrics
from hermes_cli.observability import shared_metrics_update as update_metrics


@pytest.fixture
def marks(tmp_path, monkeypatch):
    """Capture decision marks at the runtime boundary; collection on unless a test turns it off."""
    captured: list[tuple[str, dict]] = []
    policy = {"on": True}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": policy["on"]}}},
    )
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: policy["on"])
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: captured.append((mark, data)))
    store = {"saves": True}  # rows recovered from files wait for the store; a test can make it refuse

    def saved(rows):
        if store["saves"]:
            captured.extend(rows)
        return len(rows) if store["saves"] else 0

    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", saved, raising=False)
    yield SimpleNamespace(rows=captured, policy=policy, home=tmp_path / "home", store=store)


def _identity(monkeypatch, *shas: str) -> None:
    committed = int((datetime.now(timezone.utc) - timedelta(days=10)).timestamp())
    sequence = iter(shas)
    last = {"sha": shas[0]}

    def identity(refresh: bool = False) -> dict:
        last["sha"] = next(sequence, last["sha"])
        return {"sha": last["sha"], "short_sha": last["sha"][:8], "version": "0.1.0", "source": "git",
                "commit_date": committed}

    monkeypatch.setattr(update_receipt, "_code_identity", identity)


def _assert_valid(rows: list[tuple[str, dict]]) -> None:
    for mark, data in rows:
        assert contract.counter_dimensions_are_valid(contract._DECISION_MARK_METRICS[mark], data), (mark, data)


def test_failed_update_reports_the_stage_it_never_reached(marks, monkeypatch):
    _identity(monkeypatch, "a" * 40)
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("plan", "success")
    update_receipt.record_stage("snapshot", "skipped")
    update_receipt.record_stage("apply", "success", mode="git")
    # The PM preparation child died: the parent finalizes with no deps mark.
    update_receipt.finalize_pending_update_receipt(1, "completion exited 1")

    runs = [data for mark, data in marks.rows if mark == contract.UPDATE_RUN_MARK]
    stages = [data for mark, data in marks.rows if mark == contract.UPDATE_STAGE_MARK]
    assert runs == [{
        "apply_mode": "git", "duration_bucket": "lt_30s", "failed_stage": "deps",
        "from_version_age_bucket": "7d_to_30d", "kind": "cli", "outcome": "failed", "failure_class": "deps_failed",
    }]
    assert [(s["stage"], s["outcome"]) for s in stages] == [
        ("plan", "success"), ("snapshot", "skipped"), ("apply", "success")]
    _assert_valid(marks.rows)


def test_pre_pull_interpreter_parks_the_receipt_and_the_next_start_reports_it_once(marks, monkeypatch):
    (marks.home / "telemetry" / "shared_metrics").mkdir(parents=True)
    _identity(monkeypatch, "a" * 40, "b" * 40)  # the checkout moved under this interpreter
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("plan", "success")
    update_receipt.finalize_update_receipt("failed")

    assert marks.rows == []  # nothing recorded (or imported) by the pre-pull interpreter
    parked = list(update_metrics.pending_updates_dir(marks.home).glob("*.json"))
    assert len(parked) == 1

    update_metrics.report_pending_updates()
    update_metrics.report_pending_updates()
    assert [mark for mark, _ in marks.rows].count(contract.UPDATE_RUN_MARK) == 1
    assert not parked[0].exists()
    _assert_valid(marks.rows)


def test_disabled_collection_records_no_update_rows(marks, monkeypatch):
    marks.policy["on"] = False
    _identity(monkeypatch, "a" * 40)
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("plan", "success")
    update_receipt.finalize_update_receipt("success")
    update_metrics.record_desktop_update(outcome="success", failed_stage=None, duration_ms=1, mechanism="app-installer")
    assert marks.rows == []


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_dead_markers_report_killed_or_crash_and_live_ones_wait(marks):
    from gateway.status import get_process_start_time

    directory = process_metrics.markers_dir(marks.home)
    directory.mkdir(parents=True)
    dead = _dead_pid()
    killed = directory / f"cli-{dead}.json"
    killed.write_text(json.dumps({"kind": "cli", "pid": dead, "start_time": None, "state": "running"}))
    crashed = directory / "tui-2.json"
    crashed.write_text(json.dumps({
        "kind": "tui", "pid": _dead_pid(), "start_time": None, "state": "crash", "crash_class": "import_error"}))
    parent = os.getppid()
    live = directory / f"gateway-{parent}.json"
    live.write_text(json.dumps({
        "kind": "gateway", "pid": parent, "start_time": get_process_start_time(parent), "state": "running"}))

    process_metrics._report_dead_markers(marks.home, directory / "self.json")
    process_metrics._report_dead_markers(marks.home, directory / "self.json")

    rows = sorted(data["exit_kind"] + ":" + data["crash_class"] + ":" + data["process_kind"] for _, data in marks.rows)
    assert rows == ["crash:import_error:tui", "killed:none:cli"]
    assert live.exists() and not killed.exists() and not crashed.exists()
    _assert_valid(marks.rows)


def test_turn_watchdog_abort_counts_once_per_turn_in_the_turn_profile(marks, tmp_path, monkeypatch):
    agent = SimpleNamespace()
    process_metrics.record_watchdog_turn_abort(agent)  # never armed: no owning profile, no row
    process_metrics.arm_turn(agent)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))  # the watchdog thread's ambient home
    seen: list[str] = []
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", lambda mark, data: (
        marks.rows.append((mark, data)), seen.append(str(__import__("hermes_constants").get_hermes_home()))))
    process_metrics.record_watchdog_turn_abort(agent)
    process_metrics.record_watchdog_turn_abort(agent)  # the second watchdog on the same turn
    assert [data["exit_kind"] for _, data in marks.rows] == ["watchdog"]
    assert seen == [str(marks.home)]
    _assert_valid(marks.rows)


def test_begin_process_marks_then_stamps_clean_and_crash(marks, monkeypatch):
    monkeypatch.setattr(process_metrics, "_STATE", {})
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(process_metrics.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(process_metrics.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None))
    process_metrics.begin_process("cli")
    marker = process_metrics.markers_dir(marks.home) / f"cli-{os.getpid()}.json"
    assert json.loads(marker.read_text())["state"] == "running"
    sys.excepthook(ModuleNotFoundError, ModuleNotFoundError("x"), None)
    process_metrics.stamp_exit("clean")  # atexit after a crash must not hide it
    record = json.loads(marker.read_text())
    assert (record["state"], record["crash_class"]) == ("crash", "import_error")


def test_opted_out_start_purges_markers_left_while_opted_in(marks, monkeypatch):
    from hermes_cli.observability import shared_metrics_setup as setup_metrics

    left = [process_metrics.markers_dir(marks.home) / "cli-1.json", setup_metrics.markers_dir(marks.home) / "tui-1-1.json",
            marks.home / "telemetry" / "shared_metrics" / "desktop_onboarding" / "provider.completed"]  # CLI opt-out
    for marker in left:
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"pid": 1, "started_at": 0, "surface": "tui", "provider": "xai", "state": "running"}))
    monkeypatch.setattr(process_metrics, "_STATE", {})
    marks.policy["on"] = False
    process_metrics.begin_process("cli")
    assert not any(marker.parent.exists() for marker in left)  # nothing reports "abandoned"/"killed" on re-enable


def test_v3_schema_accepts_exactly_the_contract_values():
    from pathlib import Path

    import hermes_cli.observability as observability

    schema = json.loads((Path(observability.__file__).parent / "schemas/hermes.shared_metrics.v4.schema.json").read_text())
    by_name = {d["properties"]["name"]["const"]: d for d in schema["$defs"].values() if "properties" in d}
    for metric in (contract.UPDATE_RUN_METRIC, contract.UPDATE_STAGE_METRIC, contract.PROCESS_EXIT_METRIC):
        dims = by_name[metric]["properties"]["dimensions"]
        dims = (dims["oneOf"][0] if "oneOf" in dims else dims)["properties"]  # current shape first
        assert {field: set(spec["enum"]) for field, spec in dims.items()} == {
            field: set(values) for field, values in contract._COUNTER_DIMENSION_VALUES[metric].items()}


def _pre_pull_finalize(monkeypatch) -> None:
    _identity(monkeypatch, "a" * 40, "b" * 40)  # the checkout moved under this interpreter
    update_receipt.begin_update_receipt()
    update_receipt.record_step("admission", True, "operator text /home/someone/checkout")
    update_receipt.record_stage("plan", "success")
    update_receipt.finalize_update_receipt("failed")


def test_pre_pull_park_happens_only_while_opted_in_and_keeps_only_metric_fields(marks, monkeypatch):
    pending = update_metrics.pending_updates_dir(marks.home)
    pending.mkdir(parents=True)
    (pending / "earlier.json").write_text(json.dumps({"update_id": "earlier", "argv": ["hermes", "update"]}))
    marks.policy["on"] = False
    _pre_pull_finalize(monkeypatch)
    assert not pending.exists() or list(pending.iterdir()) == []  # nothing parked, the earlier one purged

    marks.policy["on"] = True
    _pre_pull_finalize(monkeypatch)
    (parked,) = pending.glob("*.json")
    text = parked.read_text()
    assert "argv" not in text and "operator text" not in text and "/home/" not in text
    update_metrics.report_pending_updates()
    assert [mark for mark, _ in marks.rows].count(contract.UPDATE_RUN_MARK) == 1


def test_one_update_id_is_counted_once_across_the_child_and_the_parked_copy(marks, monkeypatch):
    _identity(monkeypatch, "a" * 40)
    update_receipt.begin_update_receipt()
    update_receipt.record_stage("plan", "success")
    update_receipt.finalize_update_receipt("success")  # the completion child records directly
    receipt = update_receipt.read_latest_receipt()
    pending = update_metrics.pending_updates_dir(marks.home)
    pending.mkdir(parents=True)  # the parent's boundary finalize parks the SAME run as failed
    (pending / f"{receipt['update_id']}.json").write_text(json.dumps({**receipt, "outcome": "failed"}))
    update_metrics.report_pending_updates()
    assert [data["outcome"] for mark, data in marks.rows if mark == contract.UPDATE_RUN_MARK] == ["success"]
    assert list(pending.iterdir()) == []


def test_a_marker_the_store_did_not_save_is_kept_for_the_next_start(marks):
    directory = process_metrics.markers_dir(marks.home)
    directory.mkdir(parents=True)
    marker = directory / "gateway-2.json"
    marker.write_text(json.dumps({"kind": "gateway", "pid": _dead_pid(), "start_time": None, "state": "clean"}))
    marks.store["saves"] = False  # "database is locked"
    process_metrics._report_dead_markers(marks.home, directory / "self.json")
    assert marker.exists() and marks.rows == []
    marks.store["saves"] = True
    process_metrics._report_dead_markers(marks.home, directory / "self.json")
    assert not marker.exists() and [data["exit_kind"] for _, data in marks.rows] == ["clean"]


def test_graceful_gateway_exit_stamps_the_marker_clean(marks, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(process_metrics, "_STATE", {})
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(process_metrics.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(process_metrics.threading, "Thread", lambda **k: SimpleNamespace(start=lambda: None))
    process_metrics.begin_process("gateway")

    class _Exited(Exception):
        pass

    def fake_exit(code):
        raise _Exited(code)

    monkeypatch.setattr(gateway_run.os, "_exit", fake_exit)  # os._exit skips atexit: no stamp from there
    with pytest.raises(_Exited):
        gateway_run._exit_after_graceful_shutdown(1)
    marker = process_metrics.markers_dir(marks.home) / f"gateway-{os.getpid()}.json"
    assert json.loads(marker.read_text())["state"] == "clean"


def test_unrepresentable_timestamps_read_as_unknown():
    assert update_metrics._epoch(10 ** 400) is None
