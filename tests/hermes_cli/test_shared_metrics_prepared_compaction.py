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
import threading
import time
from pathlib import Path

import pytest

from agent import prepared_compaction as pc
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_snapshot import collect_install_snapshot
from tests.agent.test_prepared_compaction import _agent, _compressor, _grow, _messages

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


@pytest.fixture
def marks(monkeypatch):
    """Rows as the real producer emits them, with the profile and thread each was recorded under."""
    from hermes_cli.observability import relay_shared_metrics
    from hermes_constants import get_hermes_home

    rows = []
    lock = threading.Lock()

    def record(mark, data):
        with lock:
            rows.append((mark, dict(data), str(get_hermes_home()), threading.current_thread().name))

    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: True)
    monkeypatch.setattr(relay_shared_metrics, "record_process_mark", record)
    return rows


def _events(rows):
    for mark, data, _home, _thread in rows:
        assert mark == contract.PREPARED_COMPACTION_METRIC
        assert contract.counter_dimensions_are_valid(contract.PREPARED_COMPACTION_METRIC, data), data
    return sorted(
        (d["event"], d["origin"], d["reason"], d["failure_class"], d["wait_bucket"]) for _m, d, _h, _t in rows
    )


def _run_pass(cc, msgs, origin):
    thread = pc.maybe_prepare(_agent(cc), msgs, origin=origin)
    assert thread is not None
    thread.join(5.0)


def test_pass_and_splice_outcomes_are_counted_in_the_owning_profile(tmp_path, marks):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    owner = tmp_path / "owner"
    token = set_hermes_home_override(owner)
    try:
        msgs = _messages(30)
        spliced = _compressor()
        _run_pass(spliced, msgs, "tool-batch")
        spliced.compress(_grow(msgs, 4), current_tokens=spliced.threshold_tokens + 1)

        memory = _compressor()
        _run_pass(memory, msgs, "turn-end")
        memory.compress(msgs, current_tokens=memory.threshold_tokens + 1, memory_context="provider note")

        moved = _compressor()
        _run_pass(moved, msgs, "tool-batch")
        moved._session_id = "S2"
        moved.compress(msgs, current_tokens=moved.threshold_tokens + 1)

        reset = _compressor()
        _run_pass(reset, msgs, "tool-batch")
        reset.on_session_reset()
    finally:
        reset_hermes_home_override(token)

    pass_rows = [("pass_completed", "tool_batch", "none", "none", "none"),
                 ("pass_started", "tool_batch", "none", "none", "none")]
    assert _events(marks) == sorted([
        *pass_rows, ("used", "none", "none", "none", "none"),
        ("pass_completed", "turn_end", "none", "none", "none"), ("pass_started", "turn_end", "none", "none", "none"),
        ("auto_disabled", "none", "provider_context", "none", "none"),
        ("discarded", "none", "provider_context", "none", "none"),
        *pass_rows, ("discarded", "none", "other_session", "none", "none"),
        *pass_rows, ("discarded", "none", "session_boundary", "none", "none"),
    ])
    assert {home for _m, _d, home, _t in marks} == {str(owner)}, "every row lands in the owning profile"
    assert {t for _m, d, _h, t in marks if d["event"] == "pass_completed"} == {"compaction-prepare"}


def test_waste_modes_say_why(monkeypatch, marks):
    from agent.auxiliary_client import aux_stream_deadline
    from agent.conversation_compression_telemetry import _emit_compression_attempt_telemetry
    import agent.context_compressor as module

    msgs = _messages(30)
    # A compaction gives up waiting on a pass that runs past the host's deadline; the pass is told why it stops.
    slow = _compressor()
    slow.gate = threading.Event()
    running = pc.maybe_prepare(_agent(slow), msgs, origin="tool-batch")
    with aux_stream_deadline(time.monotonic() + 0.2):
        slow.compress(msgs, current_tokens=slow.threshold_tokens + 1)
    running.join(5.0)

    # A compaction that spliced the prepared summary and then did not commit.
    spliced = _compressor()
    _run_pass(spliced, msgs, "turn-end")
    spliced.compress(msgs, current_tokens=spliced.threshold_tokens + 1)
    attempt = spliced._last_compression_telemetry
    assert attempt["prepared_summary"] is True
    _emit_compression_attempt_telemetry(
        type("Agent", (), {"context_compressor": spliced, "session_id": "S1"})(), started_at=time.monotonic(),
        commit_status="aborted", split_status="aborted", failure_class="session_split_failed",
        attempt_seed={"attempt_id": attempt["attempt_id"], "session_id": "S1", "trigger_source": "auto"},
    )

    # A session boundary that wakes a waiting compaction is one discard, counted by the boundary.
    woken = _compressor()
    woken.gate = threading.Event()
    before = len(marks)
    running = pc.maybe_prepare(_agent(woken), msgs, origin="tool-batch")
    threading.Timer(0.2, woken.prepared_compaction.discard, args=("session reset",)).start()
    woken.compress(msgs, current_tokens=woken.threshold_tokens + 1)
    woken.gate.set()
    running.join(5.0)
    woken_rows = _events(marks[before:])
    assert [row for row in woken_rows if row[0] == "discarded"] == [
        ("discarded", "none", "session_boundary", "none", "none"),
    ]

    # A pass whose summary call fails with a terminal access error.
    failed = _compressor()
    monkeypatch.setattr(type(failed), "_generate_summary", module.ContextCompressor._generate_summary)
    denied = RuntimeError("invalid api key")
    denied.status_code = 401
    monkeypatch.setattr(module, "call_llm", lambda **_kw: (_ for _ in ()).throw(denied))
    _run_pass(failed, msgs, "tool-batch")

    rows = _events(marks)
    assert ("discarded", "none", "wait_timeout", "none", "lt_1s") in rows
    assert ("pass_cancelled", "tool_batch", "wait_timeout", "none", "none") in rows
    assert ("compaction_failed", "none", "none", "session_split_failed", "none") in rows
    assert ("pass_failed", "tool_batch", "none", "summary_auth_failure", "none") in rows
