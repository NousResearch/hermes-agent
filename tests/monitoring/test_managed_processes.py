"""Managed-process telemetry: content-free events from wherever they happen, exported by the gateway."""

from __future__ import annotations

import json

import pytest

from agent.monitoring import emitter as emitter_mod
from agent.monitoring import managed_processes as mp


@pytest.fixture
def collected(monkeypatch):
    """An attached exporter: record() emits in-process, and what it emits is returned."""
    bus = emitter_mod.MonitoringEmitter(enabled=False)
    batches: list = []
    bus.subscribe(batches.extend)
    emitter_mod.reset_emitter_for_tests(bus)
    yield lambda: (bus.flush(timeout=2.0), list(batches))[1]
    emitter_mod.reset_emitter_for_tests(None)


@pytest.fixture
def no_exporter():
    """No exporter in this process (a screen's idle check, a dashboard): record() may only spool."""
    emitter_mod.reset_emitter_for_tests(emitter_mod.MonitoringEmitter(enabled=False))
    yield
    emitter_mod.reset_emitter_for_tests(None)


def test_an_exporting_process_emits_directly_and_content_free(collected, tmp_path):
    mp.set_spooling([mp.spool_dir().parents[1]], True)  # a spool exists: emitting must still not also spool
    mp.record(mp.BOT_DESKTOP, "idle_stop", "stopped", actor="screen", reason="abandoned_lease")
    (event,) = collected()
    assert event["event"] == "managed_process"
    assert {k: event[k] for k in ("subsystem", "action", "outcome", "actor", "reason", "count")} == {
        "subsystem": "bot_desktop", "action": "idle_stop", "outcome": "stopped", "actor": "screen",
        "reason": "abandoned_lease", "count": None}
    assert set(event) == {"event", "subsystem", "action", "outcome", "actor", "reason", "count", "ts_ns"}
    assert list(mp.spool_dir().iterdir()) == [], "an exporting process must not spool as well"


def test_without_an_exporter_an_event_is_kept_only_while_a_gateway_spools_for_this_home(no_exporter):
    spool = mp.spool_dir()
    mp.record(mp.OWNED_CHILDREN, "supervisor_exit", "restarted", actor="owner", count=2)
    assert not spool.exists(), "with export off nothing may accumulate"

    mp.set_spooling([spool.parents[1]], True)
    mp.record(mp.OWNED_CHILDREN, "supervisor_exit", "restarted", actor="owner", count=2)
    (event,) = mp.drain_spool([spool.parents[1]])
    assert (event.subsystem, event.action, event.outcome, event.actor, event.count) == (
        "owned_children", "supervisor_exit", "restarted", "owner", 2)
    assert list(spool.iterdir()) == [], "a drained event must not be exported twice"

    mp.set_spooling([spool.parents[1]], False)
    assert not spool.exists()


@pytest.mark.parametrize("bad", [
    {"subsystem": "bot_desktop", "action": "idle_stop", "outcome": "exploded"},
    {"subsystem": "bot_desktop", "action": "rm -rf", "outcome": "stopped"},
    {"subsystem": "elsewhere", "action": "idle_stop", "outcome": "stopped"},
    {"subsystem": "bot_desktop", "action": "idle_stop", "outcome": "stopped", "actor": "/home/alice"},
    {"subsystem": "bot_desktop", "action": "idle_stop", "outcome": "stopped", "reason": "token=abc"},
    {"subsystem": "owned_children", "action": "orphan_reap", "outcome": "killed", "count": -1},
    {"subsystem": "owned_children", "action": "orphan_reap", "outcome": "killed", "count": "3"},
    {"subsystem": "owned_children", "action": "orphan_reap", "outcome": "killed", "count": True},
    {"event": "gateway_health", "subsystem": "bot_desktop", "action": "idle_stop", "outcome": "stopped"},
])
def test_anything_outside_the_vocabulary_is_never_exported(no_exporter, bad):
    """A spool file is written by shell scripts and could be hand-edited: only closed vocabularies and
    small counts may leave the machine, so a path or a secret in any field drops the whole event."""
    home = mp.spool_dir().parents[1]
    mp.set_spooling([home], True)
    (mp.spool_dir() / "1-1.json").write_text(json.dumps(bad), encoding="utf-8")
    (mp.spool_dir() / "2-1.json").write_text("{not json", encoding="utf-8")
    assert mp.drain_spool([home]) == []
    assert list(mp.spool_dir().iterdir()) == [], "a file that can never validate must not be re-read forever"
    if "event" not in bad:  # the same values handed to record() are dropped before anything is written
        mp.record(**bad)
        assert list(mp.spool_dir().iterdir()) == []


def test_a_spool_nobody_drains_stops_growing(no_exporter, monkeypatch):
    monkeypatch.setattr(mp, "_MAX_SPOOLED", 3)
    home = mp.spool_dir().parents[1]
    mp.set_spooling([home], True)
    for _ in range(5):
        mp.record(mp.BOT_DESKTOP, "idle_stop", "kept", actor="watcher", reason="idle")
    assert len(list(mp.spool_dir().glob("*.json"))) == 3


def test_a_shell_written_record_without_a_clock_is_stamped_when_read(no_exporter):
    """The supervisor and the launcher write ts_ns 0 when `date` is unavailable."""
    home = mp.spool_dir().parents[1]
    mp.set_spooling([home], True)
    (mp.spool_dir() / "0-42.json").write_text(
        '{"event":"managed_process","subsystem":"owned_children","action":"orphan_reap","outcome":"terminated",'
        '"actor":"supervisor","count":2,"ts_ns":0}\n', encoding="utf-8")
    (event,) = mp.drain_spool([home])
    assert event.count == 2 and event.ts_ns > 0


def test_screen_and_owned_group_gauges(monkeypatch, tmp_path):
    from hermes_cli import process_identity
    from tools.bot_desktop import lease, runtime

    homes = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    running = {str(homes[0]), str(homes[1])}
    human = {str(homes[1])}
    from hermes_constants import get_hermes_home
    monkeypatch.setattr(runtime, "is_running_locally", lambda: str(get_hermes_home()) in running)
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(
        holder=lease.HUMAN if str(get_hermes_home()) in human else lease.AGENT, viewer_id="v"))
    monkeypatch.setattr(process_identity, "_supervised_pgids", {101, 202})
    metrics = {m.name: m.value for m in mp.build_managed_process_metrics(lambda: homes)}
    assert metrics == {"hermes.bot_desktop.screens.running": 2, "hermes.bot_desktop.screens.human_held": 1,
                       "hermes.owned_children.groups": 2}


def test_the_gateway_exports_what_the_spool_holds_with_only_allowlisted_span_attributes(collected, tmp_path):
    """The path CI covers without the optional OTel SDK: spool -> the gateway's snapshot pass -> emitter ->
    span attributes (the SDK-backed loopback test is in test_gateway_health_export.py)."""
    from agent.monitoring import gateway_health_export, otlp_exporter

    mp.set_spooling([tmp_path], True)
    (mp.spool_dir(tmp_path) / "5-1.json").write_text(json.dumps({
        "event": "managed_process", "subsystem": "bot_desktop", "action": "self_stop_off", "outcome": "sleep_failed",
        "actor": "screen", "ts_ns": 5, "path": "/home/alice/.hermes"}), encoding="utf-8")
    gateway_health_export._emit_snapshot_events({"monitoring": {}}, homes=lambda: [tmp_path])
    spooled = [e for e in collected() if e["event"] == "managed_process"]
    assert len(spooled) == 1 and "path" not in spooled[0]
    assert gateway_health_export._gateway_health_event(spooled[0]), "managed-process events must reach the span exporter"
    assert otlp_exporter._span_attrs(spooled[0]) == {
        "hermes.event": "managed_process", "hermes.subsystem": "bot_desktop", "hermes.action": "self_stop_off",
        "hermes.outcome": "sleep_failed", "hermes.actor": "screen"}
    names = set(gateway_health_export._OBSERVABLE_METRIC_NAMES)
    assert {"hermes.bot_desktop.screens.running", "hermes.bot_desktop.screens.human_held",
            "hermes.owned_children.groups"} <= names, "an unlisted gauge is silently dropped"


def test_a_writer_never_brings_back_a_spool_export_removed(no_exporter, tmp_path):
    """The directory existing is the permission to collect. A writer that passed its check just before the
    export was turned off must drop the event, not recreate the directory (and with it, collection)."""
    gone = tmp_path / "monitoring" / "spool"
    mp._publish(gone, "1-1.json", "{}")
    assert not gone.exists()


def test_one_malformed_record_does_not_cost_the_valid_ones_drained_with_it(no_exporter, tmp_path):
    mp.set_spooling([tmp_path], True)
    good = {"event": "managed_process", "subsystem": "bot_desktop", "action": "idle_stop", "outcome": "kept"}
    spool = mp.spool_dir(tmp_path)
    (spool / "1-1.json").write_text(json.dumps(good), encoding="utf-8")
    (spool / "2-1.json").write_text(json.dumps({**good, "actor": ["screen"]}), encoding="utf-8")  # unhashable
    (spool / "3-1.json").write_text(json.dumps({**good, "outcome": {"a": 1}}), encoding="utf-8")
    (spool / "3-2.json").write_text("[" * 2000 + "]" * 2000, encoding="utf-8")  # deeper than the parser goes
    (spool / "4-1.json").write_text(json.dumps(good), encoding="utf-8")
    assert [e.outcome for e in mp.drain_spool([tmp_path])] == ["kept", "kept"]
    assert list(spool.iterdir()) == []


def test_one_home_listed_twice_is_counted_and_drained_once(no_exporter, monkeypatch, tmp_path):
    """A multiplexer's served set lists the launch profile, which the gateway also passes as its own home."""
    from hermes_cli import process_identity
    from tools.bot_desktop import lease, runtime
    monkeypatch.setattr(runtime, "is_running_locally", lambda: True)
    monkeypatch.setattr(lease, "get", lambda profile_key=None: lease.Lease(holder=lease.HUMAN, viewer_id="v"))
    monkeypatch.setattr(process_identity, "_supervised_pgids", set())
    twice = [tmp_path, tmp_path / "x" / ".."]
    metrics = {m.name: m.value for m in mp.build_managed_process_metrics(lambda: twice)}
    assert (metrics["hermes.bot_desktop.screens.running"], metrics["hermes.bot_desktop.screens.human_held"]) == (1, 1)
    (tmp_path / "x").mkdir()
    mp.set_spooling(twice, True)
    (mp.spool_dir(tmp_path) / "1-1.json").write_text(json.dumps(
        {"subsystem": "bot_desktop", "action": "idle_stop", "outcome": "kept"}), encoding="utf-8")
    assert len(mp.drain_spool(twice)) == 1


def test_the_shell_writers_enforce_the_same_spool_cap():
    """launcher.sh and process_supervisor.sh cannot import the cap: they spell it, and must agree."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    supervisor = (root / "hermes_cli" / "process_supervisor.sh").read_text(encoding="utf-8")
    launcher = (root / "tools" / "bot_desktop" / "launcher.sh").read_text(encoding="utf-8")
    assert re.findall(r'\[ "\$n" -lt (\d+) \]', supervisor) == [str(mp._MAX_SPOOLED)]
    assert re.findall(r"\(\( n < (\d+) \)\)", launcher) == [str(mp._MAX_SPOOLED)]


def test_a_span_carries_the_events_own_time_not_the_exports():
    """A reap spooled while the gateway was down is exported at recovery: it must not look like it happened then."""
    from agent.monitoring import otlp_exporter
    seen = []

    class _Span:
        def end(self, end_time=None):
            seen.append(("end", end_time))

    class _Tracer:
        def start_span(self, name, attributes=None, start_time=None):
            seen.append(("start", start_time))
            return _Span()

    provider = type("P", (), {"get_tracer": lambda self, name: _Tracer()})()
    otlp_exporter.export_batch(provider, [{"event": "managed_process", "subsystem": "owned_children",
                                           "action": "orphan_reap", "outcome": "killed", "ts_ns": 1700000000000000000}])
    assert seen == [("start", 1700000000000000000), ("end", 1700000000000000000)]


def test_a_failing_gauge_reader_is_visible_and_costs_only_its_own_gauges(monkeypatch, caplog):
    import logging

    from tools.bot_desktop import runtime

    def broken():
        raise PermissionError(13, "denied: /home/alice/.hermes/bot-desktop")

    monkeypatch.setattr(runtime, "is_running_locally", broken)
    with caplog.at_level(logging.WARNING, logger=mp.logger.name):
        names = {m.name for m in mp.build_managed_process_metrics(lambda: [mp.spool_dir().parents[1]])}
    assert names == {"hermes.owned_children.groups"}
    assert "error_type=PermissionError" in caplog.text and "/home/alice" not in caplog.text


def test_a_gauge_never_execs_into_a_sandbox(monkeypatch, tmp_path):
    """A sandbox-hosted screen is counted from its host marker: a hung sandbox must not stall collection."""
    from tools.bot_desktop import runtime, sandbox_host

    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: {"container": "c1"})
    monkeypatch.setattr(runtime, "_sandbox_published_env", lambda: (_ for _ in ()).throw(AssertionError("probed")))
    metrics = {m.name: m.value for m in mp.build_managed_process_metrics(lambda: [tmp_path])}
    assert metrics["hermes.bot_desktop.screens.running"] == 1


def test_a_writer_publishing_while_export_off_removes_the_spool_never_keeps_collection_on(no_exporter, monkeypatch,
                                                                                          tmp_path):
    """The deletion takes the spool out of service in one rename first: a writer landing mid-delete can no
    longer leave a directory behind that authorizes collection (rmdir failing with ENOTEMPTY)."""
    import os

    mp.set_spooling([tmp_path], True)
    live = mp.spool_dir(tmp_path)

    def rmtree_with_a_writer_in_the_middle(path, ignore_errors=False):
        for entry in os.listdir(path):
            os.unlink(os.path.join(path, entry))
        mp._publish(live, "1-1.json", "{}")  # a writer that resolved the spool before it went
        try:
            os.rmdir(path)
        except OSError:
            if not ignore_errors:
                raise

    monkeypatch.setattr(mp.shutil, "rmtree", rmtree_with_a_writer_in_the_middle)
    mp.set_spooling([tmp_path], False)
    assert not live.exists()
    mp.record(mp.BOT_DESKTOP, "idle_stop", "kept")
    assert not live.exists()


def test_temp_files_writers_died_holding_count_against_the_cap_and_are_reclaimed(no_exporter, tmp_path):
    import os
    import time

    mp.set_spooling([tmp_path], True)
    spool = mp.spool_dir(tmp_path)
    for i in range(mp._MAX_SPOOLED):
        (spool / f".record-{i}.tmp").write_text("{", encoding="utf-8")
    mp._publish(spool, "1-1.json", "{}")
    assert not (spool / "1-1.json").exists(), "abandoned temp files must count toward the cap"
    old = time.time() - mp._ABANDONED_TMP_S - 5
    for path in spool.iterdir():
        os.utime(path, (old, old))
    (spool / ".record-live.tmp").write_text("{", encoding="utf-8")  # a writer mid-publish right now
    assert mp.drain_spool([tmp_path]) == []
    assert [p.name for p in spool.iterdir()] == [".record-live.tmp"]


@pytest.mark.parametrize("ts", [10**100, 2**63, -1, 0, 5, 1_000_000_000 * 10**9])
def test_a_timestamp_outside_the_window_is_restamped_and_outside_otlp_range_never_reaches_a_span(ts):
    from agent.monitoring import otlp_exporter

    import time
    before = time.time_ns()
    event = mp._validated({"subsystem": "bot_desktop", "action": "idle_stop", "outcome": "kept", "ts_ns": ts})
    assert before <= event.ts_ns <= time.time_ns(), "a time outside the window must be the drain's"
    seen = []

    class _Tracer:
        def start_span(self, name, attributes=None, start_time=None):
            seen.append(start_time)
            return type("S", (), {"end": lambda self, end_time=None: None})()

    provider = type("P", (), {"get_tracer": lambda self, name: _Tracer()})()
    otlp_exporter.export_batch(provider, [{"event": "managed_process", "ts_ns": ts}])
    assert seen == [None if not 0 < ts < 2**63 else ts]


def test_the_host_only_screen_check_never_execs_into_a_sandbox_that_appears_meanwhile(monkeypatch, tmp_path):
    from tools.bot_desktop import runtime, sandbox_host

    reads = iter([None, {"container": "c1"}])  # absent at the first read, present at any later one
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: next(reads, {"container": "c1"}))
    monkeypatch.setattr(runtime, "_sandbox_published_env", lambda: (_ for _ in ()).throw(AssertionError("probed")))
    assert runtime.is_running_locally() is False


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX file modes")
def test_a_record_is_readable_by_a_gateway_draining_as_another_user(no_exporter, tmp_path):
    mp.set_spooling([tmp_path], True)
    mp._publish(mp.spool_dir(tmp_path), "1-1.json", "{}")
    assert (mp.spool_dir(tmp_path) / "1-1.json").stat().st_mode & 0o777 == 0o644


def test_a_failed_chmod_closes_the_descriptor_and_removes_the_temp_file(no_exporter, monkeypatch, tmp_path):
    """A long-lived dashboard failing every publish must not leak a descriptor per record."""
    import os

    mp.set_spooling([tmp_path], True)
    spool = mp.spool_dir(tmp_path)
    opened = []
    real_mkstemp = mp.tempfile.mkstemp
    monkeypatch.setattr(mp.tempfile, "mkstemp", lambda **k: opened.append(real_mkstemp(**k)) or opened[-1])
    monkeypatch.setattr(os, "fchmod", lambda *a: (_ for _ in ()).throw(PermissionError(1, "denied")), raising=False)
    mp._publish(spool, "1-1.json", "{}")
    ((fd, _),) = opened
    with pytest.raises(OSError):
        os.fstat(fd)
    assert list(spool.iterdir()) == []


def test_a_retired_spool_a_writer_kept_alive_is_swept_at_the_next_toggle(tmp_path):
    """A writer that held a descriptor into the spool as it was retired can leave the retired copy behind."""
    retired = mp.spool_dir(tmp_path).with_name("spool.removing-1-1")
    retired.mkdir(parents=True)
    (retired / "1-1.json").write_text("{}", encoding="utf-8")
    mp.set_spooling([tmp_path], False)
    assert not retired.exists()


def test_the_timestamp_window_is_thirty_days_back_and_one_day_ahead():
    import time
    hour, day = 3600 * 10**9, 86_400 * 10**9
    now = time.time_ns()
    keep = [now - hour, now + hour, now - 29 * day]
    drop = [now - 31 * day, now + 2 * day]
    for ts in keep + drop:
        event = mp._validated({"subsystem": "bot_desktop", "action": "idle_stop", "outcome": "kept", "ts_ns": ts})
        assert (event.ts_ns == ts) == (ts in keep), ts


def test_an_oversized_spool_file_costs_only_the_record_limit_in_memory(no_exporter, tmp_path):
    import tracemalloc

    mp.set_spooling([tmp_path], True)
    (mp.spool_dir(tmp_path) / "1-1.json").write_bytes(b" " * (8 * 1024 * 1024))
    tracemalloc.start()
    try:
        assert mp.drain_spool([tmp_path]) == []
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024, f"read {peak} bytes of an oversized record"
    assert list(mp.spool_dir(tmp_path).iterdir()) == []
