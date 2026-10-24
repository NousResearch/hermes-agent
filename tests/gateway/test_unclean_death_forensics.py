"""Unclean gateway death forensics — distinguish host-OOM from external SIGKILL.

Kanban t_1915c9a3, finding C1(a).

The gateway lifecycle ledger could only say "the previous life exited UNCLEANLY". That
is true but not actionable: it cannot tell anyone WHY. On 2026-10-23 the gateway pid
11533 stopped heartbeating at 22:18:34 and never came back. The leading hypothesis was a
host OOM under exhausted swap.

Reconstructing it from the box afterwards showed the hypothesis was WRONG: the kernel
journal for the whole incident window (21:00–01:00) contains no OOM record at all, while
the same box DID produce cgroup OOM kills at 01:53 and 02:08 — so the kernel was capable
of writing those records and simply did not kill anything at 22:18. The death was
external, and nothing in the gateway could have said so at the time.

These tests pin the classification: the verdict must be a READING OF EVIDENCE, and the
evidence must be real. Two properties matter most:

* a VERDICT is always produced, and it is never a guess — with no evidence at all the
  verdict is ``indeterminate``, not ``external_kill``;
* the verdict distinguishes "the kernel OOM-killed THIS process" from "the host was under
  pressure but the victim was something else" from "nothing suggests an OOM at all".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gateway import unclean_death_forensics as f  # noqa: E402


# --- the verdict logic ------------------------------------------------------------


def test_a_kernel_record_naming_the_gateway_is_host_oom(monkeypatch):
    """The kernel named this pid as its victim: that is a host OOM, conclusively."""
    monkeypatch.setattr(f, "read_cgroup_memory_events", lambda: {"oom_kill": 0})
    monkeypatch.setattr(f, "read_kernel_oom_records", lambda **kw: [
        {"victim_pid": 11533, "names_gateway": True, "line": "Killed process 11533 (hermes)"},
    ])
    monkeypatch.setattr(f, "read_heartbeat_history", lambda *a, **k: [])

    evidence = f.classify_unclean_death(prior_pid=11533)
    assert evidence["verdict"] == "host_oom"
    assert "OOM-killed" in f.format_verdict(evidence)


def test_a_nonzero_cgroup_oom_kill_counter_is_host_oom(monkeypatch):
    """The kernel's own cgroup counter says it killed something here."""
    monkeypatch.setattr(f, "read_cgroup_memory_events", lambda: {"oom_kill": 3, "oom": 2})
    monkeypatch.setattr(f, "read_kernel_oom_records", lambda **kw: [])
    monkeypatch.setattr(f, "read_heartbeat_history", lambda *a, **k: [])

    evidence = f.classify_unclean_death(prior_pid=11533)
    assert evidence["verdict"] == "host_oom"
    assert evidence["cgroup_memory_events"]["oom_kill"] == 3


def test_exhausted_swap_with_a_sibling_oom_is_memory_pressure(monkeypatch):
    """The host was hostile, but the gateway was NOT the victim.

    This is the case the 2026-10-23 incident actually was, and the one the old ledger
    flattened into "exited UNCLEANLY". The verdict must say "the host was under pressure
    but the killer is still unidentified" — NOT "host OOM", because that would name the
    gateway as the victim and send the investigation in the wrong direction.
    """
    monkeypatch.setattr(f, "read_cgroup_memory_events", lambda: {"oom_kill": 0})
    monkeypatch.setattr(f, "read_kernel_oom_records", lambda **kw: [
        {"victim_pid": 7731, "names_gateway": False, "line": "Killed process 7731 (hermes)"},
    ])
    monkeypatch.setattr(f, "read_heartbeat_history", lambda *a, **k: [])

    evidence = f.classify_unclean_death(
        prior_pid=11533,
        last_heartbeat_mem={"rss_kib": 665076, "swap_used_kib": 3800424},
    )
    assert evidence["verdict"] == "host_memory_pressure"
    summary = f.format_verdict(evidence)
    assert "still unidentified" in summary


def test_a_healthy_host_with_no_oom_evidence_is_an_external_kill(monkeypatch):
    """No OOM record, zero cgroup kills: something outside the gateway sent the signal.

    This requires POSITIVE evidence — a readable cgroup counter that says zero and a
    readable kernel log with no matching record. Absence of evidence alone is not
    enough; see the indeterminate test below.
    """
    monkeypatch.setattr(f, "read_cgroup_memory_events", lambda: {"oom_kill": 0, "oom": 0})
    monkeypatch.setattr(f, "read_kernel_oom_records", lambda **kw: [])
    monkeypatch.setattr(f, "read_heartbeat_history", lambda *a, **k: [])
    monkeypatch.setattr(f, "_swap_total_kib", lambda: 4194304)

    evidence = f.classify_unclean_death(
        prior_pid=11533,
        last_heartbeat_mem={"rss_kib": 665076, "swap_used_kib": 1024},
    )
    assert evidence["verdict"] == "external_kill"
    assert "from outside" in f.format_verdict(evidence)


def test_no_evidence_at_all_is_indeterminate_never_a_guess(monkeypatch):
    """Without the evidence to decide, the verdict is indeterminate.

    This is the property that matters most: the old ledger's failure was that silence
    was mistaken for a verdict. Claiming a cause we cannot support is worse than
    admitting we do not know.
    """
    monkeypatch.setattr(f, "read_cgroup_memory_events", lambda: {})
    monkeypatch.setattr(f, "read_kernel_oom_records", lambda **kw: [])
    monkeypatch.setattr(f, "read_heartbeat_history", lambda *a, **k: [])
    monkeypatch.setattr(f, "_swap_total_kib", lambda: None)

    evidence = f.classify_unclean_death(prior_pid=11533)
    assert evidence["verdict"] == "indeterminate"
    assert "not available" in f.format_verdict(evidence)


def test_the_verdict_is_never_absent(monkeypatch):
    """Whatever the evidence, there is always a verdict string."""
    monkeypatch.setattr(f, "read_cgroup_memory_events", lambda: {})
    monkeypatch.setattr(f, "read_kernel_oom_records", lambda **kw: [])
    monkeypatch.setattr(f, "read_heartbeat_history", lambda *a, **k: [])
    assert f.classify_unclean_death(prior_pid=1)["verdict"]


# --- the evidence collection ------------------------------------------------------


def test_the_cgroup_oom_kill_counter_is_read_from_the_gateway_unit(monkeypatch, tmp_path):
    """The counter comes from the unit's OWN cgroup, not the root cgroup.

    Reading the root cgroup would report zero kills whenever the unit's own cgroup was
    the one that OOMed — exactly inverting the conclusion.
    """
    fake_cgroup = tmp_path / "fake.service"
    fake_cgroup.mkdir()
    (fake_cgroup / "memory.events").write_text("low 0\nhigh 12\nmax 0\noom 0\noom_kill 4\n")
    monkeypatch.setattr(f, "Path", Path)
    monkeypatch.setattr(f, "_gateway_control_group", lambda: "/fake.service")

    # Patch the cgroup root to the tmp dir.
    original_read_text = f._read_text

    def _fake_read_text(path, limit=64 * 1024):
        if str(path).endswith("memory.events"):
            return (fake_cgroup / "memory.events").read_text()
        return original_read_text(path, limit)

    monkeypatch.setattr(f, "_read_text", _fake_read_text)
    events = f.read_cgroup_memory_events()
    assert events["oom_kill"] == 4
    assert events["high"] == 12


def test_a_missing_cgroup_returns_an_empty_dict_not_a_false_zero(monkeypatch):
    """No cgroup-v2 must not look like "zero OOM kills".

    An empty dict means "cannot read the counter"; a dict with oom_kill=0 means "the
    counter says zero". Conflating them would let a permissions problem produce a
    confident "external_kill" verdict.
    """
    monkeypatch.setattr(f, "_gateway_control_group", lambda: None)
    assert f.read_cgroup_memory_events() == {}


def test_kernel_oom_records_are_parsed_for_the_victim(monkeypatch):
    """The kernel's OOM line is parsed into pid / cgroup / constraint."""
    monkeypatch.setattr(f, "_kernel_oom_sources", lambda since_iso: [
        "kernel: oom-kill:constraint=CONSTRAINT_MEMCG,oom_memcg=/user.slice/x.scope,task=hermes,pid=7731,uid=1000\n"
        "kernel: Memory cgroup out of memory: Killed process 7731 (hermes) total-vm:2929268kB, anon-rss:139408kB\n"
        "kernel: some unrelated line\n",
    ])
    records = f.read_kernel_oom_records(pid=7731)
    assert len(records) == 2
    assert any(r.get("victim_pid") == 7731 for r in records)
    assert any(r.get("names_gateway") for r in records)
    assert any(r.get("constraint") == "CONSTRAINT_MEMCG" for r in records)
    assert any(r.get("oom_memcg") == "/user.slice/x.scope" for r in records)


def test_kernel_oom_records_ignore_unrelated_lines(monkeypatch):
    monkeypatch.setattr(f, "_kernel_oom_sources", lambda since_iso: [
        "kernel: hermes: all is well\nkernel: nothing to see here\n",
    ])
    assert f.read_kernel_oom_records() == []


def test_the_heartbeat_history_is_read_from_the_history_file(monkeypatch, tmp_path):
    """The history file is the primary source: it survives the death."""
    import gateway.shutdown_watchdog as sw

    history = tmp_path / "gateway.heartbeat.history"
    history.write_text(
        json.dumps({"pid": 1, "mem": {"rss_kib": 100}}) + "\n"
        + json.dumps({"pid": 1, "mem": {"rss_kib": 200}}) + "\n"
    )
    monkeypatch.setattr(sw, "get_loop_heartbeat_history_path", lambda home=None: history)

    got = f.read_heartbeat_history()
    assert len(got) == 2
    assert got[0]["mem"]["rss_kib"] == 100
    assert got[1]["mem"]["rss_kib"] == 200


def test_the_memory_trend_distinguishes_a_climb_from_a_plateau(monkeypatch):
    """The trend is what separates "the gateway ate the host's memory" from "flat"."""
    rising = [
        {"mem": {"rss_kib": 100}},
        {"mem": {"rss_kib": 200}},
        {"mem": {"rss_kib": 400}},
    ]
    trend = f._memory_trend(rising)
    assert trend["trend"] == "rising"
    assert trend["rss_kib_delta"] == 300
    assert trend["samples"] == 3

    flat = [{"mem": {"rss_kib": 400}}, {"mem": {"rss_kib": 400}}]
    assert f._memory_trend(flat)["trend"] == "flat_or_falling"

    # A single sample is not a trend.
    assert f._memory_trend([{"mem": {"rss_kib": 400}}]) == {}


def test_the_history_file_is_bounded(monkeypatch, tmp_path):
    """The heartbeat history must not grow without bound."""
    import gateway.shutdown_watchdog as sw
    history = tmp_path / "gateway.heartbeat.history"
    monkeypatch.setattr(sw, "get_loop_heartbeat_history_path", lambda home=None: history)
    for i in range(25):
        sw._append_heartbeat_history({"pid": 1, "mem": {"rss_kib": i}}, home=tmp_path)
    lines = [ln for ln in history.read_text().splitlines() if ln.strip()]
    assert len(lines) == sw._HEARTBEAT_HISTORY_LIMIT
    # The LAST record is the newest.
    assert json.loads(lines[-1])["mem"]["rss_kib"] == 24
