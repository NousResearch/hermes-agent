"""The lifecycle ledger must NAME THE KILLER on an unclean gateway death.

Measured incident (2026-09-20): the ledger logged

    Previous gateway life (pid=16874, started_at=...) exited UNCLEANLY
    (no exit path ran - SIGKILL / OOM / VM death).
    last_heartbeat_at=... last_mem=None suspected_oom=False

...and stopped there.  The macOS unified log could name the killer exactly
(``exited due to SIGKILL | sent by Python[35502]`` plus the sender's launchd
label ``ai.hermes.gateway-watchdog``), but a human had to go get it - ~40
minutes of manual forensics per incident, and only 3 of 15 unclean exits
since 09-14 were ever attributed.

These tests drive the bounded, fail-open attribution probe:
(a) it parses the REAL unified-log lines into killer/sender/sender_label;
(b) the ledger log line + persisted record carry the attribution, and say
    ``killer=unattributed`` with a reason when the probe times out or raises;
(c) the probe runs OFF the event loop thread (boot must never block).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

import pytest

from gateway import lifecycle_ledger
from gateway.lifecycle_ledger import (
    attribute_unclean_exit,
    get_lifecycle_sentinel_path,
    parse_launchd_exit_record,
    parse_launchd_spawn_label,
    record_startup,
    record_startup_async,
)

_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max; never alive

# The two verbatim lines from the 2026-09-20 incident.
_REAL_EXIT_LINES = (
    "2026-09-20 08:34:52.101946-0700 0x1f2b  Default  0x0  1  0  launchd: "
    "(ai.hermes.gateway [16874]) Service exited due to SIGKILL | sent by "
    "Python[35502], ran for 9086738ms\n"
    "2026-09-20 08:34:52.100011-0700 0x1f2b  Default  0x0  1  0  launchd: "
    "(ai.hermes.gateway [16874]) Service bootout initiated by: "
    "launchctl[36598]<-Python[35502]\n"
)
_REAL_SPAWN_LINES = (
    "2026-09-20 08:34:51.880233-0700 0x1f2b  Default  0x0  1  0  launchd: "
    "(ai.hermes.gateway-watchdog [35502]) Successfully spawned python3[35502] "
    "because interval\n"
)


def _write_sentinel(home: Path, payload: dict) -> Path:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_sentinel(home: Path) -> dict:
    return json.loads(get_lifecycle_sentinel_path(home).read_text(encoding="utf-8"))


def _unclean_sentinel(home: Path) -> None:
    _write_sentinel(home, {
        "phase": "running",
        "pid": _DEAD_PID,
        "start_time": 1000.0,
        "started_at": "2026-09-20T06:03:14+00:00",
    })


def _exit_diag_records(home: Path) -> list:
    path = home / "logs" / "gateway-exit-diag.log"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# (a) Parsing the REAL unified-log lines
# ---------------------------------------------------------------------------


def test_parses_real_launchd_exit_record_into_killer_and_sender() -> None:
    parsed = parse_launchd_exit_record(_REAL_EXIT_LINES, 16874)
    assert parsed["killer"] == "SIGKILL"
    assert parsed["sender"] == "Python[35502]"
    assert parsed["sender_pid"] == 35502


def test_parses_sender_from_bootout_line_when_no_signal_line() -> None:
    bootout_only = _REAL_EXIT_LINES.splitlines(keepends=True)[1]
    parsed = parse_launchd_exit_record(bootout_only, 16874)
    assert parsed["sender"] == "Python[35502]"
    assert parsed["sender_pid"] == 35502


def test_parses_real_spawn_line_into_sender_label() -> None:
    assert parse_launchd_spawn_label(_REAL_SPAWN_LINES, 35502) == "ai.hermes.gateway-watchdog"


def test_parsers_are_fail_open_on_unrelated_text() -> None:
    assert parse_launchd_exit_record("nothing to see here", 16874) == {}
    assert parse_launchd_spawn_label("nothing to see here", 35502) is None


def test_attribute_unclean_exit_end_to_end_against_real_log_output(monkeypatch) -> None:
    """The full probe, with the real `log show` output injected at the
    subprocess seam - no live system query, but the real parse + chain."""
    monkeypatch.setattr(lifecycle_ledger.sys, "platform", "darwin")
    calls = []

    def fake_run(argv, timeout):
        calls.append(argv)
        joined = " ".join(argv)
        return _REAL_SPAWN_LINES if "35502" in joined and "16874" not in joined else _REAL_EXIT_LINES

    monkeypatch.setattr(lifecycle_ledger, "_run_log_command", fake_run)

    out = attribute_unclean_exit(16874)
    assert out["killer"] == "SIGKILL"
    assert out["sender"] == "Python[35502]"
    assert out["sender_label"] == "ai.hermes.gateway-watchdog"
    assert len(calls) == 2  # exit record, then the sender's spawn line


def test_attribute_unclean_exit_is_unattributed_on_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_ledger.sys, "platform", "sunos5")
    out = attribute_unclean_exit(16874)
    assert out["killer"] == "unattributed"
    assert out["reason"]


def test_attribute_unclean_exit_never_raises_and_reports_reason(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle_ledger.sys, "platform", "darwin")

    def boom(argv, timeout):
        raise RuntimeError("log show exploded")

    monkeypatch.setattr(lifecycle_ledger, "_run_log_command", boom)
    out = attribute_unclean_exit(16874)
    assert out["killer"] == "unattributed"
    assert "log show exploded" in out["reason"] or "probe_failed" in out["reason"]


def test_probe_is_bounded(monkeypatch) -> None:
    """Every subprocess call must carry a timeout inside the overall bound."""
    monkeypatch.setattr(lifecycle_ledger.sys, "platform", "darwin")
    timeouts = []

    def fake_run(argv, timeout):
        timeouts.append(timeout)
        return _REAL_EXIT_LINES

    monkeypatch.setattr(lifecycle_ledger, "_run_log_command", fake_run)
    attribute_unclean_exit(16874)
    assert timeouts and all(0 < t <= lifecycle_ledger.KILL_ATTRIBUTION_TIMEOUT_S for t in timeouts)
    assert sum(timeouts) <= lifecycle_ledger.KILL_ATTRIBUTION_TIMEOUT_S + 0.001


# ---------------------------------------------------------------------------
# (b) The ledger line + record carry the attribution
# ---------------------------------------------------------------------------


def test_ledger_line_and_record_name_the_killer(tmp_path: Path, monkeypatch, caplog) -> None:
    _unclean_sentinel(tmp_path)
    monkeypatch.setattr(
        lifecycle_ledger,
        "attribute_unclean_exit",
        lambda pid, **kw: {
            "killer": "SIGKILL",
            "sender": "Python[35502]",
            "sender_label": "ai.hermes.gateway-watchdog",
        },
    )

    with caplog.at_level(logging.WARNING, logger="gateway.lifecycle_ledger"):
        evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["killer"] == "SIGKILL"
    assert evidence["kill_sender"] == "Python[35502]"
    assert evidence["kill_sender_label"] == "ai.hermes.gateway-watchdog"

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "killer=SIGKILL" in line
    assert "sender=Python[35502]" in line
    assert "sender_label=ai.hermes.gateway-watchdog" in line

    record = _exit_diag_records(tmp_path)[0]
    assert record["killer"] == "SIGKILL"
    assert record["kill_sender_label"] == "ai.hermes.gateway-watchdog"

    # Persisted on the new sentinel so `hermes gateway status`/doctor can show it.
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["prior_unclean_exit"] is True
    assert sentinel["prior_killer"] == "SIGKILL"
    assert sentinel["prior_kill_sender_label"] == "ai.hermes.gateway-watchdog"


def test_ledger_line_says_unattributed_when_probe_times_out(tmp_path: Path, monkeypatch, caplog) -> None:
    _unclean_sentinel(tmp_path)
    monkeypatch.setattr(
        lifecycle_ledger,
        "attribute_unclean_exit",
        lambda pid, **kw: {"killer": "unattributed", "reason": "timeout"},
    )

    with caplog.at_level(logging.WARNING, logger="gateway.lifecycle_ledger"):
        evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert evidence["killer"] == "unattributed"
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "killer=unattributed" in line
    assert "reason=timeout" in line


def test_probe_exception_never_breaks_the_ledger(tmp_path: Path, monkeypatch, caplog) -> None:
    _unclean_sentinel(tmp_path)

    def boom(pid, **kw):
        raise RuntimeError("probe blew up")

    monkeypatch.setattr(lifecycle_ledger, "attribute_unclean_exit", boom)

    with caplog.at_level(logging.WARNING, logger="gateway.lifecycle_ledger"):
        evidence = record_startup(home=tmp_path)

    assert evidence is not None  # the unclean finding still lands
    assert evidence.get("killer") == "unattributed"
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["prior_unclean_exit"] is True


def test_clean_boot_does_not_run_the_probe(tmp_path: Path, monkeypatch) -> None:
    _write_sentinel(tmp_path, {"phase": "exited", "pid": _DEAD_PID, "exit_code": 0})
    called = []
    monkeypatch.setattr(
        lifecycle_ledger, "attribute_unclean_exit",
        lambda pid, **kw: called.append(pid) or {},
    )
    assert record_startup(home=tmp_path) is None
    assert called == []


# ---------------------------------------------------------------------------
# (c) The probe runs OFF the event loop thread
# ---------------------------------------------------------------------------


def test_record_startup_async_runs_the_probe_off_the_loop_thread(tmp_path: Path, monkeypatch) -> None:
    _unclean_sentinel(tmp_path)
    seen = {}

    def probe(pid, **kw):
        seen["probe_thread"] = threading.current_thread()
        return {"killer": "SIGKILL", "sender": "Python[35502]", "sender_label": "lbl"}

    monkeypatch.setattr(lifecycle_ledger, "attribute_unclean_exit", probe)

    async def main():
        seen["loop_thread"] = threading.current_thread()
        return await record_startup_async(home=tmp_path)

    evidence = asyncio.run(main())
    assert evidence is not None and evidence["killer"] == "SIGKILL"
    assert seen["probe_thread"] is not seen["loop_thread"]


def test_record_startup_async_uses_to_thread(tmp_path: Path, monkeypatch) -> None:
    """Belt-and-braces: the offload goes through asyncio.to_thread, the house
    pattern (fork #759), not a bespoke executor."""
    _unclean_sentinel(tmp_path)
    monkeypatch.setattr(lifecycle_ledger, "attribute_unclean_exit", lambda pid, **kw: {})
    used = []
    real_to_thread = asyncio.to_thread

    async def spy(fn, *a, **kw):
        used.append(fn)
        return await real_to_thread(fn, *a, **kw)

    monkeypatch.setattr(lifecycle_ledger.asyncio, "to_thread", spy)

    async def main():
        return await record_startup_async(home=tmp_path)

    asyncio.run(main())
    assert used, "record_startup_async must offload via asyncio.to_thread"


# ---------------------------------------------------------------------------
# The attribution reaches the status surface
# ---------------------------------------------------------------------------


def test_memory_status_surfaces_the_killer(tmp_path: Path, monkeypatch) -> None:
    """`hermes gateway status` / doctor read this dict — the killer must be
    visible there without re-running the forensics."""
    from gateway import memory_status

    _unclean_sentinel(tmp_path)
    monkeypatch.setattr(
        lifecycle_ledger,
        "attribute_unclean_exit",
        lambda pid, **kw: {
            "killer": "SIGKILL",
            "sender": "Python[35502]",
            "sender_label": "ai.hermes.gateway-watchdog",
        },
    )
    record_startup(home=tmp_path)

    status = memory_status.collect_memory_status(home=tmp_path)
    assert status["last_boot_unclean"] is True
    assert status["last_boot_killer"] == "SIGKILL"
    assert status["last_boot_kill_sender"] == "Python[35502]"
    assert status["last_boot_kill_sender_label"] == "ai.hermes.gateway-watchdog"
