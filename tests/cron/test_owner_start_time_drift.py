"""Start-time fingerprint drift between claim and recovery reads (#117505).

``_owner_is_live()`` used to require the recomputed start-time fingerprint to be exactly
equal to the one recorded at claim time. On hosts where same-host readings drift by ~1 s
(macOS ``kern.boottime`` adjustment), every long-running execution was reconciled to
``unknown`` while its process was still alive and healthy. A mismatch is now treated as
inconclusive within a small tolerance on the epoch-derived psutil scale only — the Linux
``/proc`` tick counter is boot-relative, so a tolerance there would let a cross-reboot
recycled PID pass as the live owner (#117514 review). A recycled PID is still reaped on
every scale, and an unreadable reading is still fail-safe.
"""

from __future__ import annotations

import pytest

import cron.executions as executions_mod


@pytest.fixture(autouse=True)
def _ledger(monkeypatch, tmp_path):
    import gateway.status

    monkeypatch.setattr(executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    # The owner process exists for every row in this module; only the fingerprint decides.
    monkeypatch.setattr(gateway.status, "_pid_exists", lambda pid: True)


@pytest.fixture
def epoch_scale(monkeypatch):
    """Pin the comparison to the epoch-derived psutil scale (macOS-like) regardless of the host
    running the test, so the tolerance cases hold on the Linux CI runners too."""
    monkeypatch.setattr(executions_mod, "_start_time_tolerance_applies", lambda: True)


@pytest.fixture
def boot_scale(monkeypatch):
    """Pin the comparison to a scale without tolerance: the Linux boot-relative ``/proc``
    counter (and Windows)."""
    monkeypatch.setattr(executions_mod, "_start_time_tolerance_applies", lambda: False)


def _seed_external_running(monkeypatch, *, claimed_start_time: int) -> str:
    """Running row owned by a foreign process, with the given claim-time fingerprint."""
    with monkeypatch.context() as m:
        m.setattr(executions_mod, "_PROCESS_ID", "ext-proc")
        m.setattr(executions_mod, "_process_start_time", lambda pid: claimed_start_time)
        record = executions_mod.create_execution("job-drift", source="cron")
        executions_mod.mark_execution_running(record["id"])
    return record["id"]


def _status(execution_id: str) -> str:
    return executions_mod.latest_execution("job-drift")["status"]


@pytest.mark.parametrize(
    "drift,expect_live",
    [
        (0, True),  # exact match: the historical guarantee
        (100, True),  # 1 s drift: the reported macOS boot-time step (#117505)
        (200, True),  # tolerance boundary
        (201, False),  # just outside tolerance: presumed recycled PID
        (3600, False),  # PID-reuse scale
    ],
)
def test_drift_within_tolerance_keeps_execution_running(monkeypatch, epoch_scale, drift, expect_live):
    """Recovery only reaps a claimed/running row when the recomputed fingerprint is far from
    the claim-time one, not merely unequal to it."""
    claimed = 178864182760  # the exact reading from the report
    execution_id = _seed_external_running(monkeypatch, claimed_start_time=claimed)
    monkeypatch.setattr(executions_mod, "_process_start_time", lambda pid: claimed + drift)

    recovered = executions_mod.recover_interrupted_executions()

    if expect_live:
        assert recovered == 0
        assert _status(execution_id) == "running"
    else:
        assert recovered == 1
        assert _status(execution_id) == "unknown"


def test_unreadable_start_time_fail_safe(monkeypatch):
    """A start time that cannot be read now proves nothing about death: the row must survive."""
    execution_id = _seed_external_running(monkeypatch, claimed_start_time=178864182760)
    monkeypatch.setattr(executions_mod, "_process_start_time", lambda pid: None)

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(execution_id) == "running"


def test_boot_relative_scale_requires_exact_match(monkeypatch, boot_scale):
    """The ``/proc`` tick counter restarts near zero on every boot, so a small drift there is a
    cross-reboot recycled PID, not a misread: the row must be reconciled even inside what the
    epoch-scale tolerance would accept (#117514 review)."""
    claimed = 137  # ticks-since-boot from the previous boot's claim
    execution_id = _seed_external_running(monkeypatch, claimed_start_time=claimed)
    monkeypatch.setattr(
        executions_mod, "_process_start_time", lambda pid: claimed + 100  # 1 s "drift"
    )

    assert executions_mod.recover_interrupted_executions() == 1
    assert _status(execution_id) == "unknown"


def test_boot_relative_scale_exact_match_still_live(monkeypatch, boot_scale):
    """Exact equality keeps its historical guarantee on tolerance-free scales: a genuinely
    unchanged owner is still live."""
    claimed = 137
    execution_id = _seed_external_running(monkeypatch, claimed_start_time=claimed)
    monkeypatch.setattr(executions_mod, "_process_start_time", lambda pid: claimed)

    assert executions_mod.recover_interrupted_executions() == 0
    assert _status(execution_id) == "running"
