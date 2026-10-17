"""macOS clock corrections must not turn a live cron owner into a reused PID."""
import os
import subprocess
import sys

import pytest


@pytest.mark.platforms("macos")
def test_process_fingerprint_is_stable_across_observers_and_clock_correction(monkeypatch):
    import psutil
    import psutil._psosx as osx
    from gateway.status import get_process_start_time, _start_times_agree

    pid = os.getpid()
    monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.boot_time())
    # Keep the persisted centisecond representation: a hash changes every
    # identity, can be negative, and exceeds lossless JSON/float precision.
    legacy = int(round(psutil.Process(pid).create_time() * 100))
    before = get_process_start_time(pid)
    assert before == legacy
    assert _start_times_agree(before, legacy)
    # psutil keeps this boot timestamp per interpreter. An older desktop and a
    # newer gateway can therefore disagree after NTP, despite the same live PID.
    monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.boot_time() + 1)
    assert get_process_start_time(pid) == before
    code = (
        "import sys,psutil._psosx as osx; "
        "from gateway.status import get_process_start_time; "
        "osx.INIT_BOOT_TIME=osx.boot_time()-2; "
        "print(get_process_start_time(int(sys.argv[1])))"
    )
    observed = subprocess.check_output([sys.executable, "-c", code, str(pid)], text=True, encoding="utf-8")
    assert int(observed) == before


@pytest.mark.platforms("macos")
def test_clock_corrected_observer_does_not_reap_live_producer(monkeypatch, tmp_path):
    import cron.executions as ledger
    import psutil._psosx as osx

    monkeypatch.setattr(ledger, "EXECUTIONS_FILE", tmp_path / "executions.db")
    attempt = ledger.create_execution("cron-producer", source="builtin")
    assert ledger.mark_execution_running(attempt["id"]) is not None
    owner = ledger._PROCESS_ID
    monkeypatch.setattr(ledger, "_PROCESS_ID", "desktop-observer")
    # 5 s exceeds START_TIME_DRIFT_TOLERANCE (2 s, #117505); tolerance alone would reap the owner.
    monkeypatch.setattr(osx, "INIT_BOOT_TIME", osx.boot_time() + 5)
    assert ledger.recover_interrupted_executions() == 0
    monkeypatch.setattr(ledger, "_PROCESS_ID", owner)
    result = ledger.finish_execution(attempt["id"], success=True)
    assert result is not None
    assert result["status"] == "completed"
