"""Phase 1 cron isolation and ownership hardening regressions."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import cron.jobs as jobs
import cron.scheduler_provider as scheduler_provider


def profile_home(root: Path, name: str) -> Path:
    home = root / "profiles" / name
    (home / "cron").mkdir(parents=True)
    return home


def test_dynamic_store_follows_hermes_home_after_import(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("HERMES_HOME", str(first))
    jobs.ensure_dirs()
    jobs.create_job(name="first", prompt="x", schedule="0 1 * * *")
    assert (first / "cron" / "jobs.json").is_file()

    monkeypatch.setenv("HERMES_HOME", str(second))
    jobs.ensure_dirs()
    created = jobs.create_job(name="second", prompt="x", schedule="0 2 * * *")
    assert (second / "cron" / "jobs.json").is_file()
    assert created["owner_profile"] == "default"
    first_names = [j["name"] for j in json.loads((first / "cron" / "jobs.json").read_text())["jobs"]]
    second_names = [j["name"] for j in json.loads((second / "cron" / "jobs.json").read_text())["jobs"]]
    assert first_names == ["first"]
    assert second_names == ["second"]


def test_explicit_compatibility_path_globals_remain_honored(monkeypatch, tmp_path):
    cron_dir = tmp_path / "legacy-cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "custom-jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "custom-output")
    created = jobs.create_job(name="compat", prompt="x", schedule="0 3 * * *")
    assert created["name"] == "compat"
    assert (cron_dir / "custom-jobs.json").is_file()


def test_profile_owner_is_written_and_mismatch_fails_closed(tmp_path):
    alpha = profile_home(tmp_path, "alpha")
    beta = profile_home(tmp_path, "beta")
    with jobs.use_cron_store(alpha, owner_profile="alpha"):
        created = jobs.create_job(name="owned", prompt="x", schedule="0 1 * * *")
        assert created["owner_profile"] == "alpha"

    with pytest.raises(ValueError, match="does not match store path"):
        with jobs.use_cron_store(alpha, owner_profile="beta"):
            pass

    (beta / "cron" / "jobs.json").write_text(json.dumps({"jobs": [created]}))
    with jobs.use_cron_store(beta, owner_profile="beta"):
        with pytest.raises(RuntimeError, match="owner_profile"):
            jobs.load_jobs()


def test_legacy_job_without_owner_remains_readable(tmp_path):
    alpha = profile_home(tmp_path, "alpha")
    legacy = {"id": "legacy", "name": "legacy", "prompt": "x", "schedule": {"kind": "cron", "expr": "0 1 * * *"}}
    (alpha / "cron" / "jobs.json").write_text(json.dumps({"jobs": [legacy]}))
    with jobs.use_cron_store(alpha, owner_profile="alpha"):
        loaded = jobs.load_jobs()
    assert loaded == [legacy]


def test_nested_lock_rejects_different_store(tmp_path):
    alpha = profile_home(tmp_path, "alpha")
    beta = profile_home(tmp_path, "beta")
    with jobs.use_cron_store(alpha, owner_profile="alpha"):
        with jobs._jobs_lock():
            with jobs.use_cron_store(beta, owner_profile="beta"):
                with pytest.raises(RuntimeError, match="different cron store"):
                    with jobs._jobs_lock():
                        pass


def test_windows_lock_contention_times_out_without_entering_critical_section(
    monkeypatch, tmp_path
):
    class ContendedMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(_fd, _mode, _count):
            raise OSError("contended")

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", ContendedMsvcrt)
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0)
    entered = False
    with pytest.raises(TimeoutError):
        with jobs._jobs_lock():
            entered = True
    assert entered is False


def test_ticker_heartbeat_uses_current_store(monkeypatch, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("HERMES_HOME", str(first))
    jobs.record_ticker_heartbeat(success=True)
    monkeypatch.setenv("HERMES_HOME", str(second))
    jobs.record_ticker_heartbeat(success=True)
    assert (first / "cron" / "ticker_heartbeat").is_file()
    assert (first / "cron" / "ticker_last_success").is_file()
    assert (second / "cron" / "ticker_heartbeat").is_file()
    assert (second / "cron" / "ticker_last_success").is_file()


def test_builtin_scheduler_predicate_can_yield_without_ticking(monkeypatch):
    import cron.scheduler as scheduler_module

    calls = []
    monkeypatch.setattr(scheduler_module, "tick", lambda **kwargs: calls.append(kwargs))
    scheduler = scheduler_provider.InProcessCronScheduler()
    scheduler.set_tick_predicate(lambda: False)
    stop = threading.Event()
    thread = threading.Thread(target=scheduler.start, args=(stop,), kwargs={"interval": 0.01}, daemon=True)
    thread.start()
    time.sleep(0.04)
    stop.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert calls == []
