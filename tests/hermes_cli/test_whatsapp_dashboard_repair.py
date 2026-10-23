"""Dashboard recovery must bypass revoked credentials without losing them."""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from subprocess import Popen
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException

from hermes_cli import web_models, web_server_messaging
from hermes_cli.web_routers import messaging


@pytest.fixture
def pairing(tmp_path, monkeypatch):
    from gateway import status

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    messaging._whatsapp_onboarding_sessions.clear()
    session = home / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True)
    creds = '{"me":{"id":"15550001111:1@s.whatsapp.net"},"registered":true}'
    (session / "creds.json").write_text(creds, encoding="utf-8")
    (session / "app-state.json").write_text("preserve", encoding="utf-8")
    starts = []
    workers = []
    timers = []
    real_thread = messaging.threading.Thread

    class Timer:
        def __init__(self, interval, function, args):
            self.function, self.args = function, args
            self.cancelled = False

        def start(self):
            timers.append(self)

        def cancel(self):
            self.cancelled = True

        def fire(self):
            if not self.cancelled:
                self.function(*self.args)

    class Thread:
        def __init__(self, *, target, args, daemon, kwargs=None, name=None):
            self.args = args
            self.target = target

        def start(self):
            starts.append(self.args)
            workers.append(self)

    monkeypatch.setattr(messaging.threading, "Thread", Thread)
    monkeypatch.setattr(messaging.threading, "Timer", Timer)

    def start(**kwargs):
        return asyncio.run(messaging.start_whatsapp_onboarding(web_models.WhatsAppOnboardingStart(**kwargs)))

    def gateway_can_acquire():
        from hermes_cli._subprocess_compat import windows_hide_flags

        output = tmp_path / "lock-probe.txt"
        code = "from gateway.status import acquire_scoped_lock; import sys; print(acquire_scoped_lock('whatsapp-session', sys.argv[1])[0])"
        with output.open("w", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-c", code, str(session)], stdout=stream, stderr=subprocess.STDOUT,
                           timeout=20, check=True, creationflags=windows_hide_flags())
        return output.read_text(encoding="utf-8").strip() == "True"

    yield SimpleNamespace(home=home, session=session, creds=creds, starts=starts, workers=workers, timers=timers,
                          start=start, gateway_can_acquire=gateway_can_acquire, real_thread=real_thread)
    from gateway.status_scoped_locks import release_scoped_lock

    for record in messaging._whatsapp_onboarding_sessions.values():
        timer = record.expiry_timer
        if timer is not None:
            timer.cancel()
        release_scoped_lock("whatsapp-session", record.session_path)
    messaging._whatsapp_onboarding_sessions.clear()


def test_repair_preserves_credentials_and_starts_fresh_pairing(pairing):
    result = pairing.start(reset_session=True)

    assert result["status"] == "starting"
    assert result["qr_payload"] is None
    assert result["account_id"] is None
    assert len(pairing.starts) == 1
    assert pairing.starts[0][1] == pairing.session
    assert not (pairing.session / "creds.json").exists()
    backups = list(pairing.session.parent.glob("session-backup-*/session"))
    assert len(backups) == 1
    assert (backups[0] / "creds.json").read_text(encoding="utf-8") == pairing.creds
    assert (backups[0] / "app-state.json").read_text(encoding="utf-8") == "preserve"
    assert re.fullmatch(r"session-backup-\d{8}T\d{6}Z-.+", backups[0].parent.name)


def test_reservation_blocks_gateway_until_cancel_before_worker_start(pairing):
    result = pairing.start(reset_session=True)
    assert pairing.gateway_can_acquire() is False
    asyncio.run(messaging.cancel_whatsapp_onboarding(result["pairing_id"]))
    assert pairing.gateway_can_acquire() is True
    assert pairing.timers[0].cancelled is True


def test_unattended_setup_expires_and_releases_reservation(pairing):
    result = pairing.start(reset_session=True)
    assert pairing.gateway_can_acquire() is False
    pairing.timers[0].fire()
    assert messaging._whatsapp_onboarding_sessions[result["pairing_id"]].status == "expired"
    assert pairing.gateway_can_acquire() is True


def test_failed_process_start_releases_reservation(pairing, monkeypatch):
    result = pairing.start(reset_session=True)

    def fail_spawn(*args):
        assert pairing.gateway_can_acquire() is False
        raise OSError("cannot start bridge")

    monkeypatch.setattr(messaging, "_spawn_whatsapp_pairing_process", fail_spawn)
    messaging._run_whatsapp_pairing(*pairing.starts[0])
    assert messaging._whatsapp_onboarding_sessions[result["pairing_id"]].status == "error"
    assert pairing.gateway_can_acquire() is True


def test_recycled_pid_does_not_block_repair(pairing):
    from gateway.status import get_process_start_time

    started = get_process_start_time(os.getpid())
    assert started is not None
    (pairing.session / "bridge.pid").write_text(f"{os.getpid()}\n{started - 1}\n19877", encoding="utf-8")
    assert pairing.start(reset_session=True)["status"] == "starting"


def test_matching_pid_fingerprint_refuses_repair(pairing):
    from gateway.status import get_process_start_time

    (pairing.session / "bridge.pid").write_text(f"{os.getpid()}\n{get_process_start_time(os.getpid())}", encoding="utf-8")
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 409
    assert (pairing.session / "creds.json").exists()


class _Bridge:
    def __init__(self, lines=()):
        self.stdout = lines
        self.returncode: int | None = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_reservation_covers_connected_event_and_credential_flush(pairing, monkeypatch):
    result = pairing.start(reset_session=True)
    record = messaging._whatsapp_onboarding_sessions[result["pairing_id"]]

    def events():
        yield '{"event":"connected","user":{"id":"15550001111"}}'
        assert record.status == "connected"
        assert pairing.gateway_can_acquire() is False
        # A status poll must not release the lock before the process exits.
        asyncio.run(messaging.get_whatsapp_onboarding_status(result["pairing_id"]))
        assert pairing.gateway_can_acquire() is False

    proc = _Bridge(events())
    monkeypatch.setattr(messaging, "_spawn_whatsapp_pairing_process", lambda *args: proc)
    messaging._run_whatsapp_pairing(*pairing.starts[0])
    assert record.status == "connected"
    assert record.session_reserved is False
    assert pairing.gateway_can_acquire() is True
    assert pairing.timers[0].cancelled


def test_expiry_stops_active_writer_and_releases_reservation(pairing, monkeypatch):
    result = pairing.start(reset_session=True)
    proc = _Bridge()

    def events():
        yield '{"event":"qr","qr":"test"}'
        assert pairing.gateway_can_acquire() is False
        pairing.timers[0].fire()
        assert proc.poll() is not None
        yield '{"event":"connected"}'  # Late events cannot undo expiry.

    proc.stdout = events()
    monkeypatch.setattr(messaging, "_spawn_whatsapp_pairing_process", lambda *args: proc)
    messaging._run_whatsapp_pairing(*pairing.starts[0])
    assert messaging._whatsapp_onboarding_sessions[result["pairing_id"]].status == "expired"
    assert pairing.gateway_can_acquire() is True


def test_expiry_during_spawn_keeps_reservation_until_late_process_stops(pairing, monkeypatch):
    result = pairing.start(reset_session=True)
    proc = _Bridge()

    def spawn(*args):
        pairing.timers[0].fire()
        assert pairing.gateway_can_acquire() is False
        return proc

    monkeypatch.setattr(messaging, "_spawn_whatsapp_pairing_process", spawn)
    messaging._run_whatsapp_pairing(*pairing.starts[0])
    assert proc.poll() is not None
    assert messaging._whatsapp_onboarding_sessions[result["pairing_id"]].status == "expired"
    assert pairing.gateway_can_acquire() is True


def test_thread_start_failure_releases_reservation(pairing, monkeypatch):
    def fail_thread(*args, **kwargs):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(messaging.threading, "Thread", fail_thread)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        pairing.start(reset_session=True)
    assert pairing.gateway_can_acquire() is True
    assert pairing.timers[0].cancelled


def test_apply_waits_for_flush_before_restarting_gateway(pairing, monkeypatch):
    from hermes_cli import config

    result = pairing.start(reset_session=True)
    record = messaging._whatsapp_onboarding_sessions[result["pairing_id"]]
    record.status = "connected"
    proc = _Bridge()
    record.proc = cast(Popen, proc)
    monkeypatch.setattr(messaging.threading, "Thread", pairing.real_thread)
    monkeypatch.setattr(config, "save_env_value", lambda *args: None)
    monkeypatch.setattr(web_server_messaging, "_write_platform_enabled", lambda *args: None)

    def wait(timeout=None):
        assert pairing.gateway_can_acquire() is False
        proc.returncode = 0
        return 0

    def restart(profile=None):
        assert proc.poll() == 0
        assert pairing.gateway_can_acquire() is True
        return {"restart_started": True}

    monkeypatch.setattr(proc, "wait", wait)
    monkeypatch.setattr(web_server_messaging, "_restart_gateway_after_whatsapp_onboarding", restart)
    assert asyncio.run(messaging.apply_whatsapp_onboarding(result["pairing_id"], web_models.WhatsAppOnboardingApply()))["ok"]


def test_unverifiable_live_pid_stays_blocked(pairing, monkeypatch):
    from gateway import status

    (pairing.session / "bridge.pid").write_text(f"{os.getpid()}\n123", encoding="utf-8")
    monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 409
    assert pairing.gateway_can_acquire() is True


def test_regular_start_keeps_existing_link(pairing):
    assert pairing.start()["status"] == "connected"
    assert pairing.starts == []
    assert (pairing.session / "creds.json").read_text(encoding="utf-8") == pairing.creds
    assert list(pairing.session.parent.glob("session-backup-*")) == []


def test_repair_refuses_live_bridge_without_moving_credentials(pairing):
    (pairing.session / "bridge.pid").write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 409
    assert "bridge" in refused.value.detail.lower()
    assert (pairing.session / "creds.json").read_text(encoding="utf-8") == pairing.creds
    assert pairing.starts == []


def test_repair_refuses_pending_pairing(pairing):
    record = web_server_messaging._WhatsAppOnboardingSession(
        proc=None, mode="bot", allowed_users="", session_path=str(pairing.session),
        expires_at="2099-01-01T00:00:00Z", expires_at_ts=time.time() + 600,
        status="installing",
    )
    messaging._whatsapp_onboarding_sessions["pending"] = record
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 409
    assert record.status == "installing"
    assert (pairing.session / "creds.json").exists()


def test_repair_refuses_session_lock_owned_by_live_gateway(pairing):
    from gateway import status

    lock = status._get_scope_lock_path("whatsapp-session", str(pairing.session))
    lock.parent.mkdir(parents=True)
    # A second process is not needed to establish a refusal. Self-owned locks
    # must also remain intact instead of being borrowed and released by reset.
    record = status._build_pid_record()
    record["identity_hash"] = status._scope_hash(str(pairing.session))
    lock.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 409
    assert lock.exists()
    assert (pairing.session / "creds.json").exists()


def test_failed_backup_aborts_before_pairing(pairing, monkeypatch):
    rename = Path.rename

    def refuse(path, target):
        if path == pairing.session:
            raise PermissionError("session is locked")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", refuse)
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 500
    assert (pairing.session / "creds.json").read_text(encoding="utf-8") == pairing.creds
    assert pairing.starts == []


@pytest.mark.platforms("windows")
def test_open_credentials_refuse_backup_on_windows(pairing):
    with (pairing.session / "creds.json").open("rb"):
        with pytest.raises(HTTPException) as refused:
            pairing.start(reset_session=True)
    assert refused.value.status_code == 500
    assert (pairing.session / "creds.json").read_text(encoding="utf-8") == pairing.creds
    assert pairing.starts == []


def test_apply_retains_flushing_bridge_for_reset_guard(pairing, monkeypatch):
    from hermes_cli import config

    result = pairing.start()
    record = messaging._whatsapp_onboarding_sessions[result["pairing_id"]]
    record.proc = cast(Popen, SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(config, "save_env_value", lambda *args: None)
    monkeypatch.setattr(web_server_messaging, "_write_platform_enabled", lambda *args: None)
    monkeypatch.setattr(web_server_messaging, "_restart_gateway_after_whatsapp_onboarding", lambda profile=None: {"restart_started": True})
    asyncio.run(messaging.apply_whatsapp_onboarding(result["pairing_id"], web_models.WhatsAppOnboardingApply()))
    assert messaging._whatsapp_onboarding_sessions[result["pairing_id"]] is record
    with pytest.raises(HTTPException) as refused:
        pairing.start(reset_session=True)
    assert refused.value.status_code == 409
    assert (pairing.session / "creds.json").read_text(encoding="utf-8") == pairing.creds


def test_cancel_during_process_start_cannot_race_session_backup(pairing, monkeypatch):
    result = pairing.start(reset_session=True)
    record = messaging._whatsapp_onboarding_sessions[result["pairing_id"]]

    def spawn(session_path, mode):
        asyncio.run(messaging.cancel_whatsapp_onboarding(result["pairing_id"]))
        with pytest.raises(HTTPException) as refused:
            pairing.start(reset_session=True)
        assert refused.value.status_code == 409
        raise OSError("pairing cancelled during process startup")

    monkeypatch.setattr(messaging, "_spawn_whatsapp_pairing_process", spawn)
    messaging._run_whatsapp_pairing(*pairing.starts[0])
    assert record.status == "cancelled"
    assert record.starting_process is False
    assert pairing.start(reset_session=True)["status"] == "starting"


def test_repair_selected_profile_leaves_default_session_untouched(pairing, monkeypatch):
    from hermes_cli import web_server_profiles

    other = pairing.home / "profiles" / "work"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("model: test\n", encoding="utf-8")
    session = other / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True)
    (session / "creds.json").write_text('{"registered":true}', encoding="utf-8")
    monkeypatch.setattr(web_server_profiles, "_resolve_profile_dir", lambda profile: other)

    result = pairing.start(profile="work", reset_session=True)
    assert result["status"] == "starting"
    assert pairing.starts[0][1] == session
    assert not (session / "creds.json").exists()
    assert (pairing.session / "creds.json").read_text(encoding="utf-8") == pairing.creds
    observed = []

    def spawn(*args):
        from hermes_constants import get_hermes_home

        observed.append(get_hermes_home())
        raise OSError("stop after checking the worker profile")

    monkeypatch.setattr(messaging, "_spawn_whatsapp_pairing_process", spawn)
    pairing.workers[0].target(*pairing.starts[0])
    assert observed == [other]


def test_query_profile_is_honored_for_repair(pairing, monkeypatch):
    from fastapi import FastAPI
    import httpx

    from hermes_cli import web_server_profiles

    other = pairing.home / "profiles" / "work"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("model: test\n", encoding="utf-8")
    session = other / "platforms" / "whatsapp" / "session"
    session.mkdir(parents=True)
    (session / "creds.json").write_text('{"registered":true}', encoding="utf-8")
    monkeypatch.setattr(web_server_profiles, "_resolve_profile_dir", lambda profile: other)
    app = FastAPI()
    app.include_router(messaging.router)

    async def request():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/messaging/whatsapp/onboarding/start?profile=work", json={"reset_session": True})

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json()["status"] == "starting"
    assert pairing.starts[0][1] == session
    assert (pairing.session / "creds.json").exists()
