"""Mocked Live admission/cleanup contracts; never call a paid provider."""
import asyncio
import json
import threading
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError


def identity():
    return dict(operation_id=str(uuid4()), client_instance_id=str(uuid4()), owner_generation=0)


@pytest.fixture
def client(monkeypatch, tmp_path):
    from hermes_cli import web_server
    from hermes_cli.voice_live_admission import AdmissionLedger
    from hermes_cli.web_routers import audio
    from hermes_cli import profiles
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("alpha", "beta"):
        home = profiles.get_profile_dir(name)
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setattr(audio, "_live_admissions", AdmissionLedger())
    # Exercise production routes + auth middleware without unrelated dashboard startup workers.
    from fastapi import FastAPI
    app = FastAPI()
    app.state.auth_required = False
    app.include_router(audio.router)
    app.middleware("http")(web_server._dashboard_auth_gate)
    app.middleware("http")(web_server.auth_middleware)
    with TestClient(app) as c:
        c.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        yield c


def test_cancel_before_reordered_create_and_wrong_profile(client, monkeypatch):
    from tools import voice_live
    monkeypatch.setattr(voice_live, "create_webrtc_session", lambda *a, **k: pytest.fail("tombstone must block provider"))
    ids = identity()
    path = "/api/audio/voice-live/session"
    cancelled = client.post(path + "/cancel?profile=alpha", json=ids)
    assert cancelled.status_code == 200
    for _ in range(2):
        result = client.post(path + "?profile=alpha", json={**ids, "sdp": "offer\r\n"})
        assert result.status_code == 200
        assert result.json()["cancelled"] is True
        assert "transport" not in result.json()
    # A different profile's tombstone does not disclose or mutate the first owner.
    assert client.post(path + "/cancel?profile=beta", json=ids).json()["state"] == "closed"
    assert client.post(path + "/cancel?profile=alpha", json=ids).json()["state"] == "closed"
    assert client.post(path + "/cancel", json={**ids, "session_id": "arbitrary"}).status_code == 422
    client.headers.clear()
    assert client.post(path + "/cancel?profile=alpha", json=ids).status_code == 401


def test_held_create_cancel_retains_worker_and_never_returns_sdp(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from tools import voice_live
    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    def create(sdp, history, *, capture_cleanup):
        assert sdp == "offer\r\n"
        started.set()
        assert release.wait(5)
        capture_cleanup("live_late", lambda: cleaned.set() or True)
        return {"session": {"id": "live_late"}, "transport": {"sdp": "secret answer"}}
    monkeypatch.setattr(voice_live, "create_webrtc_session", create)
    ids = identity()
    path = "/api/audio/voice-live/session?profile=alpha"
    with ThreadPoolExecutor(max_workers=1) as pool:
        creating = pool.submit(client.post, path, json={**ids, "sdp": "offer\r\n"})
        try:
            assert started.wait(5)
            cancelled = client.post("/api/audio/voice-live/session/cancel?profile=alpha", json=ids)
            assert cancelled.json()["state"] == "cancel_requested"
        finally:
            release.set()
        answer = creating.result(timeout=5).json()
    assert cleaned.wait(5)
    assert answer["state"] == "closed"
    assert answer["finalization_confirmed"] is True
    assert "transport" not in answer
    assert client.post("/api/audio/voice-live/session/cancel?profile=alpha", json=ids).json()["state"] == "closed"


def test_active_owner_lease_outlives_600_seconds_then_expires(monkeypatch):
    """A renewed call survives the old cap; an abandoned owner still gets cleanup."""
    from fastapi import HTTPException
    from hermes_cli.voice_live_admission import AdmissionLedger
    async def scenario():
        loop = asyncio.get_running_loop()
        now, timers = 0, []
        class Timer:
            def __init__(self, when, callback, args):
                self.when, self.callback, self.args = when, callback, args
                self.cancelled = False
            def cancel(self):
                self.cancelled = True
        def call_later(seconds, callback, *args):
            timer = Timer(now + seconds, callback, args)
            timers.append(timer)
            return timer
        def advance(seconds):
            nonlocal now
            now += seconds
            for timer in list(timers):
                if timer.when <= now:
                    timers.remove(timer)
                    if not timer.cancelled:
                        timer.callback(*timer.args)
        monkeypatch.setattr(loop, "call_later", call_later)
        ledger = AdmissionLedger()
        key = (("owner",), "/profile/a", "client", 1, "op")
        created, closed = [], []
        async def provider(capture):
            created.append(True)
            capture("provider", lambda: closed.append(True) or True)
            return {"sdp": "answer"}
        await ledger.create(key, "offer", [], provider)
        advance(599)
        assert ledger.keepalive(key) == {"ok": True, "state": "active", "renewed": True}
        advance(2)
        assert ledger.entries[key].state == "active", "renewed legitimate call hit the old 600s cap"
        assert created == [True] and closed == []
        count = len(ledger.entries)
        with pytest.raises(HTTPException) as unknown:
            ledger.keepalive(("unknown",))
        assert unknown.value.status_code == 404 and len(ledger.entries) == count
        advance(599)
        await ledger.entries[key].task
        assert ledger.entries[key].state == "closed" and closed == [True]
        with pytest.raises(HTTPException) as terminal:
            ledger.keepalive(key)
        assert terminal.value.status_code == 409
        assert ledger.cancel(key)["finalization_confirmed"] is True
    asyncio.run(scenario())


def test_duplicates_conflicts_wrong_owner_and_unconfirmed():
    from hermes_cli.voice_live_admission import AdmissionLedger
    from fastapi import HTTPException
    async def run():
        ledger = AdmissionLedger()
        key = (("session", "provider", "org", "alice"), "/alpha", "instance", "op", 0)
        calls = []
        async def create(capture):
            calls.append(1)
            capture("live_id", lambda: False)
            return {"session": {"id": "live_id"}, "transport": {"sdp": "answer"}}
        a = await ledger.create(key, "offer", None, create)
        assert await ledger.create(key, "offer", None, create) == a
        assert calls == [1]
        with pytest.raises(HTTPException) as conflict:
            await ledger.create(key, "different", None, create)
        assert conflict.value.status_code == 409
        for wrong in [(("session", "provider", "org", "bob"), *key[1:]),
                      (key[0], "/beta", *key[2:]),
                      (*key[:2], "other-instance", *key[3:]),
                      (*key[:4], 1)]:
            count = len(ledger.entries)
            with pytest.raises(HTTPException) as renewal:
                ledger.keepalive(wrong)
            assert renewal.value.status_code == 404 and len(ledger.entries) == count
            assert ledger.cancel(wrong)["state"] == "closed"
            assert ledger.entries[key].state == "active"
        assert ledger.cancel(key)["state"] == "closing"
        await asyncio.shield(ledger.entries[key].task)
        assert ledger.cancel(key)["state"] == "finalization_unconfirmed"
        assert ledger.cancel(key)["finalization_confirmed"] is False
        assert calls == [1]
    asyncio.run(run())


def test_keepalive_route_is_owner_bound_and_never_admits(client, monkeypatch):
    from tools import voice_live
    from hermes_cli.web_routers import audio
    created = []
    def create(sdp, history, *, capture_cleanup):
        created.append(True)
        capture_cleanup("provider", lambda: True)
        return {"transport": {"sdp": "answer"}}
    monkeypatch.setattr(voice_live, "create_webrtc_session", create)
    ids = identity()
    path = "/api/audio/voice-live/session"
    assert client.post(path + "/keepalive?profile=alpha", json=ids).status_code == 404
    assert not audio._live_admissions.entries
    assert client.post(path + "?profile=alpha", json={**ids, "sdp": "offer"}).status_code == 200
    assert client.post(path + "/keepalive?profile=alpha", json=ids).json() == {
        "ok": True, "state": "active", "renewed": True}
    count = len(audio._live_admissions.entries)
    for url, body in [("beta", ids), ("alpha", {**ids, "owner_generation": 1})]:
        assert client.post(path + "/keepalive?profile=" + url, json=body).status_code == 404
    assert len(audio._live_admissions.entries) == count and created == [True]
    for body in [{**ids, "session_id": "arbitrary"}, {**ids, "owner_generation": True},
                 {**ids, "operation_id": "arbitrary"}, {"operation_id": ids["operation_id"]}]:
        assert client.post(path + "/keepalive?profile=alpha", json=body).status_code == 422
    client.post(path + "/cancel?profile=alpha", json=ids)
    assert client.post(path + "/keepalive?profile=alpha", json=ids).status_code == 409
    client.headers.clear()
    assert client.post(path + "/keepalive?profile=alpha", json=ids).status_code == 401


def test_failed_provider_after_capture_still_closes():
    from hermes_cli.voice_live_admission import AdmissionLedger
    async def scenario():
        ledger = AdmissionLedger()
        closed = []
        async def provider(capture):
            capture("late", lambda: closed.append(True) or True)
            raise RuntimeError("private upstream failure")
        result = await ledger.create(("owner",), "offer", None, provider)
        assert result["state"] == "closed" and closed == [True]
        assert "transport" not in result and ledger.entries[("owner",)].closer is None
    asyncio.run(scenario())


def test_create_uses_the_profile_home_captured_for_its_owner(client, monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home
    from hermes_cli.web_routers import audio
    from tools import voice_live
    original = audio._run_config_scoped
    seen, resolved = [], []
    async def move_home_after_owner_binding(profile, fn):
        result = await original(profile, fn)
        if not resolved:
            resolved.append(True)
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "replacement"))
        return result
    def create(sdp, history, *, capture_cleanup):
        seen.append(str(get_hermes_home().resolve()))
        capture_cleanup("provider", lambda: True)
        return {"transport": {"sdp": "answer"}}
    monkeypatch.setattr(audio, "_run_config_scoped", move_home_after_owner_binding)
    monkeypatch.setattr(voice_live, "create_webrtc_session", create)
    result = client.post("/api/audio/voice-live/session", json={**identity(), "sdp": "offer"})
    assert result.status_code == 200
    assert seen == [next(iter(audio._live_admissions.entries))[1]]


def test_status_does_not_seed_pool_or_write_config(monkeypatch, tmp_path):
    from hermes_cli import config, config_backups, auth
    from agent import credential_pool
    from tools import voice_live
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("voice:\n  voice_chat_mode: gpt-live\n")
    monkeypatch.setattr(config, "ensure_hermes_home", lambda: pytest.fail("status must not initialize home"))
    monkeypatch.setattr(config_backups, "backup_config", lambda *a, **k: pytest.fail("status must not write backups"))
    monkeypatch.setattr(credential_pool, "load_pool", lambda *a: pytest.fail("status must not seed a pool"))
    monkeypatch.setattr(auth, "read_credential_pool", lambda: {"openai-api": [{"access_token": "fake-persisted-key"}]})
    status = voice_live.resolve_gpt_live_status()
    assert status["mode"] == "gpt-live" and status["available"] is True
    assert "api_key" not in status and "instructions" not in status
    assert "fake-persisted-key" not in json.dumps(status)


def test_new_identity_fields_are_all_or_none_and_strict():
    from hermes_cli.web_models import VoiceLiveSessionRequest, VoiceLiveCancelRequest
    assert VoiceLiveSessionRequest(sdp="offer\r\n").sdp == "offer\r\n"
    ids = identity()
    assert VoiceLiveCancelRequest(**ids).owner_generation == 0
    for bad in [{"operation_id": ids["operation_id"]}, {**ids, "operation_id": "garbage"},
                {**ids, "owner_generation": True}, {**ids, "owner_generation": -1},
                {**ids, "owner_generation": "0"}]:
        with pytest.raises(ValidationError):
            VoiceLiveSessionRequest(sdp="offer", **bad)


def test_close_attach_uses_frozen_auth_and_requires_closed(monkeypatch):
    from tools import voice_live
    import websockets.sync.client
    captured = {}
    events = [{"type": "session.created"}, {"type": "session.closed"}]
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def send(self, event): captured["sent"] = json.loads(event)
        def recv(self, timeout):
            if events: return json.dumps(events.pop(0))
            raise TimeoutError()
    def connect(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Socket()
    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    headers = {"Authorization": "Bearer fake-original", "OpenAI-Project": "original"}
    assert voice_live.close_webrtc_session("live/id", "https://api.example/v1", headers) is True
    assert captured["url"] == "wss://api.example/v1/live/sessions/live%2Fid/attach"
    assert captured["additional_headers"] == headers
    assert captured["sent"] == {"type": "session.close"}
    assert voice_live.close_webrtc_session("live/id", "https://api.example/v1", headers) is False

def test_status_gate_and_legacy_byte_exact_offer(client, monkeypatch):
    from tools import voice_live
    calls = []
    def create(sdp, history):
        calls.append((sdp, history))
        return {"session": {"id": "legacy"}, "transport": {"sdp": "answer"}}
    monkeypatch.setattr(voice_live, "create_webrtc_session", create)
    monkeypatch.setattr(voice_live, "resolve_gpt_live_status", lambda: {"mode": "gpt-live", "available": True})
    status = client.get("/api/audio/voice-live/status?profile=alpha").json()
    assert status["supports_cancellation"] is True and status["supports_keepalive"] is True
    assert status["keepalive_interval_seconds"] == 30
    history = [{"type": "message", "content": [{"text": " unchanged "}]}]
    result = client.post("/api/audio/voice-live/session?profile=alpha", json={"sdp": "offer\r\n", "history": history})
    assert result.json()["transport"]["sdp"] == "answer"
    assert calls == [("offer\r\n", history)]


def test_upstream_error_body_never_logged_or_returned(monkeypatch, caplog):
    import io
    import urllib.error
    from email.message import Message
    from tools import voice_live
    monkeypatch.setattr(voice_live, "_resolve_credentials", lambda live: ("fake-key", "https://api.example/v1"))
    def reject(*args, **kwargs):
        raise urllib.error.HTTPError("https://api.example/v1", 400, "upstream", Message(), io.BytesIO(b"private transcript sk-secret"))
    monkeypatch.setattr(voice_live.urllib.request, "urlopen", reject)
    with pytest.raises(RuntimeError) as err:
        voice_live.create_webrtc_session("offer")
    assert "private transcript" not in str(err.value) + caplog.text
    assert "sk-secret" not in str(err.value) + caplog.text

def test_inflight_survives_tombstone_ttl_and_capacity_is_bounded():
    from hermes_cli.voice_live_admission import AdmissionLedger
    from fastapi import HTTPException
    async def run():
        ledger = AdmissionLedger(max_entries=2, max_sessions=1, tombstone_seconds=0.02)
        release, started = asyncio.Event(), asyncio.Event()
        async def create(capture):
            started.set()
            await release.wait()
            capture("late", lambda: True)
            return {"session": {"id": "late"}, "transport": {"sdp": "answer"}}
        pending = asyncio.create_task(ledger.create(("owner", "op"), "offer", None, create))
        await started.wait()
        assert ledger.entries[("owner", "op")].timer is None
        with pytest.raises(HTTPException) as pending_renewal:
            ledger.keepalive(("owner", "op"))
        assert pending_renewal.value.status_code == 409
        ledger.cancel(("owner", "op"))
        ledger.cancel(("owner", "tombstone"))
        with pytest.raises(HTTPException) as full:
            ledger.cancel(("owner", "overflow"))
        assert full.value.status_code == 429
        await asyncio.sleep(0.08)
        assert ("owner", "op") in ledger.entries
        assert ("owner", "tombstone") not in ledger.entries
        release.set()
        assert (await pending)["state"] == "closed"
        await asyncio.sleep(0.08)
        assert not ledger.entries
    asyncio.run(run())
