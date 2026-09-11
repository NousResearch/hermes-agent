"""``POST /api/audio/stt-lease`` — desktop voice input as STT warm-up/release.

The desktop acquires a lease when the mic opens so the backend pre-loads the
configured local faster-whisper model while the user is still speaking; the
transcription request then starts hot instead of paying the cold load inside
its timeout (issue #105955). Releasing the last lease drops the refcount but
— unlike TTS — never unloads the model: the engine is shared with the
gateway/CLI surfaces in the same backend process.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_beta"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (worker_home / ".env").write_text("", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_beta": worker_home}


@pytest.fixture
def client(monkeypatch, isolated_profiles):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


@pytest.fixture(autouse=True)
def _clean_stt_state():
    from tools import stt_lease, transcription_tools

    stt_lease._reset_stt_leases_for_tests()
    saved_model, saved_name = transcription_tools._local_model, transcription_tools._local_model_name
    transcription_tools._local_model, transcription_tools._local_model_name = None, None
    yield
    stt_lease._reset_stt_leases_for_tests()
    transcription_tools._local_model, transcription_tools._local_model_name = saved_model, saved_name


def _local_cfg(monkeypatch, model="tiny"):
    from tools import transcription_tools

    monkeypatch.setattr(transcription_tools, "_HAS_FASTER_WHISPER", True)
    monkeypatch.setattr(
        transcription_tools, "_load_stt_config", lambda: {"local": {"model": model}}
    )
    # Explicit local provider without going through the selection-file probe.
    monkeypatch.setattr(transcription_tools, "_get_provider", lambda cfg: "local")


def test_active_acquires_and_warms(client, monkeypatch):
    from tools import stt_lease

    warmed = []
    monkeypatch.setattr(
        stt_lease,
        "warm_stt_provider",
        lambda cfg=None, provider=None: warmed.append(1)
        or {"provider": "local", "warmed": True, "action": "loaded"},
    )

    resp = client.post("/api/audio/stt-lease", json={"lease": "desktop:voice-input:abc", "active": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["lease"] == "desktop:voice-input:abc"
    assert body["active"] is True
    assert body["leases"] == 1
    assert body["action"] == "loaded"
    assert warmed == [1]
    assert stt_lease.stt_lease_holders() == ["desktop:voice-input:abc"]


def test_inactive_releases_without_unloading(client, monkeypatch):
    from tools import stt_lease, transcription_tools

    monkeypatch.setattr(
        stt_lease, "warm_stt_provider", lambda cfg=None, provider=None: {"action": "noop", "warmed": False, "provider": "openai"}
    )
    client.post("/api/audio/stt-lease", json={"lease": "desktop:voice-input:a", "active": True})
    client.post("/api/audio/stt-lease", json={"lease": "desktop:voice-input:b", "active": True})

    # A resident model shared with other surfaces must survive the releases.
    sentinel = object()
    transcription_tools._local_model, transcription_tools._local_model_name = sentinel, "tiny"

    first = client.post("/api/audio/stt-lease", json={"lease": "desktop:voice-input:a", "active": False}).json()
    assert first["leases"] == 1
    assert transcription_tools._local_model is sentinel

    last = client.post("/api/audio/stt-lease", json={"lease": "desktop:voice-input:b", "active": False}).json()
    assert last["leases"] == 0
    assert transcription_tools._local_model is sentinel


def test_warm_failure_is_reported_not_an_http_error(client, monkeypatch):
    from tools import stt_lease

    def _boom(cfg=None, provider=None):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(stt_lease, "warm_stt_provider", _boom)
    resp = client.post("/api/audio/stt-lease", json={"lease": "desktop:voice-input:abc", "active": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["action"] == "error"
    assert "engine exploded" in body["error"]


def test_blank_lease_rejected(client):
    resp = client.post("/api/audio/stt-lease", json={"lease": "   ", "active": True})
    assert resp.status_code == 400


def test_warm_loads_configured_local_model_into_transcription_slot(monkeypatch):
    from tools import stt_lease, transcription_tools

    _local_cfg(monkeypatch, model="small")
    loaded = []
    monkeypatch.setattr(
        transcription_tools,
        "_get_or_load_local_model",
        lambda name, cfg: loaded.append((name, cfg)) or object(),
    )

    result = stt_lease.warm_stt_provider()

    assert result["warmed"] is True
    assert result["provider"] == "local"
    assert result["action"] == "loaded"
    # Same name transcription resolves (config model, normalized) and the
    # same loader transcription reads — not a shadow cache.
    assert loaded and loaded[0][0] == "small"
    assert "elapsed_ms" in result


def test_warm_reports_cached_when_model_already_resident(monkeypatch):
    from tools import stt_lease, transcription_tools

    _local_cfg(monkeypatch, model="tiny")
    sentinel = object()
    transcription_tools._local_model, transcription_tools._local_model_name = sentinel, "tiny"
    calls = []
    monkeypatch.setattr(
        transcription_tools,
        "_get_or_load_local_model",
        lambda name, cfg: calls.append(name) or sentinel,
    )

    result = stt_lease.warm_stt_provider()

    assert result == {
        "provider": "local",
        "warmed": True,
        "action": "cached",
        "elapsed_ms": result["elapsed_ms"],
    }


def test_warm_is_noop_for_cloud_providers(monkeypatch):
    from tools import stt_lease, transcription_tools

    monkeypatch.setattr(transcription_tools, "_load_stt_config", lambda: {"provider": "openai"})
    monkeypatch.setattr(transcription_tools, "_get_provider", lambda cfg: "openai")
    calls = []
    monkeypatch.setattr(
        transcription_tools, "_get_or_load_local_model", lambda name, cfg: calls.append(name)
    )

    result = stt_lease.warm_stt_provider()

    assert result["warmed"] is False
    assert result["action"] == "noop"
    assert calls == []


def test_warm_never_raises_on_engine_failure(monkeypatch):
    from tools import stt_lease, transcription_tools

    _local_cfg(monkeypatch)

    def _boom(name, cfg):
        raise RuntimeError("CUDA exploded")

    monkeypatch.setattr(transcription_tools, "_get_or_load_local_model", _boom)

    result = stt_lease.warm_stt_provider()

    assert result["warmed"] is False
    assert result["action"] == "error"
    assert "CUDA exploded" in result["error"]
