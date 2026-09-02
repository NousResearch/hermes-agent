"""``POST /api/audio/tts-lease`` — desktop speech toggles as TTS warm-up/release.

The desktop's "Read replies aloud" and voice-conversation toggles call this so
the backend can pre-load the configured TTS engine when speech is about to be
needed and unload resident local models once no surface holds a lease.
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
def _clean_leases():
    from tools import tts_tool

    tts_tool._reset_tts_leases_for_tests()
    for cache in tts_tool._LOCAL_TTS_MODEL_CACHES.values():
        cache.clear()
    yield
    tts_tool._reset_tts_leases_for_tests()
    for cache in tts_tool._LOCAL_TTS_MODEL_CACHES.values():
        cache.clear()


def test_active_acquires_and_warms(client, monkeypatch):
    from tools import tts_tool

    warmed = []
    monkeypatch.setattr(
        tts_tool,
        "warm_tts_provider",
        lambda cfg=None, provider=None: warmed.append(1) or {"provider": "piper", "warmed": True, "action": "loaded"},
    )

    resp = client.post("/api/audio/tts-lease", json={"lease": "desktop:read-aloud", "active": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["lease"] == "desktop:read-aloud"
    assert body["active"] is True
    assert body["leases"] == 1
    assert body["action"] == "loaded"
    assert warmed == [1]
    assert tts_tool.tts_lease_holders() == ["desktop:read-aloud"]


def test_inactive_releases_and_unloads_when_last(client, monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "warm_tts_provider", lambda cfg=None, provider=None: {"action": "noop", "warmed": False, "provider": "piper"})
    client.post("/api/audio/tts-lease", json={"lease": "desktop:read-aloud", "active": True})
    client.post("/api/audio/tts-lease", json={"lease": "desktop:conversation:abc", "active": True})
    tts_tool._piper_voice_cache["voice"] = object()

    first = client.post("/api/audio/tts-lease", json={"lease": "desktop:read-aloud", "active": False}).json()
    assert first["leases"] == 1
    assert first["released"] == 0
    assert len(tts_tool._piper_voice_cache) == 1

    last = client.post("/api/audio/tts-lease", json={"lease": "desktop:conversation:abc", "active": False}).json()
    assert last["leases"] == 0
    assert last["released"] == 1
    assert tts_tool._piper_voice_cache == {}


def test_warm_failure_is_reported_not_an_http_error(client, monkeypatch):
    from tools import tts_tool

    def _boom(cfg=None, provider=None):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(tts_tool, "warm_tts_provider", _boom)
    resp = client.post("/api/audio/tts-lease", json={"lease": "desktop:read-aloud", "active": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["action"] == "error"
    assert "engine exploded" in body["error"]


def test_blank_lease_rejected(client):
    resp = client.post("/api/audio/tts-lease", json={"lease": "   ", "active": True})
    assert resp.status_code == 400


def test_active_default_true(client, monkeypatch):
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "warm_tts_provider", lambda cfg=None, provider=None: {"action": "noop", "warmed": False, "provider": "x"})
    resp = client.post("/api/audio/tts-lease", json={"lease": "tui:x"})
    assert resp.json()["active"] is True
    assert tts_tool.tts_lease_holders() == ["tui:x"]


def test_acquire_resolves_provider_inside_target_profile(client, isolated_profiles, monkeypatch):
    """Warm-up must read the REQUESTING profile's tts config, like /api/audio/speak."""
    import yaml
    from tools import tts_tool

    (isolated_profiles["worker_beta"] / "config.yaml").write_text(
        yaml.safe_dump({"tts": {"provider": "kittentts"}}), encoding="utf-8"
    )
    seen = {}

    def _fake_warm(cfg=None, provider=None):
        from hermes_constants import get_hermes_home

        seen["home"] = str(get_hermes_home())
        seen["provider"] = tts_tool._get_provider(tts_tool._load_tts_config())
        return {"action": "noop", "warmed": False, "provider": seen["provider"]}

    monkeypatch.setattr(tts_tool, "warm_tts_provider", _fake_warm)
    resp = client.post("/api/audio/tts-lease?profile=worker_beta", json={"lease": "desktop:read-aloud", "active": True})
    assert resp.status_code == 200
    assert seen["home"] == str(isolated_profiles["worker_beta"])
    assert seen["provider"] == "kittentts"


def test_unknown_profile_404(client):
    resp = client.post("/api/audio/tts-lease?profile=ghost", json={"lease": "desktop:read-aloud", "active": True})
    assert resp.status_code == 404
