"""Usage-aware fill_first selection for independent subscription windows (PR #43277)."""
import json
import threading
from pathlib import Path

import httpx
import pytest

from agent.credential_pool import load_pool


@pytest.fixture
def usage_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing-codex"))
    return tmp_path


def _write_profile(home, *, threshold=98, token_prefix="key"):
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n  provider: openai-codex\n  default: gpt-6-luna\n"
        "credential_pool_strategies:\n  openai-codex: fill_first\n"
        f"codex_usage:\n  exhausted_pct: {threshold}\n")
    (home / "auth.json").write_text(json.dumps({
        "version": 1, "providers": {}, "credential_pool": {"openai-codex": [
            {"id": f"cred-{i}", "source": f"manual:test-{i}", "label": f"account-{i}",
             "auth_type": "oauth", "access_token": f"{token_prefix}-{i}", "priority": i,
             "base_url": "https://chatgpt.com/backend-api/codex"}
            for i in range(2)
        ]},
    }))


def _payload(used, seconds=18000):
    return {"plan_type": "plus", "rate_limit": {
        "primary_window": {"used_percent": used, "limit_window_seconds": seconds},
    }}


def _mock_usage(monkeypatch, payloads, *, pool=None):
    import agent.account_usage as usage

    original_client = httpx.Client
    requests = []
    def respond(request):
        token = request.headers["Authorization"].removeprefix("Bearer ")
        requests.append(token)
        assert request.url.path == "/backend-api/wham/usage"
        if pool is not None:
            # An independent pool consumer must make progress while HTTP is in flight.
            ready = threading.Event()
            def read_pool():
                pool.entries()
                ready.set()
            thread = threading.Thread(target=read_pool)
            thread.start()
            assert ready.wait(2), "usage fetch holds the pool lock"
            thread.join()
        payload = payloads.get(token)
        return httpx.Response(401 if payload is None else 200, json=payload or {})
    monkeypatch.setattr(usage.httpx, "Client",
                        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))
    return requests


@pytest.mark.parametrize("seconds", [18000, 604800, 2592000])
def test_fill_first_prefers_account_with_room_in_every_window(usage_home, monkeypatch, seconds):
    _write_profile(usage_home)
    pool = load_pool("openai-codex")
    requests = _mock_usage(monkeypatch, {
        "key-0": _payload(100, seconds), "key-1": _payload(20, seconds),
    }, pool=pool)
    assert pool.select().runtime_api_key == "key-1"
    assert pool.select().runtime_api_key == "key-1"
    assert requests == ["key-0", "key-1"]  # A second selection uses cached snapshots.


def test_usage_settings_and_cache_are_isolated_between_profiles(usage_home, monkeypatch):
    a, b = usage_home / "a", usage_home / "b"
    _write_profile(a, threshold=98)
    _write_profile(b, threshold=90)
    requests = _mock_usage(monkeypatch, {"key-0": _payload(95), "key-1": _payload(10)})
    monkeypatch.setenv("HERMES_HOME", str(a))
    assert load_pool("openai-codex").select().runtime_api_key == "key-0"
    monkeypatch.setenv("HERMES_HOME", str(b))
    assert load_pool("openai-codex").select().runtime_api_key == "key-1"
    assert requests == ["key-0", "key-1", "key-0", "key-1"]


def test_quota_failure_reprobes_before_next_rotation(usage_home, monkeypatch):
    _write_profile(usage_home)
    payloads = {"key-0": _payload(10), "key-1": _payload(10)}
    pool = load_pool("openai-codex")
    requests = _mock_usage(monkeypatch, payloads)
    assert pool.select().id == "cred-0"
    payloads["key-0"] = _payload(100)
    assert pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="key-0").id == "cred-1"
    assert requests == ["key-0", "key-1", "key-0"]


def test_usage_failure_keeps_reactive_fill_first_selection(usage_home, monkeypatch):
    _write_profile(usage_home)
    _mock_usage(monkeypatch, {})
    pool = load_pool("openai-codex")
    assert pool.select().id == "cred-0"


def test_refreshing_access_token_invalidates_cached_usage(usage_home, monkeypatch):
    _write_profile(usage_home)
    pool = load_pool("openai-codex")
    requests = _mock_usage(monkeypatch, {
        "key-0": _payload(100), "key-1": _payload(10), "replacement": _payload(10),
    })
    assert pool.select().id == "cred-1"
    auth_path = usage_home / "auth.json"
    data = json.loads(auth_path.read_text())
    data["credential_pool"]["openai-codex"][0]["access_token"] = "replacement"
    auth_path.write_text(json.dumps(data))
    assert load_pool("openai-codex").select().id == "cred-0"
    assert requests == ["key-0", "key-1", "replacement"]


def test_unusual_window_durations_keep_session_before_weekly_fallback(usage_home, monkeypatch):
    _write_profile(usage_home)
    _mock_usage(monkeypatch, {
        "key-0": _payload(100, 172800),  # A two-day weekly-tier window.
        "key-1": _payload(100, 28800),   # An eight-hour session-tier window.
    })
    assert load_pool("openai-codex").select().id == "cred-1"
