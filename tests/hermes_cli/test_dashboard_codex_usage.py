from datetime import datetime, timezone

import pytest


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"
    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_alpha": worker_home}


def test_codex_usage_serializes_account_snapshot(monkeypatch, _isolate_hermes_home):
    from agent import account_usage
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

    fetched_at = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)
    reset_at = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

    def fake_fetch(provider, *, base_url=None, api_key=None):
        assert provider == "openai-codex"
        assert base_url is None
        assert api_key is None
        return AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=fetched_at,
            title="OpenAI Codex quota",
            plan="Plus",
            windows=(
                AccountUsageWindow(
                    label="Weekly",
                    used_percent=25.0,
                    reset_at=reset_at,
                ),
            ),
            details=("Credits balance: unlimited",),
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    resp = _client().get("/api/codex/usage")

    assert resp.status_code == 200
    assert resp.json() == {
        "available": True,
        "details": ["Credits balance: unlimited"],
        "error": None,
        "fetched_at": "2026-06-22T12:00:00Z",
        "plan": "Plus",
        "provider": "openai-codex",
        "source": "usage_api",
        "title": "OpenAI Codex quota",
        "windows": [
            {
                "detail": None,
                "label": "Weekly",
                "remaining_percent": 75.0,
                "reset_at": "2026-06-23T12:00:00Z",
                "used_percent": 25.0,
            }
        ],
    }


def test_codex_usage_profile_param_switches_hermes_home(monkeypatch, isolated_profiles):
    from agent import account_usage
    from agent.account_usage import AccountUsageSnapshot
    from hermes_constants import get_hermes_home

    homes = []

    def fake_fetch(provider, *, base_url=None, api_key=None):
        homes.append(get_hermes_home())
        return AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc),
            details=("profile scoped",),
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    resp = _client().get("/api/codex/usage", params={"profile": "worker_alpha"})

    assert resp.status_code == 200
    assert homes == [isolated_profiles["worker_alpha"]]
