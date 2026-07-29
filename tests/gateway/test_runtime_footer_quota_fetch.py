from datetime import datetime, timezone

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from gateway.run import _fetch_runtime_footer_quota_snapshot


def test_runtime_footer_quota_fetch_combines_model_and_supermemory(monkeypatch):
    fetched = []
    now = datetime(2026, 7, 22, tzinfo=timezone.utc)

    def fake_fetch(provider, *, base_url=None, api_key=None):
        fetched.append((provider, base_url, api_key))
        if provider == "openai-codex":
            return AccountUsageSnapshot(
                provider="openai-codex",
                source="test",
                fetched_at=now,
                windows=(AccountUsageWindow(label="Session", used_percent=25),),
            )
        if provider == "supermemory":
            return AccountUsageSnapshot(
                provider="supermemory",
                source="test",
                fetched_at=now,
                windows=(AccountUsageWindow(label="Supermemory credits", used_percent=80, detail="$10.00 of $55.00 remaining"),),
            )
        raise AssertionError(provider)

    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm-test")
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fake_fetch)

    snapshot = _fetch_runtime_footer_quota_snapshot(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-token",
    )

    assert snapshot is not None
    assert snapshot.provider == "combined"
    assert [(window.label, window.used_percent) for window in snapshot.windows] == [
        ("Session", 25),
        ("Supermemory credits", 80),
    ]
    assert fetched == [
        ("openai-codex", "https://chatgpt.com/backend-api/codex", "codex-token"),
        ("supermemory", None, None),
    ]


def test_runtime_footer_quota_fetch_allows_supermemory_without_model_provider(monkeypatch):
    now = datetime(2026, 7, 22, tzinfo=timezone.utc)

    def fake_fetch(provider, *, base_url=None, api_key=None):
        assert provider == "supermemory"
        return AccountUsageSnapshot(
            provider="supermemory",
            source="test",
            fetched_at=now,
            windows=(AccountUsageWindow(label="Supermemory credits", used_percent=81),),
        )

    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sm-test")
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", fake_fetch)

    snapshot = _fetch_runtime_footer_quota_snapshot("custom-local")

    assert snapshot is not None
    assert snapshot.provider == "supermemory"
    assert snapshot.windows[0].label == "Supermemory credits"


def test_gateway_footer_skips_quota_fetch_when_footer_disabled(monkeypatch):
    from gateway.runtime_footer import build_footer_line, resolve_footer_config

    cfg = {"display": {"runtime_footer": {"enabled": False}}}
    resolved = resolve_footer_config(cfg, "telegram")
    called = False

    def fetch_quota():
        nonlocal called
        called = True
        return None

    quota_snapshot = fetch_quota() if resolved.get("enabled") and "quota" in (resolved.get("fields") or ()) else None
    line = build_footer_line(
        user_config=cfg,
        platform_key="telegram",
        model="openai/gpt-5.5",
        context_tokens=1,
        context_length=10,
        quota_snapshot=quota_snapshot,
        cwd="",
    )

    assert line == ""
    assert called is False
