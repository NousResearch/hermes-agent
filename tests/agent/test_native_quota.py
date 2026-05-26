import json
from datetime import datetime, timezone

from agent.native_quota import (
    active_native_quota_provider,
    clear_native_quota_cache,
    format_native_quota_statusbar,
    get_native_quota_statusbar_for_model,
    read_native_quota_summary,
    refresh_native_quota_snapshot,
    _quota_state_dirs,
)


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _write_snapshot(state_dir, provider="openai-codex", **overrides):
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "provider": provider,
        "source": "codex-usage-endpoint" if provider == "openai-codex" else "claude-oauth-usage",
        "status": "exact",
        "fetched_at": "2026-05-25T12:00:00Z",
        "windows": {
            "five_hour": {"used_percentage": 12, "resets_at": "2026-05-25T15:00:00Z"},
            "seven_day": {"used_percentage": 4.4, "resets_at": "2026-05-30T12:00:00Z"},
        },
    }
    payload.update(overrides)
    path = state_dir / f"provider-native-quota-{provider}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quota_state_dirs_are_hermes_standalone(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/Users/nicholas/.hermes/profiles/orchestrator")
    monkeypatch.delenv("HERMES_NATIVE_QUOTA_STATE_DIR", raising=False)

    paths = [str(path) for path in _quota_state_dirs(None)]

    assert "/Users/nicholas/.hermes/profiles/orchestrator/state" in paths
    assert all("/.pi/" not in path for path in paths)


def test_active_native_quota_provider_detects_codex_by_provider():
    assert active_native_quota_provider("openai-codex", "gpt-5.5") == "openai-codex"


def test_active_native_quota_provider_detects_claude_by_model():
    assert active_native_quota_provider("anthropic", "claude-opus-4.7") == "claude-cli"


def test_active_native_quota_provider_ignores_generic_gpt_without_codex_provider():
    assert active_native_quota_provider("openai", "gpt-5.5") is None


def test_read_native_quota_summary_formats_5h_and_7d(tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "state"
    _write_snapshot(state_dir)
    now = _ts("2026-05-25T12:30:00Z")

    summary = read_native_quota_summary("openai-codex", state_dirs=[state_dir], now=now)

    assert summary is not None
    assert summary.short_label == "cdx"
    assert summary.stale is True  # fetched 30m ago, beyond the 10m default
    assert format_native_quota_statusbar(summary, now=now) == "cdx~ 5h 12%↻2h30m 7d 4.4%↻5d"


def test_format_reset_countdown_carries_rounded_minutes(tmp_path):
    state_dir = tmp_path / "state"
    _write_snapshot(
        state_dir,
        windows={
            "five_hour": {"used_percentage": 12, "resets_at": "2026-05-25T13:59:30Z"},
            "seven_day": {"used_percentage": 4.4, "resets_at": "2026-05-26T11:59:30Z"},
        },
    )
    now = _ts("2026-05-25T12:00:00Z")

    summary = read_native_quota_summary("openai-codex", state_dirs=[state_dir], now=now)

    assert summary is not None
    assert format_native_quota_statusbar(summary, now=now) == "cdx 5h 12%↻2h 7d 4.4%↻1d"


def test_reader_uses_most_constrained_additional_limit_per_window(tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "state"
    _write_snapshot(
        state_dir,
        additional_limits=[
            {
                "limit_id": "codex_bengalfox",
                "limit_name": "GPT Codex",
                "windows": {
                    "five_hour": {"used_percentage": 75, "resets_at": "2026-05-25T13:15:00Z"},
                    "seven_day": {"used_percentage": 2, "resets_at": "2026-05-30T12:00:00Z"},
                },
            }
        ],
    )
    now = _ts("2026-05-25T12:30:00Z")

    summary = read_native_quota_summary("openai-codex", state_dirs=[state_dir], now=now)

    assert summary is not None
    assert format_native_quota_statusbar(summary, now=now) == "cdx~ 5h 75%↻45m 7d 4.4%↻5d"


def test_expired_windows_are_pruned(tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "state"
    _write_snapshot(
        state_dir,
        windows={
            "five_hour": {"used_percentage": 99, "resets_at": "2026-05-25T11:00:00Z"},
            "seven_day": {"used_percentage": 33, "resets_at": "2026-05-30T12:00:00Z"},
        },
    )
    now = _ts("2026-05-25T12:30:00Z")

    summary = read_native_quota_summary("openai-codex", state_dirs=[state_dir], now=now)

    assert summary is not None
    assert [window.label for window in summary.windows] == ["7d"]




def _jwt_with_account(account_id="acct_123"):
    import base64
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_refresh_codex_snapshot_is_hermes_owned(monkeypatch, tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "hermes-state"
    monkeypatch.setenv("HERMES_NATIVE_QUOTA_STATE_DIR", str(state_dir))
    calls = []

    def fake_request(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        return {
            "plan_type": "plus",
            "rate_limit": {
                "primary_window": {"used_percent": 12.5, "reset_at": _ts("2026-05-25T15:00:00Z")},
                "secondary_window": {"used_percent": 4, "reset_at": _ts("2026-05-30T12:00:00Z")},
            },
            "additional_rate_limits": [
                {
                    "metered_feature": "codex_gpt5",
                    "limit_name": "GPT Codex",
                    "rate_limit": {
                        "primary_window": {"used_percent": 66, "reset_at": _ts("2026-05-25T13:00:00Z")}
                    },
                }
            ],
        }

    path = refresh_native_quota_snapshot(
        "openai-codex",
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
        request_json=fake_request,
        codex_credentials={
            "api_key": _jwt_with_account("acct_abc"),
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )

    assert path == state_dir / "provider-native-quota-openai-codex.json"
    assert calls[0][0] == "https://chatgpt.com/backend-api/codex/usage"
    assert calls[0][1]["chatgpt-account-id"] == "acct_abc"
    summary = read_native_quota_summary("openai-codex", state_dirs=[state_dir], now=_ts("2026-05-25T12:30:00Z"))
    assert summary is not None
    assert format_native_quota_statusbar(summary, now=_ts("2026-05-25T12:30:00Z")) == "cdx~ 5h 66%↻30m 7d 4%↻5d"


def test_refresh_codex_retries_without_derived_account_id(monkeypatch, tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "hermes-state"
    monkeypatch.setenv("HERMES_NATIVE_QUOTA_STATE_DIR", str(state_dir))
    calls = []

    def fake_request(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        if "chatgpt-account-id" in headers:
            return None
        return {
            "rate_limit": {
                "primary_window": {"used_percent": 9, "reset_at": _ts("2026-05-25T13:00:00Z")},
                "secondary_window": {"used_percent": 69, "reset_at": _ts("2026-05-30T12:00:00Z")},
            }
        }

    path = refresh_native_quota_snapshot(
        "openai-codex",
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
        request_json=fake_request,
        codex_credentials={
            "api_key": _jwt_with_account("derived_rejected"),
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )

    assert path == state_dir / "provider-native-quota-openai-codex.json"
    assert calls[0][1]["chatgpt-account-id"] == "derived_rejected"
    assert "chatgpt-account-id" not in calls[1][1]
    summary = read_native_quota_summary("openai-codex", state_dirs=[state_dir], now=_ts("2026-05-25T12:30:00Z"))
    assert summary is not None
    assert format_native_quota_statusbar(summary, now=_ts("2026-05-25T12:30:00Z")) == "cdx~ 5h 9%↻30m 7d 69%↻5d"


def test_refresh_claude_snapshot_is_hermes_owned(monkeypatch, tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "hermes-state"
    monkeypatch.setenv("HERMES_NATIVE_QUOTA_STATE_DIR", str(state_dir))
    calls = []

    def fake_request(url, headers, timeout):
        calls.append((url, dict(headers), timeout))
        return {
            "five_hour": {"utilization": 9, "resets_at": "2026-05-25T15:00:00Z"},
            "seven_day": {"utilization": 14, "resets_at": "2026-05-30T12:00:00Z"},
            "seven_day_omelette": {"utilization": 51, "resets_at": "2026-05-29T12:00:00Z"},
        }

    path = refresh_native_quota_snapshot(
        "claude-cli",
        now=datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc),
        request_json=fake_request,
        claude_credentials={"accessToken": "claude-token", "source": "test"},
    )

    assert path == state_dir / "provider-native-quota-claude-cli.json"
    assert calls[0][0] == "https://api.anthropic.com/api/oauth/usage"
    assert calls[0][1]["authorization"] == "Bearer claude-token"
    summary = read_native_quota_summary("claude-cli", state_dirs=[state_dir], now=_ts("2026-05-25T12:30:00Z"))
    assert summary is not None
    assert format_native_quota_statusbar(summary, now=_ts("2026-05-25T12:30:00Z")) == "cla~ 5h 9%↻2h30m 7d 51%↻4d"


def test_get_native_quota_statusbar_for_model_returns_empty_for_non_plan_model(tmp_path):
    clear_native_quota_cache()
    state_dir = tmp_path / "state"
    _write_snapshot(state_dir)

    result = get_native_quota_statusbar_for_model(
        "openai",
        "gpt-5.5",
        state_dirs=[state_dir],
        now=_ts("2026-05-25T12:30:00Z"),
    )

    assert result == ""
