"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from gateway.runtime_footer import (
    _home_relative_cwd,
    _model_short,
    build_footer_line,
    format_runtime_footer,
    resolve_footer_config,
)


# ---------------------------------------------------------------------------
# _model_short + _home_relative_cwd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("gpt-5.4", "gpt-5.4"),
        ("", ""),
        (None, ""),
    ],
)
def test_model_short_drops_vendor_prefix(model, expected):
    assert _model_short(model) == expected


def test_home_relative_cwd_collapses_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sub = tmp_path / "projects" / "hermes"
    sub.mkdir(parents=True)
    result = _home_relative_cwd(str(sub))
    assert result == "~/projects/hermes"


def test_home_relative_cwd_leaves_abs_path_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "other"))
    result = _home_relative_cwd(str(tmp_path / "outside" / "dir"))
    assert result == str(tmp_path / "outside" / "dir")


def test_home_relative_cwd_empty_returns_empty():
    assert _home_relative_cwd("") == ""


# ---------------------------------------------------------------------------
# format_runtime_footer
# ---------------------------------------------------------------------------

def test_format_footer_all_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "projects" / "hermes"))
    (tmp_path / "projects" / "hermes").mkdir(parents=True)
    out = format_runtime_footer(
        model="openrouter/openai/gpt-5.4",
        context_tokens=68000,
        context_length=100000,
        cwd=None,  # falls back to TERMINAL_CWD env var
        fields=("model", "context_pct", "cwd"),
    )
    assert out == "gpt-5.4 · 68% · ~/projects/hermes"


@dataclass(frozen=True)
class _UsageWindow:
    label: str
    used_percent: float
    reset_at: datetime | None = None


@dataclass(frozen=True)
class _UsageSnapshot:
    windows: tuple[_UsageWindow, ...]


def test_format_footer_rich_fields_with_account_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    usage = _UsageSnapshot(
        (
            _UsageWindow("Session", 14.0, datetime.now(timezone.utc) + timedelta(hours=3)),
            _UsageWindow("Weekly", 29.0, datetime.now(timezone.utc) + timedelta(days=2)),
        )
    )
    out = format_runtime_footer(
        model="openai/gpt-5.5",
        context_tokens=214000,
        context_length=258000,
        cwd=str(tmp_path),
        fields=("model", "context", "session_limit", "weekly_limit", "cwd"),
        account_usage=usage,
    )
    assert "gpt-5.5" in out
    assert "ctx 214k/258k 83%" in out
    assert "sess 86% left/" in out
    assert "week 71% left/" in out
    assert out.endswith("~")


def test_format_footer_skips_missing_context_length():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=500,
        context_length=None,
        cwd="/tmp/wd",
        fields=("model", "context_pct", "cwd"),
    )
    # context_pct dropped silently; no "?%" artifact
    assert "%" not in out
    assert "gpt-5.4" in out
    assert "/tmp/wd" in out


def test_format_footer_context_pct_clamped_to_100():
    out = format_runtime_footer(
        model="m",
        context_tokens=500_000,  # way over
        context_length=100_000,
        cwd="",
        fields=("context_pct",),
    )
    assert out == "100%"


def test_format_footer_context_pct_never_negative():
    out = format_runtime_footer(
        model="m",
        context_tokens=-50,
        context_length=100,
        cwd="",
        fields=("context_pct",),
    )
    # Negative input => no field emitted (we require context_tokens >= 0)
    assert out == ""


def test_format_footer_empty_fields_returns_empty():
    out = format_runtime_footer(
        model="m", context_tokens=0, context_length=100,
        cwd="/x", fields=(),
    )
    assert out == ""


def test_format_footer_drops_cwd_when_empty(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=50, context_length=100,
        cwd="",
        fields=("model", "context_pct", "cwd"),
    )
    # cwd silently dropped; model + pct remain
    assert out == "gpt-5.4 · 50%"


def test_format_footer_custom_field_order():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=50, context_length=100,
        cwd="/opt/project",
        fields=("context_pct", "model"),  # swapped + no cwd
    )
    assert out == "50% · gpt-5.4"


def test_format_footer_unknown_field_silently_ignored():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=50, context_length=100,
        cwd="/x",
        fields=("model", "bogus", "context_pct"),
    )
    assert out == "gpt-5.4 · 50%"


# ---------------------------------------------------------------------------
# resolve_footer_config
# ---------------------------------------------------------------------------

def test_resolve_defaults_off_empty_config():
    cfg = resolve_footer_config({}, "telegram")
    assert cfg == {
        "enabled": False,
        "fields": ["model", "context", "session_limit", "weekly_limit", "cwd"],
        "usage_cache_seconds": 300,
        "usage_timeout_seconds": 2,
    }


def test_resolve_global_enable():
    user = {"display": {"runtime_footer": {"enabled": True}}}
    cfg = resolve_footer_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["fields"] == ["model", "context", "session_limit", "weekly_limit", "cwd"]


def test_resolve_platform_override_wins():
    user = {
        "display": {
            "runtime_footer": {"enabled": True, "fields": ["model"]},
            "platforms": {
                "slack": {"runtime_footer": {"enabled": False}},
            },
        },
    }
    # Telegram picks up the global enable
    assert resolve_footer_config(user, "telegram")["enabled"] is True
    # Slack overrides to off
    assert resolve_footer_config(user, "slack")["enabled"] is False


def test_resolve_platform_can_add_fields_only():
    user = {
        "display": {
            "runtime_footer": {"enabled": True},
            "platforms": {
                "discord": {"runtime_footer": {"fields": ["context_pct"]}},
            },
        },
    }
    tg = resolve_footer_config(user, "telegram")
    assert tg["enabled"] is True
    assert tg["fields"] == ["model", "context", "session_limit", "weekly_limit", "cwd"]
    dc = resolve_footer_config(user, "discord")
    assert dc["enabled"] is True
    assert dc["fields"] == ["context_pct"]


def test_resolve_ignores_malformed_config():
    # Non-dict runtime_footer shouldn't crash
    user = {"display": {"runtime_footer": "on"}}
    cfg = resolve_footer_config(user, "telegram")
    assert cfg["enabled"] is False


# ---------------------------------------------------------------------------
# build_footer_line — top-level entry point used by gateway/run.py
# ---------------------------------------------------------------------------

def test_build_footer_empty_when_disabled():
    out = build_footer_line(
        user_config={},
        platform_key="telegram",
        model="openai/gpt-5.4",
        context_tokens=10, context_length=100,
        cwd="/tmp",
    )
    assert out == ""


def test_build_footer_returns_rendered_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True}}},
        platform_key="telegram",
        model="openai/gpt-5.4",
        context_tokens=25, context_length=100,
        cwd=str(tmp_path / "proj"),
    )
    (tmp_path / "proj").mkdir(exist_ok=True)
    assert "gpt-5.4" in out
    assert "25%" in out


def test_build_footer_per_platform_off_suppresses():
    user = {
        "display": {
            "runtime_footer": {"enabled": True},
            "platforms": {"slack": {"runtime_footer": {"enabled": False}}},
        },
    }
    out = build_footer_line(
        user_config=user,
        platform_key="slack",
        model="openai/gpt-5.4",
        context_tokens=10, context_length=100,
        cwd="/tmp",
    )
    assert out == ""


def test_build_footer_no_data_returns_empty_even_when_enabled():
    # Enabled, but context_length is None AND cwd empty AND model empty ⇒ no fields
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True}}},
        platform_key="telegram",
        model="",
        context_tokens=0, context_length=None,
        cwd="",
    )
    # With no TERMINAL_CWD env either
    if not os.environ.get("TERMINAL_CWD"):
        assert out == ""


def test_build_footer_fetches_usage_from_config_route_when_result_omits_provider(monkeypatch):
    # Some gateway result paths historically omitted provider/base_url even
    # though the live config had a valid Codex route.  Without this fallback the
    # footer silently dropped both session and weekly limit fields.
    from agent import account_usage
    from gateway import runtime_footer

    runtime_footer._USAGE_CACHE.clear()
    calls = []

    def fake_fetch(provider, *, base_url=None, api_key=None):
        calls.append((provider, base_url, bool(api_key)))
        return _UsageSnapshot(
            (
                _UsageWindow("Session", 14.0, None),
                _UsageWindow("Weekly", 29.0, None),
            )
        )

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)
    out = build_footer_line(
        user_config={
            "model": {
                "provider": "openai-codex",
                "base_url": "https://chatgpt.com/backend-api/codex",
            },
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["session_limit", "weekly_limit"],
                    "usage_cache_seconds": 0,
                }
            },
        },
        platform_key="telegram",
        model="gpt-5.5",
        context_tokens=10,
        context_length=100,
        cwd="",
        provider=None,
        base_url=None,
    )

    assert calls == [("openai-codex", "https://chatgpt.com/backend-api/codex", False)]
    assert out == "sess 86% left · week 71% left"


def test_transient_usage_miss_keeps_last_good_snapshot(monkeypatch):
    # A single transient usage-API miss (timeout / rate-limit / empty) must not
    # blank the limit fields for the whole cache window. The footer should ride
    # out the miss by serving the last good snapshot and retry again soon.
    from agent import account_usage
    from gateway import runtime_footer

    runtime_footer._USAGE_CACHE.clear()

    seq = [
        _UsageSnapshot((_UsageWindow("Session", 14.0, None), _UsageWindow("Weekly", 29.0, None))),
        None,  # transient miss
        None,  # transient miss
        _UsageSnapshot((_UsageWindow("Session", 20.0, None), _UsageWindow("Weekly", 31.0, None))),
    ]
    idx = {"i": 0}

    def fake_fetch(provider, *, base_url=None, api_key=None):
        i = idx["i"]
        idx["i"] += 1
        return seq[i] if i < len(seq) else seq[-1]

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    cfg = {
        "model": {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"},
        "display": {
            "runtime_footer": {
                "enabled": True,
                "fields": ["session_limit", "weekly_limit"],
                # 0 cache so each call attempts a fresh fetch; the negative-TTL
                # ride-out path is what must preserve the last good data.
                "usage_cache_seconds": 0,
                "usage_timeout_seconds": 2,
            }
        },
    }

    def render():
        return build_footer_line(
            user_config=cfg,
            platform_key="telegram",
            model="gpt-5.5",
            context_tokens=10,
            context_length=100,
            cwd="",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
        )

    first = render()
    assert first == "sess 86% left · week 71% left"
    # Two transient misses in a row — must still show the last good numbers.
    assert render() == "sess 86% left · week 71% left"
    assert render() == "sess 86% left · week 71% left"
    # Recovery — fresh good data flows through again.
    assert render() == "sess 80% left · week 69% left"
