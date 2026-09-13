"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

from gateway.runtime_footer import (
    _compact_reset,
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



def test_format_footer_extended_fields_with_quota_and_underline():
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        plan="Team",
        windows=(
            AccountUsageWindow(label="5h", used_percent=20, reset_at=None),
            AccountUsageWindow(label="7d", used_percent=30, reset_at=None),
        ),
    )
    out = format_runtime_footer(
        model="openai/gpt-5.5",
        provider="openai-codex",
        account_label="openai-codex-team-main",
        context_tokens=21_800,
        context_length=200_000,
        account_usage=snapshot,
        reasoning_effort="ultra",
        cwd="",
        fields=("provider", "account", "model", "reasoning", "context", "quota"),
        underline=True,
    )
    assert out == "──────────────\nopenai-codex · team-main · gpt-5.5 · ult · ctx 21.8K/200K · 5h 80% · 7d 70%"


def test_format_footer_compacts_reset_times_for_quota():
    snapshot = AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(
                label="Current session",
                used_percent=32.2,
                reset_at=datetime.now(timezone.utc) + timedelta(hours=2, minutes=10),
            ),
        ),
    )
    out = format_runtime_footer(
        model="anthropic/claude-sonnet-4-6",
        context_tokens=21_800,
        context_length=200_000,
        account_usage=snapshot,
        cwd="",
        fields=("quota",),
    )
    assert out.startswith("5h 68% 2h")


def test_compact_reset_omits_zero_day_and_minute_components(monkeypatch):
    import gateway.runtime_footer as runtime_footer

    fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is not None else fixed_now.replace(tzinfo=None)

    monkeypatch.setattr(runtime_footer, "datetime", FixedDateTime)
    fixed_target = FixedDateTime(2026, 1, 1, tzinfo=timezone.utc)

    assert _compact_reset(fixed_target + timedelta(days=1)) == "1d"
    assert _compact_reset(fixed_target + timedelta(hours=1)) == "1h"
    assert _compact_reset(fixed_target + timedelta(seconds=30)) == "<1m"



def test_format_footer_uses_provider_specific_compact_quota_labels():
    snapshot = AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="Current week", used_percent=40, reset_at=None),
            AccountUsageWindow(label="Opus week", used_percent=50, reset_at=None),
            AccountUsageWindow(label="Sonnet week", used_percent=60, reset_at=None),
        ),
    )
    out = format_runtime_footer(
        model="anthropic/claude-opus-4-5",
        provider="anthropic",
        context_tokens=0,
        context_length=200_000,
        account_usage=snapshot,
        fields=("quota",),
    )
    assert out == "7d 60% · opus7d 50% · sonnet7d 40%"


def test_format_footer_preserves_nonstandard_quota_durations():
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="15d", used_percent=10, reset_at=None),
            AccountUsageWindow(label="17h", used_percent=20, reset_at=None),
        ),
    )

    out = format_runtime_footer(
        model="openai/gpt-5.6-sol",
        provider="openai-codex",
        context_tokens=0,
        context_length=200_000,
        account_usage=snapshot,
        fields=("quota",),
    )

    assert out == "15d 90% · 17h 80%"


def test_format_footer_includes_balance_details_for_quota():
    snapshot = AccountUsageSnapshot(
        provider="deepseek",
        source="balance_api",
        fetched_at=datetime.now(timezone.utc),
        details=("Balance: ¥110.00 CNY (granted ¥10.00, topped up ¥100.00)",),
    )

    out = format_runtime_footer(
        model="deepseek/deepseek-chat",
        provider="deepseek",
        context_tokens=0,
        context_length=200_000,
        account_usage=snapshot,
        fields=("quota",),
    )

    assert out == "balance ¥110.00 CNY"


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
    assert cfg == {"enabled": False, "fields": ["model", "context_pct", "cwd"], "underline": False}


def test_resolve_global_enable():
    user = {"display": {"runtime_footer": {"enabled": True}}}
    cfg = resolve_footer_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["fields"] == ["model", "context_pct", "cwd"]



def test_resolve_supports_underline_flag():
    user = {"display": {"runtime_footer": {"enabled": True, "underline": True}}}
    cfg = resolve_footer_config(user, "feishu")
    assert cfg["underline"] is True


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
    assert tg["fields"] == ["model", "context_pct", "cwd"]
    dc = resolve_footer_config(user, "discord")
    assert dc["enabled"] is True
    assert dc["fields"] == ["context_pct"]


def test_resolve_ignores_malformed_config():
    # Non-dict runtime_footer shouldn't crash
    user = {"display": {"runtime_footer": "on"}}
    cfg = resolve_footer_config(user, "telegram")
    assert cfg["enabled"] is False


# ---------------------------------------------------------------------------


def test_format_footer_reasoning_abbreviations():
    cases = {
        "none": "off",
        "minimal": "min",
        "low": "low",
        "medium": "med",
        "high": "high",
        "xhigh": "xhi",
        "max": "max",
        "ultra": "ult",
        "false": "off",
        "DISABLED": "off",
    }
    for effort, expected in cases.items():
        out = format_runtime_footer(
            model="m",
            context_tokens=0,
            context_length=100,
            reasoning_effort=effort,
            fields=("reasoning",),
        )
        assert out == expected, (effort, out)


def test_format_footer_reasoning_omitted_when_missing():
    out = format_runtime_footer(
        model="openai/gpt-5.5",
        context_tokens=10,
        context_length=100,
        fields=("model", "reasoning"),
    )
    assert out == "gpt-5.5"


def test_build_footer_passes_reasoning_effort():
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "reasoning"],
                }
            }
        },
        platform_key=None,
        model="openai/gpt-5.5",
        context_tokens=0,
        context_length=None,
        reasoning_effort="xhigh",
    )
    assert out == "gpt-5.5 · xhi"


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



def test_build_footer_passes_extended_runtime_metadata():
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        plan="Team",
        windows=(
            AccountUsageWindow(label="5h", used_percent=20, reset_at=None),
            AccountUsageWindow(label="7d", used_percent=30, reset_at=None),
        ),
    )
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "underline": True,
                    "fields": ["provider", "account", "model", "context", "quota"],
                }
            }
        },
        platform_key="feishu",
        model="openai/gpt-5.5",
        provider="openai-codex",
        account_label="team-main",
        context_tokens=18_200,
        context_length=400_000,
        account_usage=snapshot,
        cwd="",
    )
    assert out == "──────────────\nopenai-codex · team-main · gpt-5.5 · ctx 18.2K/400K · 5h 80% · 7d 70%"


def test_build_footer_prefers_profile_scoped_resolved_config():
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": False}}},
        platform_key="telegram",
        resolved_config={"enabled": True, "fields": ["model"], "underline": False},
        model="openai/gpt-5.5",
        context_tokens=0,
        context_length=None,
        cwd="",
    )

    assert out == "gpt-5.5"


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



# ---------------------------------------------------------------------------
# latency — opt-in wall-clock turn duration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "<1s"),
        (0.4, "<1s"),
        (0.999, "<1s"),
        (1.0, "1s"),
        (22.0, "22s"),
        (22.4, "22s"),
        (59.4, "59s"),
        (59.6, "1m00s"),
        (60.0, "1m00s"),
        (65.0, "1m05s"),
        (125.0, "2m05s"),
        (3600.0, "60m00s"),
    ],
)
def test_format_latency(seconds, expected):
    from gateway.runtime_footer import _format_latency

    assert _format_latency(seconds) == expected


def test_format_footer_latency_renders():
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=22.0,
        fields=("latency",),
    )
    assert out == "22s"


def test_format_footer_latency_skipped_when_unmeasured():
    """A call site that doesn't measure timing leaves the field out entirely."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=None,
        fields=("latency",),
    )
    assert out == ""


def test_format_footer_latency_skipped_when_negative():
    """A nonsensical (negative) duration is dropped rather than rendered."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=-1.0,
        fields=("latency",),
    )
    assert out == ""


def test_format_footer_latency_zero_renders_sub_second():
    """Zero is a real measurement (a very fast turn), not missing data."""
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=0.0,
        fields=("latency",),
    )
    assert out == "<1s"


def test_format_footer_latency_in_field_order(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=68_000,
        context_length=100_000,
        cwd=str(tmp_path),
        turn_seconds=65.0,
        fields=("model", "context_pct", "latency", "cwd"),
    )
    assert out == "gpt-5.4 · 68% · 1m05s · ~"


def test_build_footer_line_threads_turn_seconds(monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "latency"],
                }
            }
        },
        platform_key="discord",
        model="gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        turn_seconds=22.0,
    )
    assert out == "gpt-5.4 · 22s"


# ---------------------------------------------------------------------------
# Byte-stability: `latency` is opt-in, so the DEFAULT footer is unchanged.
#
# Upstream doctrine: a system prompt / rendered surface must be byte-stable for
# the life of a conversation.  Adding a field to _DEFAULT_FIELDS would silently
# change the footer text of every user who already enabled it.  These tests pin
# the default set and the exact default-config output strings.
# ---------------------------------------------------------------------------

_LEGACY_DEFAULT_FIELDS = ["model", "context_pct", "cwd"]


def test_latency_not_in_default_fields():
    from gateway.runtime_footer import _DEFAULT_FIELDS

    assert "latency" not in _DEFAULT_FIELDS
    assert list(_DEFAULT_FIELDS) == _LEGACY_DEFAULT_FIELDS


def test_resolve_footer_config_default_fields_exclude_latency():
    assert resolve_footer_config({}, "telegram")["fields"] == _LEGACY_DEFAULT_FIELDS
    assert resolve_footer_config(
        {"display": {"runtime_footer": {"enabled": True}}}, "discord"
    )["fields"] == _LEGACY_DEFAULT_FIELDS


@pytest.mark.parametrize(
    "model,tokens,window,cwd,expected",
    [
        ("openai/gpt-5.4", 50_247, 1_000_000, "/var/data", "gpt-5.4 · 5% · /var/data"),
        ("claude-opus-4-8", 68_000, 100_000, "/var/data", "claude-opus-4-8 · 68% · /var/data"),
        ("m", 0, None, "/var/data", "m · /var/data"),
        ("", 10, 100, "/var/data", "10% · /var/data"),
        ("m", 10, 100, "", "m · 10%"),
    ],
)
def test_default_footer_renders_byte_identically(
    monkeypatch, model, tokens, window, cwd, expected
):
    """Default-config output is byte-for-byte what it was before `latency`.

    Note `turn_seconds` IS supplied — proving that even when the caller
    measures timing, a default-configured footer does not show it.
    """
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    out = format_runtime_footer(
        model=model,
        context_tokens=tokens,
        context_length=window,
        cwd=cwd,
        turn_seconds=22.0,
        # fields deliberately NOT passed — exercises the default.
    )
    assert out == expected


def test_default_build_footer_line_ignores_turn_seconds(monkeypatch):
    """build_footer_line with default fields is unaffected by turn_seconds."""
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    common = dict(
        user_config={"display": {"runtime_footer": {"enabled": True}}},
        platform_key="discord",
        model="openai/gpt-5.4",
        context_tokens=50_247,
        context_length=1_000_000,
        cwd="/var/data",
    )
    baseline = build_footer_line(**common)
    with_timing = build_footer_line(**common, turn_seconds=125.0)
    assert baseline == "gpt-5.4 · 5% · /var/data"
    assert with_timing == baseline


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
