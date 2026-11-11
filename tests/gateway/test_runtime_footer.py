"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

from types import SimpleNamespace

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


# ---------------------------------------------------------------------------
# resolve_footer_config
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# build_footer_line — top-level entry point used by gateway/run.py
# ---------------------------------------------------------------------------


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
# change the footer text of every user who already enabled it.  The test below
# checks default-config output is unaffected by turn timing.
# ---------------------------------------------------------------------------


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


def test_format_footer_served_model_is_opt_in_and_skips_same_model():
    """#54864: `served_model` renders `alias → served` only when listed AND the served model
    differs from the requested one; the default field set never shows it."""
    # Default fields: served model is invisible.
    assert "→" not in format_runtime_footer(
        model="hermes-router", context_tokens=0, context_length=None, cwd="/x",
        served_model="gpt-4o-2024-11-20")
    line = format_runtime_footer(
        model="hermes-router", context_tokens=0, context_length=None, cwd="/x",
        served_model="gpt-4o-2024-11-20", fields=["served_model"])
    assert line == "hermes-router → gpt-4o-2024-11-20"
    # Hermes fallback route: requested primary → active model.
    line = format_runtime_footer(
        model="qwen/qwen3.8-max", context_tokens=0, context_length=None, cwd="/x",
        requested_model="gpt-5.6-sol", served_model="qwen/qwen3.8-max", fields=["served_model"])
    assert line == "gpt-5.6-sol → qwen/qwen3.8-max"
    # Served == requested (no header, no fallback): field skipped, nothing empty rendered.
    assert format_runtime_footer(
        model="gpt-5.4", context_tokens=0, context_length=None, cwd="/x",
        served_model=None, fields=["served_model"]) == ""


# ---------------------------------------------------------------------------
# #19922: Expanded usage metrics in runtime footer
# ---------------------------------------------------------------------------

def test_format_footer_usage_metrics():
    """Verify all usage metrics format and render properly when opted-in."""
    out = format_runtime_footer(
        model="google/gemini-3-flash-preview",
        context_tokens=15_200,
        context_length=1_000_000,
        cwd="/workspace",
        total_tokens=15_200,
        input_tokens=12_300,
        output_tokens=2_900,
        cache_read_tokens=10_000,
        reasoning_tokens=512,
        estimated_cost_usd=0.015,
        api_calls=3,
        turn_seconds=2.4,
        fields=["model", "tokens", "cost", "latency"],
    )
    assert out == "gemini-3-flash-preview · 15.2k tokens · ~$0.0150 · 2s"


def test_format_footer_detailed_breakdown_fields():
    """Verify in, out, cache, reasoning, and api_calls fields."""
    out = format_runtime_footer(
        model="gpt-5.4",
        context_tokens=1_048_576,
        context_length=2_000_000,
        total_tokens=1_048_576,
        input_tokens=1_000_000,
        output_tokens=48_576,
        cache_read_tokens=800_000,
        reasoning_tokens=16_000,
        api_calls=1,
        fields=["in", "out", "cache", "reasoning", "api_calls"],
    )
    assert out == "in: 1M · out: 48.6k · cache: 800k · reasoning: 16k · 1 call"


def test_format_footer_cost_edge_cases():
    """Verify cost formatting for zero and sub-cent values."""
    assert format_runtime_footer(
        model="m", context_tokens=0, context_length=None,
        estimated_cost_usd=0.0, fields=["cost"],
    ) == "$0.00"

    assert format_runtime_footer(
        model="m", context_tokens=0, context_length=None,
        estimated_cost_usd=0.00004, fields=["cost"],
    ) == "<$0.0001"


def test_format_footer_usage_metrics_byte_stability():
    """Default fields must ignore all usage metrics to preserve byte stability."""
    common = dict(
        model="openai/gpt-5.4",
        context_tokens=50_000,
        context_length=100_000,
        cwd="/var/data",
        total_tokens=50_000,
        input_tokens=40_000,
        output_tokens=10_000,
        cache_read_tokens=30_000,
        reasoning_tokens=2_000,
        estimated_cost_usd=0.05,
        api_calls=2,
    )
    baseline = format_runtime_footer(
        model="openai/gpt-5.4", context_tokens=50_000, context_length=100_000, cwd="/var/data",
    )
    with_metrics = format_runtime_footer(**common)
    assert baseline == with_metrics


def test_build_footer_line_threads_usage_metrics():
    """build_footer_line correctly passes usage metrics to format_runtime_footer."""
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "tokens", "cost", "api_calls"],
                }
            }
        },
        platform_key=None,
        model="anthropic/claude-sonnet-4.6",
        context_tokens=10_000,
        context_length=200_000,
        total_tokens=10_000,
        estimated_cost_usd=0.03,
        api_calls=5,
    )
    assert out == "claude-sonnet-4.6 · 10k tokens · ~$0.0300 · 5 calls"


# ---------------------------------------------------------------------------
# Gateway call-site wiring: agent_result -> build_footer_line
# ---------------------------------------------------------------------------

def test_gateway_call_site_forwards_usage_counters(monkeypatch, tmp_path):
    """The formatter can only render what the call site passes; guard the wiring.

    Regression this catches: ``build_footer_line`` grew the usage/billing kwargs while
    ``_hmwa_runtime_footer_line`` still passed only the original six, so every new field
    stayed empty in production while the formatter tests stayed green.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import gateway.run as gateway_run
    from gateway import runtime_footer, session_finite
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin

    captured: dict[str, object] = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return ""

    monkeypatch.setattr(runtime_footer, "build_footer_line", _capture)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", dict)
    monkeypatch.setattr(gateway_run, "_platform_config_key", lambda _p: "telegram")
    monkeypatch.setattr(gateway_run, "_terminal_scope_cwd", lambda _s: "/tmp")
    monkeypatch.setattr(session_finite, "finite_turn_required", lambda: False)

    agent_result = {
        "model": "anthropic/claude-sonnet-4.6",
        "last_prompt_tokens": 1200,
        "context_length": 200_000,
        "total_tokens": 1500,
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_read_tokens": 900,
        "cache_write_tokens": 50,
        "reasoning_tokens": 40,
        "estimated_cost_usd": 0.0123,
        "api_calls": 4,
    }

    GatewayTurnPersistenceMixin._hmwa_runtime_footer_line(
        None, agent_result, SimpleNamespace(platform=object()), 2.5
    )

    assert captured["total_tokens"] == 1500
    assert captured["input_tokens"] == 1200
    assert captured["output_tokens"] == 300
    assert captured["cache_read_tokens"] == 900
    assert captured["cache_write_tokens"] == 50
    assert captured["reasoning_tokens"] == 40
    assert captured["estimated_cost_usd"] == pytest.approx(0.0123)
    assert captured["api_calls"] == 4
    assert captured["turn_seconds"] == 2.5

