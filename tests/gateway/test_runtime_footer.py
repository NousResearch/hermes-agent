"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os

import pytest

from gateway.runtime_footer import (
    _home_relative_cwd,
    _model_short,
    _reasoning_short,
    build_footer_line,
    format_runtime_footer,
    outgoing_fast_applied,
    provider_consumes_credits,
    reasoning_effort_label,
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


# ---------------------------------------------------------------------------
# Credits: catalog Accounts wins over metered hosts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,base_url,expected",
    [
        ("openrouter", None, True),
        ("openai", "https://openrouter.ai/api/v1", True),
        ("xai", "https://api.x.ai/v1", True),
        ("xai-oauth", "https://api.x.ai/v1", False),
        ("xai-oauth", None, False),
        ("openai-codex", "https://api.openai.com/v1", False),
        ("nous", None, False),
        ("qwen-oauth", "https://portal.qwen.ai/v1", False),
        ("minimax-oauth", None, False),
        ("ollama", "http://127.0.0.1:11434/v1", False),
        (None, None, False),
    ],
)
def test_provider_consumes_credits_follows_catalog(provider, base_url, expected):
    assert provider_consumes_credits(provider, base_url) is expected


def test_format_footer_does_not_warn_for_xai_oauth_on_api_host():
    out = format_runtime_footer(
        model="grok-4.6",
        provider="xai-oauth",
        base_url="https://api.x.ai/v1",
        context_tokens=10,
        context_length=100,
        fields=("model", "context_pct"),
    )
    assert out == "grok-4.6 · 10%"
    assert "consuming credits" not in out


def test_format_footer_warns_for_xai_api_key_on_same_host():
    out = format_runtime_footer(
        model="grok-4.6",
        provider="xai",
        base_url="https://api.x.ai/v1",
        context_tokens=10,
        context_length=100,
        fields=("model", "context_pct"),
    )
    assert out == "grok-4.6 · 10%\n💸 consuming credits"


def test_format_footer_warns_for_openrouter():
    out = format_runtime_footer(
        model="gpt-5.6-sol",
        provider="openrouter",
        context_tokens=37,
        context_length=100,
        reasoning_effort="medium",
        fields=("model", "reasoning_effort", "context_pct"),
        separator=" • ",
    )
    assert out == "gpt-5.6-sol • 🧠 med • 37%\n💸 consuming credits"


# ---------------------------------------------------------------------------
# Reasoning + Fast fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,effort,expected",
    [
        ("reasoning_effort", "high", "gpt-5.4 · 🧠 high"),
        ("reasoning", "xhigh", "gpt-5.4 · 🧠 xhigh"),
        ("reasoning_effort", "medium", "gpt-5.4 · 🧠 med"),
        ("reasoning_effort", "minimal", "gpt-5.4 · 🧠 min"),
        ("reasoning_effort", "none", "gpt-5.4 · 🧠 off"),
        ("reasoning_effort", None, "gpt-5.4"),
    ],
)
def test_format_footer_reasoning_effort(field, effort, expected):
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=100,
        cwd="",
        reasoning_effort=effort,
        fields=("model", field),
    )
    assert out == expected


@pytest.mark.parametrize(
    "config,expected",
    [
        (None, "medium"),
        ({"enabled": False, "effort": "high"}, "none"),
        ({"enabled": True, "effort": "XHIGH"}, "xhigh"),
        ({"enabled": True}, "medium"),
    ],
)
def test_reasoning_effort_label(config, expected):
    assert reasoning_effort_label(config) == expected


def test_reasoning_short():
    assert _reasoning_short("minimal") == "min"
    assert _reasoning_short("medium") == "med"
    assert _reasoning_short("high") == "high"


def test_format_footer_fast_decorates_model_when_applied():
    out = format_runtime_footer(
        model="openai-codex/gpt-5.6-sol",
        context_tokens=37,
        context_length=100,
        cwd="/opt/data/repos/ai-life",
        reasoning_effort="medium",
        fast_mode=True,
        fields=("model", "reasoning_effort", "fast", "context_pct", "dir"),
        separator=" • ",
    )
    assert out == "gpt-5.6-sol ⚡️ • 🧠 med • 37% • ai-life"


def test_format_footer_fast_hidden_when_not_applied():
    out = format_runtime_footer(
        model="gpt-5.6-sol",
        context_tokens=37,
        context_length=100,
        reasoning_effort="low",
        fast_mode=False,
        fields=("model", "reasoning_effort", "fast", "context_pct"),
        separator=" • ",
    )
    assert out == "gpt-5.6-sol • 🧠 low • 37%"


def test_outgoing_fast_sol_when_requested():
    assert outgoing_fast_applied(
        model="gpt-5.6-sol", provider="openai-codex", requested=True
    ) is True


def test_outgoing_fast_not_applied_when_not_requested():
    assert outgoing_fast_applied(
        model="gpt-5.6-sol", provider="openai-codex", requested=False
    ) is False


def test_outgoing_fast_grok46_keeps_priority():
    assert outgoing_fast_applied(
        model="grok-4.6", provider="xai-oauth", requested=True
    ) is True


def test_outgoing_fast_older_grok_stripped():
    assert outgoing_fast_applied(
        model="grok-4.5", provider="xai-oauth", requested=True
    ) is False


def test_format_footer_grok_oauth_stale_fast_uses_effective_bit():
    out = format_runtime_footer(
        model="grok-4.5",
        provider="xai-oauth",
        base_url="https://api.x.ai/v1",
        context_tokens=40,
        context_length=100,
        reasoning_effort="high",
        fast_mode=outgoing_fast_applied(
            model="grok-4.5", provider="xai-oauth", requested=True
        ),
        fields=("model", "reasoning_effort", "fast", "context_pct"),
        separator=" • ",
    )
    assert out == "grok-4.5 • 🧠 high • 40%"
