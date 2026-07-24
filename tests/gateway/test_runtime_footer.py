"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os

import pytest

from gateway.runtime_footer import (
    _home_relative_cwd,
    _humanize_tok,
    _model_short,
    _split_provider_model,
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


def test_format_footer_unknown_field_silently_ignored():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=50, context_length=100,
        cwd="/x",
        fields=("model", "bogus", "context_pct"),
    )
    assert out == "gpt-5.4 · 50%"


# ---------------------------------------------------------------------------
# provider_model + context_full (new fields)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("claude-bridge-f3", "claude-opus-4-8", ("claude-bridge-f3", "claude-opus-4-8")),
        # provider unset, model carries the prefix -> split it
        ("", "claude-bridge-f3/claude-opus-4-8", ("claude-bridge-f3", "claude-opus-4-8")),
        (None, "openai/gpt-5.4", ("openai", "gpt-5.4")),
        # bare model, no provider anywhere
        ("", "gpt-5.4", ("", "gpt-5.4")),
        (None, None, ("", "")),
        # BOTH provider given AND model carries a prefix -> model's prefix wins
        # (no ugly triple openai-codex/claude-app/claude-opus-4-8)
        ("openai-codex", "claude-app/claude-opus-4-8", ("claude-app", "claude-opus-4-8")),
    ],
)
def test_split_provider_model(provider, model, expected):
    assert _split_provider_model(provider, model) == expected


@pytest.mark.parametrize(
    "n,expected",
    [
        (50_000, "50k"),
        (1_500, "1.5k"),
        (1_000_000, "1M"),
        (1_048_576, "1.0M"),
        (500, "500"),
        (0, "0"),
        (None, "0"),
    ],
)
def test_humanize_tok(n, expected):
    assert _humanize_tok(n) == expected


def test_format_footer_provider_model_explicit():
    out = format_runtime_footer(
        model="claude-opus-4-8",
        provider="claude-bridge-f3",
        context_tokens=0, context_length=None,
        cwd="",
        fields=("provider_model",),
    )
    assert out == "claude-bridge-f3/claude-opus-4-8"


def test_format_footer_provider_model_split_from_prefixed_model():
    # provider unset; model carries the prefix
    out = format_runtime_footer(
        model="claude-bridge-f3/claude-opus-4-8",
        provider=None,
        context_tokens=0, context_length=None,
        cwd="",
        fields=("provider_model",),
    )
    assert out == "claude-bridge-f3/claude-opus-4-8"


def test_format_footer_provider_model_bare_model_only():
    # no provider anywhere -> just the model, no leading slash
    out = format_runtime_footer(
        model="gpt-5.4",
        provider="",
        context_tokens=0, context_length=None,
        cwd="",
        fields=("provider_model",),
    )
    assert out == "gpt-5.4"


def test_format_footer_context_full():
    # both used and window humanized (50.2k/1M)
    out = format_runtime_footer(
        model="m",
        context_tokens=50_247,
        context_length=1_000_000,
        cwd="",
        fields=("context_full",),
    )
    assert out == "50.2k/1M (5%)"


def test_format_footer_context_full_no_window_shows_used_only():
    out = format_runtime_footer(
        model="m",
        context_tokens=50_247,
        context_length=None,
        cwd="",
        fields=("context_full",),
    )
    assert out == "50.2k"


def test_format_footer_context_full_no_data_dropped():
    out = format_runtime_footer(
        model="m",
        context_tokens=0,
        context_length=None,
        cwd="",
        fields=("context_full",),
    )
    assert out == ""


def test_format_footer_aces_target_layout(monkeypatch, tmp_path):
    # The exact footer Ace asked for — both counts humanized:
    #   claude-bridge-f3/claude-opus-4-8 · 50.2k/1M (5%) · ~
    monkeypatch.setenv("HOME", str(tmp_path))
    out = format_runtime_footer(
        model="claude-opus-4-8",
        provider="claude-bridge-f3",
        context_tokens=50_247,
        context_length=1_000_000,
        cwd=str(tmp_path),
        fields=("provider_model", "context_full", "cwd"),
    )
    assert out == "claude-bridge-f3/claude-opus-4-8 · 50.2k/1M (5%) · ~"


# ---------------------------------------------------------------------------
# reasoning field
# ---------------------------------------------------------------------------

def test_format_footer_reasoning_renders_with_prefix():
    out = format_runtime_footer(
        model="m", context_tokens=0, context_length=None, cwd="",
        reasoning="xhigh",
        fields=("reasoning",),
    )
    assert out == "r:xhigh"


def test_format_footer_reasoning_skipped_when_empty():
    out = format_runtime_footer(
        model="m", context_tokens=0, context_length=None, cwd="",
        reasoning="",
        fields=("reasoning",),
    )
    assert out == ""


def test_build_footer_reasoning_resolved_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # agent.reasoning_effort is the canonical source /reasoning writes to
    out = build_footer_line(
        user_config={
            "display": {"runtime_footer": {"enabled": True, "fields": ["reasoning"]}},
            "agent": {"reasoning_effort": "high"},
        },
        platform_key="discord",
        model="claude-opus-4-8",
        provider="claude-bridge-f3",
        context_tokens=0, context_length=None,
        cwd="",
    )
    assert out == "r:high"


def test_build_footer_reasoning_absent_config_drops_field(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["reasoning"]}}},
        platform_key="discord",
        model="m",
        context_tokens=0, context_length=None,
        cwd="",
    )
    # no agent.reasoning_effort -> field silently dropped -> empty footer
    assert out == ""


def test_format_footer_full_layout_with_reasoning(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = format_runtime_footer(
        model="claude-opus-4-8",
        provider="claude-bridge-f3",
        context_tokens=50_247,
        context_length=1_000_000,
        cwd=str(tmp_path),
        reasoning="xhigh",
        fields=("provider_model", "context_full", "reasoning", "cwd"),
    )
    assert out == "claude-bridge-f3/claude-opus-4-8 · 50.2k/1M (5%) · r:xhigh · ~"


# ---------------------------------------------------------------------------
# resolve_footer_config
# ---------------------------------------------------------------------------

def test_resolve_defaults_off_empty_config():
    cfg = resolve_footer_config({}, "telegram")
    assert cfg == {
        "enabled": False,
        "fields": ["provider_model", "context_full", "reasoning", "cwd"],
    }


def test_resolve_global_enable():
    user = {"display": {"runtime_footer": {"enabled": True}}}
    cfg = resolve_footer_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["fields"] == ["provider_model", "context_full", "reasoning", "cwd"]


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
    assert tg["fields"] == ["provider_model", "context_full", "reasoning", "cwd"]
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
