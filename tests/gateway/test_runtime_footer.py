"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os

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
    assert cfg == {"enabled": False, "fields": ["model", "context_pct", "cwd"]}


def test_resolve_global_enable():
    user = {"display": {"runtime_footer": {"enabled": True}}}
    cfg = resolve_footer_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["fields"] == ["model", "context_pct", "cwd"]


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


# ---------------------------------------------------------------------------
# tps field (#26877)
# ---------------------------------------------------------------------------


class TestTpsField:
    def test_tps_renders_decimal_below_100(self):
        # 50 tokens in 2 seconds = 25 t/s — under 100, decimal format.
        out = format_runtime_footer(
            model=None, context_tokens=0, context_length=None,
            fields=("tps",),
            response_tokens=50,
            elapsed_ms=2000.0,
        )
        assert out == "25.0t/s"

    def test_tps_renders_integer_at_or_above_100(self):
        # 500 tokens in 2 seconds = 250 t/s — integer when ≥100 (rate this
        # large is dominated by network/integer-scale noise).
        out = format_runtime_footer(
            model=None, context_tokens=0, context_length=None,
            fields=("tps",),
            response_tokens=500,
            elapsed_ms=2000.0,
        )
        assert out == "250t/s"

    def test_tps_skipped_when_response_tokens_zero(self):
        out = format_runtime_footer(
            model="gpt-5", context_tokens=0, context_length=None,
            fields=("tps",),
            response_tokens=0,
            elapsed_ms=2000.0,
        )
        assert out == ""

    def test_tps_skipped_when_response_tokens_none(self):
        out = format_runtime_footer(
            model="gpt-5", context_tokens=0, context_length=None,
            fields=("tps",),
            response_tokens=None,
            elapsed_ms=2000.0,
        )
        assert out == ""

    def test_tps_skipped_when_elapsed_too_small(self):
        # <50ms elapsed → degenerate denominator; skip to avoid wild numbers.
        out = format_runtime_footer(
            model=None, context_tokens=0, context_length=None,
            fields=("tps",),
            response_tokens=10,
            elapsed_ms=5.0,
        )
        assert out == ""

    def test_tps_skipped_when_elapsed_none(self):
        out = format_runtime_footer(
            model=None, context_tokens=0, context_length=None,
            fields=("tps",),
            response_tokens=50,
            elapsed_ms=None,
        )
        assert out == ""

    def test_tps_joins_with_other_fields(self):
        out = format_runtime_footer(
            model="openai/gpt-5.4",
            context_tokens=512, context_length=2048,
            cwd="",
            fields=("model", "context_pct", "tps"),
            response_tokens=80,
            elapsed_ms=4000.0,
        )
        # gpt-5.4 · 25% · 20.0t/s
        assert "gpt-5.4" in out
        assert "25%" in out
        assert "20.0t/s" in out
        assert out.count(" · ") == 2

    def test_tps_omitted_when_not_in_fields_list(self):
        # User keeps the default ``[model, context_pct, cwd]`` list — passing
        # response_tokens/elapsed_ms is a no-op, footer stays unchanged.
        baseline = format_runtime_footer(
            model="openai/gpt-5.4",
            context_tokens=512, context_length=2048,
            cwd="",
            fields=("model", "context_pct"),
        )
        with_data = format_runtime_footer(
            model="openai/gpt-5.4",
            context_tokens=512, context_length=2048,
            cwd="",
            fields=("model", "context_pct"),
            response_tokens=80,
            elapsed_ms=4000.0,
        )
        assert baseline == with_data
        assert "t/s" not in with_data

    def test_build_footer_line_threads_kwargs_through(self):
        out = build_footer_line(
            user_config={
                "display": {
                    "runtime_footer": {
                        "enabled": True,
                        "fields": ["model", "tps"],
                    }
                }
            },
            platform_key="telegram",
            model="openai/gpt-5.4",
            context_tokens=0, context_length=None,
            cwd="",
            response_tokens=60,
            elapsed_ms=3000.0,
        )
        assert "gpt-5.4" in out
        # 60 / 3 = 20 t/s
        assert "20.0t/s" in out

    def test_build_footer_line_old_callers_unaffected(self):
        # Existing callers that don't pass response_tokens/elapsed_ms still get
        # exactly today's footer.  Important for backwards-compat with anything
        # outside gateway/run.py that already invokes build_footer_line.
        out = build_footer_line(
            user_config={
                "display": {
                    "runtime_footer": {
                        "enabled": True,
                        "fields": ["model", "context_pct"],
                    }
                }
            },
            platform_key="cli",
            model="openai/gpt-5.4",
            context_tokens=100, context_length=400,
            cwd="",
        )
        assert "gpt-5.4" in out
        assert "25%" in out
        assert "t/s" not in out
