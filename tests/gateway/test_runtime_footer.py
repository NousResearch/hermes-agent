"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os

import pytest

from gateway.runtime_footer import (
    _home_relative_cwd,
    _model_short,
    apply_runtime_prefix,
    build_footer_line,
    build_prefix_line,
    build_runtime_prefix,
    format_runtime_footer,
    format_runtime_prefix,
    resolve_footer_config,
    resolve_prefix_config,
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
# runtime_prefix — first-line model markers
# ---------------------------------------------------------------------------


def test_resolve_prefix_global_enable_and_label():
    user = {
        "display": {
            "runtime_prefix": {
                "enabled": True,
                "labels": {"gpt-5.5": "[gpt5.5-custom]"},
            }
        }
    }
    cfg = resolve_prefix_config(user, "slack")
    assert cfg["enabled"] is True
    assert cfg["labels"]["gpt-5.5"] == "[gpt5.5-custom]"


def test_resolve_prefix_platform_override_wins():
    user = {
        "display": {
            "runtime_prefix": {"enabled": False, "labels": {"gpt": "[gpt]"}},
            "platforms": {"slack": {"runtime_prefix": {"enabled": True}}},
        }
    }
    cfg = resolve_prefix_config(user, "slack")
    assert cfg["enabled"] is True
    assert cfg["labels"]["gpt"] == "[gpt]"


def test_resolve_prefix_platform_label_extends_global_labels():
    user = {
        "display": {
            "runtime_prefix": {"enabled": True, "labels": {"gpt": "[gpt]"}},
            "platforms": {
                "telegram": {"runtime_prefix": {"labels": {"gpt-5.5": "[tg-gpt]"}}}
            },
        }
    }
    cfg = resolve_prefix_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["labels"]["gpt"] == "[gpt]"
    assert cfg["labels"]["gpt-5.5"] == "[tg-gpt]"


def test_resolve_prefix_accepts_legacy_label_aliases():
    user = {
        "display": {
            "runtime_prefix": {"enabled": True, "map": {"gpt-5.5": "[map-gpt]"}},
            "platforms": {
                "telegram": {"runtime_prefix": {"markers": {"glm-5.2": "[marker-glm]"}}}
            },
        }
    }
    cfg = resolve_prefix_config(user, "telegram")
    assert cfg["enabled"] is True
    assert cfg["labels"]["gpt-5.5"] == "[map-gpt]"
    assert cfg["labels"]["glm-5.2"] == "[marker-glm]"


def test_build_runtime_prefix_uses_exact_configured_label():
    assert (
        build_runtime_prefix(
            user_config={
                "display": {
                    "runtime_prefix": {
                        "enabled": True,
                        "labels": {"gpt-5.5": "[gpt5.5-custom]"},
                    }
                }
            },
            platform_key="slack",
            model="gpt-5.5",
        )
        == "[gpt5.5-custom]"
    )


def test_build_runtime_prefix_uses_default_label_when_enabled_without_label():
    assert (
        build_runtime_prefix(
            user_config={"display": {"runtime_prefix": {"enabled": True}}},
            platform_key="slack",
            model="openai/gpt-5.5",
        )
        == "[gpt5.5]"
    )


def test_build_runtime_prefix_empty_when_disabled():
    assert (
        build_runtime_prefix(
            user_config={},
            platform_key="slack",
            model="gpt-5.5",
        )
        == ""
    )


def test_format_runtime_prefix_defaults_grok_and_glm():
    assert format_runtime_prefix(model="grok-composer-2.5-fast") == "[grok]"
    assert format_runtime_prefix(model="glm-5.1") == "[glm]"
    assert format_runtime_prefix(model="glm-5.2") == "[glm2]"
    assert format_runtime_prefix(model="gpt-5.5") == "[gpt5.5]"


def test_format_runtime_prefix_longest_key_wins():
    assert (
        format_runtime_prefix(
            model="grok-composer-2.5-fast",
            labels={"grok": "[generic]", "grok-composer": "[composer]"},
        )
        == "[composer]"
    )


def test_apply_runtime_prefix_prepends_once():
    assert apply_runtime_prefix("hello", "[gpt5.5]") == "[gpt5.5] hello"
    assert apply_runtime_prefix("[gpt5.5] hello", "[gpt5.5]") == "[gpt5.5] hello"
    assert apply_runtime_prefix("  [gpt5.5] hello", "[gpt5.5]") == "  [gpt5.5] hello"


def test_build_prefix_line_compatibility_alias():
    assert (
        build_prefix_line(
            user_config={"display": {"runtime_prefix": {"enabled": True}}},
            platform_key="telegram",
            model="glm-5.1",
        )
        == "[glm]"
    )


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
