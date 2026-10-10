"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations


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
# reasoning — opt-in session effort label (#61634 sibling: never present an
# internal level as a wire level the route does not have).
# ---------------------------------------------------------------------------


def test_format_footer_reasoning_is_opt_in():
    """The default field set never renders the effort label."""
    assert "reasoning" not in format_runtime_footer(
        model="m", context_tokens=0, context_length=None, cwd="/x", reasoning="medium")
    assert format_runtime_footer(
        model="m", context_tokens=0, context_length=None, cwd="/x",
        reasoning="medium", fields=["reasoning"]) == "reasoning medium"


def test_format_footer_reasoning_skips_blank():
    """No label (None) or a blank one renders nothing — never a dangling ``reasoning``."""
    for blank in (None, "", "   "):
        assert format_runtime_footer(
            model="m", context_tokens=0, context_length=None, cwd="/x",
            reasoning=blank, fields=["reasoning"]) == ""


def test_format_footer_reasoning_in_field_order():
    out = format_runtime_footer(
        model="openai/gpt-5.4", context_tokens=68_000, context_length=100_000,
        cwd="/var/data", turn_seconds=22.0, reasoning="high",
        fields=("model", "context_pct", "latency", "reasoning"))
    assert out == "gpt-5.4 · 68% · 22s · reasoning high"


def test_reasoning_label_words_the_three_states_like_the_reasoning_command():
    """Unset config → the same "medium (default)" the /reasoning status line shows; an explicit
    disable → "none (disabled)"; otherwise the level itself."""
    from agent.i18n import t

    from gateway.runtime_footer import reasoning_label

    assert reasoning_label(None) == t("gateway.reasoning.level_default")
    assert reasoning_label({}) == t("gateway.reasoning.level_default")
    assert reasoning_label({"enabled": False, "effort": "medium"}) == t("gateway.reasoning.level_disabled")
    assert reasoning_label({"enabled": True, "effort": "high"}, "openai", "gpt-5.4") == "high"
    # Absent effort on an enabled config is the documented default, not an empty label.
    assert reasoning_label({"enabled": True}, "openai", "gpt-5.4") == "medium"


def test_reasoning_label_names_the_level_the_route_actually_sends():
    """Contract: whatever the route clamp resolves, the label says it — a Hermes-internal level
    (``ultra``) is never shown as a distinct wire level the route lacks (#61634)."""
    from agent.reasoning_effort import clamp_effort, route_supported_efforts

    from gateway.runtime_footer import reasoning_label

    route = route_supported_efforts("openai-codex", "gpt-6-sol")
    clamped = clamp_effort("ultra", route)
    expected = "ultra" if clamped == "ultra" else f"ultra (sends {clamped} on this route)"
    assert reasoning_label({"enabled": True, "effort": "ultra"}, "openai-codex", "gpt-6-sol") == expected


def test_default_build_footer_line_ignores_reasoning():
    """Byte-stability: a caller passing a label changes nothing while ``fields`` stays default."""
    common = dict(
        user_config={"display": {"runtime_footer": {"enabled": True}}},
        platform_key="discord",
        model="openai/gpt-5.4",
        context_tokens=50_247,
        context_length=1_000_000,
        cwd="/var/data",
    )
    baseline = build_footer_line(**common)
    assert baseline == "gpt-5.4 · 5% · /var/data"
    assert build_footer_line(**common, reasoning="high") == baseline
    assert build_footer_line(**common, reasoning="none (disabled)") == baseline
