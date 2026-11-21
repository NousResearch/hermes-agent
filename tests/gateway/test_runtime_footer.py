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


@pytest.mark.parametrize("platform", ["feishu", "telegram", "discord", "slack"])
def test_turn_token_footer_opts_in_per_platform(monkeypatch, platform):
    from types import SimpleNamespace

    from gateway.run_turn_runner import TurnRunner
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session import Platform, SessionSource
    from gateway.turn_context import TurnContext

    def run(_message, **_kwargs):
        agent.session_prompt_tokens += 1350
        agent.session_completion_tokens += 84
        return {"final_response": "Hello"}

    agent = SimpleNamespace(run_conversation=run, session_prompt_tokens=20_000,
                            session_completion_tokens=3000)
    runner = SimpleNamespace(session_authority=None, _consume_pending_native_image_paths=lambda key: [])
    source = SessionSource(platform=Platform(platform), chat_id="chat", user_id="user")
    turn = TurnRunner(runner, TurnContext(message="Hello", session_id="s", session_key="s", source=source))
    result = turn._run_conversation_with_approval(agent, [], None, None, None)
    result["model"] = "vendor/model"
    config = {"display": {"platforms": {platform: {
        "runtime_footer": {"enabled": True, "fields": ["model", "tokens"]},
    }}}}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    renderer = GatewayTurnPersistenceMixin()

    assert renderer._hmwa_runtime_footer_line(result, source, 1.0) == (
        "model · 1,350 in / 84 out / 1,434 total tokens"
    )
    config["display"]["platforms"].clear()
    assert renderer._hmwa_runtime_footer_line(result, source, 1.0) == ""
    config["display"]["runtime_footer"] = {"enabled": True, "fields": ["model"]}
    assert renderer._hmwa_runtime_footer_line(result, source, 1.0) == "model"


@pytest.mark.parametrize(("usage", "expected"), [
    (None, "model"),
    ({"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "model"),
    ({"input_tokens": 200, "output_tokens": 0, "total_tokens": 200},
     "model · 200 in / 0 out / 200 total tokens"),
])
def test_token_footer_skips_unreported_usage(usage, expected):
    assert format_runtime_footer(model="vendor/model", context_tokens=0, context_length=None,
                                 turn_token_usage=usage, fields=["model", "tokens"]) == expected
