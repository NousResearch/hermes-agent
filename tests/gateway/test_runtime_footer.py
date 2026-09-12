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


def test_build_footer_appends_plugin_fragments_with_sanitized_payload(monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    seen = {}

    def quota_fragment(**kwargs):
        seen.update(kwargs)
        return "5h 15% (2h51m)"

    def broken_fragment(**kwargs):
        raise RuntimeError("cache unavailable")

    manager._hooks.setdefault("append_runtime_footer", []).extend([
        lambda **kwargs: None,
        broken_fragment,
        quota_fragment,
        lambda **kwargs: "resets 2",
    ])
    out = build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        platform_key="slack", model="openai-codex/gpt-5.6", provider="openai-codex",
        context_tokens=16_000, context_length=100_000, cwd="/tmp", turn_seconds=12.0,
    )

    assert out == "gpt-5.6 · 5h 15% (2h51m) · resets 2"
    assert seen == {
        "footer": "gpt-5.6", "model": "openai-codex/gpt-5.6", "provider": "openai-codex",
        "context_tokens": 16_000, "context_length": 100_000, "cwd": "/tmp",
        "turn_seconds": 12.0, "platform": "slack",
        "telemetry_schema_version": "hermes.observer.v1",
    }


def test_build_footer_does_not_call_plugin_when_footer_is_disabled(monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager._hooks.setdefault("append_runtime_footer", []).append(
        lambda **kwargs: pytest.fail("disabled footer must not invoke plugins")
    )

    assert build_footer_line(
        user_config={}, platform_key="slack", model="gpt-5.6", provider="openai-codex",
        context_tokens=0, context_length=None,
    ) == ""


def test_append_runtime_footer_hook_is_registered_and_shell_refused(caplog):
    """This Python-only display callback must not become a shell-hook surface."""
    import logging

    from agent import shell_hooks
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS

    assert "append_runtime_footer" in VALID_HOOKS
    assert "append_runtime_footer" in SHELL_UNSUPPORTED_HOOKS
    with caplog.at_level(logging.WARNING, logger=shell_hooks.logger.name):
        specs = shell_hooks._parse_hooks_block({
            "append_runtime_footer": [{"command": "/tmp/footer.sh"}],
        })
    assert specs == []
    assert any("append_runtime_footer" in record.getMessage() for record in caplog.records)


def test_build_footer_without_listener_keeps_builtin_footer(monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)

    assert build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        platform_key="slack", model="openai-codex/gpt-5.6", context_tokens=0,
        context_length=None,
    ) == "gpt-5.6"


def test_build_footer_skips_plugin_when_builtin_footer_is_empty(monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager._hooks["append_runtime_footer"] = [
        lambda **_kwargs: pytest.fail("empty built-in footer must not invoke plugins"),
    ]

    assert build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        platform_key="slack", model=None, context_tokens=0, context_length=None,
    ) == ""


def test_build_footer_normalizes_and_bounds_plugin_fragments(monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager
    from gateway.runtime_footer import _MAX_PLUGIN_FRAGMENT_CHARS

    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    manager._hooks["append_runtime_footer"] = [
        lambda **_kwargs: None,
        lambda **_kwargs: 42,
        lambda **_kwargs: {"fragment": "not text"},
        lambda **_kwargs: " \n\t ",
        lambda **_kwargs: " quota\n resets\tsoon ",
        lambda **_kwargs: "x" * (_MAX_PLUGIN_FRAGMENT_CHARS + 20),
    ]

    assert build_footer_line(
        user_config={"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        platform_key="slack", model="openai-codex/gpt-5.6", context_tokens=0,
        context_length=None,
    ) == "gpt-5.6 · quota resets soon · " + "x" * _MAX_PLUGIN_FRAGMENT_CHARS


def test_append_runtime_footer_timeout_is_suppressed(monkeypatch):
    """A stuck display plugin cannot delay replies or create another abandoned worker."""
    import threading
    import time

    from hermes_cli import plugins
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    release = threading.Event()
    starts = []

    def blocker(**_kwargs):
        starts.append(1)
        release.wait(timeout=10.0)
        return "late"

    manager._hooks["append_runtime_footer"] = [blocker]
    kwargs = {
        "user_config": {"display": {"runtime_footer": {"enabled": True, "fields": ["model"]}}},
        "platform_key": "slack", "model": "openai-codex/gpt-5.6", "context_tokens": 0,
        "context_length": None,
    }
    try:
        started_at = time.monotonic()
        assert build_footer_line(**kwargs) == "gpt-5.6"
        assert build_footer_line(**kwargs) == "gpt-5.6"
        assert time.monotonic() - started_at < 1.0
        assert starts == [1]
    finally:
        release.set()


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
