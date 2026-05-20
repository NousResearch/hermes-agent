"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import os

import pytest

from gateway.runtime_footer import (
    _build_footer_script_payload,
    _home_relative_cwd,
    _model_short,
    _script_metrics_to_footer_kwargs,
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



def test_format_footer_renders_tokens_api_calls_and_cost():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=43000,
        context_length=100000,
        cwd="",
        total_tokens=2184,
        api_calls=3,
        estimated_cost_usd=0.0123,
        cost_status="estimated",
        fields=("model", "context_pct", "tokens", "api_calls", "cost"),
    )
    assert out == "gpt-5.4 · 43% · 2,184 tok · 3 calls · ~$0.012"



def test_format_footer_renders_token_breakdown_when_available():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        prompt_tokens=1200,
        completion_tokens=456,
        cache_read_tokens=78,
        cache_write_tokens=9,
        reasoning_tokens=12,
        fields=("token_breakdown",),
    )
    assert out == "tok in 1,200 · out 456 · cache r 78 · cache w 9 · reason 12"



def test_format_footer_omits_cost_when_unavailable():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        total_tokens=2184,
        api_calls=3,
        estimated_cost_usd=None,
        cost_status="unknown",
        fields=("tokens", "api_calls", "cost"),
    )
    assert out == "2,184 tok · 3 calls"



def test_format_footer_renders_included_cost_without_amount():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        estimated_cost_usd=None,
        cost_status="included",
        fields=("cost",),
    )
    assert out == "cost included"



def test_format_footer_omits_token_breakdown_when_missing():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="",
        fields=("model", "token_breakdown"),
    )
    assert out == "gpt-5.4"


# ---------------------------------------------------------------------------
# resolve_footer_config
# ---------------------------------------------------------------------------


def test_resolve_defaults_off_empty_config():
    cfg = resolve_footer_config({}, "telegram")
    assert cfg == {
        "enabled": False,
        "fields": ["model", "context_pct", "cwd"],
        "script": None,
    }



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


def test_resolve_footer_config_carries_script_path():
    user = {
        "display": {
            "runtime_footer": {
                "enabled": True,
                "script": "~/bin/footer-metrics.py",
            }
        }
    }
    cfg = resolve_footer_config(user, "slack")
    assert cfg["enabled"] is True
    assert cfg["script"] == "~/bin/footer-metrics.py"


def test_build_footer_script_payload_contains_runtime_metrics():
    payload = _build_footer_script_payload(
        session_id="sess-123",
        platform_key="slack",
        model="copilot/gpt-5.4",
        context_tokens=1234,
        context_length=8000,
        cwd="/tmp/project",
        total_tokens=2222,
        api_calls=5,
        estimated_cost_usd=0.015,
        cost_status="estimated",
        prompt_tokens=1000,
        completion_tokens=400,
        cache_read_tokens=800,
        cache_write_tokens=22,
        reasoning_tokens=17,
    )
    assert payload["session_id"] == "sess-123"
    assert payload["platform"] == "slack"
    assert payload["metrics"]["total_tokens"] == 2222
    assert payload["metrics"]["estimated_cost_usd"] == 0.015
    assert payload["context"]["length"] == 8000


def test_script_metrics_to_footer_kwargs_maps_known_fields_only():
    kwargs = _script_metrics_to_footer_kwargs(
        {
            "total_tokens": 9000,
            "api_calls": 7,
            "estimated_cost_usd": 0.123,
            "cost_status": "estimated",
            "prompt_tokens": 4000,
            "completion_tokens": 1000,
            "cache_read_tokens": 200,
            "cache_write_tokens": 10,
            "reasoning_tokens": 99,
            "ignored": "value",
        }
    )
    assert kwargs == {
        "total_tokens": 9000,
        "api_calls": 7,
        "estimated_cost_usd": 0.123,
        "cost_status": "estimated",
        "prompt_tokens": 4000,
        "completion_tokens": 1000,
        "cache_read_tokens": 200,
        "cache_write_tokens": 10,
        "reasoning_tokens": 99,
    }


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



def test_build_footer_supports_rich_metrics_fields():
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["model", "tokens", "api_calls", "cost", "token_breakdown"],
                }
            }
        },
        platform_key="slack",
        model="copilot/gpt-5.4-mini",
        context_tokens=0,
        context_length=None,
        cwd="",
        total_tokens=2184,
        api_calls=3,
        estimated_cost_usd=0.0123,
        cost_status="estimated",
        prompt_tokens=1200,
        completion_tokens=456,
        cache_read_tokens=78,
        cache_write_tokens=9,
        reasoning_tokens=12,
    )
    assert out == (
        "gpt-5.4-mini · 2,184 tok · 3 calls · ~$0.012 · "
        "tok in 1,200 · out 456 · cache r 78 · cache w 9 · reason 12"
    )



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


def test_build_footer_uses_script_metrics_when_configured(tmp_path):
    script = tmp_path / "footer_metrics.py"
    script.write_text(
        """
import json, sys
payload = json.load(sys.stdin)
assert payload['session_id'] == 'sess-proxy'
json.dump({
    'total_tokens': 6789,
    'api_calls': 4,
    'estimated_cost_usd': 0.042,
    'cost_status': 'estimated'
}, sys.stdout)
""".strip()
    )
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["tokens", "api_calls", "cost"],
                    "script": str(script),
                }
            }
        },
        platform_key="slack",
        session_id="sess-proxy",
        model="copilot/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="/tmp/project",
    )
    assert out == "6,789 tok · 4 calls · ~$0.042"


def test_build_footer_script_metrics_override_missing_inline_metrics(tmp_path):
    script = tmp_path / "footer_metrics.py"
    script.write_text(
        """
import json, sys
json.dump({
    'prompt_tokens': 1200,
    'completion_tokens': 300,
    'cache_read_tokens': 50,
    'reasoning_tokens': 25,
}, sys.stdout)
""".strip()
    )
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["token_breakdown"],
                    "script": str(script),
                }
            }
        },
        platform_key="slack",
        session_id="sess-proxy",
        model="copilot/gpt-5.4",
        context_tokens=0,
        context_length=None,
        cwd="/tmp/project",
    )
    assert out == "tok in 1,200 · out 300 · cache r 50 · reason 25"
