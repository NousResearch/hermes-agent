"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import copy
import pickle

import pytest

from gateway.runtime_footer import (
    FooteredReply,
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
# FooteredReply — copy, deepcopy, and pickle protocols (#158883)
# ---------------------------------------------------------------------------


def test_footered_reply_copy_and_deepcopy():
    """#158883: FooteredReply supports copy and deepcopy without TypeError in __new__."""
    reply = FooteredReply("Task complete.", "gpt-5.4 · 12% · ~/projects")
    assert reply == "Task complete.\n\ngpt-5.4 · 12% · ~/projects"
    assert reply.runtime_footer == "gpt-5.4 · 12% · ~/projects"
    assert reply._orig_text == "Task complete."

    # Shallow copy
    shallow = copy.copy(reply)
    assert shallow == reply
    assert shallow.runtime_footer == "gpt-5.4 · 12% · ~/projects"
    assert isinstance(shallow, FooteredReply)

    # Deepcopy (as performed by EventStream.publish on payload dictionaries)
    payload = {"response": reply, "settled": True}
    copied_payload = copy.deepcopy(payload)
    cloned_reply = copied_payload["response"]
    assert cloned_reply == reply
    assert cloned_reply.runtime_footer == "gpt-5.4 · 12% · ~/projects"
    assert isinstance(cloned_reply, FooteredReply)


def test_footered_reply_pickle():
    """#158883: FooteredReply roundtrips through pickle without losing footer metadata."""
    reply = FooteredReply("Summary output.", "claude-sonnet-4.6 · 45%")
    pickled = pickle.dumps(reply)
    unpickled = pickle.loads(pickled)
    assert unpickled == reply
    assert unpickled.runtime_footer == "claude-sonnet-4.6 · 45%"
    assert unpickled._orig_text == "Summary output."
    assert isinstance(unpickled, FooteredReply)


def test_footered_reply_single_arg_construction():
    """#158883: FooteredReply can be initialized with text alone when footer is omitted."""
    plain = FooteredReply("Plain response.")
    assert plain == "Plain response."
    assert plain.runtime_footer == ""
    assert plain._orig_text == "Plain response."
    assert isinstance(plain, FooteredReply)


def test_event_stream_publish_with_footered_reply():
    """#158883: EventStream / SessionEvents publish handles FooteredReply payload without TypeError."""
    from gateway.session_events import SessionEvents

    events = SessionEvents()
    reply = FooteredReply("Completed turn.", "gpt-5.4 · 10%")
    # Must not raise TypeError: FooteredReply.__new__() missing 1 required positional argument
    events.publish("test-session", {"response": reply, "settled": True})


def test_footered_reply_state_and_attribute_preservation():
    """#158883: FooteredReply preserves dynamic instance attributes across copy and pickle."""
    reply = FooteredReply("Completed task.", "gpt-5.4 · 10%")
    reply.custom_metadata = {"cached": True, "turn": 2}

    cloned = copy.deepcopy(reply)
    assert cloned == reply
    assert cloned.runtime_footer == "gpt-5.4 · 10%"
    assert cloned._orig_text == "Completed task."
    assert getattr(cloned, "custom_metadata", None) == {"cached": True, "turn": 2}

    unpickled = pickle.loads(pickle.dumps(reply))
    assert unpickled == reply
    assert unpickled.runtime_footer == "gpt-5.4 · 10%"
    assert unpickled._orig_text == "Completed task."
    assert getattr(unpickled, "custom_metadata", None) == {"cached": True, "turn": 2}


def test_footered_reply_idempotent_and_replacement_wrapping():
    """#158883: FooteredReply unwraps existing FooteredReply avoiding duplicate concatenation."""
    original = FooteredReply("Base content.", "footer-v1")
    assert original == "Base content.\n\nfooter-v1"

    # Preserves metadata when wrapped without new footer
    wrapped_no_footer = FooteredReply(original)
    assert wrapped_no_footer == "Base content.\n\nfooter-v1"
    assert wrapped_no_footer.runtime_footer == "footer-v1"
    assert wrapped_no_footer._orig_text == "Base content."

    # Replaces footer cleanly without duplicate concatenation
    replaced_footer = FooteredReply(original, "footer-v2")
    assert replaced_footer == "Base content.\n\nfooter-v2"
    assert replaced_footer.runtime_footer == "footer-v2"
    assert replaced_footer._orig_text == "Base content."

