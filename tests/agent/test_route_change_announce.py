"""Route-change announce + durable audit trail (provider-only failover + effort rider).

Companion to test_fallback_announce.py. Covers the v2 predicate and the
reasoning-effort rider added 2026-07-07:

- The announce fires on any ``(provider, model)`` route change — INCLUDING a
  same-model cross-PROVIDER failover (opus@claude-app → opus@f3), which the old
  model-only predicate silently suppressed.
- Reasoning-effort is a display RIDER: a side renders ``(effort)`` only when the
  two resolved effort LABELS differ. reasoning_config is a DICT, so the labels
  are normalized via ``_effort_label`` before comparison — a blank/inherited
  fallback effort must NOT render a spurious suffix on the common path.
- The durable sink ``_append_route_change`` records every transition with an
  ``@effort`` suffix only on a side whose (normalized) effort is non-blank.
- The two config flags (announce_route_change / announce_recovery) gate ONLY the
  chat emit, resolved by the caller and passed as ``announce_enabled``.
"""

import os
import types

import pytest

from agent.chat_completion_helpers import (
    _append_route_change,
    _effort_label,
    _emit_fallback_announce,
)


# ── _effort_label: normalize dict / str / None to a bare label ──────────────

def test_effort_label_from_dict_enabled():
    assert _effort_label({"enabled": True, "effort": "high"}) == "high"


def test_effort_label_from_dict_disabled():
    assert _effort_label({"enabled": False}) == "none"


def test_effort_label_from_string():
    assert _effort_label("xhigh") == "xhigh"


def test_effort_label_none_and_blank():
    assert _effort_label(None) == ""
    assert _effort_label("") == ""
    assert _effort_label({"enabled": True, "effort": ""}) == ""


# ── announce predicate: (provider, model) route identity ────────────────────

def _agent():
    a = types.SimpleNamespace()
    a._announced = []
    a._emit_status = lambda m: a._announced.append(m)
    a._last_fallback_announced = None
    a._last_fallback_event = None
    a._current_turn_id = "turn-1"
    return a


def test_provider_only_failover_announces():
    """opus@claude-app → opus@f3 (same model, different provider) MUST announce."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-opus-4-8", "claude-api-proxy-f3",
        old_provider="claude-app",
    )
    assert len(a._announced) == 1
    msg = a._announced[0]
    assert "claude-app/claude-opus-4-8" in msg
    assert "claude-api-proxy-f3/claude-opus-4-8" in msg


def test_noop_same_provider_same_model_silent():
    """Same provider AND model → true no-op → silent."""
    a = _agent()
    _emit_fallback_announce(
        a, "gpt-5.5", "gpt-5.5", "openai-codex", old_provider="openai-codex",
    )
    assert a._announced == []


def test_old_provider_none_same_model_silent():
    """Unknown old provider + same model → cannot prove a route change → silent."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-opus-4-8", "claude-app", old_provider=None,
    )
    assert a._announced == []


def test_dedupe_per_provider_model_pair():
    """The SAME transition twice within a turn announces once."""
    a = _agent()
    for _ in range(2):
        _emit_fallback_announce(
            a, "claude-opus-4-8", "gpt-5.5", "openai-codex",
            old_provider="claude-app",
        )
    assert len(a._announced) == 1


def test_two_distinct_transitions_both_announce():
    """A→B then B→C are distinct transitions → two announces."""
    a = _agent()
    _emit_fallback_announce(a, "opus", "gpt-5.5", "openai-codex", old_provider="claude-app")
    _emit_fallback_announce(a, "gpt-5.5", "sonnet", "anthropic", old_provider="openai-codex")
    assert len(a._announced) == 2


def test_announce_route_change_gate_off_suppresses_but_records_event():
    """announce_enabled=False suppresses the chat line; _last_fallback_event still set."""
    a = _agent()
    _emit_fallback_announce(
        a, "opus", "gpt-5.5", "openai-codex", old_provider="claude-app",
        announce_enabled=False, record_event=True,
    )
    assert a._announced == []
    assert a._last_fallback_event is not None
    assert a._last_fallback_event["new_provider"] == "openai-codex"


def test_provider_only_failover_sets_fallback_event():
    """A provider-only failover now populates _last_fallback_event (D-10)."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-opus-4-8", "claude-api-proxy-f3",
        old_provider="claude-app", record_event=True,
    )
    assert a._last_fallback_event is not None
    assert a._last_fallback_event["old_provider"] == "claude-app"
    assert a._last_fallback_event["new_provider"] == "claude-api-proxy-f3"


def test_recovery_does_not_record_fallback_event():
    """A recovery (record_event=False) must NOT stamp the causality record."""
    a = _agent()
    _emit_fallback_announce(
        a, "gpt-5.5", "claude-opus-4-8", "claude-app", old_provider="openai-codex",
        record_event=False, announce_enabled=True, kind="recovery",
    )
    assert a._last_fallback_event is None
    assert len(a._announced) == 1
    assert "recovery" in a._announced[0].lower()


# ── effort rider: (effort) suffix only when normalized labels differ ────────

def test_announce_effort_suffix_on_change():
    """Effort actually changed (high → xhigh) → both sides carry (effort)."""
    a = _agent()
    _emit_fallback_announce(
        a, "claude-opus-4-8", "gpt-5.5", "openai-codex", old_provider="claude-app",
        old_effort={"enabled": True, "effort": "high"}, new_effort="xhigh",
    )
    msg = a._announced[0]
    assert "(high)" in msg
    assert "(xhigh)" in msg


def test_announce_no_effort_suffix_when_equal():
    """Explicit equal efforts (high == high) → no suffix."""
    a = _agent()
    _emit_fallback_announce(
        a, "opus", "gpt-5.5", "openai-codex", old_provider="claude-app",
        old_effort="high", new_effort="high",
    )
    assert "(" not in a._announced[0].split("→")[-1]


def test_inherited_blank_effort_renders_no_suffix():
    """RC-A/RC-D: old side is a real reasoning_config DICT, new side blank
    (inherited). The caller normalizes blank → primary label, so labels are
    equal and NO suffix renders. Here we pass old as the DICT type prod uses and
    new as the SAME resolved label the caller would compute — proving the dict
    isn't naively compared to a string."""
    a = _agent()
    old_cfg = {"enabled": True, "effort": "high"}
    # Caller computes: new_eff = _effort_label(_fb_reasoning_effort) or _effort_label(primary)
    # With a blank fallback effort that is "" or "high" == the primary label.
    resolved_new = _effort_label("") or _effort_label(old_cfg)
    _emit_fallback_announce(
        a, "claude-opus-4-8", "claude-opus-4-8", "claude-api-proxy-f3",
        old_provider="claude-app",
        old_effort=old_cfg, new_effort=resolved_new,
    )
    msg = a._announced[0]
    assert "(high)" not in msg, msg
    assert "(" not in msg.replace("(effort)", ""), msg  # no stray suffix


# ── durable sink _append_route_change ───────────────────────────────────────

def _read_sink(home):
    p = os.path.join(home, "state", "model-route-changes.log")
    with open(p, encoding="utf-8") as fh:
        return fh.read().splitlines()


def test_sink_records_effort_both_sides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _append_route_change(
        "failover", "claude-app", "claude-opus-4-8", "openai-codex", "gpt-5.5",
        old_effort="high", new_effort="xhigh",
    )
    lines = _read_sink(str(tmp_path))
    assert len(lines) == 1
    assert "failover claude-app/claude-opus-4-8@high -> openai-codex/gpt-5.5@xhigh" in lines[0]


def test_sink_no_effort_suffix_when_inherited(tmp_path, monkeypatch):
    """Blank new effort → no @suffix on that side."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _append_route_change(
        "failover", "claude-app", "claude-opus-4-8", "claude-api-proxy-f3",
        "claude-opus-4-8", old_effort="high", new_effort="",
    )
    line = _read_sink(str(tmp_path))[0]
    assert "claude-app/claude-opus-4-8@high" in line
    assert "claude-api-proxy-f3/claude-opus-4-8 " in line + " "  # no @ after new side
    assert "@" not in line.split("-> ")[1]


def test_sink_best_effort_swallows_errors(monkeypatch):
    """An unwritable HERMES_HOME must not raise (telemetry never breaks a turn)."""
    # Point at a path that cannot be created (a file used as a dir parent).
    monkeypatch.setenv("HERMES_HOME", "/dev/null/nope")
    # Must not raise.
    _append_route_change("failover", "a", "m1", "b", "m2")


def test_sink_appends_multiple(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _append_route_change("failover", "claude-app", "opus", "openai-codex", "gpt-5.5")
    _append_route_change("recovery", "openai-codex", "gpt-5.5", "claude-app", "opus")
    lines = _read_sink(str(tmp_path))
    assert len(lines) == 2
    assert " failover " in lines[0]
    assert " recovery " in lines[1]
