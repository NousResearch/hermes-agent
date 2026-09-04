"""Tests for ``display.fallback_notifications`` collapsing.

Covers the three modes, the collapse window with a fake clock, the folded
count text, config parsing (including a YAML round-trip with a mistyped
value), and the guard that keeps all provider auth-error render sites on the
shared helper.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

from agent.notice_collapse import (
    DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS,
    FallbackNoticeState,
    collapse_fallback_notices,
    provider_auth_error_reply,
    reset_provider_auth_error_state,
    resolve_fallback_notice_interval,
    resolve_fallback_notification_mode,
)
from run_agent import AIAgent

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

HOP1 = "⚠️ Model fallback: m1 via p1 unavailable (rate limit); using m2 via p2."
HOP2 = "⚠️ Model fallback: m2 via p2 unavailable (rate limit); using m3 via p3."
HOP3 = "⚠️ Model fallback: m3 via p3 unavailable (rate limit); using m4 via p4."


@pytest.fixture(autouse=True)
def _clean_auth_state():
    reset_provider_auth_error_state()
    yield
    reset_provider_auth_error_state()


def _cfg(**display):
    return {"display": display}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_mode_defaults_to_on_when_unset():
    assert resolve_fallback_notification_mode({}) == "on"
    assert resolve_fallback_notification_mode(_cfg()) == "on"


@pytest.mark.parametrize("raw,expected", [
    ("on", "on"), ("collapse", "collapse"), ("off", "off"),
    ("  COLLAPSE  ", "collapse"), (True, "on"), (False, "off"),
])
def test_mode_accepted_values(raw, expected):
    assert resolve_fallback_notification_mode(_cfg(fallback_notifications=raw)) == expected


def test_mistyped_mode_warns_and_falls_back_to_on(caplog):
    with caplog.at_level("WARNING"):
        mode = resolve_fallback_notification_mode(_cfg(fallback_notifications="colapse"))
    assert mode == "on"
    assert "fallback_notifications" in caplog.text


def test_config_yaml_round_trip_with_mistyped_value(tmp_path, caplog):
    """A user config written and re-read must survive a typo as ``on``."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "display": {
            "fallback_notifications": "colapse",       # typo
            "fallback_notice_interval_seconds": "soon",  # not a number
        }
    }), encoding="utf-8")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    with caplog.at_level("WARNING"):
        assert resolve_fallback_notification_mode(cfg) == "on"
        assert resolve_fallback_notice_interval(cfg) == DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS
    assert "fallback_notifications" in caplog.text
    assert "fallback_notice_interval_seconds" in caplog.text


def test_valid_yaml_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "display": {"fallback_notifications": "collapse",
                    "fallback_notice_interval_seconds": 900}
    }), encoding="utf-8")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert resolve_fallback_notification_mode(cfg) == "collapse"
    assert resolve_fallback_notice_interval(cfg) == 900.0


@pytest.mark.parametrize("raw", [0, -5, "nope", None, ""])
def test_interval_falls_back_to_default(raw):
    assert resolve_fallback_notice_interval(
        _cfg(fallback_notice_interval_seconds=raw)
    ) == DEFAULT_FALLBACK_NOTICE_INTERVAL_SECONDS


def test_defaults_are_declared_in_default_config():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["display"]["fallback_notifications"] == "on"
    assert DEFAULT_CONFIG["display"]["fallback_notice_interval_seconds"] == 3600


# ---------------------------------------------------------------------------
# collapse_fallback_notices
# ---------------------------------------------------------------------------

def test_mode_on_emits_every_hop():
    state = FallbackNoticeState()
    out = collapse_fallback_notices(
        [HOP1, HOP2, HOP3], mode="on", interval=3600, state=state, now=0.0,
    )
    assert out == [HOP1, HOP2, HOP3]
    assert state.last_emitted_at is None  # window untouched in "on"


def test_mode_off_emits_nothing():
    state = FallbackNoticeState()
    out = collapse_fallback_notices(
        [HOP1, HOP2], mode="off", interval=3600, state=state, now=0.0,
    )
    assert out == []


def test_collapse_folds_hops_in_one_flush():
    """Five hops in one second produce exactly one line (spec A1)."""
    state = FallbackNoticeState()
    out = collapse_fallback_notices(
        [HOP1, HOP2, HOP3], mode="collapse", interval=3600, state=state,
        now=0.0, current_route="m4 via p4",
    )
    assert len(out) == 1
    assert out[0].startswith("⚠️ Model fallback: m1 via p1 unavailable (rate limit)")
    assert "(2 further fallbacks; now on m4 via p4.)" in out[0]


def test_collapse_first_hop_verbatim_when_alone():
    state = FallbackNoticeState()
    out = collapse_fallback_notices(
        [HOP1], mode="collapse", interval=3600, state=state, now=0.0,
        current_route="m2 via p2",
    )
    assert out == [HOP1]


def test_collapse_window_suppresses_then_folds_with_fake_clock():
    """A second burst inside the window is counted; the next allowed notice
    carries the count and the time of the previous notice."""
    state = FallbackNoticeState()
    # t=0 (09:12:00 UTC) — first notice goes out.
    t0 = 1_757_063_520.0  # 2025-09-05 09:12:00 UTC
    first = collapse_fallback_notices(
        [HOP1], mode="collapse", interval=3600, state=state, now=t0,
        current_route="m2 via p2",
    )
    assert first == [HOP1]

    # t=+60s and t=+120s — inside the window, suppressed but counted.
    for offset in (60.0, 120.0):
        assert collapse_fallback_notices(
            [HOP2, HOP3], mode="collapse", interval=3600, state=state,
            now=t0 + offset, current_route="m4 via p4",
        ) == []
    assert state.suppressed == 4

    # t=+3600s — window elapsed, the folded notice goes out.
    folded = collapse_fallback_notices(
        [HOP3], mode="collapse", interval=3600, state=state,
        now=t0 + 3600.0, current_route="m5 via p5",
    )
    assert folded == [
        "⚠️ Model fallback: 4 further fallbacks since 09:12 UTC; now on m5 via p5."
    ]
    assert state.suppressed == 0
    assert state.last_emitted_at == t0 + 3600.0


def test_collapse_empty_flush_is_a_noop():
    state = FallbackNoticeState()
    assert collapse_fallback_notices(
        [], mode="collapse", interval=3600, state=state, now=0.0) == []
    assert state.last_emitted_at is None


def test_collapse_falls_back_when_route_unknown():
    state = FallbackNoticeState(last_emitted_at=0.0, suppressed=2)
    out = collapse_fallback_notices(
        [HOP2], mode="collapse", interval=10, state=state, now=100.0,
    )
    assert "now on the fallback model." in out[0]


# ---------------------------------------------------------------------------
# Agent integration (_emit_pending_fallback_notice)
# ---------------------------------------------------------------------------

def _make_bare_agent(mode=None, interval=None, now=0.0):
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent.status_callback = None
    agent.suppress_status_output = False
    agent._mute_post_response = False
    agent._executing_tools = False
    agent._print_fn = None
    if mode is not None:
        agent.fallback_notifications = mode
    if interval is not None:
        agent.fallback_notice_interval_seconds = interval
    agent._fallback_notice_clock = lambda: agent._fake_now
    agent._fake_now = now
    agent._provider_fallback_route = ("m4", "p4")
    return agent


def test_agent_emits_every_hop_in_on_mode():
    agent = _make_bare_agent(mode="on", interval=3600)
    emitted = []
    agent._emit_status = emitted.append
    agent._pending_fallback_notice = [HOP1, HOP2, HOP3]
    agent._emit_pending_fallback_notice()
    assert emitted == [HOP1, HOP2, HOP3]


def test_agent_emits_nothing_in_off_mode():
    agent = _make_bare_agent(mode="off", interval=3600)
    emitted = []
    agent._emit_status = emitted.append
    agent._pending_fallback_notice = [HOP1, HOP2, HOP3]
    agent._emit_pending_fallback_notice()
    assert emitted == []
    assert agent._pending_fallback_notice is None


def test_agent_collapses_across_turns_with_fake_clock():
    agent = _make_bare_agent(mode="collapse", interval=3600, now=1_757_063_520.0)
    emitted = []
    agent._emit_status = emitted.append

    agent._pending_fallback_notice = [HOP1]
    agent._emit_pending_fallback_notice()
    assert emitted == [HOP1]

    # Next turn, inside the window: suppressed.
    agent._fake_now += 60
    agent._pending_fallback_notice = [HOP2, HOP3]
    agent._emit_pending_fallback_notice()
    assert emitted == [HOP1]

    # Next turn, window elapsed: one folded line.
    agent._fake_now += 3600
    agent._pending_fallback_notice = [HOP3]
    agent._emit_pending_fallback_notice()
    assert len(emitted) == 2
    assert emitted[1] == (
        "⚠️ Model fallback: 2 further fallbacks since 09:12 UTC; now on m4 via p4."
    )


def test_agent_flush_emits_collapsed_notice_outside_on_mode():
    """Terminal failure: ``on`` relies on the retry buffer, collapse/off do not
    buffer the hop line, so the flush must emit the collapsed notice itself."""
    agent = _make_bare_agent(mode="collapse", interval=3600)
    emitted = []
    agent._emit_status = emitted.append
    agent._pending_fallback_notice = [HOP1, HOP2]

    agent._flush_status_buffer()

    assert len(emitted) == 1
    assert "1 further fallbacks" in emitted[0]
    assert agent._pending_fallback_notice is None


def test_try_activate_fallback_skips_status_buffer_outside_on_mode():
    """The per-hop line stays out of the retry buffer when collapsing."""
    import inspect
    from agent import chat_completion_helpers

    src = inspect.getsource(chat_completion_helpers.try_activate_fallback)
    assert '_fallback_notice_mode' in src
    assert 'if _notice_mode == "on":\n            agent._buffer_status(notice)' in src


# ---------------------------------------------------------------------------
# Provider auth-error replies
# ---------------------------------------------------------------------------

def test_auth_reply_unchanged_in_on_mode():
    reply = provider_auth_error_reply("no creds", session_key="s1", mode="on")
    assert reply == "⚠️ Provider authentication failed: no creds"
    assert provider_auth_error_reply("no creds", session_key="s1", mode="on") == reply


def test_auth_reply_suppressed_in_off_mode():
    assert provider_auth_error_reply("no creds", session_key="s1", mode="off") == ""


def test_auth_reply_deduped_per_session_per_window():
    kw = dict(mode="collapse", interval=3600)
    assert provider_auth_error_reply("no creds", session_key="s1", now=0.0, **kw)
    # Same class, same session, inside the window → suppressed.
    assert provider_auth_error_reply("no creds", session_key="s1", now=60.0, **kw) == ""
    # Different session → its own window.
    assert provider_auth_error_reply("no creds", session_key="s2", now=60.0, **kw)
    # Different error class → not deduped against the first.
    assert provider_auth_error_reply("bad key", session_key="s1", now=60.0, **kw)
    # Window elapsed → allowed again.
    assert provider_auth_error_reply("no creds", session_key="s1", now=3600.0, **kw)


def test_gateway_reply_classes_dedupe_independently():
    from gateway.run import _gateway_provider_error_reply

    auth = _gateway_provider_error_reply("Error code: 401 invalid api key")
    assert "authentication" in auth
    # Default mode is "on" — repeats are unchanged.
    assert _gateway_provider_error_reply("Error code: 401 invalid api key") == auth


# ---------------------------------------------------------------------------
# Three-site guard
# ---------------------------------------------------------------------------

AUTH_REPLY_SITES = (
    "gateway/run.py",
    "gateway/platforms/api_server.py",
    "gateway/platforms/api_server_runs.py",
)


def test_all_provider_auth_reply_sites_use_the_shared_helper():
    """Every site that surfaces a provider auth failure must route through
    ``agent.notice_collapse`` so one outage cannot produce one reply per
    attempt. Fails loudly if a new site re-renders the string inline."""
    offenders = []
    for rel in AUTH_REPLY_SITES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if "Provider authentication failed" not in line:
                continue
            stripped = line.strip()
            # Log lines and the shared helper's own definition are fine; an
            # f-string that builds the user-facing reply is not.
            if stripped.startswith(("logger.", '"Provider authentication failed')):
                continue
            if 'f"⚠️ Provider authentication failed' in line:
                offenders.append(f"{rel}:{lineno}: {stripped}")
        if "provider_auth_error_reply" not in text and "collapse_provider_error_reply" not in text:
            offenders.append(f"{rel}: does not import the shared collapse helper")
    assert not offenders, "provider auth reply rendered without the shared helper:\n" + "\n".join(offenders)
