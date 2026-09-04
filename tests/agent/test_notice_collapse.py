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
    TERSE_SUPPRESSED_REPLY,
    FallbackNoticeState,
    collapse_fallback_notices,
    is_suppressed_notice,
    provider_auth_error_reply,
    reset_config_warning_state,
    reset_provider_auth_error_state,
    resolve_direct_reply,
    resolve_fallback_notice_interval,
    resolve_fallback_notification_mode,
    resolve_status_notice,
)
from run_agent import AIAgent

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

HOP1 = "⚠️ Model fallback: m1 via p1 unavailable (rate limit); using m2 via p2."
HOP2 = "⚠️ Model fallback: m2 via p2 unavailable (rate limit); using m3 via p3."
HOP3 = "⚠️ Model fallback: m3 via p3 unavailable (rate limit); using m4 via p4."


@pytest.fixture(autouse=True)
def _clean_auth_state():
    reset_provider_auth_error_state()
    reset_config_warning_state()
    yield
    reset_provider_auth_error_state()
    reset_config_warning_state()


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


def _agent_with_one_fallback(mode=None):
    """A real AIAgent with a one-hop fallback chain, wired for a live switch."""
    from unittest.mock import MagicMock, patch

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
    agent.client = MagicMock()
    agent.model = "gpt-5.6-sol"
    agent.provider = "openai-codex"
    if mode is not None:
        agent.fallback_notifications = mode
    return agent


def _activate_one_fallback(agent):
    """Run a real fallback hop. Returns ``(retry_buffer, pending_notices)``."""
    from unittest.mock import MagicMock, patch

    fb_client = MagicMock()
    fb_client.base_url = "https://api.z.ai/v1"
    fb_client.api_key = "fb-key"
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fb_client, "glm-5.2"),
    ):
        assert agent._try_activate_fallback() is True
    return (
        list(getattr(agent, "_retry_status_buffer", None) or []),
        list(getattr(agent, "_pending_fallback_notice", None) or []),
    )


@pytest.mark.parametrize("mode,expect_buffered", [
    ("on", True), ("collapse", False), ("off", False),
])
def test_try_activate_fallback_buffers_the_hop_line_only_in_on_mode(
    mode, expect_buffered,
):
    """The per-hop line stays out of the retry buffer when collapsing, so a
    terminal failure cannot re-expose every hop.

    Behavioural, not a source-substring assert: this survives a rename or a
    quote-style change, and fails if the gate is dropped.
    """
    agent = _agent_with_one_fallback(mode=mode)
    buffered, pending = _activate_one_fallback(agent)

    hop_lines = [b for b in buffered if "Model fallback" in str(b)]
    assert bool(hop_lines) is expect_buffered
    # The durable one-shot notice is recorded in every mode; only the
    # user-facing emission is gated, at flush time.
    assert any("Model fallback" in str(x) for x in pending)


def test_try_activate_fallback_fails_open_on_a_mock_agent():
    """A mock agent returns a truthy Mock from ``_fallback_notice_mode``. That
    must not silently skip buffering: anything that is not a known mode string
    means ``on``."""
    from unittest.mock import MagicMock

    agent = _agent_with_one_fallback()
    agent._fallback_notice_mode = lambda: MagicMock()
    buffered, _ = _activate_one_fallback(agent)

    assert any("Model fallback" in str(b) for b in buffered), (
        "unknown mode must fail open to on-mode buffering"
    )


# ---------------------------------------------------------------------------
# Provider auth-error replies
# ---------------------------------------------------------------------------

def test_auth_reply_unchanged_in_on_mode():
    reply = provider_auth_error_reply("no creds", session_key="s1", mode="on")
    assert reply == "⚠️ Provider authentication failed: no creds"
    assert provider_auth_error_reply("no creds", session_key="s1", mode="on") == reply


def test_auth_reply_suppressed_in_off_mode():
    """``off`` is intentional silence, and says so with the sentinel — it must
    never look like "this was not a provider error" to a caller."""
    reply = provider_auth_error_reply("no creds", session_key="s1", mode="off")
    assert is_suppressed_notice(reply)
    assert resolve_status_notice(reply) is None
    assert resolve_direct_reply(reply) == TERSE_SUPPRESSED_REPLY


def test_auth_reply_deduped_per_session_per_window():
    kw = dict(mode="collapse", interval=3600)
    assert provider_auth_error_reply("no creds", session_key="s1", now=0.0, **kw)
    # Same class, same session, inside the window → suppressed.
    assert is_suppressed_notice(
        provider_auth_error_reply("no creds", session_key="s1", now=60.0, **kw)
    )
    # Different session → its own window.
    assert provider_auth_error_reply("no creds", session_key="s2", now=60.0, **kw)
    # Window elapsed → allowed again.
    assert provider_auth_error_reply("no creds", session_key="s1", now=3600.0, **kw)


def test_auth_reply_dedupes_on_the_error_class_not_the_exception_text():
    """F2: provider 401 bodies carry per-request ids and timestamps. If any of
    that text reaches the dedupe key, nothing ever collapses and the box still
    spams — which is the entire bug this patch exists to fix."""
    kw = dict(mode="collapse", interval=3600)
    first = provider_auth_error_reply(
        "Error code: 401 — req_a1b2c3 at 2026-09-04T10:00:01Z",
        session_key="s1", now=0.0, **kw,
    )
    assert "401" in first
    for n, text in enumerate((
        "Error code: 401 — req_zz9 at 2026-09-04T10:00:09Z",
        "Error code: 401 — req_qq7 at 2026-09-04T10:01:44Z, org org-xyz",
    ), start=1):
        assert is_suppressed_notice(
            provider_auth_error_reply(text, session_key="s1", now=60.0 * n, **kw)
        ), f"varying exception text must still dedupe: {text}"


def test_suppression_sentinel_is_not_deliverable_text():
    """The sentinel must be impossible to mistake for a reply."""
    from agent.notice_collapse import SUPPRESSED_NOTICE

    assert "\x00" in SUPPRESSED_NOTICE
    assert not is_suppressed_notice("")
    assert not is_suppressed_notice(None)
    assert not is_suppressed_notice("⚠️ Provider authentication failed: x")


def test_gateway_reply_classes_unchanged_in_on_mode():
    from gateway.run import _gateway_provider_error_reply

    auth = _gateway_provider_error_reply("Error code: 401 invalid api key")
    assert "authentication" in auth
    # Default mode is "on" — repeats are unchanged.
    assert _gateway_provider_error_reply("Error code: 401 invalid api key") == auth


def test_gateway_reply_classes_dedupe_independently_in_collapse_mode():
    """F6: in collapse mode a second auth error in the window is suppressed,
    but a rate limit is a different class and still gets through."""
    from agent.notice_collapse import collapse_provider_error_reply
    from gateway.run import _gateway_provider_error_reply

    kw = dict(mode="collapse", interval=3600.0)
    seen = []

    def _capture(reply, **kwargs):
        kwargs.update(kw)
        kwargs.setdefault("now", 60.0 * len(seen))
        out = collapse_provider_error_reply(reply, **kwargs)
        seen.append(out)
        return out

    import gateway.run as gr
    original = gr.collapse_provider_error_reply if hasattr(gr, "collapse_provider_error_reply") else None
    assert original is None  # imported inside the function, so patch the module

    import agent.notice_collapse as nc
    real = nc.collapse_provider_error_reply
    nc.collapse_provider_error_reply = _capture
    try:
        auth1 = _gateway_provider_error_reply("Error code: 401 invalid api key", "chat-a")
        auth2 = _gateway_provider_error_reply("Error code: 401 invalid api key", "chat-a")
        rate = _gateway_provider_error_reply("Rate limited after 5 retries", "chat-a")
        auth_other = _gateway_provider_error_reply(
            "Error code: 401 invalid api key", "chat-b",
        )
    finally:
        nc.collapse_provider_error_reply = real

    assert "authentication" in auth1
    assert is_suppressed_notice(auth2), "second auth in the window must collapse"
    assert "rate-limiting" in rate, "a different class must not dedupe against auth"
    assert "authentication" in auth_other, "a different chat has its own window"


def test_gateway_surface_key_is_per_chat_not_per_platform():
    """F3: keyed on the platform alone, one person's error 20 minutes ago
    silences a different person in a different chat."""
    from gateway.run import _gateway_surface_key

    class _P:
        name = "SLACK"

    a = _gateway_surface_key(_P(), "C123")
    b = _gateway_surface_key(_P(), "C999")
    assert a != b
    assert a.startswith("SLACK")
    # No chat in hand → coarser platform-only key, documented fallback.
    assert _gateway_surface_key(_P()) == "SLACK"
    assert _gateway_surface_key(_P(), "") == "SLACK"


def test_two_chats_on_one_platform_do_not_share_a_window():
    """End to end through the real gateway renderer: chat B must still be told
    about its own auth failure after chat A was told about one."""
    from gateway.run import _gateway_provider_error_reply, _gateway_surface_key

    class _P:
        name = "SLACK"

    kw = dict(mode="collapse", interval=3600.0)
    raw = "Error code: 401 invalid api key"

    import agent.notice_collapse as nc
    real = nc.collapse_provider_error_reply

    def _capture(reply, **kwargs):
        kwargs.update(kw)
        return real(reply, **kwargs)

    nc.collapse_provider_error_reply = _capture
    try:
        a1 = _gateway_provider_error_reply(raw, _gateway_surface_key(_P(), "C-a"))
        b1 = _gateway_provider_error_reply(raw, _gateway_surface_key(_P(), "C-b"))
        a2 = _gateway_provider_error_reply(raw, _gateway_surface_key(_P(), "C-a"))
    finally:
        nc.collapse_provider_error_reply = real

    assert "authentication" in a1
    assert "authentication" in b1, "chat B shares no window with chat A"
    assert is_suppressed_notice(a2), "chat A is still inside its own window"


# ---------------------------------------------------------------------------
# F1: suppression is never a raw-error fallthrough, and never silence on a
# reply to a message a person actually sent.
# ---------------------------------------------------------------------------

def test_direct_reply_never_falls_through_to_the_raw_provider_error():
    """The exact defect: the sanitizer withholds a repeat, the caller reads
    that as "not a provider error" and prints ``result['error']`` — the raw,
    unredacted provider body — straight into chat."""
    from gateway.run import (
        _is_suppressed_gateway_notice,
        _terse_suppressed_gateway_reply,
    )
    from agent.notice_collapse import SUPPRESSED_NOTICE

    raw_error = "Error code: 401 sk-live-DEADBEEF req_a1b2 org-acme"
    sanitized = SUPPRESSED_NOTICE

    # The shape both call sites now use.
    if _is_suppressed_gateway_notice(sanitized):
        final = _terse_suppressed_gateway_reply()
    elif not sanitized:
        final = f"⚠️ {raw_error}"
    else:
        final = sanitized

    assert final == TERSE_SUPPRESSED_REPLY
    assert "sk-live" not in final
    assert "401" not in final
    assert final.strip(), "a direct reply must never be silence"
    assert len(final.splitlines()) == 1, "terse means one line"


def test_both_final_response_call_sites_resolve_suppression():
    """Guard the two ``_sanitize_gateway_final_response`` call sites: each must
    check for suppression BEFORE any raw-error fallback."""
    import re

    src = (REPO_ROOT / "gateway/run.py").read_text(encoding="utf-8")
    calls = [
        m.start() for m in re.finditer(r"_sanitize_gateway_final_response\(", src)
        if not src[max(0, m.start() - 4):m.start()].endswith("def ")
    ]
    assert len(calls) == 2, f"expected 2 call sites, found {len(calls)}"
    for pos in calls:
        window = src[pos:pos + 700]
        assert "_is_suppressed_gateway_notice" in window, (
            "a _sanitize_gateway_final_response call site does not resolve "
            "suppression before falling back:\n" + window[:400]
        )


def test_only_the_status_path_collapses_to_nothing():
    """``_prepare_gateway_status_message`` is unsolicited chatter, so it may
    return None; the final-response path may not."""
    assert resolve_status_notice(
        __import__("agent.notice_collapse", fromlist=["x"]).SUPPRESSED_NOTICE
    ) is None
    assert resolve_status_notice("hello") == "hello"
    assert resolve_status_notice("") is None
    assert resolve_direct_reply("hello") == "hello"


# ---------------------------------------------------------------------------
# Three-site guard
# ---------------------------------------------------------------------------

AUTH_REPLY_MARKER = "Provider authentication failed"

# The renderer itself, and the tests that assert on its output.
GUARD_EXEMPT = ("agent/notice_collapse.py",)


SHARED_RENDERERS = ("collapse_provider_error_reply", "provider_auth_error_reply")


def find_inline_auth_reply_renders(root: pathlib.Path, base: pathlib.Path = None):
    """Return every string literal under ``root`` that builds the auth reply
    outside a logging call and outside the shared render path.

    An ``ast`` walk, not a substring scan, so quote style, the emoji, an
    implicit concatenation, ``.format()``, ``%`` and a brand new file in any
    subdirectory are all covered — the old line-based guard, which matched one
    exact double-quoted f-string in three hardcoded files, let every one of
    those through.

    A function that hands its text to one of ``SHARED_RENDERERS`` IS the shared
    path, so its literals are exempt. The exemption is per function, not per
    file: a new render site in a file that happens to use the helper elsewhere
    is still caught.
    """
    import ast

    base = base or root
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(base).as_posix()
        if rel in GUARD_EXEMPT or ("gateway/" + rel) in GUARD_EXEMPT:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - unparseable file
            continue

        exempt = set()
        for fn_node in ast.walk(tree):
            if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {
                getattr(c.func, "attr", None) or getattr(c.func, "id", None)
                for c in ast.walk(fn_node) if isinstance(c, ast.Call)
            }
            if names & set(SHARED_RENDERERS):
                for sub_node in ast.walk(fn_node):
                    exempt.add(id(sub_node))

        logged = set()
        for node in ast.walk(tree):
            # Anything textual passed to a logger is diagnostics, not a reply.
            if isinstance(node, ast.Call):
                fn = node.func
                name = getattr(fn, "attr", None) or getattr(fn, "id", None) or ""
                target = getattr(fn, "value", None)
                is_logger = name in {
                    "debug", "info", "warning", "warn", "error",
                    "exception", "critical", "log",
                } and (
                    "log" in str(getattr(target, "id", "")).lower()
                    or "log" in str(getattr(target, "attr", "")).lower()
                )
                if is_logger:
                    for sub_node in ast.walk(node):
                        logged.add(id(sub_node))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if AUTH_REPLY_MARKER not in node.value:
                continue
            if id(node) in logged or id(node) in exempt:
                continue
            offenders.append(
                f"{rel}:{getattr(node, 'lineno', 0)}: {node.value.strip()[:70]!r}"
            )
    return offenders


def test_no_gateway_site_renders_the_auth_reply_inline():
    """Every site that surfaces a provider auth failure must route through
    ``agent.notice_collapse`` so one outage cannot produce one reply per
    attempt. Fails loudly if any file under ``gateway/`` re-renders the string
    inline, in any quoting or formatting style."""
    offenders = find_inline_auth_reply_renders(REPO_ROOT / "gateway", base=REPO_ROOT)
    assert not offenders, (
        "provider auth reply rendered without the shared collapse helper:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("bypass", [
    '''reply = f"⚠️ Provider authentication failed: {exc}"''',
    """reply = f'Provider authentication failed: {exc}'""",
    '''reply = "Provider authentication failed: " + str(exc)''',
    '''reply = "Provider authentication failed: {}".format(exc)''',
    '''reply = "Provider authentication failed: %s" % exc''',
    '''reply = ("Provider authentication " "failed for this session")''',
])
def test_the_guard_catches_a_deliberate_bypass(tmp_path, bypass):
    """The old guard matched one exact double-quoted f-string in three
    hardcoded files; every form below walked straight past it."""
    fake_gateway = tmp_path / "gateway" / "platforms"
    fake_gateway.mkdir(parents=True)
    (fake_gateway / "sneaky_new_site.py").write_text(
        "def render(exc):\n    " + bypass + "\n    return reply\n",
        encoding="utf-8",
    )
    offenders = find_inline_auth_reply_renders(tmp_path / "gateway", base=tmp_path)
    assert offenders, f"guard missed a bypass: {bypass}"
    assert "sneaky_new_site.py" in offenders[0]


def test_the_guard_does_not_flag_logger_calls(tmp_path):
    """Diagnostics are not user-facing replies."""
    fake_gateway = tmp_path / "gateway"
    fake_gateway.mkdir(parents=True)
    (fake_gateway / "noisy.py").write_text(
        'import logging\nlogger = logging.getLogger(__name__)\n'
        'def f(exc):\n    logger.warning("Provider authentication failed for %s", exc)\n',
        encoding="utf-8",
    )
    assert find_inline_auth_reply_renders(fake_gateway, base=tmp_path) == []


def test_every_auth_site_still_imports_the_shared_helper():
    """Complements the ast guard: the three known sites must actually call the
    renderer, not merely avoid the literal."""
    missing = [
        rel for rel in (
            "gateway/run.py",
            "gateway/platforms/api_server.py",
            "gateway/platforms/api_server_runs.py",
        )
        if "provider_auth_error_reply" not in (REPO_ROOT / rel).read_text(encoding="utf-8")
        and "collapse_provider_error_reply" not in (REPO_ROOT / rel).read_text(encoding="utf-8")
    ]
    assert not missing, f"auth reply sites not on the shared helper: {missing}"


# ---------------------------------------------------------------------------
# F1 end to end, through the real gateway functions
# ---------------------------------------------------------------------------

def _collapse_mode(monkeypatch, interval=3600.0):
    """Force the gateway's provider-error renderer into collapse mode."""
    import agent.notice_collapse as nc

    monkeypatch.setattr(nc, "resolve_fallback_notification_mode", lambda cfg=None: "collapse")
    monkeypatch.setattr(nc, "resolve_fallback_notice_interval", lambda cfg=None: interval)


RAW_401 = "Error code: 401 - invalid api key sk-live-DEADBEEF req_a1b2 org-acme"


def test_sanitizer_signals_suppression_instead_of_returning_empty(monkeypatch):
    """The final-response sanitizer must hand its caller something it can tell
    apart from "not a provider error" — otherwise the caller falls through to
    the raw provider body."""
    from gateway.run import _sanitize_gateway_final_response
    from gateway.config import Platform

    _collapse_mode(monkeypatch)
    first = _sanitize_gateway_final_response(Platform.TELEGRAM, RAW_401, "chat-1")
    assert "authentication" in first and "sk-live" not in first

    second = _sanitize_gateway_final_response(Platform.TELEGRAM, RAW_401, "chat-1")
    assert is_suppressed_notice(second), "suppression must be distinguishable"
    assert second != "", "an empty string is what caused the raw-error leak"

    # A different chat is a different window (F3), end to end.
    other = _sanitize_gateway_final_response(Platform.TELEGRAM, RAW_401, "chat-2")
    assert "authentication" in other


def test_status_path_is_the_only_one_that_collapses_to_nothing(monkeypatch):
    """Unsolicited status chatter may go silent; nobody asked for it."""
    from gateway.run import _prepare_gateway_status_message
    from gateway.config import Platform

    _collapse_mode(monkeypatch)
    first = _prepare_gateway_status_message(Platform.TELEGRAM, "warn", RAW_401, "chat-1")
    assert first and "authentication" in first
    second = _prepare_gateway_status_message(Platform.TELEGRAM, "warn", RAW_401, "chat-1")
    assert second is None
    assert not is_suppressed_notice(second), "the sentinel must never escape"


def test_auth_sites_resolve_suppression_before_returning_a_reply():
    """The three auth render sites answer a message a person sent, so each must
    pass the collapsed reply through ``resolve_direct_reply``."""
    missing = []
    for rel in (
        "gateway/run.py",
        "gateway/platforms/api_server.py",
        "gateway/platforms/api_server_runs.py",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if "provider_auth_error_reply" in text and "resolve_direct_reply" not in text:
            missing.append(rel)
    assert not missing, (
        "auth reply sites that can return an empty body during an outage: "
        f"{missing}"
    )
