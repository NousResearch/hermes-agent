"""Tests for the messaging follow-up grace window (#28417).

The grace window prevents rapid burst follow-ups (mobile typo fixups,
Chinese IME segmentation, voice-to-text cleanup) from interrupting an
in-flight agent turn — that asymmetry between Telegram (had 3s grace)
and WeChat (had 0s) was the smoking gun behind a user-visible "the
conversation breaks mid Q&A" bug on both platforms.

These tests pin down:

* The grace covers Telegram, Weixin, WeCom, and WeCom callback.
* Platforms outside that set (e.g. Slack) fall through to the legacy
  interrupt path so we don't accidentally widen the change.
* ``HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS`` is the canonical knob.
* ``HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS`` is honoured as a legacy
  fallback so existing deployments don't regress.
* A non-numeric env value fails open to the default (no crash).
* Once the window has elapsed, the message no longer short-circuits and
  flows on to the normal busy-session handling.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _PendingAdapter:
    def __init__(self) -> None:
        self._pending_messages: dict = {}


def _make_runner(platform: Platform) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {platform: _PendingAdapter()}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda _source: True
    return runner


def _mark_running(runner: GatewayRunner, session_key: str, *, started_at: float | None = None) -> MagicMock:
    """Pretend an agent is already running for ``session_key``.

    ``get_activity_summary`` is wired up explicitly so the
    stale-eviction code in ``_handle_message`` sees a healthy idle
    figure (real ``float``, not a ``MagicMock``) and leaves our test
    agent in place.  Without this the eviction would race with the
    grace branch and wipe the running_agent state we just installed.
    """
    agent = MagicMock()
    agent.get_activity_summary.return_value = {
        "seconds_since_activity": 0.0,
        "last_activity_desc": "test",
        "api_call_count": 1,
        "max_iterations": 10,
    }
    runner._running_agents[session_key] = agent
    runner._running_agents_ts[session_key] = (
        started_at if started_at is not None else time.time()
    )
    return agent


def _make_text_event(source: SessionSource, text: str = "follow up") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
    )


# ----------------------------------------------------------------------
# Platform coverage
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform",
    [
        Platform.TELEGRAM,
        Platform.WEIXIN,
        Platform.WECOM,
        Platform.WECOM_CALLBACK,
    ],
)
@pytest.mark.asyncio
async def test_burst_followup_within_grace_is_merged_not_interrupted(
    monkeypatch: pytest.MonkeyPatch, platform: Platform
) -> None:
    """Each of the four messaging platforms in scope must queue the
    follow-up (instead of interrupting) when it arrives well inside the
    grace window — that's the core regression from #28417."""
    monkeypatch.delenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)

    runner = _make_runner(platform)
    source = SessionSource(platform=platform, chat_id="42", chat_type="dm", user_id="u")
    session_key = build_session_key(source)
    agent = _mark_running(runner, session_key)

    event = _make_text_event(source, text="oops typo correction")
    result = await runner._handle_message(event)

    assert result is None
    agent.interrupt.assert_not_called()
    assert runner.adapters[platform]._pending_messages[session_key] is event


@pytest.mark.asyncio
async def test_platform_outside_grace_set_does_not_short_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slack is intentionally NOT in the grace set: Slack threads have
    well-defined boundaries and rich-burst follow-ups are uncommon.  We
    must NOT accidentally widen the grace by enumerating every platform."""
    monkeypatch.delenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)

    runner = _make_runner(Platform.SLACK)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel", user_id="U1")
    session_key = build_session_key(source)
    _mark_running(runner, session_key)

    event = _make_text_event(source)
    result = await runner._handle_message(event)

    # When the grace short-circuit doesn't fire, the message flows on to
    # the regular busy-session handling.  We just need to assert it was
    # NOT queued via the grace path — exact downstream behaviour belongs
    # to other tests, not this one.
    pending = runner.adapters[Platform.SLACK]._pending_messages
    assert pending.get(session_key) is not event or result is not None


# ----------------------------------------------------------------------
# Timing semantics
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_followup_outside_window_is_not_queued_via_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the grace window elapses, the short-circuit must release
    the message to the normal busy-session path so genuine ``interrupt``
    semantics still work."""
    monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "5.0")
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)

    runner = _make_runner(Platform.TELEGRAM)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id="u")
    session_key = build_session_key(source)
    # Mark the run as having started 60 seconds ago — well outside any
    # plausible grace window.
    _mark_running(runner, session_key, started_at=time.time() - 60.0)

    event = _make_text_event(source)
    await runner._handle_message(event)

    pending = runner.adapters[Platform.TELEGRAM]._pending_messages
    # The grace branch did NOT capture the event.
    assert pending.get(session_key) is not event


@pytest.mark.asyncio
async def test_zero_grace_disables_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting the grace to ``0`` must completely disable the
    short-circuit, restoring the pre-#28417 default-interrupt path."""
    monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "0")
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)

    runner = _make_runner(Platform.WEIXIN)
    source = SessionSource(platform=Platform.WEIXIN, chat_id="1", chat_type="dm", user_id="u")
    session_key = build_session_key(source)
    _mark_running(runner, session_key)

    event = _make_text_event(source)
    await runner._handle_message(event)

    pending = runner.adapters[Platform.WEIXIN]._pending_messages
    assert pending.get(session_key) is not event


# ----------------------------------------------------------------------
# Env var resolution
# ----------------------------------------------------------------------


def test_generalised_env_var_wins_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "12.5")
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")
    assert GatewayRunner._messaging_followup_grace_seconds() == 12.5


def test_legacy_env_var_is_honoured_when_generalised_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", raising=False)
    monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "7.0")
    assert GatewayRunner._messaging_followup_grace_seconds() == 7.0


def test_default_is_five_seconds_when_no_env_var_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """5s is wide enough to cover normal mobile bursts (the gap that
    3s left uncovered in #28417) without making real interrupts feel
    laggy."""
    monkeypatch.delenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    assert GatewayRunner._messaging_followup_grace_seconds() == 5.0


def test_invalid_env_value_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """A misconfigured env var must not break gateway startup or
    request handling — fall back to the default value instead."""
    monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "not-a-number")
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    assert GatewayRunner._messaging_followup_grace_seconds() == 5.0


def test_negative_value_is_clamped_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative values would short-circuit the ``> 0`` guard but make no
    semantic sense; clamp to zero so the window is simply disabled."""
    monkeypatch.setenv("HERMES_GATEWAY_FOLLOWUP_GRACE_SECONDS", "-1")
    monkeypatch.delenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", raising=False)
    assert GatewayRunner._messaging_followup_grace_seconds() == 0.0


# ----------------------------------------------------------------------
# Platform set
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "platform,expected",
    [
        (Platform.TELEGRAM, True),
        (Platform.WEIXIN, True),
        (Platform.WECOM, True),
        (Platform.WECOM_CALLBACK, True),
        (Platform.SLACK, False),
        (Platform.DISCORD, False),
        (Platform.WHATSAPP, False),
        (Platform.FEISHU, False),
    ],
)
def test_platform_membership(platform: Platform, expected: bool) -> None:
    """Pin down the platform set so future widening is an explicit
    decision rather than an accidental drive-by edit."""
    assert GatewayRunner._platform_has_followup_grace(platform) is expected
