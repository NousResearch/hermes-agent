"""
Tests for the Slack ``cron_continuable_surface`` extra key and its pairing warning.

``cron_continuable_surface: in_channel`` (paired with ``reply_in_thread: false``)
lets a continuable cron job deliver FLAT into a channel — no dedicated thread —
so a plain channel reply continues the job via the shared-channel session
``(slack, channel_id, None)``. See specs/cron-inchannel-continuable decisions
D1/D4/D5/D6.

- ``_cron_continuable_surface`` resolves the key: default ``"thread"``, coerces
  any unrecognised value to ``"thread"`` (fail safe), only ``"in_channel"``
  opts in.
- ``supports_inchannel_continuable`` is True on Slack (it has both a flat-reply
  outbound gate and a whole-channel inbound session bucket).
- ``_warn_if_inchannel_without_flat_reply`` warns (D5: warn, not hard-require)
  when ``in_channel`` is set without ``reply_in_thread: false`` — the misconfig
  fails SAFE to a threaded continuation, so it is a warning, not a rejection.
"""

import logging
import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_user_token_warning.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _make_adapter(extra):
    """object.__new__ skips __init__ (heavy setup) — established slack-test
    pattern. Attach a minimal config carrying only the ``extra`` dict."""
    adapter = object.__new__(SlackAdapter)
    cfg = MagicMock()
    cfg.extra = dict(extra)
    adapter.config = cfg
    return adapter


# --- capability flag -------------------------------------------------------

def test_slack_declares_inchannel_capability():
    """Slack has both halves the in_channel surface needs, so the class-level
    capability flag the cron scheduler reads generically must be True."""
    assert SlackAdapter.supports_inchannel_continuable is True


# --- surface resolver ------------------------------------------------------

def test_surface_defaults_to_thread():
    adapter = _make_adapter({})
    assert adapter._cron_continuable_surface() == "thread"


def test_surface_in_channel_opts_in():
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"})
    assert adapter._cron_continuable_surface() == "in_channel"


def test_surface_in_channel_case_and_whitespace_insensitive():
    adapter = _make_adapter({"cron_continuable_surface": "  In_Channel  "})
    assert adapter._cron_continuable_surface() == "in_channel"


def test_surface_explicit_thread():
    adapter = _make_adapter({"cron_continuable_surface": "thread"})
    assert adapter._cron_continuable_surface() == "thread"


def test_surface_unrecognised_value_coerces_to_thread():
    """Fail safe: any value that isn't 'in_channel' resolves to 'thread'."""
    adapter = _make_adapter({"cron_continuable_surface": "bogus"})
    assert adapter._cron_continuable_surface() == "thread"


# --- pairing warning (D5: warn, not hard-require) --------------------------

def test_warns_when_in_channel_without_flat_reply(caplog):
    """in_channel set, reply_in_thread left at its True default → warn."""
    adapter = _make_adapter({"cron_continuable_surface": "in_channel"})
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_inchannel_without_flat_reply("Acme")
    matched = [r for r in caplog.records
               if "cron_continuable_surface=in_channel" in r.message
               and "reply_in_thread=false" in r.message]
    assert matched


def test_warns_when_in_channel_with_reply_in_thread_true(caplog):
    """Explicit reply_in_thread: true alongside in_channel → still warn."""
    adapter = _make_adapter(
        {"cron_continuable_surface": "in_channel", "reply_in_thread": True}
    )
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_inchannel_without_flat_reply("Acme")
    assert any("cron_continuable_surface=in_channel" in r.message
               for r in caplog.records)


def test_no_warning_when_properly_paired(caplog):
    """in_channel + reply_in_thread: false is the correct pairing → silent."""
    adapter = _make_adapter(
        {"cron_continuable_surface": "in_channel", "reply_in_thread": False}
    )
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_inchannel_without_flat_reply("Acme")
    assert not any("cron_continuable_surface=in_channel" in r.message
                   for r in caplog.records)


def test_no_warning_when_surface_is_thread(caplog):
    """Default thread surface never warns about the pairing."""
    adapter = _make_adapter({"reply_in_thread": True})
    with caplog.at_level(logging.WARNING):
        adapter._warn_if_inchannel_without_flat_reply("Acme")
    assert not any("cron_continuable_surface=in_channel" in r.message
                   for r in caplog.records)
