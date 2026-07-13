"""Mixin contract tests: GatewayRunner inherits Telegram topic-mode helpers via MRO.

After the Telegram topic-mode mixin extraction (god-file decomposition Phase 4, Cluster B-1),
the 11 topic-mode helpers live on ``GatewayTelegramTopicMixin``:

  - ``_telegram_topic_mode_enabled``
  - ``_is_telegram_topic_root_lobby``
  - ``_is_telegram_topic_lane``
  - ``_should_send_telegram_lobby_reminder``
  - ``_telegram_topic_root_lobby_message``
  - ``_telegram_topic_root_new_message``
  - ``_telegram_topic_new_header``
  - ``_record_telegram_topic_binding``
  - ``_sync_telegram_topic_binding``
  - ``_recover_telegram_topic_thread_id``
  - ``_normalize_source_for_session_key``

These tests assert behavior contracts (invariants), not snapshots:
  - The methods must resolve on a ``GatewayRunner`` instance via the MRO.
  - The mixin itself must expose all 11 methods.
  - The mixin must NOT depend on any ``self.*`` state for these 11 methods that
    would prevent them from being invoked or behaving safely on uninitialized/empty
    objects (where they fail closed or return defaults defensively).
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.telegram_topic_mixin import GatewayTelegramTopicMixin

_METHOD_NAMES = (
    "_telegram_topic_mode_enabled",
    "_is_telegram_topic_root_lobby",
    "_is_telegram_topic_lane",
    "_should_send_telegram_lobby_reminder",
    "_telegram_topic_root_lobby_message",
    "_telegram_topic_root_new_message",
    "_telegram_topic_new_header",
    "_record_telegram_topic_binding",
    "_sync_telegram_topic_binding",
    "_recover_telegram_topic_thread_id",
    "_normalize_source_for_session_key",
)


def test_mixin_exposes_all_eleven_methods():
    """Each named method must exist as an attribute on the mixin."""
    missing = [n for n in _METHOD_NAMES if not hasattr(GatewayTelegramTopicMixin, n)]
    assert not missing, f"Mixin missing methods: {missing}"


def test_runner_resolves_methods_via_mro():
    """A bare ``object.__new__(GatewayRunner)`` shell (no ``__init__``) must
    still resolve each method through the MRO — proves the mixin is wired into
    ``GatewayRunner``'s bases."""
    shell = object.__new__(GatewayRunner)
    missing = [n for n in _METHOD_NAMES if not hasattr(shell, n)]
    assert not missing, f"GatewayRunner shell missing methods via MRO: {missing}"


def test_methods_resolve_to_mixin_not_runner():
    """The methods must resolve to the *mixin*'s function objects, not stale
    copies left on ``GatewayRunner`` itself. After extraction, run.py must
    not re-declare them."""
    for name in _METHOD_NAMES:
        runner_attr = getattr(GatewayRunner, name, None)
        mixin_attr = getattr(GatewayTelegramTopicMixin, name, None)
        assert runner_attr is not None, f"{name} not reachable on GatewayRunner"
        assert mixin_attr is not None, f"{name} not defined on the mixin"
        assert runner_attr is mixin_attr, (
            f"{name} on GatewayRunner is not the mixin's implementation — "
            f"either a stale duplicate was left in run.py or the mixin is "
            f"not in the MRO. Expected identical function objects."
        )


def test_topic_methods_dont_require_init_state():
    """Methods must handle empty/uninitialized objects gracefully by returning
    safe defaults or failing closed, instead of raising AttributeError due to
    hard dependencies on __init__ state.
    """
    shell = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        chat_id="12345",
        user_id="67890",
        thread_id="1",
    )

    # _telegram_topic_mode_enabled should fail closed (return False) if self._session_db is missing
    assert shell._telegram_topic_mode_enabled(source) is False

    # _is_telegram_topic_root_lobby should return False since topic mode is disabled (fails closed)
    assert shell._is_telegram_topic_root_lobby(source) is False
    assert shell._is_telegram_topic_lane(source) is False

    # _should_send_telegram_lobby_reminder should auto-initialize its tracking dict and return True on first call
    assert not hasattr(shell, "_telegram_lobby_reminder_ts")
    assert shell._should_send_telegram_lobby_reminder(source) is True
    assert hasattr(shell, "_telegram_lobby_reminder_ts")

    # String messages should be returned cleanly
    assert "reserved for system commands" in shell._telegram_topic_root_lobby_message()
    assert "start a new parallel" in shell._telegram_topic_root_new_message()

    # Headers require topic lane to be True, which fails closed here
    assert shell._telegram_topic_new_header(source) is None

    # Sync and record bindings should run without crashing even if database is missing
    mock_entry = MagicMock()
    shell._record_telegram_topic_binding(source, mock_entry)  # no-op
    shell._sync_telegram_topic_binding(source, mock_entry, reason="test")  # no-op

    # Recover thread ID should return None if topic mode is disabled/db missing
    assert shell._recover_telegram_topic_thread_id(source) is None

    # Normalization should return the original source unchanged if recovery returns None
    assert shell._normalize_source_for_session_key(source) is source
