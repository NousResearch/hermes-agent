"""Regression tests for #30479 — session-scoped /model and /reasoning overrides
silently lost on Telegram forum/DM topics and after compression session splits.

Root cause: ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
``_recover_telegram_topic_thread_id`` (lobby/stripped reply -> the user's
last-active bound topic) *before* deriving the session key for a message turn.
The ``/model`` and ``/reasoning`` command handlers derived their override key
from the raw inbound ``event.source``, skipping that recovery — so the override
was stored under one key and the next message turn read a different key, and the
override was dropped.

Fix: both command handlers normalize the source via
``_normalize_source_for_session_key`` before deriving the override key, so
storage and read keys are identical.
"""

import threading
from unittest.mock import MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


def _make_runner(recovered_thread_id=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = None
    runner.session_store = None
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    # Stub topic recovery: returns the bound topic id for a lobby message,
    # None otherwise (the real method's contract).
    runner._recover_telegram_topic_thread_id = MagicMock(return_value=recovered_thread_id)
    return runner


def _topic_dm_source(thread_id):
    """A Telegram DM in topic mode. thread_id="" / "1" == General/lobby."""
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="555",
        chat_name="Forum DM",
        chat_type="dm",
        user_id="user-1",
        thread_id=thread_id,
    )


def test_normalize_rewrites_lobby_thread_to_bound_topic():
    """A lobby (stripped) reply gets pinned to the user's bound topic id."""
    runner = _make_runner(recovered_thread_id="42")
    src = _topic_dm_source(thread_id="")  # lobby/General — no message_thread_id

    normalized = runner._normalize_source_for_session_key(src)

    assert normalized.thread_id == "42"
    # Original source is left untouched (we return a copy).
    assert src.thread_id == ""


