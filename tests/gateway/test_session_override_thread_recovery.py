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


def test_normalize_passthrough_when_no_recovery():
    """No recovery -> source returned unchanged (identity)."""
    runner = _make_runner(recovered_thread_id=None)
    src = _topic_dm_source(thread_id="42")

    normalized = runner._normalize_source_for_session_key(src)

    assert normalized is src


def test_normalize_swallows_recovery_exceptions():
    """Recovery raising must not break the command — return the raw source."""
    runner = _make_runner()
    runner._recover_telegram_topic_thread_id = MagicMock(side_effect=RuntimeError("boom"))
    src = _topic_dm_source(thread_id="")

    normalized = runner._normalize_source_for_session_key(src)

    assert normalized is src


def test_override_key_matches_message_turn_key_after_recovery():
    """The bug, end to end at the key level.

    /model arrives as a lobby reply (thread_id="").  The next message turn
    runs recovery and lands on the bound topic ("42").  After the fix, the
    key the command stores under must equal the key the message turn reads.
    """
    runner = _make_runner(recovered_thread_id="42")

    # --- /model command path (raw inbound is a lobby reply) ---
    command_source = _topic_dm_source(thread_id="")
    normalized_command_source = runner._normalize_source_for_session_key(command_source)
    # _session_key_for_source falls back to build_session_key when there is no
    # session_store; emulate that resolution here directly.
    command_key = build_session_key(normalized_command_source)

    # --- next message turn path (recovery already applied to source) ---
    message_turn_source = _topic_dm_source(thread_id="42")
    message_turn_key = build_session_key(message_turn_source)

    assert command_key == message_turn_key

    # And the orphaning the bug caused: storing under the RAW (pre-recovery)
    # key would NOT be found by the message turn.
    raw_key = build_session_key(command_source)
    assert raw_key != message_turn_key
