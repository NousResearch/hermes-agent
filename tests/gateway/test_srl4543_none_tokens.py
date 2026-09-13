"""SRL-4543 Gate B rodada 3: tokens=None/[] must reset every identity ContextVar safely.

Regresses the exact bug found in Gate B rodada 2 (Kimi): clear_session_vars(None) raised
TypeError from len(None) BEFORE any ContextVar reset ran, leaving a previous turn's identity
(HERMES_SESSION_USER_ID etc.) leaking into the next admission on the same task/thread.
"""
import pytest

from gateway.session_context import (
    clear_session_vars,
    get_session_env,
    set_session_vars,
)


def test_clear_session_vars_none_resets_identity_without_raising(caplog):
    # Simulate a prior turn that left identity bound (e.g. admission raised before returning
    # tokens, or a caller that legitimately has nothing to unwind).
    tokens = set_session_vars(user_id="bob@example.invalid")
    # tokens=None must NOT raise TypeError, and must still reset every var to baseline.
    clear_session_vars(None)
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert "tokens was None" in caplog.text or "falsy" in caplog.text
    # Clean up the still-open outer token from set_session_vars above.
    clear_session_vars(tokens)


def test_clear_session_vars_empty_list_resets_identity_without_raising(caplog):
    set_session_vars(user_id="carol@example.invalid")
    clear_session_vars([])
    assert get_session_env("HERMES_SESSION_USER_ID") == ""


def test_clear_session_vars_none_does_not_leak_into_next_turn():
    """The concurrent scenario from the original SRL-4543 diagnosis: turn A's identity must
    never survive into turn B's context after a malformed clear."""
    set_session_vars(user_id="alice@example.invalid")
    clear_session_vars(None)  # simulates a broken/incomplete unwind for turn A
    # Turn B admits its own identity in the same context.
    tokens_b = set_session_vars(user_id="bob@example.invalid")
    assert get_session_env("HERMES_SESSION_USER_ID") == "bob@example.invalid"
    clear_session_vars(tokens_b)
    assert get_session_env("HERMES_SESSION_USER_ID") == ""


def test_clear_session_vars_malformed_short_list_still_raises_after_reset():
    """A genuinely truncated (but non-empty) tokens list is still a caller bug and must still
    raise ValueError -- but only AFTER every var was reset to baseline."""
    tokens = set_session_vars(user_id="dave@example.invalid")
    truncated = tokens[:1]
    with pytest.raises(ValueError, match="tokens length mismatch"):
        clear_session_vars(truncated)
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
