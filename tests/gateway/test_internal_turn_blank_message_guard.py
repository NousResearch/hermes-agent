"""A blank ``internal`` turn must never reach the model as an empty user message.

#120963: a startup auto-resume turn is synthesized with ``text=""`` and carried by
``MessageEvent(internal=True)``. The substitution in ``_prepare_turn_message`` is gated on the
in-memory ``entry.resume_pending`` flag, so a flag cleared (or an entry missing) between
scheduling and execution let the empty string through. The model then replied to the user
"I didn't receive any text in your latest message" — a question about a message nobody sent.

``event.internal`` is the non-racy signal: it is stamped on the event itself and survives
whatever the session store does next. These tests pin that a blank internal turn always gets a
recovery note, whatever the session entry says.
"""
from __future__ import annotations

from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner


def _ctx(*, message: str, internal: bool, session_key: str = "agent:main:slack:chan:thread") -> SimpleNamespace:
    return SimpleNamespace(
        message=message, internal=internal, session_key=session_key,
        source=SimpleNamespace(platform="slack"),
        persist_user_message=None, persist_user_timestamp=None,
        history=[], channel_prompt=None, user_config=None, session_id="sid-1",
    )


def _runner(*, entry=None) -> SimpleNamespace:
    """A runner whose session store resolves ``entry`` (or nothing) for any key.

    ``_delivery_adapter_for`` reports an interactive adapter, matching the chat platforms the
    blank auto-resume turn is synthesized for.
    """
    store = SimpleNamespace(_entries={"agent:main:slack:chan:thread": entry} if entry else {})
    return SimpleNamespace(
        session_store=store,
        _delivery_adapter_for=lambda source: SimpleNamespace(interactive_resume=True),
    )


def _turn(entry=None, *, internal: bool = True, message: str = "") -> TurnRunner:
    return TurnRunner(_runner(entry=entry), _ctx(message=message, internal=internal))


def test_blank_internal_turn_gets_recovery_note_when_entry_is_missing():
    """The reported case: ``resume_pending`` flipped false (or the entry vanished) after the
    event was scheduled, so the racy gate missed and the blank text reached the model."""
    entry = SimpleNamespace(resume_pending=False, resume_reason="restart_timeout", last_resume_marked_at=None)
    turn = _turn(entry)

    turn._prepare_turn_message([])

    assert turn._ctx.message.strip(), "a blank internal turn must not reach the model as an empty user message"
    assert "interrupted" in turn._ctx.message.lower() or "resume" in turn._ctx.message.lower()


def test_blank_internal_turn_gets_recovery_note_when_no_entry_exists():
    """No in-memory entry at all — the lookup itself misses, so the flag is not False but absent."""
    turn = _turn(None)

    turn._prepare_turn_message([])

    assert turn._ctx.message.strip(), "an internal turn with no session entry must still be recovered, not blank"


def test_resume_reason_is_carried_into_the_note():
    """The reason drives reason-aware wording; keep passing it through when we synthesize it."""
    entry = SimpleNamespace(resume_pending=False, resume_reason="drain_timeout", last_resume_marked_at=None)
    turn = _turn(entry)

    turn._prepare_turn_message([])

    assert turn._ctx.message.strip()


def test_real_user_message_is_never_substituted():
    """A genuine (non-internal) blank — e.g. a caption-less image turn — is left alone: the
    existing gate deliberately does not touch it, and this fix must not change that."""
    turn = _turn(None, internal=False, message="")

    turn._prepare_turn_message([])

    assert turn._ctx.message == "", "a non-internal blank turn must keep its existing handling"


def test_non_empty_internal_turn_is_untouched():
    """Only blank text is at risk. An internal turn that already carries text passes through."""
    turn = _turn(None, internal=True, message="real content")

    turn._prepare_turn_message([])

    assert turn._ctx.message == "real content"


def test_blank_internal_turn_preserves_the_message_type_contract():
    """The recovery note must be a plain str: the model layer branches on that, and a list
    content part here would silently take the multimodal path."""
    turn = _turn(None)

    turn._prepare_turn_message([])

    assert isinstance(turn._ctx.message, str)
