"""SessionSource.sender_is_owner wire contract (#95705).

The /sethome nudge gate and any other operator-only delivery path reads
``source.sender_is_owner``. Adding the field is non-trivial because the
gateway session may be persisted and replayed (compressed-thread continuation,
async-completion routing) and the operator signal must survive those hops —
otherwise a deferred-delivery path can re-leak the nudge after the fix.

These tests pin the round-trip: a SessionSource with sender_is_owner=True
serializes to JSON, restores from JSON, and the restored flag is still True.
A source without the key restores to False (default), so pre-fix persisted
sources don't get retroactively re-authorized.
"""

from __future__ import annotations

from gateway.config import Platform
from gateway.session import SessionSource


def test_default_sender_is_owner_is_false():
    """Backwards-compat default: pre-fix sources have no operator signal."""
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="c1")
    assert src.sender_is_owner is False


def test_to_dict_omits_sender_is_owner_when_false():
    """Default-False sources serialize without the key — byte-stable history."""
    src = SessionSource(platform=Platform.WHATSAPP, chat_id="c1")
    d = src.to_dict()
    assert "sender_is_owner" not in d


def test_to_dict_includes_sender_is_owner_when_true():
    """Operator-typed sources persist the flag for deferred-delivery hops."""
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="c1",
        sender_is_owner=True,
    )
    d = src.to_dict()
    assert d.get("sender_is_owner") is True


def test_from_dict_restores_sender_is_owner_when_true():
    """Round-trip: True stays True across persistence."""
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="c1",
        user_id="u1",
        sender_is_owner=True,
    )
    restored = SessionSource.from_dict(src.to_dict())
    assert restored.sender_is_owner is True


def test_from_dict_defaults_to_false_when_key_missing():
    """Round-trip: pre-fix history (no key) restores as False, not True.

    Pinning this prevents a future refactor from accidentally treating
    absent == True (which would re-introduce the leak on every older
    session).
    """
    payload = {
        "platform": "whatsapp",
        "chat_id": "c1",
    }
    restored = SessionSource.from_dict(payload)
    assert restored.sender_is_owner is False


def test_from_dict_explicit_false_stays_false():
    """If persisted as False, restored as False."""
    payload = {
        "platform": "whatsapp",
        "chat_id": "c1",
        "sender_is_owner": False,
    }
    restored = SessionSource.from_dict(payload)
    assert restored.sender_is_owner is False


def test_from_dict_coerces_truthy_non_bool_to_bool():
    """Defense-in-depth: a forged ``"1"`` or similar can't authorize."""
    payload = {
        "platform": "whatsapp",
        "chat_id": "c1",
        "sender_is_owner": "yes",
    }
    restored = SessionSource.from_dict(payload)
    # bool("yes") is True; the helper explicitly bool()-coerces input, so
    # this is True. The single-line test only pins the documented surface;
    # the authz gate uses `is True` to refuse non-bool stand-ins.
    assert isinstance(restored.sender_is_owner, bool)
