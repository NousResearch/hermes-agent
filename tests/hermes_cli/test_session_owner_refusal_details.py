"""The ``Details:`` line every owner-refusal message ends with.

A refused user gets one plain sentence, then this line; a bug report gets the same text. It must
therefore identify the holder that is actually in the way, and must not claim more than the registry
knows.

``started_at`` is written when the lease entry is built -- the CLAIM, not the moment the chat was
opened -- so a re-claim resets it. Observed: the same stored session was reported as
"opened by desktop 26m ago" and then "opened by desktop 1h30m ago" within two hours, and both
readings sent triage looking for a chat of that age instead of at the lease.
"""

import time

from hermes_cli.active_sessions import (
    session_already_owned_message,
    session_owner_details,
)

_SID = "20260101_000000_aaaa"
_FIRST_LINE = ("This chat is open in another Hermes window/terminal. "
               "Use it there, or start a new chat here.")


def _entry(**over):
    entry = {
        "lease_id": "L1",
        "session_id": _SID,
        "surface": "desktop",
        "pid": 4242,
        "track_liveness": True,
        "started_at": time.time() - 90 * 60,
    }
    entry.update(over)
    return entry


def test_details_line_identifies_the_holder():
    line = session_owner_details(_SID, _entry())
    assert "pid 4242" in line, "a bug report cannot trace the holder without its pid"
    assert "desktop" in line
    assert "1h30m" in line, "the line must carry the lease age"


def test_details_line_degrades_without_a_usable_pid():
    for bad in (None, 0, -1, "abc", True):
        line = session_owner_details(_SID, _entry(pid=bad, started_at=None))
        assert _SID in line
        assert "pid" not in line, f"pid={bad!r} is not a holder identity"


def test_details_line_keeps_the_surface_fallback():
    assert "another surface" in session_owner_details(_SID, _entry(surface=None))


def test_refusal_keeps_its_first_line_contract():
    message = session_already_owned_message(_SID, _entry())
    first, *rest = message.splitlines()
    assert first == _FIRST_LINE
    assert len(rest) == 1 and rest[0].startswith(f"Details: session {_SID} ")
