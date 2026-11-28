"""The owner-refusal ``Details:`` line says what the age measures and when a Desktop hold ends (#154162).

The registry records when a holder CLAIMED a chat, not its last activity, so "opened by desktop 13m ago"
read as "idle for 13 minutes" while that window had written a turn moments before. A liveness-tracked
(Desktop) hold is also released by the idle reaper; without saying so the user concludes the chat is stuck
until that window closes.
"""

import time

from hermes_cli.active_sessions import session_already_owned_message, session_owner_details


def _entry(**extra):
    return {"surface": "desktop", "started_at": time.time() - 13 * 60, **extra}


def test_details_say_claimed_not_opened():
    details = session_owner_details("sid-1", _entry())
    assert details.startswith("Details: session sid-1 claimed by desktop 13m ago.")
    assert "opened by" not in details


def test_a_liveness_tracked_hold_says_it_is_released_when_idle():
    details = session_owner_details("sid-1", _entry(track_liveness=True))
    assert "released automatically" in details and "idle" in details


def test_an_untracked_hold_makes_no_release_promise():
    # A CLI/TUI lease is held until that process exits; the idle reaper never releases it.
    assert "released" not in session_owner_details("sid-1", _entry())


def test_the_user_sentence_stays_the_first_line():
    lines = session_already_owned_message("sid-1", _entry(track_liveness=True)).splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("This chat is open in another Hermes window/terminal.")
    assert lines[1].startswith("Details: ")
