"""Two id-less review guards never share one registry slot.

``prepare_background_review_run`` publishes a run under ``(profile, session)``. A guard that
carries no session id (a cold ``/refine``'s per-session ``SimpleNamespace``,
``gateway/session_refine.py``) used to collapse to ``(profile, '')``, so while one such review ran
every other anonymous guard in the profile was refused as its duplicate. An anonymous guard is
keyed by object instead, like ``run_agent._review_queue_key``; a second publication on the SAME
guard still joins the running review, and a finished run frees its slot.
"""

from types import SimpleNamespace

from agent import review_admission
from agent.background_review import (
    finish_background_review_run,
    prepare_background_review_run,
)


def test_two_anonymous_guards_both_publish_and_the_same_guard_still_joins():
    first, second = SimpleNamespace(), SimpleNamespace()
    run_a = prepare_background_review_run(first)
    assert run_a is not None
    try:
        run_b = prepare_background_review_run(second)
        try:
            assert run_b is not None, (
                "the second anonymous guard was refused as a duplicate"
            )
            assert run_a._review_owner_key != run_b._review_owner_key
            assert second._background_review_run is run_b
            # No shared anonymous bucket: nothing is published under the empty session id.
            assert review_admission.current_review_run("") is None
            # The same guard joins its running review instead of starting another.
            assert prepare_background_review_run(first) is None
            assert first._background_review_run is run_a
        finally:
            if run_b is not None:
                finish_background_review_run(second, run_b)
    finally:
        finish_background_review_run(first, run_a)
    assert first._background_review_run is None
    # A finished run frees the guard's slot.
    again = prepare_background_review_run(first)
    assert again is not None
    finish_background_review_run(first, again)
