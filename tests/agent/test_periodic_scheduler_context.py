"""Context propagation contract for the shared periodic scheduler."""

from contextvars import ContextVar
import threading

from agent.periodic_scheduler import PeriodicScheduler


def test_periodic_callback_keeps_schedule_context():
    """A callback must run in the Context that owned the scheduled handle."""
    scheduler = PeriodicScheduler()
    profile_scope = ContextVar("periodic_scheduler_profile_scope", default="launch-profile")
    observed = []
    finished = threading.Event()

    def callback():
        observed.append(profile_scope.get())
        finished.set()
        return False

    token = profile_scope.set("served-profile")
    try:
        handle = scheduler.schedule(callback, 0.01)
    finally:
        profile_scope.reset(token)

    try:
        assert finished.wait(2.0), "periodic callback did not run"
        assert observed == ["served-profile"], (
            "periodic callback lost the Context captured at schedule time"
        )
    finally:
        handle.cancel(wait=1.0)
