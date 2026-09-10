"""Session-identity fencing for the async dialectic prefetch (_spawn_dialectic /
_consume_pending_dialectic) — the sibling of recall_sync.py's _recall_generation fence,
but scoped to session-identity changes only (never per-turn) since dialectic results are
meant to survive across turn boundaries."""
import threading
import time

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


class DialecticManager:
    """Fake session manager whose dialectic_query() can be held open with a barrier."""

    def __init__(self):
        self.calls = []

    def dialectic_query(self, session, prompt, **kwargs):
        self.calls.append((session, prompt, kwargs))
        return f"dialectic-result-for:{session}"


def make_provider(**options):
    provider = HonchoMemoryProvider()
    cfg = HonchoClientConfig(timeout=5, **options)
    provider._config = cfg
    provider._manager = DialecticManager()
    provider._session_key = "session-a"
    provider._session_initialized = True
    for name in ("recall_mode", "injection_frequency", "context_cadence",
                 "dialectic_cadence", "dialectic_depth", "dialectic_depth_levels", "reasoning_heuristic"):
        setattr(provider, f"_{name}", getattr(cfg, name))
    return provider


def test_session_switch_during_fetch_prevents_publish_and_cross_session_leak():
    """A dialectic fetch fired for session A must not land in _prefetch_result (and
    therefore must never be consumable by session B) if on_session_switch fires while
    the background HTTP call is still in flight."""
    provider = make_provider()
    entered, release = threading.Event(), threading.Event()

    def blocked(session, prompt, **kwargs):
        entered.set()
        assert release.wait(3), "test setup: release was never signalled"
        return f"LEAKED CONTENT FROM {session}"

    provider._manager.dialectic_query = blocked
    provider.on_turn_start(1, "Plan the garden")
    thread = provider._spawn_dialectic(
        "Plan the garden", thread_name="test-dialectic", fired_at=1, log_label="test prefetch")
    assert entered.wait(2)

    # Session switch lands while the fetch above is still blocked mid-flight — this is the
    # exact race: honcho_sync's queue_prefetch() fired a background dialectic thread, then a
    # /new, /branch, /resume, or compression-triggered session switch happened before it returned.
    provider.on_session_switch("session-b")
    provider._session_key = "session-b"
    provider._turn_count = 1  # new session, its own turn 1 (turn numbers can repeat)

    release.set()
    thread.join(timeout=3)
    assert not thread.is_alive()

    # The stale session-A result must never have been published into shared state...
    assert provider._prefetch_result == ""
    assert provider._prefetch_result_fired_at == -999
    assert provider._prefetch_result_generation is None
    # ...so session B's own consume call gets nothing, never session A's leaked content.
    assert provider._consume_pending_dialectic() == ""


def test_session_switch_between_publish_and_consume_discards_result():
    """Even if a result already landed in _prefetch_result before the switch (publish raced
    ahead of on_session_switch), _consume_pending_dialectic must still reject it once the
    generation has moved on — the switch may occur in the gap between publish and consume."""
    provider = make_provider()
    provider.on_turn_start(1, "Plan the garden")
    thread = provider._spawn_dialectic(
        "Plan the garden", thread_name="test-dialectic", fired_at=1, log_label="test prefetch")
    thread.join(timeout=3)
    assert provider._prefetch_result == "dialectic-result-for:session-a"

    # Now the session switches before anyone calls _consume_pending_dialectic() for it.
    provider.on_session_switch("session-b")
    provider._session_key = "session-b"
    provider._turn_count = 1

    assert provider._consume_pending_dialectic() == ""


def test_normal_cross_turn_consumption_is_not_broken_by_the_fence():
    """Regression guard for the fence itself: a dialectic result fired on turn N, with no
    session switch, must still be consumable on a later turn within the same session — the
    fence must key off session-identity events only, never off on_turn_start."""
    provider = make_provider()
    provider.on_turn_start(1, "Plan the garden")
    thread = provider._spawn_dialectic(
        "Plan the garden", thread_name="test-dialectic", fired_at=1, log_label="test prefetch")
    thread.join(timeout=3)
    assert provider._prefetch_result == "dialectic-result-for:session-a"

    # Turn advances within the SAME session (no session switch) — result must still be usable.
    provider.on_turn_start(2, "Debug the compiler")
    assert provider._consume_pending_dialectic() == "dialectic-result-for:session-a"


def test_note_dialectic_failure_not_counted_against_new_session():
    """A late failure from an abandoned (pre-switch) dialectic fetch must not widen the
    NEW session's empty-streak backoff."""
    provider = make_provider()
    entered, release = threading.Event(), threading.Event()

    def blocked_then_raise(session, prompt, **kwargs):
        entered.set()
        assert release.wait(3)
        raise RuntimeError("simulated backend failure for the old session")

    provider._manager.dialectic_query = blocked_then_raise
    provider.on_turn_start(1, "Plan the garden")
    thread = provider._spawn_dialectic(
        "Plan the garden", thread_name="test-dialectic", fired_at=1, log_label="test prefetch")
    assert entered.wait(2)

    provider.on_session_switch("session-b")
    provider._session_key = "session-b"
    streak_before = provider._dialectic_empty_streak

    release.set()
    thread.join(timeout=3)
    assert not thread.is_alive()

    assert provider._dialectic_empty_streak == streak_before
