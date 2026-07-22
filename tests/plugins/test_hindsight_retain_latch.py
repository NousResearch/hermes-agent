"""Retain latch after shutdown() must not outlive the session it protected.

``shutdown()`` sets ``_shutting_down`` so retains can't race interpreter
teardown. But on a long-lived process (the messaging gateway) the same
provider instance serves many sessions, and the ``sync_turn()`` /
``queue_prefetch()`` gates returned before anything could clear the flag —
one shutdown() silently dropped every later session's writes while reads
kept working. ``on_session_switch()`` is the revive point: a new session is
affirmative proof the process is not tearing down (atexit never starts new
sessions).
"""

import importlib
import json

hindsight = importlib.import_module("plugins.memory.hindsight")
HindsightMemoryProvider = hindsight.HindsightMemoryProvider


def _make_provider(monkeypatch):
    """Bare provider wired so sync_turn() reaches the writer hermetically."""
    provider = HindsightMemoryProvider()
    provider._session_id = "sess-a"
    provider._document_id = "sess-a-doc-1"
    # Keep the retain path off the network: no API probe, no client, and no
    # initialize()-only config fields (these tests pin lifecycle, not kwargs).
    monkeypatch.setattr(
        provider, "_resolve_retain_target", lambda fallback: (fallback, None)
    )
    monkeypatch.setattr(
        provider,
        "_build_retain_kwargs",
        lambda content, **kw: {"content": content, **kw},
    )
    retained = []
    monkeypatch.setattr(
        provider, "_run_hindsight_operation", lambda op: retained.append(op)
    )
    # No atexit registration from tests.
    monkeypatch.setattr(provider, "_register_atexit", lambda: None)
    return provider, retained


def _turn(provider, text):
    provider.sync_turn(text, "ack", session_id=provider._session_id)
    provider._retain_queue.join()


def test_sync_turn_retains_before_shutdown(monkeypatch):
    provider, retained = _make_provider(monkeypatch)
    _turn(provider, "hello")
    assert len(retained) == 1
    provider.shutdown()


def test_sync_turn_dropped_after_shutdown(monkeypatch):
    provider, retained = _make_provider(monkeypatch)
    _turn(provider, "hello")
    provider.shutdown()
    provider.sync_turn("after teardown", "ack", session_id="sess-a")
    assert len(retained) == 1  # dropped: teardown protection holds


def test_session_switch_revives_retains_after_shutdown(monkeypatch):
    provider, retained = _make_provider(monkeypatch)
    _turn(provider, "hello")
    provider.shutdown()

    provider.on_session_switch("sess-b")
    assert not provider._shutting_down.is_set()
    provider._retain_queue.join()
    flushed = len(retained)  # the switch may re-flush the old session's turns
    assert flushed >= 1

    provider._session_id = "sess-b"
    _turn(provider, "next session's turn")
    assert len(retained) == flushed + 1  # the new session's write landed

    # The revived write carries the NEW session's turn, not stale state.
    turns = provider._session_turns
    assert len(turns) == 1
    assert "next session's turn" in json.loads(turns[0])[0]["content"]


def test_session_switch_flushes_buffered_turns_after_shutdown(monkeypatch):
    # retain_every_n_turns > 1 buffers turns; a switch after shutdown() must
    # still flush the old session's buffer (same data-loss class).
    provider, retained = _make_provider(monkeypatch)
    provider._retain_every_n_turns = 5
    provider.sync_turn("buffered", "ack", session_id="sess-a")
    provider.shutdown()
    assert len(retained) == 0

    provider.on_session_switch("sess-b")
    provider._retain_queue.join()
    assert len(retained) == 1  # the old session's buffered turn flushed
