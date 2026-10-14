"""Invariant tests: a resumed turn must not lose the images the interrupted turn had.

``native_image_paths`` is consume-once. A turn interrupted by a gateway restart empties it,
and the resume turn is built text-only, so the transcript keeps the ``[Image attached at: …]``
handle while the request carries no pixels - the model then reports it can see an image that
is not there. These pin the contract that the resume branch re-arms the still-readable paths,
and only those.
"""
from types import SimpleNamespace

from gateway.run_turn_runner import TurnRunner


def _runner_with(state):
    class _Runner:
        def _session_state(self, session_key):
            return state

    return _Runner()


def _state(paths):
    return SimpleNamespace(persistent=SimpleNamespace(native_image_paths=list(paths)))


def _handle(path):
    return {"role": "user", "content": "[Image attached at: %s]\ndescribe this" % path}


def test_resume_rearms_images_dropped_by_the_interrupted_turn(tmp_path):
    """The paths the interrupted turn consumed come back, newest first, without duplicates."""
    older = tmp_path / "one.png"
    newer = tmp_path / "two.png"
    older.write_bytes(b"x")
    newer.write_bytes(b"x")

    state = _state([])
    turn = TurnRunner(_runner_with(state), None)

    # History is oldest -> newest, so the most recent image must be re-armed first.
    turn._rearm_resume_images_from_history(
        "sess:1", [_handle(older), {"role": "assistant", "content": "ok"}, _handle(newer)],
    )

    assert state.persistent.native_image_paths == [str(newer), str(older)]

    # A second resume must not duplicate what is already re-armed.
    turn._rearm_resume_images_from_history("sess:1", [_handle(older), _handle(newer)])
    assert state.persistent.native_image_paths == [str(newer), str(older)]


def test_resume_leaves_queued_paths_and_unreadable_paths_alone(tmp_path):
    """Paths the adapter already queued win, and a path that no longer exists is not re-armed."""
    queued = tmp_path / "queued.png"
    queued.write_bytes(b"x")
    missing = tmp_path / "gone.png"          # deliberately never written

    state = _state([str(queued)])
    turn = TurnRunner(_runner_with(state), None)
    turn._rearm_resume_images_from_history("sess:1", [_handle(missing)])

    assert state.persistent.native_image_paths == [str(queued)]

    # With nothing queued, an unreadable path is still dropped rather than re-attached.
    empty = _state([])
    turn = TurnRunner(_runner_with(empty), None)
    turn._rearm_resume_images_from_history("sess:1", [_handle(missing)])
    assert empty.persistent.native_image_paths == []

    # No session key, no history, and non-string content are all no-ops.
    for history in ([], None, [{"role": "user", "content": ["not", "a", "string"]}]):
        untouched = _state([])
        TurnRunner(_runner_with(untouched), None)._rearm_resume_images_from_history(
            "sess:1" if history is not None else None, history,
        )
        assert untouched.persistent.native_image_paths == []