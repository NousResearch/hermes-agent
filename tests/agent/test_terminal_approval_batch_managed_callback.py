"""The desktop terminal batch must not drop Relay's managed-callback marker.

``bind_prepared_dispatch`` wraps the dispatch in a Context captured while Relay was *outside* the
tool callback (depth 0). Re-entering that Context from inside the callback wipes
``_MANAGED_CALLBACK_DEPTH``, so nested auxiliary calls — the smart-approval guardian — resolve a
managed Relay attempt and die with "Synchronous Relay LLM execution cannot run on an event-loop
thread": every flagged command in a desktop batch escalates to a human prompt instead of being
auto-approved, while the same command issued alone is fine.

The invariant comes from ``relay_runtime``'s own contract: work forwarded via ``copy_context()``
inside a managed callback still runs unmanaged. These tests pin it at the binder.

Follow-up to #82313, which routes nested calls inside a managed callback to the unmanaged path on the
strength of ``_MANAGED_CALLBACK_DEPTH`` — the batch binder was dropping that marker (see also #77244,
closed as completed by #82313).
"""
import asyncio
from types import SimpleNamespace

from agent import relay_runtime
from agent import terminal_approval_batch as batch


class _Slot:
    """Minimal slot: the binder only needs a batch agent, a tid list and the cancel check."""

    def __init__(self):
        self.batch = SimpleNamespace(
            agent=SimpleNamespace(
                session_id="probe-session", _current_turn_id="probe-turn", _interrupt_requested=False
            )
        )
        self.tids = []

    def check_cancelled(self):
        pass


def _bind_recording_depth(seen):
    return batch.bind_prepared_dispatch(
        lambda *args, **kwargs: seen.setdefault("depth", relay_runtime._MANAGED_CALLBACK_DEPTH.get())
    )


def test_bound_dispatch_preserves_managed_callback_marker():
    """Invoked from inside a managed callback (loop running, guard set), the bound dispatch must
    still see depth > 0 — otherwise the guardian takes the managed path and escalates."""
    token = batch._slot.set(_Slot())
    try:
        seen = {}
        bound = _bind_recording_depth(seen)

        async def managed_callback():
            # Mirrors agent/relay_tools.py: the tool callback runs synchronously inside the
            # pipeline's asyncio.run loop, with managed_callback_guard() entered.
            with relay_runtime.managed_callback_guard():
                assert relay_runtime._MANAGED_CALLBACK_DEPTH.get() == 1
                bound("python3 -c \"print('x')\"")

        asyncio.run(managed_callback())
        assert seen["depth"] == 1, "marker lost: nested aux calls would run managed and escalate"
    finally:
        batch._slot.reset(token)


def test_bound_dispatch_leaves_unmanaged_invocation_unmarked():
    """Outside a managed callback the binder must not claim one — asserting depth would suppress
    Relay instrumentation for the whole subtree."""
    token = batch._slot.set(_Slot())
    try:
        seen = {}
        bound = _bind_recording_depth(seen)
        bound("echo hi")
        assert seen["depth"] == 0
    finally:
        batch._slot.reset(token)
