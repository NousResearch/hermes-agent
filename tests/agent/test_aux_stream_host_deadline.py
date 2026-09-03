"""#99692 — the streamed auxiliary summary must not outlive its compression host.

Background
----------
``run_compress_context_with_progress_timeout`` arms a wall-clock deadline on the
``CompressionCommitFence`` (``set_total_ceiling_seconds``), whose docstring calls
it "the wall-clock deadline **shared by the host and worker**".  Only the host
ever read it.

``8207862212`` (fix(compression): stop timeout paths from blocking retries)
closed the first half: a cancelled fence now releases the compression OWNER,
which frees the pool slot and the session lease.  It left the second half open
by design — its own comment says the isolated provider daemon runs on "until
the auxiliary stream's longer absolute ceiling expires".

That ceiling is ``_aux_stream_total_ceiling`` = ``max(600, 4 * aux_timeout)``:
>= the default host ceiling (600s) for every configured timeout, and it starts
counting later (after pool admission, serialization, prompt build and TTFT).
So the daemon holding the socket is *always* still streaming when its host gives
up — 2400s with the reporter's ``auxiliary.compression.timeout: 600`` — billing
every token of a summary the fence is already guaranteed to refuse, and stacking
one fresh orphan per turn because the session never shrank.

These tests pin the missing half of that shared deadline: the stream consumer
must stop at the host's deadline, including on the isolated provider daemon
that ``_run_protected_sync_provider_call`` spawns.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import auxiliary_client as aux
from agent.conversation_compression import (
    DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS,
    CompressionCommitFence,
)


def _chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp-1",
        model="test-model",
        usage=None,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=None,
                delta=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
    )


class _Stream:
    """Chunk iterator that records how far the consumer drained it."""

    def __init__(self, count: int = 50) -> None:
        self._count = count
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        for _ in range(self._count):
            self.yielded += 1
            yield _chunk("x")

    def close(self) -> None:
        self.closed = True


class _AsyncStream(_Stream):
    async def __aiter__(self):  # pragma: no cover - exercised via asyncio.run
        for _ in range(self._count):
            self.yielded += 1
            yield _chunk("x")


# ── The structural gap the bug lives in ──────────────────────────────────


def test_stream_ceiling_structurally_outlives_the_default_host_ceiling():
    """The worker's own budget is >= the host's for every configured timeout.

    This is the arithmetic that guarantees the orphan: there is no aux timeout
    for which ``_aux_stream_total_ceiling`` lands below the 600s default host
    ceiling, and the reporter's ``auxiliary.compression.timeout: 600`` puts it
    at 2400s — a 30-minute window in which an abandoned provider daemon keeps
    streaming a summary nobody can commit.
    """
    for aux_timeout in (None, 0, 30.0, 120.0, 300.0):
        assert (
            aux._aux_stream_total_ceiling(aux_timeout)
            >= DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
        )
    assert aux._aux_stream_total_ceiling(600.0) == 2400.0
    assert (
        aux._aux_stream_total_ceiling(600.0)
        - DEFAULT_CONTEXT_TOTAL_CEILING_SECONDS
        == 1800.0
    )


# ── The fence must publish the deadline it already owns ──────────────────


def test_commit_fence_publishes_its_shared_deadline():
    fence = CompressionCommitFence()
    assert fence.deadline_monotonic is None

    fence.set_total_ceiling_seconds(600.0)
    published = fence.deadline_monotonic
    assert published is not None
    assert 590.0 < published - time.monotonic() <= 600.0
    assert not fence.deadline_exceeded

    fence.set_total_ceiling_seconds(0.001)
    time.sleep(0.01)
    assert fence.deadline_exceeded
    assert fence.deadline_monotonic <= time.monotonic()


# ── The stream consumer must honour it ───────────────────────────────────


def test_streamed_summary_stops_at_an_elapsed_host_deadline():
    """A host that already gave up must not leave the worker streaming on."""
    stream = _Stream(count=50)
    with aux.aux_stream_deadline(time.monotonic() - 1.0):
        with pytest.raises(TimeoutError) as excinfo:
            aux._aggregate_chat_stream(stream, model="m", total_ceiling=2400.0)

    # "timed out" keeps _is_timeout_error classification identical to a
    # request timeout, so the existing recovery chains are unchanged.
    assert "timed out" in str(excinfo.value)
    assert "host compression deadline" in str(excinfo.value)
    # Stopped on the first frame instead of draining the whole stream, and the
    # HTTP response was closed rather than left dangling.
    assert stream.yielded == 1
    assert stream.closed is True


def test_streamed_summary_runs_to_completion_under_a_live_host_deadline():
    stream = _Stream(count=5)
    with aux.aux_stream_deadline(time.monotonic() + 600.0):
        response = aux._aggregate_chat_stream(
            stream, model="m", total_ceiling=2400.0
        )
    assert response.choices[0].message.content == "xxxxx"
    assert stream.yielded == 5


def test_no_host_deadline_keeps_the_historical_ceiling_behaviour():
    """Every non-compression aux caller must be byte-for-byte unchanged."""
    stream = _Stream(count=5)
    response = aux._aggregate_chat_stream(stream, model="m", total_ceiling=2400.0)
    assert response.choices[0].message.content == "xxxxx"
    assert stream.yielded == 5

    # An installed-then-exited scope must not leak into the next call.
    with aux.aux_stream_deadline(time.monotonic() - 1.0):
        pass
    stream2 = _Stream(count=3)
    assert (
        aux._aggregate_chat_stream(
            stream2, model="m", total_ceiling=2400.0
        ).choices[0].message.content
        == "xxx"
    )


def test_none_deadline_is_a_no_op_passthrough():
    """Callers wire the scope unconditionally; a fenceless call must not break."""
    stream = _Stream(count=3)
    with aux.aux_stream_deadline(None):
        response = aux._aggregate_chat_stream(
            stream, model="m", total_ceiling=2400.0
        )
    assert response.choices[0].message.content == "xxx"


def test_nested_none_inherits_rather_than_escaping_the_host_deadline():
    """A fenceless aux call nested inside a fenced one stays bounded.

    ``None`` means "I have no deadline of my own", not "clear the one in
    force" — mirroring ``_aux_thread_local_hook``'s passthrough contract. If it
    cleared, any nested auxiliary call made during compression would escape the
    host ceiling that the whole attempt is supposed to live inside.
    """
    outer = time.monotonic() - 1.0
    stream = _Stream(count=50)
    with aux.aux_stream_deadline(outer):
        with aux.aux_stream_deadline(None):
            assert aux._current_aux_stream_deadline() == outer
            with pytest.raises(TimeoutError):
                aux._aggregate_chat_stream(stream, model="m", total_ceiling=2400.0)
    assert stream.yielded == 1


def test_deadline_scope_restores_the_previous_value():
    outer = time.monotonic() + 900.0
    with aux.aux_stream_deadline(outer):
        assert aux._current_aux_stream_deadline() == outer
        with aux.aux_stream_deadline(time.monotonic() + 10.0):
            assert aux._current_aux_stream_deadline() != outer
        assert aux._current_aux_stream_deadline() == outer
    assert aux._current_aux_stream_deadline() is None


def test_async_stream_mirror_honours_the_host_deadline():
    """The async consumer must not drift from the sync one."""
    stream = _AsyncStream(count=50)

    async def _run():
        with aux.aux_stream_deadline(time.monotonic() - 1.0):
            return await aux._aggregate_chat_stream_async(
                stream, model="m", total_ceiling=2400.0
            )

    with pytest.raises(TimeoutError):
        asyncio.run(_run())
    assert stream.yielded == 1


# ── The isolated provider daemon must inherit it ─────────────────────────


def test_protected_provider_daemon_inherits_the_host_deadline():
    """``_run_protected_sync_provider_call`` runs the stream on ANOTHER thread.

    Thread-locals do not cross that boundary, so without explicit propagation
    the fix would be inert on exactly the path large-session compression takes
    (protected + hard-cancel source installed).
    """
    seen: dict[str, object] = {}

    def _callback(_kwargs):
        seen["deadline"] = aux._current_aux_stream_deadline()
        seen["thread"] = threading.current_thread().name
        return "ok"

    deadline = time.monotonic() + 42.0
    cancel_event = threading.Event()
    with aux.aux_progress_hook(lambda: None), aux.aux_interrupt_protection(
        cancel_event=cancel_event
    ), aux.aux_stream_deadline(deadline):
        assert aux._run_protected_sync_provider_call(_callback, {}) == "ok"

    assert seen["thread"] == "hermes-protected-aux-provider"
    assert seen["deadline"] == deadline


# ── The compression worker must actually install it ──────────────────────


def _summary_dispatch_source() -> str:
    from agent import conversation_compression

    path = Path(inspect.getsourcefile(conversation_compression))
    return path.read_text(encoding="utf-8")


def test_compression_summary_dispatch_installs_the_fence_deadline():
    """Source guard: the wiring is one line and trivially droppable.

    A behavioural test would have to drive the whole ``compress_context`` body
    (durable lock, watermark, telemetry, commit). This asserts the seam itself:
    the same ``with`` statement that installs the progress hook must also
    install the stream deadline.
    """
    tree = ast.parse(_summary_dispatch_source())
    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        names = set()
        for item in node.items:
            call = item.context_expr
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                names.add(call.func.id)
        if "aux_progress_hook" in names:
            assert "aux_stream_deadline" in names, (
                "the summary dispatch scope installs the progress hook but not "
                "the host stream deadline — #99692 would regress"
            )
            wired = True
    assert wired, "summary dispatch scope not found"
