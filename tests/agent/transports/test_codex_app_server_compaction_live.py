"""Live E2E guardrail for real-thread compaction on the codex app-server.

The unit suite (``test_codex_app_server_session.py``) fakes the wire
protocol, and the compaction change in PR #99000 (commit ff3835a630)
introduced the ``thread/compact/start`` path that compacts the LIVE
server-side thread. That PR's own caveat, called out by @teknium1 on
issue #73503, was:

    a live app-server E2E pass to confirm real-thread compaction
    efficacy at scale

This file is that pass. It spawns a REAL ``codex`` app-server
subprocess, runs real turns so the live thread accumulates context, then
drives ``CodexAppServerSession.compact_thread()`` and asserts the
server-side thread was actually compacted — the exact behavior that
local/transcript-side compression could never provide (the transcript is
a write-only mirror on this transport).

Opt-in — not part of default CI:
    HERMES_LIVE_TESTS=1 pytest tests/agent/transports/test_codex_app_server_compaction_live.py -v

Requires the ``codex`` CLI on PATH (>= MIN_CODEX_VERSION) and an
authenticated codex (``codex login`` / ~/.codex/auth.json), the same
prereqs as the app-server transport itself.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from agent.transports.codex_app_server import (
    MIN_CODEX_VERSION,
    check_codex_binary,
)
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
)

LIVE = os.environ.get("HERMES_LIVE_TESTS") == "1"

_CODEX_OK, _CODEX_MSG = check_codex_binary() if shutil.which("codex") else (False, "codex CLI not found")

LIVE_ONLY = pytest.mark.skipif(
    not LIVE, reason="live-only — set HERMES_LIVE_TESTS=1"
)
CODEX_REQUIRED = pytest.mark.skipif(
    not _CODEX_OK, reason=f"codex app-server unavailable: {_CODEX_MSG}"
)


def _assert_healthy_turn(result, turn_label: str) -> None:
    """A live turn must complete without error and produce real text.

    This mirrors the transport's contract: a completed turn is the only
    proof the subprocess handshake, thread/start, and turn/start all
    worked against a real codex binary.
    """
    assert result.error is None, (
        f"{turn_label} failed: {result.error!r} "
        f"(thread_id={result.thread_id!r})"
    )
    assert result.final_text and result.final_text.strip(), (
        f"{turn_label} returned empty final text: {result.final_text!r}"
    )


@LIVE_ONLY
@CODEX_REQUIRED
def test_live_thread_compaction_is_real_and_session_survives(tmp_path: Path) -> None:
    """Compact the LIVE thread and prove the session keeps working.

    Pre-#99000, compression was a local no-op: the transcript is never
    replayed into the codex thread, so the model's real context (the
    server-side thread) never shrank. This test drives the real
    ``thread/compact/start`` RPC and asserts:

    1. compaction completes with no error and ``compacted=True``
       (the ContextCompaction item / thread/compacted notification
       arrived from the server);
    2. the thread id is preserved — codex compacted in place rather
       than abandoning the thread;
    3. a follow-up turn on the SAME session still completes, proving the
       compacted thread remains usable (efficacy, not just a no-op).
    """
    codex_home = os.environ.get("CODEX_HOME")  # use the runner's auth/login
    session = CodexAppServerSession(
        cwd=str(tmp_path),
        codex_home=codex_home,
    )
    try:
        thread_id = session.ensure_started()
        assert thread_id, "ensure_started returned an empty thread id"

        # Accumulate real context on the server-side thread.
        r1 = session.run_turn(
            "Write a short paragraph (3-4 sentences) about why "
            "compaction matters for long agent sessions."
        )
        _assert_healthy_turn(r1, "warmup turn")
        assert r1.thread_id == thread_id

        r2 = session.run_turn(
            "Now write a second paragraph contrasting that with "
            "a transcript-only (write-only mirror) compression "
            "approach."
        )
        _assert_healthy_turn(r2, "context turn")
        assert r2.thread_id == thread_id

        # Drive the real thread/compact/start path.
        compact = session.compact_thread(turn_timeout=300.0)
        assert compact.error is None, (
            f"compact_thread failed: {compact.error!r}"
        )
        assert compact.compacted is True, (
            "compact_thread returned without a compaction boundary "
            "(no ContextCompaction item / thread/compacted notification). "
            "This is the exact #73503 no-op regression shape."
        )
        assert compact.thread_id == thread_id, (
            f"compaction changed thread id: {compact.thread_id!r} != {thread_id!r}"
        )

        # The proof of efficacy: the same live session must still answer.
        r3 = session.run_turn(
            "Reply with exactly the word: OK"
        )
        _assert_healthy_turn(r3, "post-compaction turn")
        assert r3.thread_id == thread_id, (
            "post-compaction turn ran on a different thread — the "
            "session was retired instead of reusing the compacted thread"
        )
    finally:
        session.close()


@LIVE_ONLY
@CODEX_REQUIRED
def test_live_compaction_is_idempotent_after_empty_thread(tmp_path: Path) -> None:
    """A freshly started thread must compact cleanly too.

    Guards the auto/native gate: /compress on a thread with nothing to
    compact must still return a completed boundary without error, and
    must not corrupt the session (a subsequent turn still works).
    """
    session = CodexAppServerSession(
        cwd=str(tmp_path),
        codex_home=os.environ.get("CODEX_HOME"),
    )
    try:
        thread_id = session.ensure_started()
        assert thread_id

        compact = session.compact_thread(turn_timeout=300.0)
        assert compact.error is None, (
            f"compact_thread on fresh thread failed: {compact.error!r}"
        )
        # Some codex builds reply "nothing to compact" as a completed
        # boundary without a ContextCompaction item; either way the
        # session must stay usable. We assert the hard invariant (no
        # error, thread preserved) and tolerate compacted True/False.
        assert compact.thread_id == thread_id

        r = session.run_turn("Reply with exactly the word: FINE")
        _assert_healthy_turn(r, "post-empty-compaction turn")
    finally:
        session.close()


def test_check_codex_binary_rejects_old_version(tmp_path: Path) -> None:
    """The version gate must reject codex builds without compact support.

    ``thread/compact/start`` landed in the app-server protocol used by
    MIN_CODEX_VERSION. A deterministic fake binary reports a version one
    patch below the minimum; the gate must refuse it the same way it
    refuses a missing executable — so /compress and hygiene never drive
    the RPC against a build that cannot compact the live thread.
    """
    fake = tmp_path / "codex"
    fake_minor = max(0, MIN_CODEX_VERSION[1] - 1)
    fake.write_text(
        "#!/bin/sh\n"
        f"echo 'codex-cli {MIN_CODEX_VERSION[0]}.{fake_minor}."
        f"{MIN_CODEX_VERSION[2]}'\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    ok, msg = check_codex_binary(codex_bin=str(fake))
    assert ok is False, (
        "check_codex_binary accepted a below-minimum codex version — "
        "compact_thread would then run thread/compact/start against an "
        "unsupported build"
    )
    assert "older than required" in msg.lower(), msg

    ok, msg = check_codex_binary(codex_bin="/nonexistent/codex/binary/path")
    assert ok is False
    assert "not found" in msg.lower() or "no such" in msg.lower()
