"""Regression for #118120: blocking memory-provider hook inside the compaction
commit fence makes a session uninterruptible.

``compress_context`` called ``MemoryManager.on_session_switch`` while holding
the ``CompressionCommitFence`` lock (between ``begin_commit`` and
``finish_commit``). A wedged provider (e.g. hindsight's embedded client
blocking 180s on daemon start) therefore wedged the fence, and the
interrupt path (``cancel_before_commit``) blocks on that same lock — so
the cancel could never land and only SIGKILL got out.

Contract pinned here: while a provider's ``on_session_switch`` is still
running, ``cancel_before_commit`` must return promptly (the hook runs
OUTSIDE the fence), and the switch notification must still fire exactly
once afterwards (deferred, not dropped).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import CompressionCommitFence
from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from hermes_state import SessionDB


class _StuckProvider(MemoryProvider):
    """Provider whose switch hook blocks until the test releases it."""

    def __init__(self, fence: CompressionCommitFence):
        self._fence = fence
        self.entered = threading.Event()
        self.released = threading.Event()
        self.calls: list = []
        self.fence_held_at_hook: list = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return "stuck"

    def is_available(self) -> bool:
        return True

    def get_tool_schemas(self):  # type: ignore[override]
        return []

    def initialize(self, agent=None, **kwargs) -> bool:  # type: ignore[override]
        return True

    def build_system_prompt(self) -> str:  # type: ignore[override]
        return ""

    def sync_turn(self, *args, **kwargs) -> None:  # type: ignore[override]
        pass

    def on_session_end(self, messages) -> None:  # type: ignore[override]
        pass

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:  # type: ignore[override]
        self.calls.append((new_session_id, kwargs.get("reason")))
        self.entered.set()
        # Simulate the wedged daemon start: stay inside the hook.
        assert self.released.wait(timeout=60), "test harness stalled"
        # Non-blocking probe: could we take the fence right now? If not,
        # this hook is running while the commit fence is held (the bug).
        acquired = self._fence._lock.acquire(blocking=False)
        if acquired:
            self._fence._lock.release()
        self.fence_held_at_hook.append(not acquired)


def _build_agent(db: SessionDB, session_id: str, provider: _StuckProvider):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/v1",
            model="test/model",
            platform="cli",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    mm = MemoryManager()
    mm._providers.append(provider)
    agent._memory_manager = mm
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


def test_cancel_lands_while_provider_hook_still_running(tmp_path: Path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        sid = "FENCE_HOOK_118120"
        db.create_session(sid, source="cli")
        fence = CompressionCommitFence()
        provider = _StuckProvider(fence)
        agent = _build_agent(db, sid, provider)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]

        errors: list = []

        def worker():
            try:
                agent._compress_context(
                    msgs, "sys", approx_tokens=120_000, commit_fence=fence
                )
            except Exception as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        t = threading.Thread(target=worker, name="compress-118120", daemon=True)
        t.start()
        try:
            assert provider.entered.wait(timeout=60), "switch hook never ran"
            # Bound the stuck hook so a still-broken fence fails fast
            # instead of deadlocking the test (cancel blocks <-> release
            # waits for cancel).
            threading.Timer(6.0, provider.released.set).start()
            # Interrupt path (mirrors agent/interrupt_control: blocking
            # cancel_before_commit). Must NOT wait out the stuck provider.
            t0 = time.monotonic()
            fence.cancel_before_commit()
            cancel_elapsed = time.monotonic() - t0
            assert cancel_elapsed < 3.0, (
                f"cancel blocked {cancel_elapsed:.1f}s behind the provider hook: "
                "hook still runs inside the commit fence (#118120)"
            )
        finally:
            provider.released.set()
            t.join(timeout=60)

        assert not errors, f"compression worker raised: {errors!r}"
        assert not t.is_alive(), "compression worker did not finish"
        # Deferred, not dropped: exactly one switch with the boundary reason.
        assert provider.calls == [(sid, "compression")], provider.calls
        # And it ran outside the fence.
        assert provider.fence_held_at_hook == [False], (
            "on_session_switch ran while the commit fence was held (#118120)"
        )
    finally:
        db.close()
