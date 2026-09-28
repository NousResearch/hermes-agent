"""Regression for #36908: the repeated-compression warning must reach the
TUI / gateway, not just CLI stdout.

When a session is compressed >= 2 times, ``compress_context`` warns that
accuracy may degrade. That warning used to go through ``_vprint`` (stdout
only), so the Ink TUI / Telegram / Discord never saw it — unlike the two
other compression warnings in the same module, which route through
``_emit_status`` (and store ``_compression_warning`` for late-bound
gateway replay). This pins the warning onto the gateway-aware channel.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, compression_count: int):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = compression_count
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


def test_repeated_compression_warning_routed_through_emit_status(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_36908"
    db.create_session(sid, source="cli")

    # compression_count == 2 → the "compressed N times" warning should fire.
    agent = _build_agent_with_db(db, sid, compression_count=2)

    emitted: list[str] = []
    agent._emit_status = lambda message: emitted.append(message)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    # The warning reached the gateway-aware channel...
    assert any("compressed 2 times" in m.lower() for m in emitted), (
        f"repeated-compression warning not emitted via _emit_status: {emitted}"
    )
    # ...and was stored for late-bound gateway status_callback replay.
    assert "compressed 2 times" in (getattr(agent, "_compression_warning", "") or "").lower()


def test_no_warning_below_threshold(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_36908_ONCE"
    db.create_session(sid, source="cli")

    # compression_count == 1 → no repeated-compression warning.
    agent = _build_agent_with_db(db, sid, compression_count=1)
    emitted: list[str] = []
    agent._emit_status = lambda message: emitted.append(message)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert not any("compressed" in m.lower() and "times" in m.lower() for m in emitted)


def test_lossless_engine_is_not_told_its_accuracy_degraded(tmp_path: Path) -> None:
    """#53000: an engine that keeps compacted turns retrievable (LCM-style) declares
    ``lossless_compaction``; the warning must then stop claiming accuracy loss — that
    names a cause the engine does not have and sends users to /new for nothing.
    """
    from agent.context_engine import ContextEngine

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_53000"
    db.create_session(sid, source="cli")

    agent = _build_agent_with_db(db, sid, compression_count=2)

    # A real engine object whose CLASS declares itself lossless (how a plugin ships it):
    # the production code must read the declaration off the class, not the instance.
    class _LosslessEngine:
        lossless_compaction = True
        compression_count = 2
        last_prompt_tokens = 0
        last_completion_tokens = 0
        _last_summary_error = None
        _last_compress_aborted = False
        _last_aux_model_failure_model = None
        _last_aux_model_failure_error = None

        def compress(self, *args, **kwargs):
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "user", "content": "tail"},
            ]

    agent.context_compressor = _LosslessEngine()

    emitted: list[str] = []
    agent._emit_status = lambda message: emitted.append(message)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert any("compacted 2 times" in m.lower() for m in emitted), (
        f"lossless engine still got the lossy warning: {emitted}"
    )
    assert not any("accuracy may degrade" in m.lower() for m in emitted)
    assert "compacted 2 times" in (getattr(agent, "_compression_warning", "") or "").lower()
    # Default stays lossy: the base engine class declares nothing.
    assert ContextEngine.lossless_compaction is False
