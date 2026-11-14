"""Regression for #36908: the repeated-compression warning must reach the
TUI / gateway, not just CLI stdout.

When a session is compressed >= 2 times, ``compress_context`` warns that
accuracy may degrade. That warning used to go through ``_vprint`` (stdout
only), so the Ink TUI / Telegram / Discord never saw it — unlike the two
other compression warnings in the same module, which reach ``status_callback``
(and store ``_compression_warning`` for late-bound gateway replay). This pins
the warning onto the gateway-aware channel, as a warning.
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


def test_repeated_compression_warning_reaches_status_callback_as_warning(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_36908"
    db.create_session(sid, source="cli")

    # compression_count == 2 → the "compressed N times" warning should fire.
    agent = _build_agent_with_db(db, sid, compression_count=2)

    from agent.conversation_compression import compaction_status_kind

    emitted: list[tuple[str, str]] = []
    agent.status_callback = lambda kind, message=None: emitted.append((kind, message))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    # The warning reached the gateway-aware channel as a warning, never as compaction progress
    # (it follows the done edge, where a "compacting" kind would open a compaction that never ends).
    warned = [(k, m) for k, m in emitted if "compressed 2 times" in str(m).lower()]
    assert [k for k, _ in warned] == ["warn"], emitted
    assert [compaction_status_kind(k, m) for k, m in warned] == ["warn"]
    # ...and was stored for late-bound gateway status_callback replay.
    assert "compressed 2 times" in (getattr(agent, "_compression_warning", "") or "").lower()


def test_no_warning_below_threshold(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_36908_ONCE"
    db.create_session(sid, source="cli")

    # compression_count == 1 → no repeated-compression warning.
    agent = _build_agent_with_db(db, sid, compression_count=1)
    emitted: list[str] = []
    agent.status_callback = lambda kind, message=None: emitted.append(str(message))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert not any("compressed" in m.lower() and "times" in m.lower() for m in emitted)
