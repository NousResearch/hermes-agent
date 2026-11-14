"""Regression for #36908: the repeated-compression warning must reach the
TUI / gateway, not just CLI stdout.

When a session is compressed at least ``compression.warn_after_compressions``
times (default 2; 0 disables), ``compress_context`` warns that accuracy may
degrade. That warning used to go through ``_vprint`` (stdout only), so the Ink
TUI / Telegram / Discord never saw it. It now reaches ``status_callback`` as
kind ``warn`` (and stores ``_compression_warning`` for late-bound gateway
replay), and must not be reclassified as compaction progress.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(
    db: SessionDB,
    session_id: str,
    compression_count: int,
    *,
    warn_after_compressions: int = 2,
):
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
    agent.compression_warn_after_compressions = warn_after_compressions
    return agent


def _run_compress(agent) -> list[tuple[str, object]]:
    emitted: list[tuple[str, object]] = []
    agent.status_callback = lambda kind, message=None: emitted.append((kind, message))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)
    return emitted


def _repeated_compression_warnings(emitted) -> list[tuple[str, object]]:
    """The 'compressed N times' warning only.

    Accepts ``(kind, message)`` pairs or a bare message. Compaction progress
    mentions compression too, but it is a ``CompactionProgressText`` mark, not
    this warning.
    """
    from agent.context_engine import CompactionProgressText
    from agent.conversation_compression import compaction_status_kind

    warnings: list[tuple[str, object]] = []
    for item in emitted:
        if isinstance(item, tuple) and len(item) == 2:
            kind, message = item
        else:
            kind, message = "", item
        if isinstance(message, CompactionProgressText):
            continue
        if compaction_status_kind(str(kind), message) == "compacting":
            continue
        text = str(message).lower()
        if "compressed" in text and "times" in text:
            warnings.append((str(kind), message))
    return warnings


def test_repeated_compression_warning_reaches_status_callback_as_warning(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_36908"
    db.create_session(sid, source="cli")

    # compression_count == 2 → the default threshold fires exactly once.
    agent = _build_agent_with_db(db, sid, compression_count=2)
    emitted = _run_compress(agent)

    from agent.conversation_compression import compaction_status_kind

    # The warning reached the gateway-aware channel as a warning, never as compaction progress
    # (it follows the done edge, where a "compacting" kind would open a compaction that never ends).
    warned = _repeated_compression_warnings(emitted)
    assert [k for k, _ in warned] == ["warn"], emitted
    assert [compaction_status_kind(k, m) for k, m in warned] == ["warn"]
    assert "compressed 2 times" in str(warned[0][1]).lower()
    # ...and was stored for late-bound gateway status_callback replay.
    assert "compressed 2 times" in (getattr(agent, "_compression_warning", "") or "").lower()


def test_no_warning_below_threshold(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_36908_ONCE"
    db.create_session(sid, source="cli")

    # compression_count == 1 → no repeated-compression warning.
    agent = _build_agent_with_db(db, sid, compression_count=1)
    emitted = _run_compress(agent)

    assert _repeated_compression_warnings(emitted) == []


def test_custom_warn_after_compressions_defers_warning(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_53876_DEFER"
    db.create_session(sid, source="cli")

    agent = _build_agent_with_db(db, sid, compression_count=4, warn_after_compressions=5)
    assert _repeated_compression_warnings(_run_compress(agent)) == []


def test_custom_warn_after_compressions_fires_at_threshold(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_53876_FIRE"
    db.create_session(sid, source="cli")

    agent = _build_agent_with_db(db, sid, compression_count=5, warn_after_compressions=5)
    emitted = _run_compress(agent)

    from agent.conversation_compression import compaction_status_kind

    warned = _repeated_compression_warnings(emitted)
    assert [k for k, _ in warned] == ["warn"], emitted
    assert [compaction_status_kind(k, m) for k, m in warned] == ["warn"]
    assert "compressed 5 times" in str(warned[0][1]).lower()
    assert "compressed 5 times" in (getattr(agent, "_compression_warning", "") or "").lower()


def test_warn_after_compressions_zero_disables_warning(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "PARENT_53876_OFF"
    db.create_session(sid, source="cli")

    agent = _build_agent_with_db(db, sid, compression_count=99, warn_after_compressions=0)
    assert _repeated_compression_warnings(_run_compress(agent)) == []
