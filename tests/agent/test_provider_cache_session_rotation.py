"""Regression: provider prompt-cache scope must survive session rotation.

Compaction rotates ``agent.session_id`` so the old durable session DB row can
be finalized (``conversation_compression.py`` reassigns it to a fresh id). But
Codex/OpenAI prompt caching keys off ``agent.provider_cache_session_id``, which
is initialized once per agent instance and must NOT rotate — otherwise every
compaction silently drops the provider-side cache bucket for the continuing
in-memory conversation. This asserts the cache scope is left untouched across a
real ``_compress_context`` rotation.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
    """Mirror tests/agent/test_compression_logging_session_context.py's harness."""
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
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # The default is in-place compaction, which deliberately preserves the
    # durable session id. Pin the legacy rotation path this regression covers.
    agent.compression_in_place = False
    return agent


def test_provider_cache_scope_survives_compression_rotation(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PARENT_PROVIDER_CACHE_SESSION"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)

    # init_agent seeds the provider cache scope from the initial session id.
    assert agent.provider_cache_session_id == parent_sid
    initial_cache_scope = agent.provider_cache_session_id

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    agent._compress_context(messages, "sys", approx_tokens=120_000)

    # The durable session id actually rotated (sanity — otherwise the
    # invariant assertion below would be vacuous).
    assert agent.session_id != parent_sid

    # The provider-side cache scope must NOT have followed the rotation: it is
    # the stable routing key for Codex/OpenAI prompt caching.
    assert agent.provider_cache_session_id == initial_cache_scope, (
        "provider_cache_session_id rotated with the session id; provider "
        "prompt-cache buckets would be dropped on every compaction "
        f"(scope was {initial_cache_scope!r}, now "
        f"{agent.provider_cache_session_id!r})."
    )
