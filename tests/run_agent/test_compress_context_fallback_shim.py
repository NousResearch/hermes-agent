"""Item 1 regression — the run_agent._compress_context fallback shim must be loud.

Before the fix, _compress_context wrapped the imports of _DB_PERSISTED_MARKER
(agent.context_compressor) and the scoped-identity stamping helper
(agent.conversation_compression._stamp_scoped_twins) in a try/except that silently defined local
fallbacks (a hard-coded ``"_db_persisted"`` literal and a local copy of the
identity helper) with NO logging. If the canonical constant/helper is renamed
or removed upstream, the import raises, the fallback silently keeps stamping
with the stale literal, and the stamping key splits from the flush's — the
duplicate-row bug this PR fixes returns with no error anywhere.

The fix imports both symbols UNCONDITIONALLY (no fallback), so a
renamed/removed symbol must fail the wrapper loudly with ImportError before
any stamping happens.

The ``already_present`` outcome is load-bearing: it keeps compress_context
from touching the deleted module-global name (which would raise NameError on
BOTH pre- and post-fix code and make the test non-discriminating), because the
stamp block at conversation_compression.py:3834-3859 — the ONLY in-module use
of _stamp_scoped_twins — is skipped for already_present.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import agent.conversation_compression as conversation_compression
from agent.conversation_compression import CompressionCommitFence
from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str, platform: str = "telegram"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform=platform,
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    # A real user row in the stub return makes _ensure_compressed_has_user_turn
    # return `already_present`, so the in-module stamp block (the only user of
    # _stamp_scoped_twins inside compress_context) is skipped and
    # the deleted name is referenced ONLY by the run_agent shim import.
    compressor.compress.return_value = [
        {"role": "user", "content": "real user row"},
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
    # ROTATION fallback path — pin in_place=False so the fork-rotation path is
    # exercised regardless of the global default (flipped to True in #38763).
    agent.compression_in_place = False
    return agent


class TestCompressContextFallbackShim:
    def test_compress_context_shim_import_failure_is_loud(
        self, tmp_path: Path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = "PARENT_ROT_SHIM_LOUD"
        db.create_session(parent, source="cli")
        db.append_message(parent, "user", "persisted question")
        db.append_message(parent, "assistant", "persisted answer")

        loaded = db.get_messages_as_conversation(parent)
        messages = [*loaded, {"role": "user", "content": "live question"}]

        agent = _build_agent_with_db(db, parent)
        agent._persist_user_message_idx = len(messages) - 1

        # Delete the canonical helper from its defining module: only the shim
        # import can still reference it (already_present skips the in-module
        # stamp block). Pre-fix the except branch silently defines a fallback;
        # post-fix the unconditional import must raise ImportError.
        monkeypatch.delattr(
            conversation_compression, "_stamp_scoped_twins"
        )
        with pytest.raises(ImportError):
            agent._compress_context(
                messages,
                "sys",
                approx_tokens=120_000,
                commit_fence=CompressionCommitFence(),
            )
