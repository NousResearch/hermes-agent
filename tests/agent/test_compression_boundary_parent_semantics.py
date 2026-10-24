"""#134827 — in-place compaction must not pass the session as its own lineage parent.

``_finish_compaction_boundary`` derives the boundary parent as
``_old_sid or agent.session_id`` — but ``old_session_id`` is bound only when
the compressor rotates to a continuation child. On the in-place path (session
id unchanged) that expression resolves to the session ITSELF, so memory
providers received ``on_session_switch(..., parent_session_id=<own id>)`` and
external memory stores (e.g. Hindsight retain) tagged the session as its own
parent — a lineage self-cycle.

In-place parent semantics for the MEMORY provider leg are ``""`` (the
``on_session_switch`` contract sentinel for "no parent":
``parent_session_id: str = ""``). Rotation keeps ``parent=<old session id>``.
The context-engine notification legs are out of scope here: forwarding the
same id in-place is intentional there (the boundary is real).
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingSwitchProvider(MemoryProvider):
    """Records every on_session_switch invocation verbatim."""

    def __init__(self) -> None:
        self.switches: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "recorder"

    def is_available(self) -> bool:
        return True

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def initialize(self, agent: Any = None, **kwargs) -> bool:  # type: ignore[override]
        return True

    def build_system_prompt(self) -> str:  # type: ignore[override]
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, **kwargs) -> None:  # type: ignore[override]
        pass

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self.switches.append({"new_session_id": new_session_id, **kwargs})


class _InPlaceSuccessCompressor:
    """Minimal conforming compressor (same shape as abort-state tests)."""

    _last_compress_aborted = False
    _last_summary_error = None
    compression_count = 1
    _last_compression_made_progress = True
    _last_summary_fallback_used = False
    last_compression_rough_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    awaiting_real_usage_after_compression = False

    def compress(self, _messages, **_kwargs):
        return [
            {"role": "user", "content": "[summary] earlier state"},
            {"role": "assistant", "content": "retained tail"},
        ]


def _make_mock_compressor() -> MagicMock:
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
    return compressor


def _make_agent(session_db, session_id: str, manager: MemoryManager, compressor):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    # skip_memory=True leaves _memory_manager None; inject a real manager with
    # a recording provider so the boundary's provider leg is observable.
    agent._memory_manager = manager
    agent.context_compressor = compressor
    agent._compression_feasibility_checked = True
    return agent


def _compression_switches(provider: _RecordingSwitchProvider):
    return [s for s in provider.switches if s.get("reason") == "compression"]


class TestCompressionBoundaryParentSemantics:
    def test_in_place_memory_provider_parent_is_empty(self):
        """In-place compaction: the memory leg must see parent="" (no parent),
        never the session reporting itself as its own parent (#134827)."""
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        manager = MemoryManager()
        provider = _RecordingSwitchProvider()
        manager._providers.append(provider)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "PARENT_IP_134827"
            db.create_session(sid, source="cli")
            agent = _make_agent(db, sid, manager, _InPlaceSuccessCompressor())
            agent.compression_in_place = True

            messages = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
            ]
            agent._flush_messages_to_session_db(messages, [])
            compacted, _ = compress_context(agent, messages, "system", approx_tokens=100_000)
            assert agent._last_compaction_in_place is True

            switches = _compression_switches(provider)
            assert switches, (
                f"no compression on_session_switch reached the provider: {provider.switches}"
            )
            assert len(switches) == 1
            call = switches[0]
            assert call["new_session_id"] == sid  # in-place: id unchanged
            assert call["parent_session_id"] == "", (
                "in-place compaction must not report the session as its own "
                f"memory lineage parent (got parent_session_id={call['parent_session_id']!r})"
            )
            assert call["reset"] is False
            db.close()

    def test_rotation_memory_provider_parent_is_old_sid(self):
        """Rotation: the memory leg must keep parent=<old session id> — the
        fix is scoped to the in-place leg only."""
        from agent.conversation_compression import compress_context
        from hermes_state import SessionDB

        manager = MemoryManager()
        provider = _RecordingSwitchProvider()
        manager._providers.append(provider)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "state.db")
            sid = "PARENT_ROT_134827"
            db.create_session(sid, source="cli")
            agent = _make_agent(db, sid, manager, _make_mock_compressor())
            # ROTATION path — pin in_place=False regardless of the global default.
            agent.compression_in_place = False

            messages = [
                {"role": "user", "content": f"m{i} " + "x" * 60} for i in range(20)
            ]
            compacted, _ = compress_context(agent, messages, "system", approx_tokens=120_000)

            switches = _compression_switches(provider)
            assert switches, (
                f"no compression on_session_switch reached the provider: {provider.switches}"
            )
            call = switches[-1]
            assert call["parent_session_id"] == sid, (
                "rotation must keep the old session id as the memory lineage parent "
                f"(got parent_session_id={call['parent_session_id']!r})"
            )
            assert call["new_session_id"] != sid, "rotation must move to a child session id"
            assert call["reset"] is False
            db.close()
