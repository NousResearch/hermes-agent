"""ACP metadata exposure for Hermes session provenance and compaction state."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp.schema import SessionInfoUpdate, TextContentBlock

from acp_adapter.provenance import build_hermes_session_meta
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def _session_provenance(field_meta: dict) -> dict:
    return field_meta["hermes"]["sessionProvenance"]


def test_build_hermes_session_meta_describes_compression_lineage(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="acp", session_kind="acp", creator_kind="acp")
    db.append_message("root", role="user", content="before compaction")
    db.end_session("root", "compression")
    db.create_session(
        "tip",
        source="acp",
        parent_session_id="root",
        session_kind="continuation",
        creator_kind="compression",
    )
    db.append_message("tip", role="user", content="after compaction")

    field_meta = build_hermes_session_meta(
        db,
        acp_session_id="root",
        hermes_session_id="tip",
    )

    provenance = _session_provenance(field_meta)
    assert provenance["acpSessionId"] == "root"
    assert provenance["hermesSessionId"] == "tip"
    assert provenance["parentHermesSessionId"] == "root"
    assert provenance["rootHermesSessionId"] == "root"
    assert provenance["sessionKind"] == "continuation"
    assert provenance["creatorKind"] == "compression"
    assert provenance["isUserFacing"] is True
    assert provenance["compressionDepth"] == 1
    assert provenance["lineageHermesSessionIds"] == ["root", "tip"]

    compaction = field_meta["hermes"]["compaction"]
    assert compaction == {
        "lastMode": "split",
        "compressionDepth": 1,
        "currentHermesSessionId": "tip",
    }


@pytest.mark.asyncio
async def test_new_load_resume_and_list_sessions_expose_hermes_provenance_meta(tmp_path):
    manager = SessionManager(
        agent_factory=lambda: SimpleNamespace(model="gpt-test", provider="test-provider"),
        db=SessionDB(tmp_path / "state.db"),
    )
    agent = HermesACPAgent(session_manager=manager)

    new_resp = await agent.new_session(cwd="/work")
    new_meta = _session_provenance(new_resp.field_meta)
    assert new_meta["acpSessionId"] == new_resp.session_id
    assert new_meta["hermesSessionId"] == new_resp.session_id
    assert new_meta["sessionKind"] == "acp"
    assert new_meta["creatorKind"] == "acp"

    state = manager.get_session(new_resp.session_id)
    assert state is not None
    state.history.append({"role": "user", "content": "hello provenance"})
    manager.save_session(state.session_id)

    load_resp = await agent.load_session(cwd="/work", session_id=new_resp.session_id)
    assert _session_provenance(load_resp.field_meta)["acpSessionId"] == new_resp.session_id

    resume_resp = await agent.resume_session(cwd="/work", session_id=new_resp.session_id)
    assert _session_provenance(resume_resp.field_meta)["hermesSessionId"] == new_resp.session_id

    listed = await agent.list_sessions(cwd="/work")
    assert listed.sessions
    listed_meta = _session_provenance(listed.sessions[0].field_meta)
    assert listed_meta["acpSessionId"] == new_resp.session_id
    assert listed_meta["sessionKind"] == "acp"


@pytest.mark.asyncio
async def test_prompt_emits_session_info_update_when_auto_compression_rotates_hermes_session(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"), db=db)
    agent = HermesACPAgent(session_manager=manager)
    conn = MagicMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock(return_value=None)
    agent._conn = conn

    new_resp = await agent.new_session(cwd="/work")
    state = manager.get_session(new_resp.session_id)
    assert state is not None
    state.agent.context_compressor = MagicMock(context_length=100_000)
    state.agent.tools = []
    state.agent._cached_system_prompt = ""

    def rotate_session(**_kwargs):
        db.end_session(state.session_id, "compression")
        db.create_session(
            "tip-session",
            source="acp",
            parent_session_id=state.session_id,
            session_kind="continuation",
            creator_kind="compression",
        )
        state.agent.session_id = "tip-session"
        return {
            "final_response": "done",
            "messages": [{"role": "assistant", "content": "done"}],
        }

    state.agent.run_conversation.side_effect = rotate_session
    conn.session_update.reset_mock()

    with patch("agent.title_generator.maybe_auto_title"):
        resp = await agent.prompt(
            prompt=[TextContentBlock(type="text", text="please continue")],
            session_id=new_resp.session_id,
        )

    assert resp.stop_reason == "end_turn"
    metadata_updates = [
        call.kwargs["update"]
        for call in conn.session_update.await_args_list
        if isinstance(call.kwargs.get("update"), SessionInfoUpdate)
    ]
    assert metadata_updates
    provenance = _session_provenance(metadata_updates[-1].field_meta)
    assert provenance["acpSessionId"] == new_resp.session_id
    assert provenance["hermesSessionId"] == "tip-session"
    assert provenance["parentHermesSessionId"] == new_resp.session_id
    assert provenance["compressionDepth"] == 1
    assert metadata_updates[-1].field_meta["hermes"]["compaction"]["lastMode"] == "split"


@pytest.mark.asyncio
async def test_manual_compact_emits_in_place_compaction_metadata(tmp_path):
    manager = SessionManager(
        agent_factory=lambda: MagicMock(name="MockAIAgent"),
        db=SessionDB(tmp_path / "state.db"),
    )
    agent = HermesACPAgent(session_manager=manager)
    conn = MagicMock()
    conn.session_update = AsyncMock()
    agent._conn = conn

    new_resp = await agent.new_session(cwd="/work")
    state = manager.get_session(new_resp.session_id)
    assert state is not None
    state.history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    state.agent.compression_enabled = True
    state.agent._session_db = manager._get_db()
    state.agent._cached_system_prompt = "system"
    state.agent.tools = []
    state.agent.context_compressor = MagicMock(context_length=100_000)
    state.agent._compress_context.return_value = ([{"role": "system", "content": "summary"}], None)

    conn.session_update.reset_mock()
    resp = await agent.prompt(
        prompt=[TextContentBlock(type="text", text="/compact")],
        session_id=new_resp.session_id,
    )

    assert resp.stop_reason == "end_turn"
    metadata_updates = [
        call.kwargs["update"]
        for call in conn.session_update.await_args_list
        if isinstance(call.kwargs.get("update"), SessionInfoUpdate)
    ]
    assert metadata_updates
    compaction = metadata_updates[-1].field_meta["hermes"]["compaction"]
    assert compaction["lastMode"] == "in_place"
    assert compaction["inPlaceCount"] == 1
    assert compaction["currentHermesSessionId"] == new_resp.session_id
