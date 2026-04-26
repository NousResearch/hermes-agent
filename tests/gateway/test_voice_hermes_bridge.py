import pytest

from gateway.voice_hermes_bridge import HerVoiceBridge, her_voice_tool_declarations
from hermes_state import SessionDB


def test_voice_tool_declarations_include_required_contract():
    names = {tool["name"] for tool in her_voice_tool_declarations()}
    assert names == {
        "run_hermes_agent",
        "create_issue",
        "list_issues",
        "send_whatsapp",
        "send_discord",
        "get_memory",
        "get_calendar",
        "read_email",
    }


@pytest.mark.asyncio
async def test_run_hermes_agent_uses_voice_platform_and_session(tmp_path):
    calls = []

    async def run_agent(**kwargs):
        calls.append(kwargs)
        return ({"final_response": "Done"}, {"total_tokens": 3})

    bridge = HerVoiceBridge(
        run_agent=run_agent,
        session_db_factory=lambda: SessionDB(db_path=tmp_path / "state.db"),
    )

    result = await bridge.call(
        "run_hermes_agent",
        {"task": "Crée une issue test", "context": "Depuis le vocal", "session_id": "voice-test"},
    )

    assert result["ok"] is True
    assert result["session_id"] == "voice-test"
    assert result["response"] == "Done"
    assert calls[0]["platform"] == "voice"
    assert calls[0]["session_id"] == "voice-test"
    assert "Context:\nDepuis le vocal" in calls[0]["user_message"]


@pytest.mark.asyncio
async def test_get_memory_returns_shared_voice_and_text_results(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("text-1", source="cli")
        db.append_message("text-1", role="user", content="factures avril à vérifier")
        db.create_session("voice-1", source="voice")
        db.append_message("voice-1", role="user", content="factures mai à vérifier")

        async def run_agent(**kwargs):
            raise AssertionError("run_agent should not be called")

        bridge = HerVoiceBridge(run_agent=run_agent, session_db_factory=lambda: db)
        result = await bridge.call("get_memory", {"query": "factures", "limit": 10})

        assert result["ok"] is True
        assert result["count"] == 2
        channels = {item["channel"] for item in result["memories"]}
        assert channels == {"text", "voice"}
    finally:
        db.close()
