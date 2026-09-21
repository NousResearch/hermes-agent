"""Private notebooks must follow the gateway conversation, not the profile."""

import json
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource, build_session_key
from run_agent import AIAgent
from tools.memory_tool import memory_tool
from plugins.memory.holographic import HolographicMemoryProvider


def test_buyer_memory_survives_session_rotation_without_crossing_chats(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  memory_enabled: true\n  user_profile_enabled: true\n"
        "  scope: conversation\n  provider: ''\n"
    )
    monkeypatch.setattr("run_agent.OpenAI", MagicMock())
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})

    def buyer(number, session):
        source = SessionSource(
            platform=Platform.WHATSAPP, chat_type="dm", chat_id=number, user_id=number,
        )
        return AIAgent(
            model="gpt-4.1", provider="openrouter", api_mode="chat_completions",
            api_key="test-only", base_url="http://test.invalid", quiet_mode=True,
            skip_context_files=True, enabled_toolsets=["memory"],
            session_id=session, platform="whatsapp", user_id=number,
            chat_id=number, chat_type="dm", gateway_session_key=build_session_key(source),
        )

    alice = buyer("100000001@s.whatsapp.net", "alice-first")
    try:
        saved = json.loads(memory_tool(
            action="add", target="user", content="My private project codename is AMBER-LANTERN.",
            store=alice._memory_store,
        ))
        assert saved["success"]
        bob = buyer("100000002@s.whatsapp.net", "bob-first")
        returning_alice = buyer("100000001@s.whatsapp.net", "alice-new-session")
        try:
            assert "AMBER-LANTERN" not in bob._build_system_prompt()
            assert "AMBER-LANTERN" in returning_alice._build_system_prompt()
        finally:
            bob.shutdown_memory_provider()
            returning_alice.shutdown_memory_provider()
    finally:
        alice.shutdown_memory_provider()


def test_holographic_facts_and_mirrored_notes_stay_in_the_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("memory:\n  scope: conversation\n")
    config = {"db_path": str(tmp_path / "legacy-shared.db"), "hrr_dim": 64}
    alice, bob, returning_alice = [HolographicMemoryProvider(config=config) for _ in range(3)]
    try:
        alice.initialize("first", gateway_session_key="agent:main:whatsapp:dm:100000001")
        bob.initialize("second", gateway_session_key="agent:main:whatsapp:dm:100000002")
        returning_alice.initialize("third", gateway_session_key="agent:main:whatsapp:dm:100000001")
        result = json.loads(alice.handle_tool_call(
            "fact_store", {"action": "add", "content": "The private project is AMBER LANTERN"},
        ))
        assert result["status"] == "added"
        alice.on_memory_write("add", "user", "Buyer preference is confidential emerald finish")
        assert json.loads(bob.handle_tool_call("fact_store", {"action": "list"}))["facts"] == []
        assert "AMBER" not in bob.prefetch("AMBER LANTERN")
        assert "AMBER" in returning_alice.prefetch("AMBER LANTERN")
        assert not (tmp_path / "legacy-shared.db").exists()
    finally:
        for provider in (alice, bob, returning_alice):
            provider.shutdown()


def test_missing_conversation_identity_cannot_open_global_memory(tmp_path, monkeypatch):
    from tools.memory_tool import MemoryStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("memory:\n  scope: conversation\n")
    with pytest.raises(ValueError, match="trusted gateway session key"):
        MemoryStore()
    provider = HolographicMemoryProvider(config={"hrr_dim": 64})
    try:
        with pytest.raises(ValueError, match="trusted gateway session key"):
            provider.initialize("unidentified")
        assert not (tmp_path / "memory_store.db").exists()
    finally:
        provider.shutdown()


@pytest.mark.parametrize("batch", [False, True])
def test_pending_memory_approval_cannot_cross_buyers(tmp_path, monkeypatch, batch):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools.memory_tool import MemoryStore

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  scope: conversation\n  write_approval: true\n"
    )
    alice = MemoryStore(gateway_session_key="agent:main:whatsapp:dm:100000001")
    bob = MemoryStore(gateway_session_key="agent:main:whatsapp:dm:100000002")
    alice.load_from_disk()
    bob.load_from_disk()
    op = {"action": "add", "content": "Private codename AMBER-LANTERN"}
    args = {"operations": [op]} if batch else op
    staged = json.loads(memory_tool(target="user", store=alice, **args))
    assert staged["staged"]
    assert "AMBER-LANTERN" not in handle_pending_subcommand("memory", ["pending"], memory_store=bob)
    handle_pending_subcommand("memory", ["approve", staged["pending_id"]], memory_store=bob)
    assert bob.user_entries == []
    approved = handle_pending_subcommand("memory", ["approve", staged["pending_id"]], memory_store=alice)
    assert "Approved 1" in approved
    assert alice.user_entries == [op["content"]]
