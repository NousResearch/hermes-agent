"""Chat scoping and tools-only behavior for holographic memory."""

from pathlib import Path

from plugins.memory.holographic import HolographicMemoryProvider


def test_db_path_template_is_sanitized_and_separates_dm_from_group(tmp_path):
    config = {
        "db_path": str(tmp_path / "legacy.db"),
        "db_path_template": "$HERMES_HOME/holographic/{profile}/{platform}/{chat}-{user}.db",
        "hrr_dim": 64,
    }

    dm = HolographicMemoryProvider(config=config)
    dm.initialize(
        "dm-session",
        hermes_home=tmp_path,
        agent_identity="personal/profile",
        platform="telegram",
        chat_id="dm/../../123",
        user_id="user:42",
    )
    dm_path = dm._store.db_path
    dm.shutdown()

    group = HolographicMemoryProvider(config=config)
    group.initialize(
        "group-session",
        hermes_home=tmp_path,
        agent_identity="personal/profile",
        platform="telegram",
        chat_id="group/../../456",
        user_id="user:42",
    )
    group_path = group._store.db_path
    group.shutdown()

    assert dm_path == tmp_path / "holographic/personal-profile/telegram/dm-123-user-42.db"
    assert group_path == tmp_path / "holographic/personal-profile/telegram/group-456-user-42.db"
    assert dm_path != group_path
    assert dm_path.is_file()
    assert group_path.is_file()
    assert not Path(config["db_path"]).exists()


def test_db_path_behavior_is_preserved_without_template(tmp_path):
    db_path = tmp_path / "legacy.db"
    provider = HolographicMemoryProvider(config={"db_path": str(db_path), "hrr_dim": 64})
    provider.initialize("legacy-session", hermes_home=tmp_path, chat_id="ignored")
    assert provider._store.db_path == db_path
    provider.shutdown()


def test_tools_mode_disables_prefetch_but_keeps_explicit_tools(tmp_path):
    provider = HolographicMemoryProvider(
        config={
            "db_path": str(tmp_path / "tools.db"),
            "memory_mode": "tools",
            "auto_extract": False,
            "hrr_dim": 64,
        }
    )
    provider.initialize("tools-session")
    provider._retriever.search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("tools mode must not search during prefetch")
    )

    assert provider.prefetch("private query") == ""
    assert "Use fact_store" in provider.system_prompt_block()
    assert {schema["name"] for schema in provider.get_tool_schemas()} == {
        "fact_store",
        "fact_feedback",
    }
    assert '"status": "added"' in provider.handle_tool_call(
        "fact_store", {"action": "add", "content": "The user prefers local storage"}
    )
    provider.shutdown()
