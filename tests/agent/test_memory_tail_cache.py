"""Memory-only rebuilds preserve the prefix through frozen plugin/timestamp bytes."""

from agent.system_prompt import (
    build_system_prompt_parts,
    restore_plugin_prompt_sections,
)
from hermes_cli.plugins import RenderedPluginSystemPromptSection
from tools.memory_tool_store import MemoryStore
from tests.agent.test_system_prompt import _make_agent


def test_memory_reload_preserves_every_preceding_prompt_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memory = tmp_path / "memories"
    memory.mkdir()
    path = memory / "MEMORY.md"
    path.write_text("The preferred drink is tea.", encoding="utf-8")
    store = MemoryStore()
    store.load_from_disk()
    section = RenderedPluginSystemPromptSection(
        id="cache-test", content="frozen plugin", position="after_memory", plugin="test"
    )
    agent = _make_agent(
        skip_context_files=True,
        _memory_store=store,
        _memory_enabled=True,
        _user_profile_enabled=True,
        session_id="20260905_090000_test",
        _plugin_system_prompt_sections_snapshot=(section,),
    )
    before = build_system_prompt_parts(agent)
    old_block = store.format_for_system_prompt("memory")
    prefix, sep, tail = before["volatile"].partition(old_block)
    assert sep and not tail
    assert "frozen plugin" in prefix and "Conversation started:" in prefix
    path.write_text("The preferred drink is coffee.", encoding="utf-8")
    # A disk write alone must not change a running conversation's snapshot.
    assert build_system_prompt_parts(agent) == before
    store.load_from_disk()
    after = build_system_prompt_parts(agent)
    new_block = store.format_for_system_prompt("memory")
    assert new_block != old_block
    assert after["stable"] == before["stable"]
    assert after["context"] == before["context"]
    assert after["volatile"] == prefix + new_block
    restored = _make_agent()
    restore_plugin_prompt_sections(restored, "\n\n".join(after.values()))
    assert [
        (s.id, s.content) for s in restored._plugin_system_prompt_sections_snapshot
    ] == [(section.id, section.content)]
