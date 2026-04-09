from pathlib import Path

from plugins.memory.obsidian import ObsidianMemoryProvider
from tools.memory_tool import MemoryStore


def _seed_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    store = MemoryStore()
    store.load_from_disk()
    store.add("user", "User prefers concise updates")
    store.add("memory", "SetVenue dashboard route is /dashboard/owner")
    return store


def test_obsidian_provider_bootstraps_workspace_and_mirrors_builtin_memory(monkeypatch, tmp_path):
    _seed_memory(monkeypatch, tmp_path)
    vault = tmp_path / "vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    provider = ObsidianMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path / "hermes-home"), platform="cli")

    workspace = vault / "Hermes"
    assert (workspace / "user-profile.md").exists()
    assert (workspace / "active-projects.md").exists()
    assert (workspace / "decisions-log.md").exists()
    assert (workspace / "current-focus.md").exists()

    user_text = (workspace / "user-profile.md").read_text(encoding="utf-8")
    decisions_text = (workspace / "decisions-log.md").read_text(encoding="utf-8")
    assert "User prefers concise updates" in user_text
    assert "SetVenue dashboard route is /dashboard/owner" in decisions_text


def test_obsidian_provider_prefetch_returns_bootstrap_and_query_relevant_notes(monkeypatch, tmp_path):
    _seed_memory(monkeypatch, tmp_path)
    vault = tmp_path / "vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    provider = ObsidianMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path / "hermes-home"), platform="cli")

    first = provider.prefetch("", session_id="s1")
    assert "Obsidian external memory" in first
    assert "current-focus" in first
    assert "user-profile" in first

    second = provider.prefetch("What is the SetVenue dashboard route?", session_id="s1")
    assert "SetVenue dashboard route is /dashboard/owner" in second


def test_obsidian_provider_updates_mirrored_notes_after_memory_write(monkeypatch, tmp_path):
    _seed_memory(monkeypatch, tmp_path)
    vault = tmp_path / "vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    provider = ObsidianMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path / "hermes-home"), platform="cli")

    store = MemoryStore()
    store.load_from_disk()
    store.add("memory", "Kaizen is the system motto")
    provider.on_memory_write("add", "memory", "Kaizen is the system motto")

    decisions_text = (vault / "Hermes" / "decisions-log.md").read_text(encoding="utf-8")
    assert "Kaizen is the system motto" in decisions_text


def test_obsidian_provider_writes_bounded_current_focus_snapshot(monkeypatch, tmp_path):
    _seed_memory(monkeypatch, tmp_path)
    vault = tmp_path / "vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    provider = ObsidianMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path / "hermes-home"), platform="telegram")
    provider.sync_turn("Let’s improve the memory system", "I’ll build the Obsidian provider.")
    provider.on_session_end(
        [
            {"role": "user", "content": "Let’s improve the memory system"},
            {"role": "assistant", "content": "I’ll build the Obsidian provider."},
        ]
    )

    focus_text = (vault / "Hermes" / "current-focus.md").read_text(encoding="utf-8")
    assert "Recent user focus" in focus_text
    assert "Let’s improve the memory system" in focus_text
    assert "I’ll build the Obsidian provider." in focus_text
    assert "Platform: telegram" in focus_text


def test_obsidian_provider_current_focus_uses_rolling_buffer_when_session_end_messages_empty(monkeypatch, tmp_path):
    _seed_memory(monkeypatch, tmp_path)
    vault = tmp_path / "vault"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    provider = ObsidianMemoryProvider()
    provider.initialize(session_id="s2", hermes_home=str(tmp_path / "hermes-home"), platform="cli")
    provider.sync_turn("We need continuity without retelling context", "Use a structured Obsidian layer owned by Hermes.")
    provider.on_session_end([])

    focus_text = (vault / "Hermes" / "current-focus.md").read_text(encoding="utf-8")
    assert "We need continuity without retelling context" in focus_text
    assert "Use a structured Obsidian layer owned by Hermes." in focus_text
    assert "No recent user messages captured." not in focus_text
    assert "No recent assistant output captured." not in focus_text
