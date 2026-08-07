"""Tests for the holographic memory provider — Episode hierarchy and pronoun resolution."""

import json

import pytest

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore


# ---------------------------------------------------------------------------
# pronoun_resolve
# ---------------------------------------------------------------------------


def test_pronoun_resolve_english():
    assert FactRetriever.pronoun_resolve("I prefer dark mode") == "the user prefer dark mode"
    assert (
        FactRetriever.pronoun_resolve("My project uses pytest")
        == "the user's project uses pytest"
    )
    assert FactRetriever.pronoun_resolve("they told me the news") == "they told the user the news"
    assert (
        FactRetriever.pronoun_resolve("that book is mine")
        == "that book is the user's"
    )
    # "I am" pattern
    assert FactRetriever.pronoun_resolve("I am ready") == "the user am ready"
    # "I can" pattern
    assert FactRetriever.pronoun_resolve("I can help") == "the user can help"


def test_pronoun_resolve_chinese():
    assert FactRetriever.pronoun_resolve("我喜欢深色模式") == "the user喜欢深色模式"
    assert FactRetriever.pronoun_resolve("我用python") == "the user用python"
    assert FactRetriever.pronoun_resolve("我的电脑很快") == "the user的电脑很快"
    assert FactRetriever.pronoun_resolve("我想去旅行") == "the user想去旅行"
    assert FactRetriever.pronoun_resolve("我觉得不错") == "the user觉得不错"
    assert FactRetriever.pronoun_resolve("我认为可以") == "the user认为可以"
    assert FactRetriever.pronoun_resolve("我希望成功") == "the user希望成功"
    # 我们应该不被替换
    assert FactRetriever.pronoun_resolve("我们应该合作") == "我们应该合作"


def test_pronoun_resolve_mixed():
    text = "I like 我喜欢 both"
    result = FactRetriever.pronoun_resolve(text)
    assert "the user" in result
    assert "I" not in result
    assert "我" not in result


# ---------------------------------------------------------------------------
# Episode CRUD (store.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=str(db))
    return s


def test_add_and_list_episodes(store):
    eid1 = store.add_episode("First session", session_id="s1", topic="testing")
    eid2 = store.add_episode("Second session", session_id="s2")

    episode = store.get_episode(eid1)
    assert episode is not None
    assert episode["title"] == "First session"
    assert episode["session_id"] == "s1"
    assert episode["topic"] == "testing"

    episodes = store.list_episodes(limit=5)
    assert len(episodes) >= 2
    # Both should appear in the list
    ep_ids = {ep["episode_id"] for ep in episodes}
    assert eid1 in ep_ids
    assert eid2 in ep_ids


def test_search_episodes(store):
    store.add_episode("Debugging pytest", session_id="s3")
    store.add_episode("Writing tests", session_id="s4")
    store.add_episode("Deploy service", session_id="s5")

    results = store.search_episodes("test")
    assert len(results) >= 1
    titles = [r["title"] for r in results]
    assert any("test" in t.lower() for t in titles)


def test_get_current_episode(store):
    eid_a = store.add_episode("Ep A", session_id="sx")
    import time
    time.sleep(2)  # SQLite CURRENT_TIMESTAMP has second granularity
    eid_b = store.add_episode("Ep B", session_id="sx")  # newer
    latest = store.get_current_episode("sx")
    assert latest is not None
    assert latest["title"] == "Ep B"
    assert latest["episode_id"] == eid_b


def test_link_fact_to_episode(store):
    eid = store.add_episode("Linking", session_id="s9")
    fid = store.add_fact("test fact content")
    assert store.link_fact_to_episode(fid, eid) is True
    # Duplicate link returns False
    assert store.link_fact_to_episode(fid, eid) is False

    # fact_count updated
    ep = store.get_episode(eid)
    assert ep["fact_count"] == 1


def test_get_episode_facts(store):
    eid = store.add_episode("Multi-fact", session_id="s10")
    f1 = store.add_fact("Fact Alpha")
    f2 = store.add_fact("Fact Beta")
    store.link_fact_to_episode(f1, eid)
    store.link_fact_to_episode(f2, eid)

    facts = store.get_episode_facts(eid)
    assert len(facts) == 2
    contents = {f["content"] for f in facts}
    assert "Fact Alpha" in contents
    assert "Fact Beta" in contents


def test_update_and_remove_episode(store):
    eid = store.add_episode("Original title", session_id="s11")
    assert store.update_episode(eid, title="Updated title") is True
    ep = store.get_episode(eid)
    assert ep["title"] == "Updated title"

    assert store.update_episode(eid, topic="new topic") is True
    ep = store.get_episode(eid)
    assert ep["topic"] == "new topic"

    assert store.remove_episode(eid) is True
    assert store.get_episode(eid) is None


# ---------------------------------------------------------------------------
# episode_propagate (retrieval.py)
# ---------------------------------------------------------------------------


def test_episode_propagate_adds_sibling_facts(tmp_path):
    """Sibling facts from the same episode are appended to result set."""
    db = tmp_path / "propagate.db"
    store = MemoryStore(db_path=str(db))
    retriever = FactRetriever(store=store)

    # Create an episode
    eid = store.add_episode("Propagation test", session_id="s12")

    # Add three facts — two linked to the episode, one not
    f1 = store.add_fact("Python is used for the backend")
    f2 = store.add_fact("TypeScript is used for the frontend")
    f3 = store.add_fact("Dogs are great pets")  # unrelated, no episode
    store.link_fact_to_episode(f1, eid)
    store.link_fact_to_episode(f2, eid)
    # f3 is intentionally not linked to any episode

    # Search only matches f1
    results = retriever.search("Python", limit=10)
    matched_ids = {r["fact_id"] for r in results}
    assert f1 in matched_ids, "f1 should match 'Python' directly"

    # f2 should be pulled in via episode propagation
    assert f2 in matched_ids, "f2 should be propagated from same episode"
    # f3 should NOT appear (no episode link)
    assert f3 not in matched_ids, "f3 is unrelated and should not appear"


def test_episode_propagate_no_episodes_noop(tmp_path):
    """When no episodes exist, propagation is a no-op."""
    db = tmp_path / "noep.db"
    store = MemoryStore(db_path=str(db))
    retriever = FactRetriever(store=store)

    store.add_fact("some content")
    results = retriever.search("some")  # should not crash
    assert len(results) == 1


# ---------------------------------------------------------------------------
# on_memory_write & _handle_fact_store (pronoun resolve + episode linking)
# ---------------------------------------------------------------------------


@pytest.fixture
def provider(tmp_path):
    db = tmp_path / "holographic.db"
    p = HolographicMemoryProvider(config={
        "db_path": str(db),
        "auto_extract": False,
    })
    p.initialize("session-test-1")
    return p


def test_on_memory_write_resolves_pronouns(provider):
    """Built-in memory writes are pronoun-resolved before storage."""
    provider.on_memory_write("add", "user", "I prefer dark mode")
    # Verify the stored fact
    facts = provider._store.list_facts(limit=5)
    contents = [f["content"] for f in facts]
    assert any("the user prefer dark mode" in c for c in contents), \
        f"Expected pronoun-resolved content, got: {contents}"
    # Original with literal "I" should NOT be present
    assert not any(c.startswith("I prefer") for c in contents)


def test_on_memory_write_links_to_episode(provider):
    """Facts added via on_memory_write link to the current episode."""
    provider.on_memory_write("add", "memory", "Project uses Docker Compose")
    facts = provider._store.list_facts(limit=5)
    assert len(facts) >= 1

    episodes = provider._store.list_episodes(limit=5)
    assert len(episodes) >= 1, "Episode should be auto-created"

    ep_facts = provider._store.get_episode_facts(episodes[0]["episode_id"])
    assert len(ep_facts) >= 1


def test_handle_fact_store_add_resolves_pronouns(provider):
    """fact_store(add) applies pronoun resolution."""
    provider._handle_fact_store({
        "action": "add",
        "content": "My favorite editor is vim",
    })
    facts = provider._store.list_facts(limit=5)
    assert any("the user's favorite editor is vim" in f["content"] for f in facts)


def test_handle_fact_store_episodes_action(provider):
    """fact_store(action='episodes') returns episode listing."""
    provider.on_memory_write("add", "user", "test fact for episode")
    result = json.loads(provider._handle_fact_store({"action": "episodes"}))
    assert "episodes" in result
    assert result["count"] >= 1


def test_handle_fact_store_episodes_search(provider):
    """fact_store(action='episodes', query=...) searches episodes."""
    # Create an episode with a recognizable title (via on_memory_write)
    provider.on_memory_write("add", "user", "unique-searchable-fact")

    result = json.loads(provider._handle_fact_store({
        "action": "episodes",
        "query": "unique",
    }))
    # It searches titles — the auto-generated title is a timestamp, not "unique"
    # so this tests the code path, not necessarily a hit
    assert "episodes" in result


# ---------------------------------------------------------------------------
# system_prompt_block: should mention episodes action when facts exist
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_episodes(provider):
    provider.on_memory_write("add", "user", "fact for prompt test")
    block = provider.system_prompt_block()
    assert "episodes" in block.lower(), \
        "system prompt should mention episodes browsing"


def test_system_prompt_empty_no_crash(provider):
    """system_prompt_block works even with empty store."""
    block = provider.system_prompt_block()
    assert isinstance(block, str)
    assert len(block) > 0
