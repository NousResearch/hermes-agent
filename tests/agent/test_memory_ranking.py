from __future__ import annotations

from agent.memory_ranking import memory_key, reinforce_access, rerank_memories


NOW = 1_700_000_000.0
DAY = 86_400.0


def test_memory_key_is_stable_and_normalized():
    assert memory_key("  User Prefers Concise Replies ") == memory_key("user prefers concise replies")
    assert len(memory_key("anything")) == 16


def test_rerank_memories_combines_query_overlap_and_backend_score():
    items = [
        {
            "memory": "The deployment route is https://example.invalid",
            "score": 0.35,
            "metadata": {"category": "service", "last_verified": NOW - DAY},
        },
        {
            "memory": "The user likes short replies",
            "score": 0.5,
            "metadata": {"category": "preference", "last_verified": NOW - DAY},
        },
    ]

    ranked = rerank_memories(items, query="deployment route", now=NOW)

    assert ranked[0]["memory"].startswith("The deployment route")
    assert ranked[0]["freshness"] == "fresh"
    assert ranked[0]["category"] == "service"
    assert ranked[0]["durability"] == "stable"
    assert ranked[0]["rank_input"] == 0
    assert ranked[0]["score"] > ranked[1]["score"]


def test_stable_memories_decay_less_than_working_memories():
    old = NOW - 120 * DAY
    items = [
        {
            "memory": "Stable service route for the dashboard",
            "score": 0.8,
            "metadata": {"durability": "stable", "last_verified": old},
        },
        {
            "memory": "Working note from a stale experiment",
            "score": 0.8,
            "metadata": {"durability": "working", "last_verified": old},
        },
    ]

    ranked = rerank_memories(items, query="", now=NOW)

    assert ranked[0]["durability"] == "stable"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert all(item["freshness"] == "stale" for item in ranked)


def test_current_context_records_are_protected_from_strong_decay():
    old = NOW - 120 * DAY
    ranked = rerank_memories(
        [
            {
                "memory": "Current context checkpoint for active migration",
                "score": 0.7,
                "metadata": {"durability": "working", "last_verified": old},
            }
        ],
        now=NOW,
    )

    assert ranked[0]["category"] == "current_context"
    assert ranked[0]["score"] > 0.5


def test_reinforce_access_updates_caller_owned_state():
    state = {}
    returned = reinforce_access(state, ["Remember the deployment route", "Remember the deployment route"], timestamp=NOW)
    key = memory_key("Remember the deployment route")

    assert returned is state
    assert state["version"] == 1
    assert state["memories"][key]["last_accessed"] == NOW
    assert state["memories"][key]["access_count"] == 2


def test_rerank_memories_does_not_mutate_input_items():
    items = [{"memory": "Preference: concise", "score": 0.5, "metadata": {}}]

    ranked = rerank_memories(items, now=NOW)

    assert "freshness" not in items[0]
    assert ranked[0]["freshness"] == "untracked"
