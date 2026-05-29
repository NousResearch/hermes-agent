"""Tests for the local Ebbinghaus memory plugin."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.ebbinghaus import (  # noqa: E402
    EbbinghausMemoryProvider,
    EbbinghausMemoryStore,
    forgetting_retention,
)


def test_forgetting_retention_uses_exponential_curve():
    assert forgetting_retention(0, 3) == 1.0
    assert forgetting_retention(6, 3) == pytest.approx(math.exp(-2))
    assert 0 < forgetting_retention(30, 3) < 0.001


def test_store_encodes_deduplicates_and_recalls(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(tmp_path / "memory.db", time_fn=lambda: clock["now"])

    first = store.remember(
        "Use uv for Python dependency management on this PC.",
        tags="python,uv,dev",
        salience=0.9,
    )
    duplicate = store.remember(
        "Use uv for Python dependency management on this PC.",
        tags="python,tooling",
        salience=0.7,
    )

    assert duplicate["memory_id"] == first["memory_id"]
    assert duplicate["status"] == "reinforced"
    assert store.stats()["count"] == 1

    results = store.recall("python uv dependency", reinforce=False)
    assert results
    assert results[0]["memory_id"] == first["memory_id"]
    assert "uv" in results[0]["cues"]
    assert "tooling" in results[0]["tags"]

    store.close()


def test_rehearsal_restores_retention_after_decay(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db",
        base_stability_days=1.0,
        time_fn=lambda: clock["now"],
    )
    memory = store.remember("Telegram sitrep should be sent after Hermes startup.", salience=0.4)

    clock["now"] += 20 * 86400
    before = store.get(memory["memory_id"])["retention"]
    rehearsed = store.rehearse(memory_id=memory["memory_id"])[0]

    assert before < 0.05
    assert rehearsed["retention"] == pytest.approx(1.0)
    assert rehearsed["rehearsal_count"] == 1

    store.close()


def test_decay_can_prune_forgotten_memories(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db",
        base_stability_days=0.5,
        decay_threshold=0.1,
        time_fn=lambda: clock["now"],
    )
    memory = store.remember("Temporary setup detail that should fade.", salience=0.1)

    clock["now"] += 8 * 86400
    result = store.decay(prune=True)

    assert result["pruned"] == [memory["memory_id"]]
    assert store.stats()["count"] == 0

    store.close()


def test_sleep_cycle_rehearses_important_memories_and_prunes_low_value_traces(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db",
        base_stability_days=0.5,
        decay_threshold=0.1,
        time_fn=lambda: clock["now"],
    )
    durable = store.remember(
        "User prefers memory maintenance to model human sleep consolidation.",
        tags="user-preference,memory-design",
        salience=0.95,
    )
    ephemeral = store.remember(
        "One-time OAuth consent URL state=abc123.",
        tags="ephemeral,oauth",
        salience=0.1,
    )

    clock["now"] += 8 * 86400
    report = store.sleep_cycle(
        prune=True,
        rehearse_threshold=0.8,
        forget_threshold=0.2,
        salience_keep_threshold=0.75,
    )

    assert report["mode"] == "sleep_cycle"
    assert durable["memory_id"] in report["rehearsed"]
    assert ephemeral["memory_id"] in report["pruned"]
    assert store.get(durable["memory_id"])["retention"] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        store.get(ephemeral["memory_id"])

    store.close()


def test_provider_exposes_sleep_cycle_tool_action(tmp_path):
    provider = EbbinghausMemoryProvider({"db_path": str(tmp_path / "provider.db")})
    provider.initialize("session-1", hermes_home=str(tmp_path))

    add_result = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {
                "action": "remember",
                "content": "Important preference should survive sleep consolidation.",
                "tags": "user-preference",
                "salience": 0.9,
            },
        )
    )
    result = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {"action": "sleep", "prune": True, "rehearse_threshold": 1.0},
        )
    )

    assert result["mode"] == "sleep_cycle"
    assert add_result["memory_id"] in result["rehearsed"]

    provider.shutdown()


def test_provider_tools_and_prefetch(tmp_path):
    provider = EbbinghausMemoryProvider(
        {
            "db_path": str(tmp_path / "provider.db"),
            "max_prefetch": 3,
            "min_prefetch_score": 0.01,
        }
    )
    provider.initialize("session-1", hermes_home=str(tmp_path))

    add_result = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {
                "action": "remember",
                "content": "Hermes WebUI is protected by HTTP 401 before login.",
                "tags": "hermes,webui,auth",
                "salience": 0.85,
            },
        )
    )
    assert add_result["status"] == "remembered"

    recall_result = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {"action": "recall", "query": "webui auth", "limit": 2},
        )
    )
    assert recall_result["results"][0]["memory_id"] == add_result["memory_id"]
    assert recall_result["results"][0]["retrieval_count"] == 1

    prefetch = provider.prefetch("How is Hermes WebUI protected?")
    assert "Ebbinghaus Memory" in prefetch
    assert "HTTP 401" in prefetch

    provider.shutdown()


def test_memory_provider_discovery_loads_ebbinghaus():
    from plugins.memory import discover_memory_providers, load_memory_provider

    names = [name for name, _desc, _available in discover_memory_providers()]
    assert "ebbinghaus" in names

    provider = load_memory_provider("ebbinghaus")
    assert provider is not None
    assert provider.name == "ebbinghaus"
    assert provider.is_available()
