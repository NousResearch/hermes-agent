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


def test_legacy_iso_timestamps_do_not_break_stats_or_sleep(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(tmp_path / "memory.db", time_fn=lambda: clock["now"])
    memory = store.remember("Legacy import used ISO timestamps.", salience=0.8)
    store._conn.execute(
        """
        UPDATE memories
        SET created_at = ?, updated_at = ?, last_rehearsed_at = NULL, last_retrieved_at = NULL
        WHERE memory_id = ?
        """,
        ("2026-06-11T11:28:25.014828", "2026-06-11T11:28:25.014828", memory["memory_id"]),
    )
    store._conn.commit()

    assert store.stats()["count"] == 1
    result = store.get(memory["memory_id"])
    assert result["age_days"] >= 0

    report = store.sleep_cycle(prune=True)
    assert report["mode"] == "sleep_cycle"
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

    # Keep durable retention between forget and rehearse thresholds (forget-first policy).
    clock["now"] += 1 * 86400
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


def test_sleep_limit_and_archive_mode(tmp_path):
    from plugins.memory.ebbinghaus.policies import CapacityPolicy, EbbinghausPolicies, SleepPolicy

    clock = {"now": 1_700_000_000.0}
    policies = EbbinghausPolicies(
        base_stability_days=0.5,
        decay_threshold=0.1,
        sleep=SleepPolicy(
            rehearse_threshold=0.8,
            forget_threshold=0.2,
            salience_keep_threshold=0.75,
            limit=1000,
            prune_mode="archive",
            max_sleep_rehearsals=4,
            max_negative_sleep_rehearsals=1,
        ),
        capacity=CapacityPolicy(max_active_memories=5000, max_archived_memories=20000),
    )
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db",
        time_fn=lambda: clock["now"],
        policies=policies,
    )
    for idx in range(5):
        store.remember(f"low value trace {idx}", salience=0.1, valence=0.0)
    clock["now"] += 8 * 86400
    report = store.sleep_cycle(limit=1000, prune_mode="archive")
    assert report["reviewed"] <= 1000
    assert report["pruned"] == []
    assert report["archived"]
    assert store.stats()["archived_count"] == len(report["archived"])
    assert store.recall("low value", include_archived=False) == []
    assert store.recall("low value", include_archived=True)
    store.close()


def test_max_sleep_rehearsals_and_negative_cap(tmp_path):
    from plugins.memory.ebbinghaus.policies import EbbinghausPolicies, SleepPolicy

    clock = {"now": 1_700_000_000.0}
    policies = EbbinghausPolicies(
        base_stability_days=0.5,
        sleep=SleepPolicy(
            rehearse_threshold=0.9,
            forget_threshold=0.05,
            salience_keep_threshold=0.5,
            limit=100,
            prune_mode="none",
            max_sleep_rehearsals=2,
            max_negative_sleep_rehearsals=1,
            negative_valence_threshold=-0.6,
            negative_reinforcement_multiplier=0.25,
        ),
    )
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db",
        time_fn=lambda: clock["now"],
        policies=policies,
    )
    pos = store.remember("Positive durable preference.", salience=0.95, valence=0.8)
    neg = store.remember("Strongly negative episode detail.", salience=0.95, valence=-0.9)

    for _ in range(4):
        clock["now"] += 8 * 86400
        store.sleep_cycle(prune_mode="none")

    pos_row = store.get(pos["memory_id"])
    neg_row = store.get(neg["memory_id"])
    assert pos_row["sleep_rehearsal_count"] <= 2
    assert neg_row["sleep_rehearsal_count"] <= 1
    store.close()


def test_protected_capacity_error(tmp_path):
    from plugins.memory.ebbinghaus.policies import CapacityPolicy, EbbinghausPolicies
    from plugins.memory.ebbinghaus.store import CapacityError

    policies = EbbinghausPolicies(
        capacity=CapacityPolicy(max_active_memories=2, max_archived_memories=20),
    )
    store = EbbinghausMemoryStore(tmp_path / "memory.db", policies=policies)
    store.remember("Pinned A", tags="pinned", salience=0.9)
    store.remember("Pinned B", tags="safety-critical", salience=0.9)
    with pytest.raises(CapacityError):
        store.remember("Overflow candidate", salience=0.5)
    store.close()


def test_dream_preview_apply_provenance(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(tmp_path / "memory.db", time_fn=lambda: clock["now"])
    a = store.remember("Consent required before VRChat body edit A.", tags="consent,vrchat", salience=0.8, valence=-0.7)
    b = store.remember("Consent required before VRChat body edit B.", tags="consent,vrchat", salience=0.75, valence=-0.65)
    store._conn.execute(
        "UPDATE memories SET dream_candidate = 1 WHERE memory_id IN (?, ?)",
        (a["memory_id"], b["memory_id"]),
    )
    store._conn.commit()

    before_count = store.stats()["count"]
    preview = store.dream_preview()
    assert preview["mode"] == "dream_preview"
    assert preview["clusters"]
    # Preview must not mutate memories rows (ledger goes to dream_previews only).
    assert store.stats()["count"] == before_count
    preview2 = store.dream_preview()
    assert store.stats()["count"] == before_count
    assert len(preview2["clusters"]) <= 8
    # Durable preview ledger for apply-after-reopen.
    ledger = store._conn.execute("SELECT COUNT(*) AS c FROM dream_previews").fetchone()
    assert int(ledger["c"]) >= 1

    cluster = preview["clusters"][0]
    applied = store.dream_apply(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "source_memory_ids": cluster["source_memory_ids"],
                "summary": "Confirm consent before VRChat body manipulation.",
                "tags": ["dream-summary", "semantic", "consent", "safety"],
                "salience": 0.8,
                "valence": -0.1,
            }
        ]
    )
    assert applied["mode"] == "dream_apply"
    assert applied["applied"][0]["status"] == "applied"
    semantic_id = applied["applied"][0]["semantic_memory_id"]
    assert store.get(semantic_id)["memory_type"] == "semantic"
    assert store.get(a["memory_id"])["state"] == "archived"
    again = store.dream_apply(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "source_memory_ids": cluster["source_memory_ids"],
                "summary": "Confirm consent before VRChat body manipulation.",
                "tags": ["dream-summary", "semantic", "consent", "safety"],
                "salience": 0.8,
                "valence": -0.1,
            }
        ]
    )
    assert again["applied"][0]["status"] == "idempotent"
    # Same source-set, different summary text → still idempotent by provenance.
    again2 = store.dream_apply(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "source_memory_ids": cluster["source_memory_ids"],
                "summary": "Alternate wording of the same consent lesson.",
                "tags": ["dream-summary", "semantic", "consent"],
                "salience": 0.7,
                "valence": -0.1,
            }
        ]
    )
    assert again2["applied"][0]["status"] == "idempotent"
    store.close()


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "legacy.db"
    conn = __import__("sqlite3").connect(str(db))
    conn.executescript(
        """
        CREATE TABLE memories (
            memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            encoded TEXT NOT NULL,
            cues TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            salience REAL DEFAULT 0.6,
            valence REAL DEFAULT 0.0,
            strength REAL DEFAULT 1.0,
            rehearsal_count INTEGER DEFAULT 0,
            retrieval_count INTEGER DEFAULT 0,
            source TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_rehearsed_at REAL,
            last_retrieved_at REAL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO memories (
            content, encoded, cues, tags, salience, valence, strength,
            created_at, updated_at, last_rehearsed_at
        ) VALUES ('legacy row', '{}', '', 'legacy', 0.5, 0.0, 1.0, 100.0, 100.0, 100.0)
        """
    )
    conn.commit()
    conn.close()

    store = EbbinghausMemoryStore(db)
    assert store.stats()["count"] == 1
    assert store.get(1)["state"] == "active"
    store.close()
    store2 = EbbinghausMemoryStore(db)
    assert store2.stats()["count"] == 1
    store2.close()


def test_sleep_review_limit_and_progress_across_cycles(tmp_path):
    from plugins.memory.ebbinghaus.policies import CapacityPolicy, EbbinghausPolicies, SleepPolicy

    clock = {"now": 1_700_000_000.0}
    policies = EbbinghausPolicies(
        base_stability_days=0.2,
        sleep=SleepPolicy(
            rehearse_threshold=0.9,
            forget_threshold=0.5,
            salience_keep_threshold=0.99,
            limit=100,
            prune_mode="archive",
            max_sleep_rehearsals=0,
            max_negative_sleep_rehearsals=0,
        ),
        capacity=CapacityPolicy(max_active_memories=5000, max_archived_memories=20000),
    )
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db", time_fn=lambda: clock["now"], policies=policies
    )
    for idx in range(250):
        store.remember(f"low retention batch item {idx}", salience=0.05)
    clock["now"] += 10 * 86400
    first = store.sleep_cycle(limit=100, prune_mode="archive")
    second = store.sleep_cycle(limit=100, prune_mode="archive")
    assert first["reviewed"] <= 100
    assert second["reviewed"] <= 100
    assert set(first["forgotten"]) != set(second["forgotten"]) or first["archived"]
    store.close()


def test_prune_mode_delete_only_physically_deletes(tmp_path):
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db",
        base_stability_days=0.2,
        time_fn=lambda: clock["now"],
    )
    mem = store.remember("delete me please", salience=0.05)
    clock["now"] += 20 * 86400
    report = store.sleep_cycle(
        prune_mode="delete",
        rehearse_threshold=0.9,
        forget_threshold=0.5,
        salience_keep_threshold=0.99,
    )
    assert mem["memory_id"] in report["pruned"]
    assert report["archived"] == []
    with pytest.raises(KeyError):
        store.get(mem["memory_id"])
    store.close()


def test_negative_reinforcement_multiplier_and_forget_exact(tmp_path):
    from plugins.memory.ebbinghaus.policies import EbbinghausPolicies, SleepPolicy

    clock = {"now": 1_700_000_000.0}
    policies = EbbinghausPolicies(
        sleep=SleepPolicy(negative_valence_threshold=-0.6, negative_reinforcement_multiplier=0.25)
    )
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db", time_fn=lambda: clock["now"], policies=policies
    )
    neg = store.remember("neg episode", salience=0.5, valence=-0.9)
    pos = store.remember("pos episode", salience=0.5, valence=0.5)
    neg_row = store._conn.execute(
        "SELECT * FROM memories WHERE memory_id = ?", (neg["memory_id"],)
    ).fetchone()
    pos_row = store._conn.execute(
        "SELECT * FROM memories WHERE memory_id = ?", (pos["memory_id"],)
    ).fetchone()
    assert store._reinforcement_gain(neg_row, kind="rehearsal") == pytest.approx(0.25 * 0.25)
    assert store._reinforcement_gain(pos_row, kind="rehearsal") == pytest.approx(0.25)
    assert store.forget(neg["memory_id"]) is True
    with pytest.raises(KeyError):
        store.get(neg["memory_id"])
    store.close()


def test_plugin_config_defaults_and_tool_overrides(tmp_path):
    provider = EbbinghausMemoryProvider(
        {
            "db_path": str(tmp_path / "cfg.db"),
            "sleep": {
                "limit": 7,
                "rehearse_threshold": 0.4,
                "forget_threshold": 0.15,
                "salience_keep_threshold": 0.85,
                "prune_mode": "archive",
            },
        }
    )
    provider.initialize("s1", hermes_home=str(tmp_path))
    defaults = provider._sleep_defaults()
    assert defaults["limit"] == 7
    assert defaults["rehearse_threshold"] == 0.4
    result = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {"action": "sleep", "limit": 3, "prune_mode": "none"},
        )
    )
    assert result["reviewed"] <= 3
    assert result["prune_mode"] == "none"
    stats = json.loads(provider.handle_tool_call("ebbinghaus_memory", {"action": "stats"}))
    assert "active_count" in stats
    assert "max_active_memories" in stats
    assert "capacity_blocked" in stats
    provider.shutdown()


def test_stats_and_persistence_across_reopen(tmp_path):
    db = tmp_path / "persist.db"
    clock = {"now": 1_700_000_000.0}
    store = EbbinghausMemoryStore(db, time_fn=lambda: clock["now"])
    mem = store.remember("persist me", salience=0.9, valence=0.2, tags="pinned")
    clock["now"] += 2 * 86400
    store.sleep_cycle(
        prune_mode="none",
        rehearse_threshold=0.95,
        forget_threshold=0.05,
        salience_keep_threshold=0.5,
        max_sleep_rehearsals=1,
    )
    mid = mem["memory_id"]
    before = store.get(mid)
    store.close()
    store2 = EbbinghausMemoryStore(db, time_fn=lambda: clock["now"])
    after = store2.get(mid)
    assert after["state"] == before["state"]
    assert after["sleep_rehearsal_count"] == before["sleep_rehearsal_count"]
    assert after["sleep_rehearsal_count"] >= 0
    stats = store2.stats()
    assert stats["protected_count"] >= 1
    assert "oldest_active_age_days" in stats
    store2.close()


def test_safety_critical_not_deleted_for_negative_valence(tmp_path):
    from plugins.memory.ebbinghaus.policies import CapacityPolicy, EbbinghausPolicies, SleepPolicy

    clock = {"now": 1_700_000_000.0}
    policies = EbbinghausPolicies(
        base_stability_days=0.2,
        sleep=SleepPolicy(
            rehearse_threshold=0.9,
            forget_threshold=0.5,
            salience_keep_threshold=0.5,
            prune_mode="delete",
            max_sleep_rehearsals=0,
            max_negative_sleep_rehearsals=0,
        ),
        capacity=CapacityPolicy(max_active_memories=50, max_archived_memories=100),
    )
    store = EbbinghausMemoryStore(
        tmp_path / "memory.db", time_fn=lambda: clock["now"], policies=policies
    )
    lesson = store.remember(
        "Never skip consent checks on VRChat avatar edits.",
        tags="safety-critical,consent",
        salience=0.95,
        valence=-0.9,
    )
    clock["now"] += 30 * 86400
    report = store.sleep_cycle(prune_mode="delete")
    assert lesson["memory_id"] not in report["pruned"]
    assert store.get(lesson["memory_id"])["state"] == "active"
    store.close()


def test_sql_injection_inputs_are_parameterized_and_rejected(tmp_path):
    from plugins.memory.ebbinghaus.store import _assert_safe_identifier, _placeholders

    store = EbbinghausMemoryStore(tmp_path / "memory.db")
    evil = "x'; DROP TABLE memories;--"
    remembered = store.remember(evil, tags="probe", salience=0.5)
    assert store.stats()["count"] == 1
    hits = store.recall("DROP TABLE", reinforce=False)
    assert hits and hits[0]["memory_id"] == remembered["memory_id"]
    # Content with NUL is rejected rather than interpolated.
    with pytest.raises(ValueError):
        store.remember("bad\x00content", salience=0.5)
    with pytest.raises(ValueError):
        store.list_memories(state="active; DROP TABLE memories")
    with pytest.raises(ValueError):
        _assert_safe_identifier("state;DROP TABLE memories")
    with pytest.raises(ValueError):
        _placeholders(0)
    with pytest.raises(ValueError):
        store.forget(0)
    # Table still intact after dangerous-looking inputs.
    assert store.stats()["count"] == 1
    store.close()


def test_remember_idempotent_duplicate_and_capacity_details(tmp_path):
    from plugins.memory.ebbinghaus.policies import CapacityPolicy, EbbinghausPolicies
    from plugins.memory.ebbinghaus.store import CapacityError

    policies = EbbinghausPolicies(
        capacity=CapacityPolicy(max_active_memories=2, max_archived_memories=20),
    )
    store = EbbinghausMemoryStore(tmp_path / "memory.db", policies=policies)
    first = store.remember("same fact about uv", tags="python", salience=0.8)
    second = store.remember("same fact about uv", tags="tooling", salience=0.7)
    assert first["memory_id"] == second["memory_id"]
    assert second["status"] == "reinforced"
    assert store.stats()["count"] == 1
    store.remember("Pinned A", tags="pinned", salience=0.9)
    # Fill remaining slot then block with only protected overflow candidates.
    store2 = EbbinghausMemoryStore(
        tmp_path / "cap.db",
        policies=EbbinghausPolicies(
            capacity=CapacityPolicy(max_active_memories=1, max_archived_memories=10)
        ),
    )
    store2.remember("Pinned only", tags="pinned", salience=0.9)
    with pytest.raises(CapacityError) as excinfo:
        store2.remember("cannot fit", salience=0.5)
    assert excinfo.value.details.get("capacity_blocked") is True
    assert "protected_categories" in excinfo.value.details
    store.close()
    store2.close()


def test_dream_apply_rolls_back_on_bad_source_and_clamps_valence(tmp_path):
    store = EbbinghausMemoryStore(tmp_path / "memory.db")
    a = store.remember("Source A consent lesson alpha.", tags="consent", salience=0.8, valence=-0.8)
    b = store.remember("Source B consent lesson beta.", tags="consent", salience=0.75, valence=-0.7)
    store._conn.execute(
        "UPDATE memories SET dream_candidate = 1 WHERE memory_id IN (?, ?)",
        (a["memory_id"], b["memory_id"]),
    )
    store._conn.commit()
    preview = store.dream_preview()
    cluster = preview["clusters"][0]
    before = store.stats()["count"]
    with pytest.raises(ValueError):
        store.dream_apply(
            [
                {
                    "cluster_id": cluster["cluster_id"],
                    "source_memory_ids": [999999, 999998],
                    "summary": "Should fail source validation.",
                    "salience": 0.5,
                    "valence": -0.1,
                }
            ]
        )
    assert store.stats()["count"] == before
    with pytest.raises(ValueError):
        store.dream_apply(
            [
                {
                    "cluster_id": cluster["cluster_id"],
                    "source_memory_ids": cluster["source_memory_ids"],
                    "summary": "Overconfident salience should be rejected.",
                    "salience": 0.99,
                    "valence": -0.1,
                }
            ]
        )
    applied = store.dream_apply(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "source_memory_ids": cluster["source_memory_ids"],
                "summary": "Reusable consent boundary without graphic detail.",
                "tags": ["dream-summary", "semantic", "consent"],
                "salience": 0.8,
                "valence": -0.95,
            }
        ]
    )
    semantic = store.get(applied["applied"][0]["semantic_memory_id"])
    assert semantic["valence"] >= -0.20
    store.close()


def test_dream_preview_survives_store_reopen(tmp_path):
    db = tmp_path / "dream_persist.db"
    store = EbbinghausMemoryStore(db)
    a = store.remember("Episode A about consent boundary.", tags="consent", salience=0.8, valence=-0.7)
    b = store.remember("Episode B about consent boundary.", tags="consent", salience=0.75, valence=-0.65)
    store._conn.execute(
        "UPDATE memories SET dream_candidate = 1 WHERE memory_id IN (?, ?)",
        (a["memory_id"], b["memory_id"]),
    )
    store._conn.commit()
    preview = store.dream_preview()
    cluster = preview["clusters"][0]
    store.close()

    store2 = EbbinghausMemoryStore(db)
    applied = store2.dream_apply(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "source_memory_ids": cluster["source_memory_ids"],
                "summary": "Confirm consent before irreversible edits.",
                "tags": ["dream-summary", "semantic", "consent"],
                "salience": 0.8,
                "valence": -0.1,
            }
        ]
    )
    assert applied["applied"][0]["status"] == "applied"
    again = store2.dream_apply(
        [
            {
                "cluster_id": cluster["cluster_id"],
                "source_memory_ids": cluster["source_memory_ids"],
                "summary": "Confirm consent before irreversible edits.",
                "tags": ["dream-summary", "semantic", "consent"],
                "salience": 0.8,
                "valence": -0.1,
            }
        ]
    )
    assert again["applied"][0]["status"] == "idempotent"
    store2.close()


def test_provenance_blocks_purge_of_sole_source(tmp_path):
    from plugins.memory.ebbinghaus.policies import CapacityPolicy, EbbinghausPolicies

    clock = {"now": 1_700_000_000.0}
    policies = EbbinghausPolicies(
        capacity=CapacityPolicy(
            max_active_memories=50,
            max_archived_memories=1,
            archive_retention_days=1,
        )
    )
    store = EbbinghausMemoryStore(
        tmp_path / "prov.db", time_fn=lambda: clock["now"], policies=policies
    )
    src = store.remember("Raw episodic detail for provenance.", tags="consent", salience=0.7)
    semantic = store.remember(
        "Reusable consent lesson from dream apply path.",
        tags="dream-summary,semantic",
        salience=0.7,
    )
    store._conn.execute(
        "UPDATE memories SET memory_type = 'semantic' WHERE memory_id = ?",
        (semantic["memory_id"],),
    )
    store._conn.execute(
        """
        INSERT INTO memory_provenance
            (semantic_memory_id, source_memory_id, relation, created_at)
        VALUES (?, ?, 'dream-derived', ?)
        """,
        (semantic["memory_id"], src["memory_id"], clock["now"]),
    )
    store._archive_memory(src["memory_id"], reason="dream-consolidated", now=clock["now"])
    store._conn.commit()
    # Force purge pressure: old archive + over archive cap.
    clock["now"] += 10 * 86400
    purged = store._purge_old_archives(clock["now"])
    assert src["memory_id"] not in purged
    assert store.get(src["memory_id"])["state"] == "archived"
    store.close()


def test_e2e_provider_reopen_with_temp_hermes_home(tmp_path):
    home = tmp_path / "hermes_home"
    home.mkdir()
    db = home / "ebbinghaus_memory.db"
    provider = EbbinghausMemoryProvider({"db_path": str(db), "auto_encode_turns": False})
    provider.initialize("e2e-session", hermes_home=str(home))
    first = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {
                "action": "remember",
                "content": "Operator prefers archive sleep mode.",
                "tags": "user-profile",
                "salience": 0.9,
            },
        )
    )
    sleep = json.loads(
        provider.handle_tool_call(
            "ebbinghaus_memory",
            {"action": "sleep", "limit": 10, "prune_mode": "archive"},
        )
    )
    assert sleep["mode"] == "sleep_cycle"
    provider.shutdown()

    provider2 = EbbinghausMemoryProvider({"db_path": str(db)})
    provider2.initialize("e2e-session-2", hermes_home=str(home))
    stats = json.loads(provider2.handle_tool_call("ebbinghaus_memory", {"action": "stats"}))
    assert stats["count"] >= 1
    recalled = json.loads(
        provider2.handle_tool_call(
            "ebbinghaus_memory",
            {"action": "recall", "query": "archive sleep", "limit": 3},
        )
    )
    assert any(item["memory_id"] == first["memory_id"] for item in recalled["results"])
    provider2.shutdown()