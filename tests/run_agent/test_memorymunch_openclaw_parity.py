import importlib.util
import json
import threading
import time
from pathlib import Path


PLUGIN_PATH = Path.home() / ".hermes" / "plugins" / "memorymunch" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("memorymunch_plugin_parity_under_test", PLUGIN_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_raw_search_rows_injected_when_curator_unavailable(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-curator-unavailable"
    provider._scope_entity = "scope-a"
    provider._domain = "general"

    monkeypatch.delenv("HERMES_MEMORYMUNCH_CURATOR_MODEL_ENABLE", raising=False)
    monkeypatch.setattr(
        provider,
        "_active_session_rows",
        lambda session_id, query="": [
            {
                "id": "active-1",
                "source": "ACTIVE_SESSION_LEDGER",
                "provenance_class": "ACTIVE_SESSION_LEDGER_CURRENT",
                "content_preview": "Current session asked for OpenClaw parity.",
            }
        ],
    )
    monkeypatch.setattr(
        provider,
        "_run_readonly_recall",
        lambda query, max_results=6: {
            "results": [
                {
                    "id": "raw-noise",
                    "source": "vault,activation",
                    "provenance_class": "OWN_SCOPE",
                    "content_preview": "Al Cooke closed on 245 Lake View Drive and has unrelated income streams.",
                }
            ]
        },
    )

    context = provider._compose_prefetch_context("OpenClaw MemoryMunch curator parity", "sid-curator-unavailable")

    assert context.count("<memorymunch-briefing") == 1
    assert "CURATOR_UNAVAILABLE" in context
    assert "Current session asked for OpenClaw parity" in context
    assert "Lake View Drive" not in context
    assert "income streams" not in context


def test_pasted_memory_context_is_stripped_before_prefetch_query(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-prefetch-sanitize"
    provider._scope_entity = "scope-a"
    provider._domain = "general"
    seen = {}
    events = []

    monkeypatch.setattr(provider, "_active_session_rows", lambda session_id, query="": [])

    def fake_recall(query, max_results=6):
        seen["query"] = query
        return {"results": []}

    monkeypatch.setattr(provider, "_run_readonly_recall", fake_recall)
    monkeypatch.setattr(provider, "_append_session_event", lambda session_id, event, **kwargs: events.append((event, kwargs)))

    provider._compose_prefetch_context(
        """
        Explain MemoryMunch OpenClaw parity.
        <memory-context>
        <memorymunch-briefing>
        - unrelated theology/email/property atom [source=vault,activation; atom=noise]
        </memorymunch-briefing>
        </memory-context>
        """,
        "sid-prefetch-sanitize",
    )

    assert "theology" not in seen["query"]
    assert "email" not in seen["query"]
    assert "property atom" not in seen["query"]
    assert "Explain MemoryMunch OpenClaw parity" in seen["query"]
    assert any(event == "prefetch_query_sanitized" for event, _ in events)


def test_briefing_marks_old_session_atoms_history_only_never_live_intent():
    mm = load_plugin()

    briefing = mm.format_memorymunch_briefing(
        [
            {
                "id": "active::sid-current::1",
                "source_session_id": "sid-current",
                "provenance_class": "ACTIVE_SESSION_LEDGER_CURRENT",
                "content_preview": "Current live user asked for MemoryMunch build.",
            },
            {
                "id": "ledger::sid-old::1",
                "source_session_id": "sid-old",
                "provenance_class": "ACTIVE_SESSION_LINEAGE",
                "content_preview": "Old session user wanted a different task.",
            },
        ],
        active_session_id="sid-current",
        scope_entity="tg-test",
    )

    assert "current_intent=live_user_msg@active_session_id_only" in briefing
    assert "current_session_yes" not in briefing
    assert "source_session_id=sid-current; current_session=yes; intent_scope=active_session_context" in briefing
    assert "source_session_id=sid-old; current_session=no; intent_scope=history_only_never_live_intent_task_state" in briefing


def test_janitor_runs_every_turn_review_mode(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor-every-turn"
    events = []
    calls = []

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_ENABLE", "0")
    monkeypatch.delenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", raising=False)
    monkeypatch.setattr(provider, "_append_jsonl", lambda session_id, row: None)
    monkeypatch.setattr(provider, "_append_session_event", lambda session_id, event, **kwargs: events.append((event, kwargs)))
    monkeypatch.setattr(provider, "_maybe_live_capture_exchange", lambda session_id, user, assistant: None)

    def fake_janitor(exchange_text, **kwargs):
        calls.append(kwargs)
        return {
            "hermes_mode": "approved_openclaw_janitor_apply",
            "proposed_actions": {"archive": []},
            "status": "APPLIED",
            "live_db_write": True,
            "live_vault_write": True,
        }

    monkeypatch.setattr(provider, "run_janitor_cycle", fake_janitor)

    provider.sync_turn("User request", "Assistant answer", session_id="sid-janitor-every-turn")

    janitor_events = [(event, payload) for event, payload in events if event == "janitor_cycle_completed"]
    assert janitor_events
    assert not any(event == "janitor_cycle_skipped" for event, _ in events)
    assert calls[-1]["apply"] is True
    assert calls[-1]["approval_phrase"] == "AL_DIRECT_APPROVAL"
    assert janitor_events[-1][1]["live_db_write"] is True
    assert janitor_events[-1][1]["live_vault_write"] is True
    assert janitor_events[-1][1]["status"] == "APPLIED"


def test_capture_live_write_ignores_env_disable(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()

    monkeypatch.setenv("HERMES_MEMORYMUNCH_LIVE_WRITE_ENABLE", "0")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_AUTO_CAPTURE_ENABLE", "0")

    assert provider._live_capture_enabled() is True
    assert provider._capture_mode() == "live"
    assert provider._janitor_mode() == "live"


def test_curator_model_parity_uses_search_and_deep_read_before_injection(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-curator-model"
    provider._scope_entity = "scope-a"
    provider._domain = "general"
    provider._recent_exchanges["sid-curator-model"] = [
        {"user": "Need OpenClaw MemoryMunch curator parity", "assistant": "Working on it", "source": "ACTIVE_SESSION_LEDGER"}
    ]

    bridge_calls = []

    def fake_bridge(tool, args, timeout=180):
        bridge_calls.append((tool, args))
        if tool == "smart_search":
            return {"result": {"results": [{"id": "atom-1", "content": "Relevant curator atom", "search_score": 0.99}]}}
        if tool == "get_memory":
            return {"result": {"memory": {"id": args["memory_id"], "content": "Deep read relevant curator atom"}}}
        raise AssertionError(tool)

    def fake_model(role, system_prompt, user_prompt, timeout=180):
        assert role == "curator"
        assert "memorymunch_search_and_read" in system_prompt
        assert "Deep read relevant curator atom" in user_prompt
        assert "CURRENT USER MESSAGE" in user_prompt
        return "Curator briefing: use only the deep-read relevant curator atom."

    monkeypatch.setenv("HERMES_MEMORYMUNCH_CURATOR_MODEL_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)
    monkeypatch.setattr(provider, "_call_memorymunch_worker_model", fake_model)

    context = provider._compose_prefetch_context("OpenClaw MemoryMunch curator parity", "sid-curator-model")

    assert "Curator briefing" in context
    assert any(call[0] == "smart_search" for call in bridge_calls)
    assert any(call[0] == "get_memory" for call in bridge_calls)
    assert "curator_mode=model" in context


def test_search_and_deep_read_fetches_top_atoms_in_parallel(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._scope_entity = "scope-a"

    started = []
    lock = threading.Lock()
    all_started = threading.Event()

    def fake_bridge(tool, args, timeout=180):
        if tool == "smart_search":
            return {
                "result": {
                    "results": [
                        {"id": "atom-1", "content": "one", "search_score": 0.99},
                        {"id": "atom-2", "content": "two", "search_score": 0.98},
                        {"id": "atom-3", "content": "three", "search_score": 0.97},
                    ],
                    "_meta": {"timing_ms": 10},
                }
            }
        if tool == "get_memory":
            with lock:
                started.append(args["memory_id"])
                if len(started) == 3:
                    all_started.set()
            assert all_started.wait(0.5), "deep reads did not overlap like original bounded parallel fanout"
            return {"result": {"memory": {"id": args["memory_id"], "content": f"full {args['memory_id']}"}}}
        raise AssertionError(tool)

    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)

    result = provider._search_and_deep_read("parallel curator parity", max_results=15, deep_read_count=3)

    assert sorted(started) == ["atom-1", "atom-2", "atom-3"]
    assert len(result["deep_reads"]) == 3
    assert result["_meta"]["deep_read_mode"] == "parallel"


def test_janitor_prefetch_fetches_prescans_and_context_in_parallel(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor-parallel"
    provider._scope_entity = "scope-a"

    events = []
    lock = threading.Lock()
    get_started = threading.Event()
    search_started = threading.Event()

    def fake_bridge(tool, args, timeout=180):
        if tool == "smart_cleanup":
            return {"result": {"duplicates": [{"id": "dup-1", "similar_to": "dup-2", "content_preview": "duplicate"}], "stale": [{"id": "stale-1", "content_preview": "stale"}], "correction_events": [], "edge_heavy": [], "orphan_edges": 0}}
        if tool == "get_memory":
            with lock:
                events.append((tool, args["memory_id"], time.perf_counter()))
            get_started.set()
            assert search_started.wait(0.5), "janitor prescan get_memory did not overlap with context smart_search"
            return {"result": {"memory": {"id": args["memory_id"], "content": "full candidate"}}}
        if tool == "smart_search":
            with lock:
                events.append((tool, "context", time.perf_counter()))
            search_started.set()
            assert get_started.wait(0.5), "janitor context smart_search did not overlap with prescan get_memory"
            return {"result": {"results": [{"id": "context-1", "content": "context atom"}]}}
        raise AssertionError(tool)

    def fake_model(role, system_prompt, user_prompt, timeout=180):
        assert role == "janitor"
        assert "full candidate" in user_prompt
        assert "context atom" in user_prompt
        return json.dumps({"archive": [], "edge_cleanup": False, "edge_prune": []})

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_MODEL_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)
    monkeypatch.setattr(provider, "_call_memorymunch_worker_model", fake_model)

    result = provider.run_janitor_cycle("User: parallel janitor\nBot: ok", apply=False)

    assert result["hermes_mode"] == "openclaw_janitor_model_review"
    assert any(event[0] == "get_memory" for event in events)
    assert any(event[0] == "smart_search" for event in events)


def test_janitor_model_cycle_builds_review_plan_without_mutation(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor"
    provider._scope_entity = "scope-a"
    provider._domain = "general"

    bridge_calls = []

    def fake_bridge(tool, args, timeout=180):
        bridge_calls.append((tool, args))
        if tool == "smart_cleanup":
            return {"result": {"duplicates": [{"id": "dup-1", "similar_to": "dup-2", "content_preview": "duplicate"}], "stale": [], "correction_events": [], "edge_heavy": [], "orphan_edges": 0}}
        if tool == "get_memory":
            return {"result": {"memory": {"id": args["memory_id"], "content": "duplicate full content"}}}
        if tool == "smart_search":
            return {"result": {"results": []}}
        raise AssertionError(f"mutation should not run in review mode: {tool}")

    def fake_model(role, system_prompt, user_prompt, timeout=180):
        assert role == "janitor"
        assert "memorymunch_archive" in system_prompt
        assert "duplicate full content" in user_prompt
        return json.dumps({"archive": ["dup-1"], "edge_cleanup": True, "edge_prune": []})

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_MODEL_ENABLE", "1")
    monkeypatch.delenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", raising=False)
    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)
    monkeypatch.setattr(provider, "_call_memorymunch_worker_model", fake_model)

    result = provider.run_janitor_cycle("User: duplicate cleanup\nBot: ok", apply=False)

    assert result["hermes_mode"] == "openclaw_janitor_model_review"
    assert result["live_db_write"] is False
    assert result["proposed_actions"]["archive"] == ["dup-1"]
    assert any(call[0] == "smart_cleanup" for call in bridge_calls)
    assert all(call[0] not in {"archive_memory", "edge_cleanup", "edge_prune"} for call in bridge_calls)


def test_janitor_apply_refuses_protected_atoms_even_with_al_direct_approval(tmp_path, monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor"
    provider._scope_entity = "scope-a"
    rollback = tmp_path / "rollback"
    (rollback / "db").mkdir(parents=True)
    (rollback / "vault").mkdir(parents=True)

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_MODEL_ENABLE", "1")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_janitor_model_review", lambda exchange_text, max_candidates=20: {"archive": ["system::janitor-prompt#procedural"], "edge_cleanup": False, "edge_prune": []})

    result = provider.run_janitor_cycle(
        "User: duplicate cleanup\nBot: ok",
        apply=True,
        approval_phrase="AL_DIRECT_APPROVAL",
        rollback_pack_path=str(rollback),
    )

    assert result["status"] == "BLOCKED"
    assert "protected_atom_requested" in result["blocked_by"]
    assert result["live_db_write"] is False


def test_janitor_apply_accepts_al_direct_approval_with_rollback(tmp_path, monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor"
    rollback = tmp_path / "rollback"
    (rollback / "db").mkdir(parents=True)
    (rollback / "vault").mkdir(parents=True)
    calls = []

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_janitor_model_review", lambda exchange_text, max_candidates=20: {"archive": ["dup-1"], "edge_cleanup": True, "edge_prune": []})

    def fake_bridge(tool, args, timeout=180):
        calls.append((tool, args))
        if tool == "archive_memory":
            return {"result": {"ok": True, "vault_archived": True, "archived_vault_path": "/vault/_archived/dup-1.md"}}
        return {"result": {"ok": True}}

    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)

    result = provider.run_janitor_cycle(
        "User: duplicate cleanup\nBot: ok",
        apply=True,
        approval_phrase="AL_DIRECT_APPROVAL",
        rollback_pack_path=str(rollback),
    )

    assert result["status"] == "APPLIED"
    assert result["live_vault_write"] is True
    assert result["vault_archive_status"] == "vault_moved_true"
    assert result["vault_archive_counts"] == {"requested": 1, "moved": 1, "not_found": 0, "failed": 0}
    assert result["vault_archive_details"][0]["memory_id"] == "dup-1"
    assert result["vault_archive_details"][0]["status"] == "vault_moved_true"
    assert result["vault_archive_details"][0]["archived_vault_path"] == "/vault/_archived/dup-1.md"
    assert ("archive_memory", {"memory_id": "dup-1"}) in calls
    assert any(tool == "edge_cleanup" for tool, _ in calls)


def test_janitor_apply_telemetry_distinguishes_no_vault_file(monkeypatch, tmp_path):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor-telemetry-no-file"
    rollback = tmp_path / "rollback"
    (rollback / "db").mkdir(parents=True)
    (rollback / "vault").mkdir(parents=True)

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_janitor_model_review", lambda exchange_text, max_candidates=20: {"archive": ["dup-no-file"], "edge_cleanup": False, "edge_prune": []})
    monkeypatch.setattr(provider, "_run_original_bridge", lambda tool, args, timeout=180: {"result": {"ok": True, "memory_id": args["memory_id"], "vault_archived": False, "vault_reason": "vault_atom_not_found"}})

    result = provider.run_janitor_cycle(
        "User: cleanup\nBot: ok",
        apply=True,
        approval_phrase="AL_DIRECT_APPROVAL",
        rollback_pack_path=str(rollback),
    )

    assert result["status"] == "APPLIED"
    assert result["live_db_write"] is True
    assert result["live_vault_write"] is False
    assert result["vault_archive_status"] == "no_vault_file_to_move"
    assert result["vault_archive_counts"] == {"requested": 1, "moved": 0, "not_found": 1, "failed": 0}
    assert result["vault_archive_details"] == [{"memory_id": "dup-no-file", "status": "no_vault_file_to_move", "reason": "vault_atom_not_found"}]


def test_janitor_apply_telemetry_distinguishes_vault_move_failed(monkeypatch, tmp_path):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor-telemetry-failed"
    rollback = tmp_path / "rollback"
    (rollback / "db").mkdir(parents=True)
    (rollback / "vault").mkdir(parents=True)

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_janitor_model_review", lambda exchange_text, max_candidates=20: {"archive": ["dup-failed"], "edge_cleanup": False, "edge_prune": []})
    monkeypatch.setattr(provider, "_run_original_bridge", lambda tool, args, timeout=180: {"result": {"ok": True, "memory_id": args["memory_id"], "vault_archived": False, "vault_reason": "permission_denied"}})

    result = provider.run_janitor_cycle(
        "User: cleanup\nBot: ok",
        apply=True,
        approval_phrase="AL_DIRECT_APPROVAL",
        rollback_pack_path=str(rollback),
    )

    assert result["status"] == "APPLIED"
    assert result["live_db_write"] is True
    assert result["live_vault_write"] is False
    assert result["vault_archive_status"] == "vault_move_failed"
    assert result["vault_archive_counts"] == {"requested": 1, "moved": 0, "not_found": 0, "failed": 1}
    assert result["vault_archive_details"] == [{"memory_id": "dup-failed", "status": "vault_move_failed", "reason": "permission_denied"}]


def test_maybe_janitor_cycle_logs_vault_archive_telemetry(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor-event-telemetry"
    events = []

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", "1")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPROVAL_PHRASE", "AL_DIRECT_APPROVAL")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_ROLLBACK_PACK", "/tmp/rollback-proof")
    monkeypatch.setattr(provider, "_append_session_event", lambda session_id, event, **kwargs: events.append((event, kwargs)))
    monkeypatch.setattr(provider, "run_janitor_cycle", lambda *args, **kwargs: {
        "status": "APPLIED",
        "hermes_mode": "approved_openclaw_janitor_apply",
        "live_db_write": True,
        "live_vault_write": False,
        "vault_archive_status": "no_vault_file_to_move",
        "vault_archive_counts": {"requested": 1, "moved": 0, "not_found": 1, "failed": 0},
        "vault_archive_details": [{"memory_id": "dup-no-file", "status": "no_vault_file_to_move", "reason": "vault_atom_not_found"}],
    })

    provider._maybe_janitor_cycle("sid-janitor-event-telemetry", "User cleanup", "Assistant ok")

    event, payload = events[-1]
    assert event == "janitor_cycle_completed"
    assert payload["vault_archive_status"] == "no_vault_file_to_move"
    assert payload["vault_archive_counts"] == {"requested": 1, "moved": 0, "not_found": 1, "failed": 0}
    assert payload["vault_archive_details"][0]["reason"] == "vault_atom_not_found"


def test_live_capture_sanitizes_recalled_memory_before_ingest(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-capture-sanitize"
    provider._scope_entity = "tg-7475127948"
    provider._domain = "general"

    captured = {}

    def fake_bridge(tool, args, timeout=180):
        assert tool == "ingest_exchange"
        captured.update(args)
        return {"result": {"exchange_id": "conv::safe", "facts_stored": len(args.get("facts") or []), "facts_failed": 0, "_meta": {"total_atoms_created": 1}}}

    monkeypatch.setenv("HERMES_MEMORYMUNCH_LIVE_WRITE_ENABLE", "1")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_AUTO_CAPTURE_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_original_bridge", fake_bridge)

    assistant = """
Here is the answer.
<memorymunch-briefing isolation="soft" scope_entity="tg-7475127948" domain="general">
OWN_SCOPE:
- Al Cooke closed on 245 Lake View Drive and has unrelated income streams. [source=vault,activation; atom=noise]
</memorymunch-briefing>
Final answer should not persist recalled briefing facts.
"""

    user = """
Fix MemoryMunch capture sanitizer.
<memory-context>
<memorymunch-briefing isolation="soft" scope_entity="tg-7475127948" domain="general">
OWN_SCOPE:
- Al Cooke closed on 245 Lake View Drive and has unrelated income streams. [source=vault,activation; atom=noise]
</memorymunch-briefing>
</memory-context>
"""

    provider._maybe_live_capture_exchange("sid-capture-sanitize", user, assistant)

    assert "245 Lake View" not in captured["user_message"]
    assert "income streams" not in captured["user_message"]
    assert "memory-context" not in captured["user_message"]
    assert "245 Lake View" not in captured["bot_response"]
    assert "income streams" not in captured["bot_response"]
    assert "memorymunch-briefing" not in captured["bot_response"]
    assert all("245 Lake View" not in fact.get("fact", "") for fact in captured.get("facts") or [])
