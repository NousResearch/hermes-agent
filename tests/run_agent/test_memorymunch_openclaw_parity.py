import importlib.util
import json
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


def test_janitor_runs_every_turn_review_mode(monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor-every-turn"
    events = []

    monkeypatch.delenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", raising=False)
    monkeypatch.setattr(provider, "_append_jsonl", lambda session_id, row: None)
    monkeypatch.setattr(provider, "_append_session_event", lambda session_id, event, **kwargs: events.append((event, kwargs)))
    monkeypatch.setattr(provider, "_maybe_live_capture_exchange", lambda session_id, user, assistant: None)
    monkeypatch.setattr(
        provider,
        "run_janitor_cycle",
        lambda exchange_text, **kwargs: {
            "hermes_mode": "openclaw_janitor_model_review",
            "proposed_actions": {"archive": []},
            "live_db_write": False,
            "live_vault_write": False,
        },
    )

    provider.sync_turn("User request", "Assistant answer", session_id="sid-janitor-every-turn")

    janitor_events = [(event, payload) for event, payload in events if event == "janitor_cycle_completed"]
    assert janitor_events
    assert janitor_events[-1][1]["live_db_write"] is False
    assert janitor_events[-1][1]["status"] == "REVIEWED"


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
    assert ("archive_memory", {"memory_id": "dup-1"}) in calls
    assert any(tool == "edge_cleanup" for tool, _ in calls)


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
