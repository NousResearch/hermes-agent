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


def test_janitor_apply_gate_blocks_without_exact_approval(tmp_path, monkeypatch):
    mm = load_plugin()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-janitor"
    provider._scope_entity = "scope-a"
    rollback = tmp_path / "rollback"
    (rollback / "db").mkdir(parents=True)
    (rollback / "vault").mkdir(parents=True)

    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_MODEL_ENABLE", "1")
    monkeypatch.setenv("HERMES_MEMORYMUNCH_JANITOR_APPLY_ENABLE", "1")
    monkeypatch.setattr(provider, "_run_janitor_model_review", lambda exchange_text, max_candidates=20: {"archive": ["dup-1"], "edge_cleanup": False, "edge_prune": []})

    result = provider.run_janitor_cycle(
        "User: duplicate cleanup\nBot: ok",
        apply=True,
        approval_phrase="wrong",
        rollback_pack_path=str(rollback),
    )

    assert result["status"] == "BLOCKED"
    assert result["expected_approval_phrase"] == "APPROVE memorymunch janitor apply sid-janitor"
    assert result["live_db_write"] is False


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
