import copy
import importlib.util
import json
from pathlib import Path


PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "context_engine" / "context_governor" / "__init__.py"


def load_plugin_module():
    spec = importlib.util.spec_from_file_location("context_governor_plugin_under_test", PLUGIN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_memory_archive_payloads_include_exact_content_and_receipt_metadata(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", "true")
    engine = mod.ContextGovernorEngine()
    response = {
        "receipt": {"receipt_id": "ctxr_1", "semantic_memory_fact_ids": []},
        "allocation_plan": {
            "archived_item_ids": ["item-1"],
            "items": [
                {
                    "item_id": "item-1",
                    "item_type": "ToolResult",
                    "authority_class": "EvidenceCritical",
                    "content_kind": "CommandOutput",
                }
            ],
        },
        "exact_store": [
            {
                "item_id": "item-1",
                "content": "durable exact content NEEDLE",
                "content_blake3": "b3abc",
            }
        ],
    }

    payloads = engine._semantic_memory_archive_payloads(response)

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["namespace"] == "projects"
    assert payload["source"] == "context-governor receipt ctxr_1 item item-1"
    assert "ctxr_1" in payload["content"]
    assert "item-1" in payload["content"]
    assert "b3abc" in payload["content"]
    assert "durable exact content NEEDLE" in payload["content"]


def test_semantic_memory_archive_payloads_honor_namespace_override(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", "true")
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_NAMESPACE", "context_governor_bench")
    engine = mod.ContextGovernorEngine()
    response = {
        "receipt": {"receipt_id": "ctxr_ns", "semantic_memory_fact_ids": []},
        "allocation_plan": {
            "archived_item_ids": ["item-ns"],
            "items": [{"item_id": "item-ns", "item_type": "Decision", "authority_class": "DurableMemoryCandidate", "content_kind": "Text"}],
        },
        "exact_store": [{"item_id": "item-ns", "content": "Decision: isolated namespace", "content_blake3": "b3ns"}],
    }

    payloads = engine._semantic_memory_archive_payloads(response)

    assert payloads[0]["namespace"] == "context_governor_bench"


def test_semantic_memory_archive_posts_facts_and_updates_receipt(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", "true")
    engine = mod.ContextGovernorEngine()
    response = {
        "receipt": {"receipt_id": "ctxr_2", "semantic_memory_fact_ids": []},
        "allocation_plan": {
            "archived_item_ids": ["item-2"],
            "items": [{"item_id": "item-2", "item_type": "Decision", "authority_class": "DurableMemoryCandidate", "content_kind": "Text"}],
        },
        "exact_store": [{"item_id": "item-2", "content": "Decision: persist me", "content_blake3": "b3def"}],
    }
    calls = []

    def fake_post_json(path, payload, timeout=10):
        calls.append((path, payload, timeout))
        if path == "/search":
            return {"ok": True, "results": []}
        if path == "/add":
            return {"ok": True, "fact_id": "fact-123"}
        raise AssertionError(path)

    monkeypatch.setattr(engine, "_semantic_memory_post_json", fake_post_json)

    engine._archive_response_to_semantic_memory(response)

    assert any(path == "/add" for path, _payload, _timeout in calls)
    assert response["receipt"]["semantic_memory_fact_ids"] == ["fact-123"]
    assert not response["receipt"].get("warnings")


def test_semantic_memory_archive_dedupes_by_content_blake3(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", "true")
    engine = mod.ContextGovernorEngine()
    response = {
        "receipt": {"receipt_id": "ctxr_3", "semantic_memory_fact_ids": []},
        "allocation_plan": {
            "archived_item_ids": ["item-3"],
            "items": [{"item_id": "item-3", "item_type": "Decision", "authority_class": "DurableMemoryCandidate", "content_kind": "Text"}],
        },
        "exact_store": [{"item_id": "item-3", "content": "Decision: already persisted", "content_blake3": "b3ghi"}],
    }
    calls = []

    def fake_post_json(path, payload, timeout=10):
        calls.append(path)
        if path == "/search":
            return {"ok": True, "results": [{"result_id": "fact:existing", "content": "content_blake3: b3ghi"}]}
        if path == "/add":
            raise AssertionError("dedupe should skip /add")
        raise AssertionError(path)

    monkeypatch.setattr(engine, "_semantic_memory_post_json", fake_post_json)

    engine._archive_response_to_semantic_memory(response)

    assert calls == ["/search"]
    assert response["receipt"]["semantic_memory_fact_ids"] == ["fact:existing"]


def test_hard_cascade_failure_retries_soft_warn(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_BUDGET_MODE", "hard_cascade")
    engine = mod.ContextGovernorEngine()
    monkeypatch.setattr(engine, "_ensure_binary", lambda: None)
    monkeypatch.setattr(engine, "_persist_response", lambda response: None)
    monkeypatch.setattr(engine, "_archive_response_to_semantic_memory", lambda response: None)
    calls = []

    def fake_run(args, request):
        calls.append(request["policy"]["budget_mode"])
        if request["policy"]["budget_mode"] == "hard_cascade":
            raise mod.subprocess.CalledProcessError(1, ["context-governor", "compact"], stderr="budget exceeded")
        return {
            "receipt": {"receipt_id": "ctxr_retry", "compacted_approx_tokens": 42},
            "compacted_messages": [{"role": "assistant", "content": "summary"}, {"role": "user", "content": "latest"}],
            "allocation_plan": {"items": [], "archived_item_ids": []},
            "exact_store": [],
        }

    monkeypatch.setattr(engine, "_run_context_governor", fake_run)

    result = engine.compress([
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "latest"},
    ], current_tokens=20000)

    assert calls == ["hard_cascade", "soft_warn"]
    assert result[-1] == {"role": "user", "content": "latest"}
    assert engine.last_prompt_tokens == 42


def test_context_governor_plugin_exposes_diff_receipts_and_status_tools():
    mod = load_plugin_module()
    engine = mod.ContextGovernorEngine()
    tool_names = {schema["name"] for schema in engine.get_tool_schemas()}

    assert "context_governor_search" in tool_names
    assert "context_governor_expand" in tool_names
    assert "context_governor_diff" in tool_names
    assert "context_governor_receipts" in tool_names
    assert "context_governor_status" in tool_names


def test_soft_warn_retry_failure_fails_open_to_original_messages(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_BUDGET_MODE", "hard_cascade")
    engine = mod.ContextGovernorEngine()
    request = {"policy": {"budget_mode": "hard_cascade"}}
    messages = [{"role": "user", "content": "latest"}]
    calls = []

    def always_fail(args, request):
        calls.append(request["policy"]["budget_mode"])
        raise mod.subprocess.CalledProcessError(1, ["context-governor", "compact"], stderr="still failing")

    monkeypatch.setattr(engine, "_run_context_governor", always_fail)

    response = engine._run_compact_with_fallback(request, messages)

    assert calls == ["hard_cascade", "soft_warn"]
    assert response["receipt"]["receipt_id"].startswith("ctxr_fail_open_")
    assert response["compacted_messages"] == [{"id": "m0", "role": "user", "content": "latest"}]
    assert any("still failing" in warning for warning in response["receipt"]["warnings"])


def test_store_failure_does_not_abort_successful_compaction(monkeypatch):
    mod = load_plugin_module()
    engine = mod.ContextGovernorEngine()
    monkeypatch.setattr(engine, "_ensure_binary", lambda: None)
    monkeypatch.setattr(engine, "_archive_response_to_semantic_memory", lambda response: None)
    monkeypatch.setattr(engine, "_persist_response", lambda response: (_ for _ in ()).throw(mod.subprocess.CalledProcessError(1, ["store"], stderr="disk full")))
    monkeypatch.setattr(engine, "_run_context_governor", lambda args, request: {
        "receipt": {"receipt_id": "ctxr_ok", "compacted_approx_tokens": 42},
        "compacted_messages": [{"role": "assistant", "content": "summary"}],
        "allocation_plan": {"items": [], "archived_item_ids": []},
        "exact_store": [],
    })

    result = engine.compress([
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "latest"},
    ], current_tokens=20000)

    assert result[-1] == {"role": "user", "content": "latest"}
    assert engine.compression_count == 1
    assert engine.last_prompt_tokens == 42


def test_persisted_receipt_matches_final_returned_messages_after_latest_user_guard(monkeypatch):
    mod = load_plugin_module()
    engine = mod.ContextGovernorEngine()
    monkeypatch.setattr(engine, "_ensure_binary", lambda: None)
    monkeypatch.setattr(engine, "_archive_response_to_semantic_memory", lambda response: None)
    captured = []
    monkeypatch.setattr(engine, "_persist_response", lambda response: captured.append(copy.deepcopy(response)))
    monkeypatch.setattr(engine, "_run_context_governor", lambda args, request: {
        "receipt": {"receipt_id": "ctxr_missing_latest", "compacted_approx_tokens": 42},
        "compacted_messages": [{"id": "m0", "role": "assistant", "content": "summary only"}],
        "allocation_plan": {"items": [], "archived_item_ids": []},
        "exact_store": [],
    })

    returned = engine.compress([
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "latest"},
    ], current_tokens=20000)

    assert returned[-1] == {"role": "user", "content": "latest"}
    assert captured[0]["compacted_messages"][-1]["role"] == "user"
    assert captured[0]["compacted_messages"][-1]["content"] == "latest"


def test_python_semantic_archive_clears_no_memory_sink_warning(monkeypatch):
    mod = load_plugin_module()
    monkeypatch.setenv("HERMES_CONTEXT_GOVERNOR_SEMANTIC_MEMORY_ENABLED", "true")
    engine = mod.ContextGovernorEngine()
    response = {
        "receipt": {
            "receipt_id": "ctxr_warn",
            "semantic_memory_fact_ids": [],
            "warnings": ["archive requested but no memory sink was supplied; semantic-memory IDs are intentionally empty"],
        },
        "allocation_plan": {
            "archived_item_ids": ["item-1"],
            "items": [{"item_id": "item-1", "item_type": "Decision", "authority_class": "DurableMemoryCandidate", "content_kind": "Text"}],
        },
        "exact_store": [{"item_id": "item-1", "content": "Decision: persist", "content_blake3": "b3warn"}],
    }

    def fake_post_json(path, payload, timeout=10):
        if path == "/search":
            return {"ok": True, "results": []}
        if path == "/add":
            return {"ok": True, "fact_id": "fact-warn"}
        raise AssertionError(path)

    monkeypatch.setattr(engine, "_semantic_memory_post_json", fake_post_json)

    engine._archive_response_to_semantic_memory(response)

    assert response["receipt"]["semantic_memory_fact_ids"] == ["fact-warn"]
    assert not any("no memory sink" in warning for warning in response["receipt"]["warnings"])


def test_fail_open_receipt_ids_are_unique():
    mod = load_plugin_module()
    engine = mod.ContextGovernorEngine()
    request = {"policy": {"budget_mode": "hard_cascade"}}
    messages = [{"role": "user", "content": "latest"}]

    first = engine._original_messages_response(request, messages, "first")
    second = engine._original_messages_response(request, messages, "second")

    assert first["receipt"]["receipt_id"] != second["receipt"]["receipt_id"]

