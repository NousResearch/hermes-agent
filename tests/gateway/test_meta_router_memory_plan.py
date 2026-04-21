from starlette.testclient import TestClient

import gateway.meta_router_runtime as runtime
import gateway.meta_router_server as server


def _capture_events(monkeypatch):
    events = []
    monkeypatch.setattr(runtime, "_init_logger", lambda: None)
    monkeypatch.setattr(runtime, "_ALS_LOGGING", True)
    monkeypatch.setattr(runtime, "_log_event_fn", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(runtime, "_make_request_id_fn", lambda: "rid-memory")
    return events


def test_make_route_decision_assigns_history_tools_for_progress_queries(monkeypatch):
    _capture_events(monkeypatch)

    decision = runtime.make_route_decision(
        "What did we decide before and what progress has already been made on this meta-router task?",
        source="gateway",
        surface="telegram",
    )

    assert decision.memory_need == "history"
    assert "memory_search" in decision.required_tools
    assert "memory_get" in decision.optional_tools
    assert decision.max_memory_steps == 2


def test_make_route_decision_assigns_history_and_docs_for_audit_queries(monkeypatch):
    _capture_events(monkeypatch)

    decision = runtime.make_route_decision(
        "Inspect current code, docs, reports, and past progress for remaining blockers in the meta-router runtime.",
        source="cli",
        surface="cli",
    )

    assert decision.memory_need == "history+docs"
    assert decision.required_tools == ["memory_search", "qmd__query"]
    assert decision.optional_tools == ["memory_get", "qmd__get"]
    assert "qmd" in decision.memory_authority


def test_make_route_decision_assigns_wiki_memory_authority_for_vault_queries(monkeypatch):
    _capture_events(monkeypatch)

    decision = runtime.make_route_decision(
        "Update the Obsidian wiki vault pages for this memory design and keep the curated knowledge consistent.",
        source="gateway",
        surface="telegram",
    )

    assert decision.memory_need == "wiki+history"
    assert "memory-wiki" in decision.memory_authority
    assert "qmd__query" in decision.required_tools


def test_classify_response_exposes_memory_plan_and_prepend_text(monkeypatch):
    monkeypatch.setattr(
        server,
        "make_route_decision",
        lambda text, source="api", surface="http", session_id=None: runtime.RouteDecision(
            request_id="rid-123",
            type="research",
            mode="execute",
            directive="[META-ROUTER | research | execute]",
            confidence=0.88,
            primary="som",
            secondary=None,
            budget_multiplier=1.0,
            routing_artifact_version="candidate-0002",
            bypassed=False,
            bypass_reason="",
            memory_need="history+docs",
            memory_authority=["active-memory", "memory-core", "qmd"],
            required_tools=["memory_search", "qmd__query"],
            optional_tools=["memory_get", "qmd__get"],
            skip_tools=["open-brain__recall"],
            max_memory_steps=2,
            memory_policy_version="mr-memory-v1",
        ),
    )
    client = TestClient(server.app)

    resp = client.post("/classify", json={"text": "Find the prior report and continue from there"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["memory_need"] == "history+docs"
    assert data["required_tools"] == ["memory_search", "qmd__query"]
    assert data["prepend_text"].startswith("[META-ROUTER | research | execute]")
    assert "[META-MEMORY]" in data["prepend_text"]


def test_make_route_decision_bypasses_internal_reset_flush_prompt(monkeypatch):
    _capture_events(monkeypatch)

    decision = runtime.make_route_decision(
        "[System: This session is about to be automatically reset due to inactivity or a scheduled daily reset. The conversation context will be cleared after this turn.]",
        source="cli",
        surface="cli",
    )

    assert decision.bypassed is True
    assert decision.bypass_reason == "internal-reset-flush"
