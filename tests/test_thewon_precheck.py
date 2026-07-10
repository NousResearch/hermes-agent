from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from agent import thewon_precheck as tp


def test_should_run_thewon_precheck_detects_thewon_terms():
    assert tp.should_run_thewon_precheck("TheWon 에이전트 고도화 절차 만들자")
    assert tp.should_run_thewon_precheck("LLM Wiki를 실제 적용해")
    assert not tp.should_run_thewon_precheck("일반적인 파이썬 문법 질문")


def test_apply_precheck_response_block_prefixes_response():
    bundle = tp.TheWonPrecheckBundle(
        required=True,
        query="TheWon 에이전트",
        kh_summary="[harness] 1건",
        wiki_source="wiki",
        wiki_reason="hit",
        wiki_hits=[{"title": "Universal LLM Wiki Usage", "confidence": "high"}],
        read_files=["/tmp/universal-llm-wiki-usage.md"],
        kh_query_executed=True,
        wiki_route_executed=True,
        relevant_sources_read=True,
    )

    response = tp.apply_precheck_response_block("본문", bundle)

    assert response is not None
    assert response.startswith("[Precheck]")
    assert "KH: [harness] 1건" in response
    assert "Universal LLM Wiki Usage (high)" in response
    assert response.endswith("본문")


def test_run_thewon_precheck_uses_kh_and_wiki(monkeypatch, tmp_path):
    monkeypatch.setattr(tp, "_ensure_thewon_import_paths", lambda: True)

    wiki_file = tmp_path / "wiki.md"
    wiki_file.write_text("# Wiki\ncontent", encoding="utf-8")

    class FakeKnowledgeHub:
        def __init__(self, agent_code=None):
            self.agent_code = agent_code

        def smart_ask_formatted(self, query, top_k_per_source=3, min_sources=2):
            return "[harness] 1건"

    class Hit:
        title = "Universal LLM Wiki Usage"
        confidence = "high"
        score = 0.42
        path = str(wiki_file)

    class FakeWikiRouter:
        def route(self, query, threshold=0.2):
            return {"source": "wiki", "reason": "fake", "hits": [Hit()]}

    system_mod = types.ModuleType("System")
    shared_mod = types.ModuleType("System.shared")
    kh_mod = types.ModuleType("System.shared.knowledge_hub")
    setattr(kh_mod, "KnowledgeHub", FakeKnowledgeHub)
    wiki_mod = types.ModuleType("System.shared.wiki_router")
    setattr(wiki_mod, "WikiRouter", FakeWikiRouter)

    monkeypatch.setitem(sys.modules, "System", system_mod)
    monkeypatch.setitem(sys.modules, "System.shared", shared_mod)
    monkeypatch.setitem(sys.modules, "System.shared.knowledge_hub", kh_mod)
    monkeypatch.setitem(sys.modules, "System.shared.wiki_router", wiki_mod)

    bundle = tp.run_thewon_precheck("TheWon 에이전트 절차", agent_code="Hermes")

    assert bundle.required is True
    assert bundle.kh_summary == "[harness] 1건"
    assert bundle.wiki_source == "wiki"
    assert bundle.wiki_hits[0]["title"] == "Universal LLM Wiki Usage"
    assert bundle.read_files == [str(wiki_file)]
    assert bundle.kh_query_executed is True
    assert bundle.wiki_route_executed is True
    assert bundle.relevant_sources_read is True
    assert bundle.ok is True
    assert bundle.errors == []


def test_format_precheck_context_is_ephemeral_and_compact():
    bundle = tp.TheWonPrecheckBundle(
        required=True,
        query="TheWon",
        kh_summary="[harness] 1건",
        wiki_source="wiki",
        wiki_reason="fake",
        wiki_hits=[{"title": "A", "confidence": "high"}],
        read_files=["/tmp/a.md"],
        kh_query_executed=True,
        wiki_route_executed=True,
        relevant_sources_read=True,
    )

    context = tp.format_precheck_context(bundle)

    assert context.startswith("[TheWon KH+LLM Wiki precheck]")
    assert '"kh": "[harness] 1건"' in context
    assert '"wiki_source": "wiki"' in context
    assert '"precheck_pass": true' in context


def test_apply_precheck_response_block_blocks_when_required_object_incomplete():
    bundle = tp.TheWonPrecheckBundle(
        required=True,
        query="TheWon",
        kh_summary="",
        wiki_source="",
        errors=["KnowledgeHub failed"],
    )

    response = tp.apply_precheck_response_block("모델이 만든 일반 답변", bundle)

    assert response is not None
    assert "Gate: FAIL" in response
    assert "runtime hard gate blocked" in response
    assert "모델이 만든 일반 답변" not in response


def test_codex_app_server_path_applies_precheck_block(monkeypatch):
    from agent import codex_runtime

    bundle = tp.TheWonPrecheckBundle(
        required=True,
        query="TheWon",
        kh_summary="[harness] 1건",
        wiki_source="wiki",
        wiki_hits=[{"title": "A", "confidence": "high"}],
        read_files=["/tmp/a.md"],
        kh_query_executed=True,
        wiki_route_executed=True,
        relevant_sources_read=True,
    )

    class FakeSession:
        def run_turn(self, user_input):
            assert "TheWon" in user_input
            return SimpleNamespace(
                projected_messages=[],
                tool_iterations=0,
                final_text="본문",
                interrupted=False,
                error=None,
                thread_id="thread-1",
                turn_id="turn-1",
                should_retire=False,
            )

    agent = SimpleNamespace(
        _codex_session=FakeSession(),
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _thewon_precheck_bundle=bundle,
        _sync_external_memory_for_turn=lambda **kwargs: None,
        _spawn_background_review=lambda **kwargs: None,
    )
    monkeypatch.setattr(codex_runtime, "_record_codex_app_server_usage", lambda agent, turn: {})

    result = codex_runtime.run_codex_app_server_turn(
        agent,
        user_message="TheWon 요청",
        original_user_message="TheWon 요청",
        messages=[],
        effective_task_id="task-1",
    )

    assert result["final_response"].startswith("[Precheck]")
    assert "Gate: PASS" in result["final_response"]
    assert result["final_response"].endswith("본문")
