from __future__ import annotations

import json
from typing import Any


def test_search_fiscal_sources_blocks_unverified_precise_cgi_article(monkeypatch):
    from tools import hermes_governance_tools as mod

    events: list[dict[str, Any]] = []

    def _fake_get_code_article_version_tool(arguments: dict[str, Any]) -> str:
        return json.dumps(
            {
                "ok": True,
                "tool_name": "get_code_article_version",
                "result": {
                    "article_ref": "39",
                    "code_name": "Code général des impôts",
                    "fact_date": "2024-01-15",
                    "version_found": False,
                    "in_force_on_fact_date": False,
                    "coverage_status": "not_verified",
                    "vigueur_start": None,
                    "vigueur_end": None,
                    "text": None,
                    "source_url": None,
                    "source_ids": [],
                    "selection_mode": "exact_num_article+vigueur_window",
                },
            },
            ensure_ascii=False,
        )

    def _fake_log_audit_event_tool(**kwargs) -> str:
        events.append(kwargs)
        return json.dumps({"ok": True}, ensure_ascii=False)

    def _unexpected_fallback(*args, **kwargs) -> str:
        raise AssertionError("fallback search must not run for a precise CGI article with fact_date")

    monkeypatch.setattr(mod, "get_code_article_version_tool", _fake_get_code_article_version_tool)
    monkeypatch.setattr(mod, "log_audit_event_tool", _fake_log_audit_event_tool)
    monkeypatch.setattr(mod, "search_legal_sources_tool", _unexpected_fallback)

    raw = mod.search_fiscal_sources_tool(
        query="article 39 CGI pénalités non déductibles",
        source_type="cgi",
        fact_date="2024-01-15",
    )
    payload = json.loads(raw)
    result = payload["result"]

    assert payload["ok"] is True
    assert result["success"] is False
    assert result["coverage_status"] == "not_verified"
    assert result["blocking_reason"] == "code_article_version_not_verified"
    assert result["article_ref"] == "39"
    assert result["fact_date"] == "2024-01-15"
    assert result["source_links"] == []
    assert result["sources"] == []
    assert events, "audit trail must capture version guard failure"
    assert events[0]["event_type"] == "source_verification"


def test_search_fiscal_sources_returns_verified_precise_cgi_article(monkeypatch):
    from tools import hermes_governance_tools as mod

    def _fake_get_code_article_version_tool(arguments: dict[str, Any]) -> str:
        return json.dumps(
            {
                "ok": True,
                "tool_name": "get_code_article_version",
                "result": {
                    "article_ref": "39",
                    "code_name": "Code général des impôts",
                    "fact_date": "2026-03-01",
                    "version_found": True,
                    "in_force_on_fact_date": True,
                    "coverage_status": "verified",
                    "vigueur_start": "21/02/2026",
                    "vigueur_end": "01/09/2026",
                    "text": "ARTICLE 39 VERSION VERIFIED",
                    "source_url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006307555",
                    "source_ids": ["LEGIARTI000006307555"],
                    "selection_mode": "exact_num_article+vigueur_window",
                },
            },
            ensure_ascii=False,
        )

    def _unexpected_fallback(*args, **kwargs) -> str:
        raise AssertionError("fallback search must not run for a precise CGI article with fact_date")

    monkeypatch.setattr(mod, "get_code_article_version_tool", _fake_get_code_article_version_tool)
    monkeypatch.setattr(mod, "search_legal_sources_tool", _unexpected_fallback)

    raw = mod.search_fiscal_sources_tool(
        query="article 39 CGI",
        source_type="cgi",
        fact_date="2026-03-01",
    )
    payload = json.loads(raw)
    result = payload["result"]

    assert payload["ok"] is True
    assert result["success"] is True
    assert result["coverage_status"] == "verified"
    assert result["primary_sources_verified"] is True
    assert result["article_ref"] == "39"
    assert result["fact_date"] == "2026-03-01"
    assert result["result_text"] == "ARTICLE 39 VERSION VERIFIED"
    assert result["source_links"] == [
        "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006307555"
    ]
    assert result["sources"] == [
        {
            "type": "code_article_version",
            "code_name": "Code général des impôts",
            "article_ref": "39",
            "fact_date": "2026-03-01",
            "source_url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006307555",
        }
    ]


def test_search_fiscal_sources_non_article_query_keeps_fallback_path(monkeypatch):
    from tools import hermes_governance_tools as mod

    calls: list[dict[str, Any]] = []

    def _fake_search_legal_sources_tool(**kwargs) -> str:
        calls.append(kwargs)
        return json.dumps(
            {
                "success": True,
                "corpus": "code",
                "tool_name": "mcp_openlegi_rechercher_code",
                "result_text": "fallback search result",
                "source_links": ["https://example.test/source"],
                "source_count": 1,
            },
            ensure_ascii=False,
        )

    def _unexpected_guard(arguments: dict[str, Any]) -> str:
        raise AssertionError("version guard must not run when no precise article is present")

    monkeypatch.setattr(mod, "search_legal_sources_tool", _fake_search_legal_sources_tool)
    monkeypatch.setattr(mod, "get_code_article_version_tool", _unexpected_guard)

    raw = mod.search_fiscal_sources_tool(
        query="amende non déductible IS",
        source_type="cgi",
        fact_date="2024-01-15",
    )
    payload = json.loads(raw)
    result = payload.get("result", payload)

    assert result["success"] is True
    assert payload["tool_name"] == "search_fiscal_sources"
    assert result.get("tool_name") in {"search_fiscal_sources", "mcp_openlegi_rechercher_code"}
    assert calls, "fallback search must remain active for non-article fiscal queries"
