"""Regression tests for MemoryMunch soft-wall no-bleed behavior."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_memorymunch_module():
    plugin_path = Path("/home/alcoo/.hermes/plugins/memorymunch/__init__.py")
    spec = importlib.util.spec_from_file_location("memorymunch_plugin_under_test", plugin_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_soft_wall_caps_outward_rows_and_marks_them_non_identity_promotable():
    mm = _load_memorymunch_module()
    rows = [
        {
            "id": "own-current",
            "provenance_class": "OWN_SCOPE",
            "source": "ACTIVE_SESSION_LEDGER",
            "activation_weight": 0.2,
            "hop_depth": 0,
            "content_preview": "Current session says build async compaction now.",
        }
    ]
    rows.extend(
        {
            "id": f"outward-{idx}",
            "provenance_class": "GRAPH_LINKED_OUTWARD",
            "source": "GRAPH_MEMORY",
            "activation_weight": 0.99,
            "hop_depth": 1,
            "linked_from": "own-current",
            "content_preview": f"Outward memory {idx}: identity/preference claim from another scope.",
        }
        for idx in range(3)
    )

    briefing = mm.format_memorymunch_briefing(rows, scope_entity="current-user")

    assert "OWN_SCOPE:" in briefing
    assert briefing.count("Outward memory") <= 2
    assert "identity_promotable=false" in briefing
    assert "assertion_authority=background_linked_context" in briefing
    assert briefing.index("OWN_SCOPE:") < briefing.index("GRAPH_LINKED_OUTWARD:")


def test_soft_wall_renders_session_lineage_lower_than_current_ledger():
    mm = _load_memorymunch_module()
    rows = [
        {
            "id": "current",
            "provenance_class": "ACTIVE_SESSION_LEDGER_CURRENT",
            "source": "ACTIVE_SESSION_LEDGER",
            "activation_weight": 1.0,
            "hop_depth": 0,
            "content_preview": "Current turn asks for compaction fix.",
        },
        {
            "id": "lineage",
            "provenance_class": "ACTIVE_SESSION_LINEAGE",
            "source": "SESSIONDB_LINEAGE",
            "activation_weight": 0.8,
            "hop_depth": 0,
            "content_preview": "Older parent session context.",
        },
    ]

    briefing = mm.format_memorymunch_briefing(rows, scope_entity="current-user")

    assert "ACTIVE_SESSION_LEDGER_CURRENT:" in briefing
    assert "ACTIVE_SESSION_LINEAGE:" in briefing
    assert briefing.index("ACTIVE_SESSION_LEDGER_CURRENT:") < briefing.index("ACTIVE_SESSION_LINEAGE:")


def test_prefetch_cache_is_query_and_scope_aware(monkeypatch):
    mm = _load_memorymunch_module()
    provider = mm.MemoryMunchProvider()
    provider._session_id = "sid-1"
    provider._scope_entity = "scope-a"
    provider._domain = "general"
    provider._harness = "hermes/cli/primary/default"
    provider._prefetch_cache["sid-1"] = {
        "query": "old query",
        "scope_entity": "scope-a",
        "domain": "general",
        "harness": "hermes/cli/primary/default",
        "context": "STALE_QUERY_CONTEXT",
    }

    def cold_context(query, session_id, *, prefetch_cache="cold"):
        return f"COLD_CONTEXT query={query} cache={prefetch_cache}"

    monkeypatch.setattr(provider, "_compose_prefetch_context", cold_context)

    context = provider.prefetch("new query", session_id="sid-1")

    assert "STALE_QUERY_CONTEXT" not in context
    assert "COLD_CONTEXT query=new query" in context
