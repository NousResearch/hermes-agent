"""Fork-vendored LCM context-engine loader smoke."""

from __future__ import annotations

from pathlib import Path

from agent.context_engine import ContextEngine
from plugins.context_engine import load_context_engine


def _close_engine(engine) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()


def test_load_context_engine_lcm_returns_context_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    engine = load_context_engine("lcm")
    try:
        assert engine is not None
        assert isinstance(engine, ContextEngine)
        assert engine.name == "lcm"
        assert not getattr(type(engine), "__abstractmethods__", set())

        for attr in (
            "last_prompt_tokens",
            "last_completion_tokens",
            "last_total_tokens",
            "threshold_tokens",
            "context_length",
            "compression_count",
        ):
            assert hasattr(engine, attr)
    finally:
        if engine is not None:
            _close_engine(engine)


def test_lcm_tool_schemas_expose_recall_and_status_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    engine = load_context_engine("lcm")
    try:
        assert engine is not None
        schemas = engine.get_tool_schemas()
        functions = [schema.get("function", schema) for schema in schemas]
        names = {schema.get("name") for schema in functions}
        assert {
            "lcm_grep",
            "lcm_load_session",
            "lcm_describe",
            "lcm_expand",
            "lcm_expand_query",
            "lcm_status",
            "lcm_doctor",
        } <= names
        for schema, function_schema in zip(schemas, functions):
            assert schema.get("type") == "function"
            assert function_schema.get("name")
            assert function_schema.get("description")
            assert function_schema.get("parameters", {}).get("type") == "object"
    finally:
        if engine is not None:
            _close_engine(engine)


def test_lcm_provenance_records_required_metadata():
    provenance = Path("plugins/context_engine/lcm/VENDORED_FROM.txt").read_text(encoding="utf-8")

    required_fragments = [
        "github.com/stephenschoettler/hermes-lcm",
        "03b74f84440be99164ce3e2cd929917bc9550bfe",
        "v0.16.2",
        "Ingest-audit verdict: PASS",
        "License: cleared for private fork use",
        "PRD-6 I-10 upstream security-drift metadata",
        "last_upstream_security_check: 2026-06-16",
        "checked_by: Apollo",
        "checked_upstream_head: a744da693febf689f78fcb24ade70f04e1eb3e3e",
        "upstream_drift_since_source: 2 commits",
        "next_check_due: 2026-09-16",
    ]
    for fragment in required_fragments:
        assert fragment in provenance
