"""Actual-route projection for bounded kanban retry provenance (worker session usage ledger).

The projection must come from the worker's own main-loop usage rows, never from the
configured primary, and must fail closed (unknown) rather than guess.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from hermes_cli import cli_single_query as single_query
from hermes_state import SessionDB

_TOOLS = [{"type": "function", "function": {"name": "safe_tool", "parameters": {"type": "object"}}}]


def _schema_sha256(tools) -> str:
    return hashlib.sha256(
        json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_main_loop_route_pairs_exclude_auxiliary_and_configured_rows(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        session_db.create_session("worker-session-actual", source="cli", model="configured-primary")
        session_db.update_token_counts(
            "worker-session-actual", input_tokens=10, output_tokens=1,
            model="fallback-model", billing_provider="fallback-provider", api_call_count=1,
        )
        session_db.record_auxiliary_usage(
            "worker-session-actual", "vision", input_tokens=4,
            model="vision-model", billing_provider="vision-provider",
        )

        assert session_db.main_loop_route_pairs("worker-session-actual") == [
            {"provider": "fallback-provider", "model": "fallback-model"},
        ]
    finally:
        session_db.close()


def test_main_loop_route_pairs_fail_closed_on_a_credential_shaped_identifier(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        session_db.create_session("worker-session-unsafe", source="cli", model="configured-primary")
        session_db.update_token_counts(
            "worker-session-unsafe", input_tokens=10, output_tokens=1,
            model="https://credential.invalid/model", billing_provider="fallback-provider",
            api_call_count=1,
        )
        stored = session_db._read_all(
            "SELECT model FROM session_model_usage WHERE session_id = ? AND task = ''",
            ("worker-session-unsafe",),
        )
        assert [row["model"] for row in stored] == ["https://credential.invalid/model"]

        assert session_db.main_loop_route_pairs("worker-session-unsafe") == []
    finally:
        session_db.close()


def test_cli_projection_uses_worker_session_routes_and_never_the_configured_route(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        session_db.create_session("worker-session-actual", source="cli", model="configured-primary")
        session_db.update_token_counts(
            "worker-session-actual", input_tokens=10, output_tokens=1,
            model="fallback-model", billing_provider="fallback-provider", api_call_count=1,
        )
        cli = SimpleNamespace(
            session_id="originating-session",
            model="configured-primary",
            agent=SimpleNamespace(
                session_id="worker-session-actual", tools=_TOOLS, _session_db=session_db,
                model="configured-primary", provider="configured-provider",
            ),
        )

        provenance = single_query._build_kanban_retry_provenance(cli)
    finally:
        session_db.close()

    assert provenance["worker_session_id"] == "worker-session-actual"
    assert provenance["actual_routes"] == [{"provider": "fallback-provider", "model": "fallback-model"}]
    assert "configured-primary" not in json.dumps(provenance)
    assert "configured-provider" not in json.dumps(provenance)
    assert provenance["schema_sha256"] == _schema_sha256(_TOOLS)
    assert "schema_sha256" not in provenance["unknown_fields"]
    assert "source_generation_id" in provenance["unknown_fields"]


def test_cli_projection_records_routes_unknown_without_an_actual_main_loop_row(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    try:
        session_db.create_session("worker-session-empty", source="cli", model="configured-primary")
        cli = SimpleNamespace(
            session_id="originating-session",
            agent=SimpleNamespace(
                session_id="worker-session-empty", tools=_TOOLS, _session_db=session_db,
            ),
        )

        provenance = single_query._build_kanban_retry_provenance(cli)
    finally:
        session_db.close()

    assert provenance["actual_routes"] == []
    assert "actual_routes" in provenance["unknown_fields"]
    assert "configured-primary" not in json.dumps(provenance)
