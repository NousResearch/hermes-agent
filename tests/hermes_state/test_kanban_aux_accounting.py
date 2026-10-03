"""Kanban board aux calls (specify/decompose) attribute their LLM usage.

The board lane runs outside any agent turn (CLI command, dashboard route,
gateway dispatcher tick), so without an explicitly published accounting
context the aux-client chokepoint drops the usage row and per-task cost
views cannot see it.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from hermes_state import SessionDB


def _fake_response(model="board-model", prompt=800, completion=90):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
    )


def _usage_rows(db, session_id):
    with db._lock:
        rows = db._conn.execute(
            "SELECT task, model, input_tokens, output_tokens, api_call_count"
            " FROM session_model_usage WHERE session_id = ? ORDER BY task",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


class TestKanbanAuxAccounting:
    def test_call_aux_publishes_accounting_context(self, tmp_path, monkeypatch):
        from hermes_cli import kanban_specify

        def fake_call_llm(*, task, **kwargs):
            # The recorder runs inside the call: the context must already be bound.
            from agent.aux_accounting import record_aux_usage

            record_aux_usage(_fake_response(), task)
            return _fake_response()

        monkeypatch.setattr(
            kanban_specify, "_load_triage_task", lambda tid: (object(), "")
        )
        import agent.auxiliary_client as aux_client

        with mock.patch.object(
            aux_client, "call_llm", side_effect=fake_call_llm
        ) as call:
            raw, reason = kanban_specify._call_aux(
                "specify",
                "T-1",
                aux_task="triage_specifier",
                system="s",
                user="u",
                max_tokens=100,
                timeout=30,
            )
        assert raw == "{}"
        call.assert_called_once()
        # The recorder wrote through the ambient context onto the isolated home's
        # state.db (the hermetic conftest repins _default_db_path per test).
        from hermes_state import _default_db_path

        db = SessionDB(_default_db_path())
        try:
            rows = _usage_rows(db, kanban_specify._kanban_aux_session_id())
        finally:
            db.close()
        assert rows and rows[0]["task"] == "triage_specifier"
        assert rows[0]["input_tokens"] == 800

    def test_accounting_context_reset_after_call(self, tmp_path, monkeypatch):
        from agent.aux_accounting import get_accounting_context
        from hermes_cli import kanban_specify

        def fake_call_llm(*, task, **kwargs):
            assert get_accounting_context() is not None
            return _fake_response()

        import agent.auxiliary_client as aux_client

        with mock.patch.object(aux_client, "call_llm", side_effect=fake_call_llm):
            kanban_specify._call_aux(
                "specify",
                "T-2",
                aux_task="kanban_decomposer",
                system="s",
                user="u",
                max_tokens=100,
                timeout=30,
            )
        assert get_accounting_context() is None

    def test_session_id_is_stable_per_process(self):
        from hermes_cli import kanban_specify

        assert (
            kanban_specify._kanban_aux_session_id()
            == kanban_specify._kanban_aux_session_id()
        )
        assert kanban_specify._kanban_aux_session_id().startswith("kanban-aux-")
