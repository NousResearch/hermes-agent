import os

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


def test_single_query_exit_code_success():
    assert cli._single_query_exit_code({"completed": True, "final_response": "ok"}) == 0
    assert cli._single_query_exit_code(None) == 0


def test_single_query_exit_code_failed_non_kanban_is_generic(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    result = {"failed": True, "failure_reason": "rate_limit", "error": "quota"}
    assert cli._single_query_exit_code(result) == 1


def test_single_query_exit_code_failed_kanban_rate_limit_is_tempfail(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    result = {"failed": True, "failure_reason": "rate_limit", "error": "429"}
    assert cli._single_query_exit_code(result) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_single_query_exit_code_failed_kanban_billing_is_tempfail(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    result = {"failed": True, "failure_reason": "billing", "error": "credits"}
    assert cli._single_query_exit_code(result) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_single_query_exit_code_failed_kanban_model_error_is_nonzero_not_protocol_ok(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    result = {"failed": True, "failure_reason": "model_not_found", "error": "404"}
    assert cli._single_query_exit_code(result) == 1
