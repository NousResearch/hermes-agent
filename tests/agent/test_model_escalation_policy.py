"""Policy tests for automatic initial model routing.

These tests exercise the pure selector logic only; they do not make live LLM
calls.
"""

from __future__ import annotations

from agent.model_escalation_policy import (
    SelectionDecision,
    count_valid_file_references,
    select_initial_model,
)


def test_default_routine_task_stays_on_mini():
    d = select_initial_model("Write a short summary of this paragraph.")
    assert isinstance(d, SelectionDecision)
    assert d.selected_model == "gpt-5.4-mini"
    assert d.reason == "default_mini"


def test_high_risk_intent_goes_to_gpt54():
    d = select_initial_model("This is a production security rollback for credentials.")
    assert d.selected_model == "gpt-5.4"
    assert d.reason == "hard_complex"


def test_three_file_references_go_to_gpt54():
    d = select_initial_model("Fix these files: src/app.py, src/api.py, src/db.py.")
    assert d.selected_model == "gpt-5.4"
    assert d.reason == "hard_complex"


def test_single_words_do_not_force_gpt54():
    for text in ["API", "SQL", "error", "test", "refactor", "integration"]:
        d = select_initial_model(text)
        assert d.selected_model == "gpt-5.4-mini", text
        assert d.reason == "default_mini", text


def test_invalid_pseudo_file_tokens_do_not_increase_file_count():
    text = "Pseudo files: version.1 release.2 build.3 just.words here"
    assert count_valid_file_references(text) == 0
    d = select_initial_model(text)
    assert d.selected_model == "gpt-5.4-mini"
    assert d.reason == "default_mini"


def test_reason_matches_selected_model_for_architecture_trigger():
    d = select_initial_model(
        "Need architecture guidance and multi-file refactor across src/app.py, src/api.py, src/db.py"
    )
    assert d.selected_model == "gpt-5.4"
    assert d.reason == "hard_complex"
