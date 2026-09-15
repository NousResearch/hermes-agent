"""Behavior-contract tests for the Rule 2 task_mode heuristic classifier.

Every assertion is a *relationship* between an input (title/body) and the
resulting mode, never a frozen snapshot of the whole function's behavior on
one exact string. See AGENTS.md "Behavior contracts over snapshots" /
"Don't write change-detector tests".
"""

from __future__ import annotations

from hermes_cli.kanban_task_mode import classify_task_mode


def test_return_value_is_always_one_of_the_three_valid_modes():
    for title, body in [
        ("fix typo in readme", "s/hte/the/"),
        ("Why does the dispatcher retry forever?", ""),
        ("Redesign the auth subsystem", "Touches auth.py, session.py and db.py."),
        ("", None),
    ]:
        mode = classify_task_mode(title, body)
        assert mode in ("autocomplete", "chat", "agent")


def test_question_phrased_title_classifies_as_chat():
    mode = classify_task_mode("What does the retry breaker do here?", "")
    assert mode == "chat"


def test_explain_request_classifies_as_chat_even_with_a_filename_mentioned():
    # A single file mention alongside an explanation request must not tip
    # this into "agent" — question-phrasing signals win over an incidental
    # filename.
    mode = classify_task_mode(
        "Can you explain why kanban_db.py retries on failure?", "No code change needed.",
    )
    assert mode == "chat"


def test_architecture_keyword_with_multi_file_scope_classifies_as_agent():
    mode = classify_task_mode(
        "Refactor the dispatcher architecture",
        "Spans kanban_db.py, kanban_decompose.py and kanban_db_graph.py; "
        "needs a migration plan and new tests across the module.",
    )
    assert mode == "agent"


def test_two_or_more_distinct_file_mentions_classifies_as_agent_without_keywords():
    mode = classify_task_mode(
        "Wire the new field through",
        "Update kanban_db.py and kanban_decompose.py so the field round-trips.",
    )
    assert mode == "agent"


def test_short_single_file_mechanical_body_classifies_as_autocomplete():
    mode = classify_task_mode("Fix typo in docstring", "Fix the typo in kanban_db.py's docstring.")
    assert mode == "autocomplete"


def test_short_body_with_no_file_mention_defaults_to_autocomplete():
    mode = classify_task_mode("Rename the helper", "Rename _foo to _bar.")
    assert mode == "autocomplete"


def test_longer_body_with_no_strong_signal_and_no_multi_file_scope_is_not_agent():
    # A longer, signal-free body isn't a short mechanical edit, but also
    # carries no agent-level scope signal — it must not be classified agent.
    mode = classify_task_mode(
        "Update the changelog",
        "Add an entry summarizing the last release cycle's notable fixes "
        "and mention the contributors who reported the issues we closed.",
    )
    assert mode != "agent"


def test_classification_depends_only_on_title_and_body_content():
    # Same signal-bearing content, different casing/whitespace -> same mode
    # (a content relationship, not a frozen literal value).
    a = classify_task_mode("WHY DOES THIS FAIL?", "")
    b = classify_task_mode("why does this fail?", "")
    assert a == b == "chat"


def test_never_raises_on_missing_or_empty_body():
    # body=None must be handled the same as body="" — no exception either way.
    mode_none = classify_task_mode("Some task", None)
    mode_empty = classify_task_mode("Some task", "")
    assert mode_none in ("autocomplete", "chat", "agent")
    assert mode_none == mode_empty
