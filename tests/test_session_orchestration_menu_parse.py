"""Tests for session_orchestration.menu_parse (answerable needs-input).

Ported from the so-MCP orchestrator tests (test_hermes_so_mcp.py) plus the
new ``extract()`` tuple contract.
"""

import json

from session_orchestration.menu_parse import (
    extract,
    extract_menu_options,
    extract_needs_input_context,
    resolve_menu_answer,
)

_MENU_PANE = (
    " Proposal 1: add structured degraded-consult provenance. Decision?\n"
    "\n"
    "────────────────────────────────────────────────────────────────\n"
    "│ Accept (Recommended)                                          │\n"
    "│    Apply edits to the agent instructions now.                 │\n"
    "│ Refine                                                        │\n"
    "│    Ask a follow-up question before deciding.                  │\n"
    "│ Defer                                                         │\n"
    "│    Keep it in the retro, no edits now.                        │\n"
    "│ Reject                                                        │\n"
    "│    Drop the proposal entirely.                                │\n"
    "│ Other (type your own)                                         │\n"
    "────────────────────────────────────────────────────────────────\n"
    " up/down navigate  enter select  esc cancel\n"
    "────────────────────────────────────────────────────────────────"
)


def test_extract_menu_options_returns_ordered_labels():
    assert extract_menu_options(_MENU_PANE) == [
        "Accept (Recommended)",
        "Refine",
        "Defer",
        "Reject",
        "Other (type your own)",
    ]


def test_extract_menu_options_skips_indented_descriptions():
    # Indented description rows must never leak into the option list.
    for desc in (
        "Apply edits to the agent instructions now.",
        "Keep it in the retro, no edits now.",
        "Drop the proposal entirely.",
    ):
        assert desc not in extract_menu_options(_MENU_PANE)


def test_extract_menu_options_empty_for_bare_prompt():
    assert extract_menu_options("Proceed?\n❯") == []


def test_extract_needs_input_context_cleans_menu():
    out = extract_needs_input_context(_MENU_PANE)
    assert "Proposal 1: add structured degraded-consult provenance. Decision?" in out
    assert "Accept (Recommended)" in out
    assert "Apply edits to the agent instructions now." in out
    # Box-drawing rules and nav footer are stripped.
    assert "─" not in out
    assert "│" not in out
    assert "enter select" not in out
    assert "esc cancel" not in out


def test_extract_needs_input_context_falls_back_on_unknown_shape():
    raw = "some freeform prompt asking a question ❯"
    assert extract_needs_input_context(raw) == raw.strip()


def test_extract_menu_tuple():
    question, options, is_menu = extract(_MENU_PANE)
    assert is_menu is True
    assert options == [
        "Accept (Recommended)",
        "Refine",
        "Defer",
        "Reject",
        "Other (type your own)",
    ]
    # The question prose survives; option labels are removed from it so they
    # aren't duplicated when a caller renders the numbered list.
    assert "Proposal 1" in question
    assert "Accept (Recommended)" not in question


def test_extract_free_form_prompt_is_not_a_menu():
    question, options, is_menu = extract("Which branch should I use? ❯")
    assert is_menu is False
    assert options == []
    assert "Which branch should I use?" in question


def test_extract_empty_pane():
    question, options, is_menu = extract("")
    assert is_menu is False
    assert options == []
    assert question == ""


# Real omp v16.2 `ask` menu capture (2026-07-01 live smoke). Exercises: two
# stacked boxes (a transient "Ask" render + the live footer-anchored menu),
# Nerd-Font glyph prefixes on labels, a 4-space-indented description row, and
# a 2-space-indented final option. Footer-anchored parse must pick the live
# menu, strip glyphs, keep all four options, and pull the real question.
_OMP_V16_ASK_PANE = (
    "──────────────────────────────────────────────\n"
    "  Update Available\n"
    "  New version 16.2.13 is available. Run: omp update\n"
    "──────────────────────────────────────────────\n"
    "\n"
    "  Clarifying task requirements\n"
    "\n"
    "  I realize that I need to comply with the user's request by asking...\n"
    "\n"
    "╭─── Ask 1 questions ──────────────────────────╮\n"
    "├─── [task_goal] · options:3 ──────────────────┤\n"
    "│  What task should I work on after this clarification?          │\n"
    "│   Code change                                                  │\n"
    "│    ↳ Modify implementation, tests, or tooling in this repo.    │\n"
    "│   Investigation                                                │\n"
    "│    ↳ Read-only debugging, audit, or explanation.              │\n"
    "╰──────────────────────────────────────────────╯\n"
    "\n"
    "  ⠇ Clarifying intent ⟨esc⟩\n"
    "\n"
    "──────────────────────────────────────────────\n"
    "  What task should I work on after this clarification?\n"
    "──────────────────────────────────────────────\n"
    "│  Code change (Recommended)                                 │\n"
    "│    Modify implementation, tests, or tooling in this repo.      │\n"
    "│  Investigation                                             │\n"
    "│    Read-only debugging, audit, or explanation.                 │\n"
    "│  Plan/review                                               │\n"
    "│    Create or assess a plan before implementation.              │\n"
    "│  Other (type your own)                                         │\n"
    "──────────────────────────────────────────────\n"
    " up/down navigate  enter select  esc cancel\n"
    "──────────────────────────────────────────────"
)


_OMP_FREE_FORM_PANE = (
    "──────────────────────────────────\n"
    " Update Available\n"
    " New version 16.2.13 is available. Run: omp update\n"
    "──────────────────────────────────\n"
    "\n"
    " Clarifying user needs\n"
    " I need to ask a clarifying question and then wait for the user's response...\n"
    "\n"
    " What task do you want me to work on after this clarification?\n"
    "\n"
    " Hermes nudge: your session appears to have stalled. If you are waiting...\n"
    "\n"
    " Still waiting on the clarifying answer: what task should I work on?\n"
    "\n"
    "╭──  GPT-5.5 ·  med   ~/dev/z-harness   main ──╮\n"
    "╰─                                            ─╯"
)


def test_extract_free_form_omp_question_isolated():
    question, options, is_menu = extract(_OMP_FREE_FORM_PANE)
    assert is_menu is False
    assert options == []
    # The actual question is isolated from reasoning / banner / nudge / chrome.
    assert question == "Still waiting on the clarifying answer: what task should I work on?"
    assert "omp update" not in question
    assert "Hermes nudge" not in question
    assert "GPT-5.5" not in question


def test_extract_real_omp_v16_ask_menu():
    question, options, is_menu = extract(_OMP_V16_ASK_PANE)
    assert is_menu is True
    assert options == [
        "Code change (Recommended)",
        "Investigation",
        "Plan/review",
        "Other (type your own)",
    ]
    assert question == "What task should I work on after this clarification?"
    # Nerd-Font glyphs must not leak into labels.
    assert "" not in "".join(options)
    assert "" not in "".join(options)


# --- resolve_menu_answer -----------------------------------------------------

def _menu_row(**over):
    row = {
        "state": "WAITING_USER",
        "last_input_kind": "menu",
        "last_options": json.dumps(["Accept", "Defer", "Reject"]),
    }
    row.update(over)
    return row


def test_resolve_menu_answer_valid_number_escapes_and_names_label():
    drive_text, pre_keys = resolve_menu_answer(_menu_row(), "2")
    assert pre_keys == ["Escape"]
    assert "option 2" in drive_text
    assert "Defer" in drive_text


def test_resolve_menu_answer_trailing_period_ok():
    drive_text, pre_keys = resolve_menu_answer(_menu_row(), "1.")
    assert pre_keys == ["Escape"]
    assert "Accept" in drive_text


def test_resolve_menu_answer_out_of_range_escapes_as_free_text():
    # A menu is up; an out-of-range number still must Escape the menu (else the
    # spinner keeps the pane "busy") and is treated as the user's typed answer.
    drive_text, pre_keys = resolve_menu_answer(_menu_row(), "9")
    assert pre_keys == ["Escape"]
    assert "9" in drive_text
    assert "typed answer" in drive_text


def test_resolve_menu_answer_non_numeric_escapes_as_free_text():
    # Free text on a menu = the "Other (type your own)" case: Escape the menu,
    # then paste the answer.
    drive_text, pre_keys = resolve_menu_answer(_menu_row(), "let's defer this")
    assert pre_keys == ["Escape"]
    assert "let's defer this" in drive_text


def test_resolve_menu_answer_free_form_prompt_untouched():
    row = _menu_row(last_input_kind="prompt", last_options=json.dumps([]))
    drive_text, pre_keys = resolve_menu_answer(row, "2")
    assert pre_keys is None
    assert drive_text == "2"


def test_resolve_menu_answer_not_waiting_untouched():
    row = _menu_row(state="RUNNING")
    drive_text, pre_keys = resolve_menu_answer(row, "2")
    assert pre_keys is None
    assert drive_text == "2"


def test_resolve_menu_answer_bad_options_json_falls_through():
    row = _menu_row(last_options="not-json")
    drive_text, pre_keys = resolve_menu_answer(row, "1")
    assert pre_keys is None
    assert drive_text == "1"
