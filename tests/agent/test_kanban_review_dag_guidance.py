"""Prompt contract: Kanban workers run a continuously automated review DAG.

Policy (operator-mandated): the board must keep moving without a human in the
loop. A coding worker no longer parks its own card in ``blocked`` with a
``review-required:`` reason and waits for a person — it creates a *ready*
review card for a real installed reviewer profile and completes its own card.
Review outcomes fan out as more cards (remediation → fresh review), so the
dependency graph is expressed with ``parents``/DAG edges rather than blocked
state.

``blocked`` keeps its narrow, genuine meaning (irretrievable human input, hard
capability/access wall, truly transient failure after retries) — these tests
must not be "fixed" by deleting the block guidance, only by removing the
*mandatory review block* workflow.
"""

from agent.prompt_builder import KANBAN_GUIDANCE


def _g() -> str:
    return KANBAN_GUIDANCE.lower()


# --- the old human-gated review block is gone -------------------------------


def test_no_mandatory_review_required_block_instruction():
    """The `kanban_block(reason="review-required: ...")` handoff is retired."""
    assert "review-required" not in _g()


def test_completion_step_does_not_route_code_changes_into_blocked():
    guidance = _g()
    # The old text told coding workers to "end with kanban_block(...)".
    assert "end with `kanban_block" not in guidance
    # ...and justified it as more honest than completing. Both must be gone.
    assert "auto-completing" not in guidance


# --- the automated review DAG is present ------------------------------------


def test_reviewable_code_creates_a_ready_review_card():
    guidance = _g()
    assert "review card" in guidance
    assert "ready" in guidance
    # Assigned to a real installed reviewer profile, normally pr-reviewer.
    assert "pr-reviewer" in guidance


def test_review_card_carries_exact_evidence():
    guidance = _g()
    for token in ("workspace", "repo", "base", "head", "acceptance"):
        assert token in guidance, token


def test_implementation_card_completes_and_lists_created_cards():
    guidance = _g()
    assert "created_cards" in guidance
    # Completion of the implementation card is explicit, not implied.
    assert "kanban_complete" in guidance


def test_reviewer_pass_completes_and_block_creates_remediation():
    guidance = _g()
    assert "pass" in guidance
    assert "remediation" in guidance
    # BLOCK verdict routes back to the original implementation profile.
    assert "original implementation profile" in guidance


def test_remediation_creates_a_fresh_exact_head_review_card():
    guidance = _g()
    assert "fresh" in guidance and "exact-head" in guidance


def test_dependencies_use_parents_dag_not_blocked():
    guidance = _g()
    assert "parents=" in guidance
    assert "not `blocked`" in guidance or "not blocked" in guidance


# --- blocked stays narrow ---------------------------------------------------


def test_blocked_is_reserved_for_three_named_conditions():
    guidance = _g()
    assert "irretrievable human input" in guidance
    assert "capability" in guidance and "access wall" in guidance
    assert "transient failure after retries" in guidance


def test_blocked_anti_uses_are_enumerated():
    guidance = _g()
    for anti in (
        "dispatcher fence",
        "status update",
        "review queue",
        "ordinary dependency wait",
        "substitute for child tasks",
    ):
        assert anti in guidance, anti


def test_missing_reviewer_profile_surfaces_needs_input_not_an_invented_assignee():
    guidance = _g()
    assert "needs_input" in guidance
    assert "invent" in guidance


# --- preserved invariants ---------------------------------------------------


def test_unknown_assignees_must_still_be_discovered_and_validated():
    guidance = _g()
    assert "hermes profile list" in guidance
    assert "silently" in guidance


def test_genuine_block_path_is_retained():
    """Do not weaken true safety blocks: kanban_block still exists."""
    assert "kanban_block" in _g()
