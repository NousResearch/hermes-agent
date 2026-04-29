import pytest

from agent.retrieval_budget import RetrievalBudget


def test_broad_search_blocked_when_not_allowed():
    budget = RetrievalBudget(
        max_retrieval_calls=4,
        max_broad_search_calls=1,
        max_subtree_expansions=2,
        max_total_retrieval_seconds=25,
        recommended_sequence=["exact_path", "known_subtree", "session_search"],
        allow_broad_search=False,
    )

    allowed, reason = budget.can_attempt("broad_search")

    assert allowed is False
    assert reason == "broad_search_disabled"


def test_stage_order_requires_prior_attempts():
    budget = RetrievalBudget(
        max_retrieval_calls=4,
        max_broad_search_calls=1,
        max_subtree_expansions=2,
        max_total_retrieval_seconds=25,
        recommended_sequence=["exact_path", "known_subtree", "session_search"],
        allow_broad_search=False,
    )

    allowed, reason = budget.can_attempt("session_search")
    assert allowed is False
    assert reason == "stage_order"

    budget.record_attempt("exact_path", "read_file", seconds=0.3, outcome="empty")
    allowed, reason = budget.can_attempt("known_subtree")
    assert allowed is True
    assert reason is None

    budget.record_attempt("known_subtree", "search_files", seconds=0.4, outcome="empty")
    allowed, reason = budget.can_attempt("session_search")
    assert allowed is True
    assert reason is None


def test_successful_prior_stage_blocks_unnecessary_escalation():
    budget = RetrievalBudget(
        max_retrieval_calls=4,
        max_broad_search_calls=1,
        max_subtree_expansions=2,
        max_total_retrieval_seconds=25,
        recommended_sequence=["exact_path", "known_subtree", "session_search"],
        allow_broad_search=False,
    )

    budget.record_attempt("exact_path", "read_file", seconds=0.2, outcome="success")
    allowed, reason = budget.can_attempt("known_subtree")

    assert allowed is False
    assert reason == "previous_stage_succeeded"


def test_budget_exhaustion_by_calls_and_time():
    budget = RetrievalBudget(
        max_retrieval_calls=2,
        max_broad_search_calls=1,
        max_subtree_expansions=2,
        max_total_retrieval_seconds=1,
        recommended_sequence=["exact_path", "known_subtree"],
        allow_broad_search=False,
    )

    budget.record_attempt("exact_path", "read_file", seconds=0.6, outcome="empty")
    budget.record_attempt("known_subtree", "search_files", seconds=0.5, outcome="empty")

    allowed, reason = budget.can_attempt("session_search")
    assert allowed is False
    assert reason in {"max_retrieval_calls", "max_total_retrieval_seconds"}


def test_subtree_expansion_limit_is_enforced():
    budget = RetrievalBudget(
        max_retrieval_calls=5,
        max_broad_search_calls=1,
        max_subtree_expansions=1,
        max_total_retrieval_seconds=25,
        recommended_sequence=["known_subtree", "session_search"],
        allow_broad_search=False,
    )

    budget.record_attempt("known_subtree", "search_files", seconds=0.2, outcome="empty")
    allowed, reason = budget.can_attempt("known_subtree")

    assert allowed is False
    assert reason == "max_subtree_expansions"
