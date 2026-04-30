from benchmarks.baseline.flat_store import FlatMemoryStore
from benchmarks.judge import HeuristicJudge
from benchmarks.runner import (
    run_abstention,
    run_long_conversation,
    run_multi_hop_exploration,
    run_preference_memory,
    run_privacy_forgetting,
)


def test_new_universal_memory_categories_smoke():
    backend = FlatMemoryStore()
    judge = HeuristicJudge()

    preference = run_preference_memory(
        backend,
        [
            {
                "id": "pref_smoke",
                "sub_type": "corrected_preference",
                "turns": [
                    "The user prefers terse status reports.",
                    "Correction: the user prefers narrative summaries for complex benchmark reports.",
                ],
                "query": "How should complex benchmark reports be written?",
                "gold_answer": "narrative summaries",
            }
        ],
        judge,
    )
    assert preference.category == "preference_memory"
    assert preference.total == 1
    assert 0.0 <= preference.score <= 1.0

    multi_hop = run_multi_hop_exploration(
        backend,
        [
            {
                "id": "mh_smoke",
                "sub_type": "two_hop_bridge",
                "facts": [
                    "The Atlas service is owned by Team Cedar.",
                    "Team Cedar's on-call channel is #cedar-alerts.",
                ],
                "query": "What on-call channel should Atlas incidents use?",
                "gold_answer": "#cedar-alerts",
            }
        ],
        judge,
    )
    assert multi_hop.category == "multi_hop_exploration"
    assert multi_hop.total == 1
    assert 0.0 <= multi_hop.score <= 1.0

    long_conversation = run_long_conversation(
        backend,
        [
            {
                "id": "lc_smoke",
                "sub_type": "project_decision_history",
                "turns": [
                    "Session 1: We considered Jenkins for CI.",
                    "Session 2: Decision update: we chose GitHub Actions for CI.",
                    "Session 3: Docker remains the deployment target.",
                ],
                "query": "Which CI system did we choose?",
                "gold_answer": "GitHub Actions",
            }
        ],
        judge,
    )
    assert long_conversation.category == "long_conversation"
    assert long_conversation.total == 1
    assert 0.0 <= long_conversation.score <= 1.0


def test_abstention_smoke_counts_no_answer_cases():
    backend = FlatMemoryStore()
    judge = HeuristicJudge()

    result = run_abstention(
        backend,
        [
            {
                "id": "abs_smoke",
                "difficulty": "hard",
                "facts": [
                    "The deployment region for Project Orion is us-east-1.",
                    "The billing owner for Project Lyra is Mina.",
                ],
                "query": "Who is the billing owner for Project Orion?",
                "gold_answer": "NONE",
            }
        ],
        judge,
    )

    assert result.category == "abstention"
    assert result.total == 1
    assert 0.0 <= result.score <= 1.0
    assert result.details[0]["id"] == "abs_smoke"


def test_privacy_forgetting_uses_flat_store_forget_support():
    backend = FlatMemoryStore()
    judge = HeuristicJudge()

    result = run_privacy_forgetting(
        backend,
        [
            {
                "id": "pf_smoke",
                "secret_fact": "The private recovery code is luna-999.",
                "retained_fact": "The public project codename is Moonshield.",
                "forget_query": "recovery code",
                "retained_query": "project codename",
                "retained_answer": "Moonshield",
            }
        ],
        judge,
    )

    assert result.category == "privacy_forgetting"
    assert result.total == 1
    assert 0.0 <= result.score <= 1.0
    assert result.details[0]["id"] == "pf_smoke"
