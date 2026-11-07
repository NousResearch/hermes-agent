"""Evidence transport + deterministic fixture policy, not live-model task ownership."""
import json

import pytest

from agent.turn_continuation import CONTINUATION_RECONCILIATION_NUDGE
from tests.agent.review_chain_fixture import FIELDS, replay_evidence
from tests.agent.test_degenerate_final_recovery import loop_agent
from tests.agent.test_review_continuation import SUGGESTION


def test_review_request_changes_reaches_diagnostic_and_second_review(loop_agent):
    result, calls, accepted, sent = replay_evidence(loop_agent)
    assert result["final_response"] == "Verified artifact-one with verification-one; review second-review-one approved."
    assert accepted == list(FIELDS)
    assert [name for name, _, _ in calls] == list(FIELDS)
    assert [args for _, args, _ in calls] == [
        {"candidate_id": "candidate-one", "read_only": True},
        {"artifact_id": "artifact-one", "read_only": True},
        {"artifact_id": "artifact-one", "test_report_id": "tests-one", "read_only": True},
        {"artifact_id": "artifact-one", "review_id": "review-one", "flaw_id": "flaw-one",
         "target_id": "cancel-order", "read_only": True},
        {"artifact_id": "artifact-one", "trace_id": "trace-one", "read_only": True},
        {"artifact_id": "artifact-one", "verification_id": "verification-one", "read_only": True},
    ]
    assert len(sent) == 8
    assert len({call_id for _, _, call_id in calls}) == 6
    assert all(rows[0] == sent[0][0] for rows in sent)
    assert sent[4][-1]["content"] == CONTINUATION_RECONCILIATION_NUDGE
    assert any(row.get("content") == SUGGESTION and row.get("finish_reason") != "stop"
               for row in result["messages"])
    assert any(row.get("display_kind") == "steer" and "Gateway message origin" in row["content"]
               for row in result["messages"])
    review = [json.loads(row["content"]) for row in sent[3] if row["role"] == "tool"][-1]
    assert review["status"] == "REQUEST_CHANGES" and review["review_id"] == "review-one"


@pytest.mark.parametrize("mutation", ["unrelated", "unknown_arg", "swapped", "reordered", "write"])
def test_invalid_actual_calls_cannot_complete(loop_agent, mutation):
    def mutate(name, args):
        if name != "compare_trace":
            return name, args
        if mutation == "unrelated":
            return "web_search", {"query": "unrelated weather"}
        if mutation == "unknown_arg":
            return name, {**args, "command": "publish"}
        if mutation == "swapped":
            return name, {**args, "artifact_id": args["review_id"], "review_id": args["artifact_id"]}
        if mutation == "reordered":
            return "verify_trace", {"artifact_id": args["artifact_id"], "trace_id": "trace-one", "read_only": True}
        return name, {**args, "read_only": False}
    result, calls, accepted, _ = replay_evidence(loop_agent, mutate_call=mutate)
    assert result["final_response"].startswith("Blocked:")
    assert accepted == ["inspect_candidate", "read_test_report", "review_artifact"]
    assert len(calls) == 4  # the rejected call really reached the independent boundary


@pytest.mark.parametrize("mutation", ["missing", "wrong_artifact", "missing_id", "unavailable"])
def test_invalid_review_cannot_complete(loop_agent, mutation):
    def mutate(review):
        if mutation == "missing":
            return {"kind": "missing_review"}
        if mutation == "wrong_artifact":
            review["artifact_id"] = "unrelated-artifact"
        if mutation == "missing_id":
            review.pop("review_id")
        if mutation == "unavailable":
            review["trace_available"] = False
        return review
    result, calls, accepted, sent = replay_evidence(loop_agent, review_mutation=mutate)
    assert result["final_response"].startswith("Blocked:")
    assert len(calls) == len(accepted) == 3
    assert len(sent) == 4


def test_different_identifiers_and_diagnostic_propagate(loop_agent):
    result, calls, _, _ = replay_evidence(loop_agent, suffix="alternate", target="cancel-delivery")
    assert calls[3][1] == {"artifact_id": "artifact-alternate", "review_id": "review-alternate",
                           "flaw_id": "flaw-alternate", "target_id": "cancel-delivery", "read_only": True}
    assert calls[4][1]["trace_id"] == "trace-alternate"
    assert calls[5][1]["verification_id"] == "verification-alternate"
    assert result["final_response"] == "Verified artifact-alternate with verification-alternate; review second-review-alternate approved."


def test_review_only_target_mutation_changes_dispatched_arguments(loop_agent):
    result, calls, accepted, _ = replay_evidence(
        loop_agent, review_mutation=lambda review: {**review, "target_id": "other-trace"},
    )
    assert calls[3][0] == "compare_trace"
    assert calls[3][1]["target_id"] == "other-trace"
    assert accepted == ["inspect_candidate", "read_test_report", "review_artifact"]
    assert result["final_response"].startswith("Blocked:")


def test_failed_actual_trace_verification_cannot_complete(loop_agent):
    result, calls, accepted, sent = replay_evidence(loop_agent, failed_verification=True)
    assert result["final_response"] == "Blocked: cancellation verification failed."
    assert accepted[-1] == "verify_trace" and len(calls) == 5
    verification = [json.loads(row["content"]) for row in sent[-1] if row["role"] == "tool"][-1]
    assert verification["status"] == "FAIL"
    trace = next(json.loads(row["content"]) for row in sent[-1]
                 if row["role"] == "tool" and json.loads(row["content"])["kind"] == "trace")
    assert trace["events"] == ["cancel_requested", "halted", "dispatch"]


def test_interrupt_after_nudge_before_dispatch_blocks_tools(loop_agent):
    _, calls, accepted, sent = replay_evidence(loop_agent, interrupt_after_nudge=True)
    assert sent[4][-1]["content"] == CONTINUATION_RECONCILIATION_NUDGE
    assert len(sent) == 5
    assert len(calls) == len(accepted) == 3


def test_textual_stop_is_consumed_by_fixture_policy_before_action(loop_agent):
    result, calls, _, sent = replay_evidence(loop_agent, stop_after_review=True)
    assert "Stop now" in sent[-1][-1]["content"]
    assert result["final_response"] == "Stopped by human request."
    assert len(calls) == 3
