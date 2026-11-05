"""Read-only synthetic evidence policy through the real loop, NOT live-model reasoning.

The provider sees only wire messages. The tool boundary independently checks names,
exact parsed arguments and order. Only the premature final is a scripted injection.
"""
import json
from copy import deepcopy
from unittest.mock import patch

from agent.turn_continuation import CONTINUATION_RECONCILIATION_NUDGE
from tests.agent.test_degenerate_final_recovery import _final
from tests.agent.test_review_continuation import REQUEST, SUGGESTION, gateway_steer
from tests.agent.test_run_agent import _mock_response, _mock_tool_call

FIELDS = {
    "inspect_candidate": ("candidate_id",),
    "read_test_report": ("artifact_id",),
    "review_artifact": ("artifact_id", "test_report_id"),
    "compare_trace": ("artifact_id", "review_id", "flaw_id", "target_id"),
    "verify_trace": ("artifact_id", "trace_id"),
    "review_verification": ("artifact_id", "verification_id"),
}
SCHEMAS = [{"type": "function", "function": {
    "name": name, "description": "Fixture-only read-only evidence lookup; no writes or publication.",
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {**{key: {"type": "string"} for key in keys},
                                  "read_only": {"type": "boolean", "enum": [True]}},
                   "required": [*keys, "read_only"]},
}} for name, keys in FIELDS.items()]


def evidence_policy(messages):
    """Choose from model-visible evidence, never invocation indices or a queued plan."""
    latest_user = next(row["content"] for row in reversed(messages) if row["role"] == "user")
    if "Stop now" in latest_user:
        return "Stopped by human request."
    try:
        evidence = [json.loads(row["content"]) for row in messages if row["role"] == "tool"]
    except (ValueError, TypeError):
        return "Blocked: tool did not return structured evidence."
    if not evidence:
        request = next(row["content"] for row in messages if row["role"] == "user")
        return "inspect_candidate", {"candidate_id": json.loads(request.split("\nFixture target: ")[1])["candidate_id"]}
    current = evidence[-1]
    kind = current["kind"]
    if kind == "inspection":
        return "read_test_report", {"artifact_id": current["artifact_id"]}
    if kind == "test_report":
        return "review_artifact", {"artifact_id": current["artifact_id"], "test_report_id": current["test_report_id"]}
    if kind == "review":
        artifact = next(e["artifact_id"] for e in evidence if e["kind"] == "inspection")
        if (current.get("artifact_id") != artifact or current.get("status") != "REQUEST_CHANGES"
                or not current.get("trace_available") or not current.get("review_id")):
            return "Blocked: valid artifact review and available trace required."
        return "compare_trace", {key: current[key] for key in FIELDS["compare_trace"]}
    if kind == "trace":
        return "verify_trace", {"artifact_id": current["artifact_id"], "trace_id": current["trace_id"]}
    if kind == "verification":
        if current["status"] != "PASS":
            return "Blocked: cancellation verification failed."
        return "review_verification", {"artifact_id": current["artifact_id"], "verification_id": current["verification_id"]}
    if kind == "verification_review":
        verification = next(e for e in evidence if e["kind"] == "verification")
        if (current["status"] == "APPROVE" and verification["status"] == "PASS"
                and current["verification_id"] == verification["verification_id"]):
            return f"Verified {verification['artifact_id']} with {verification['verification_id']}; review {current['review_id']} approved."
    return "Blocked: missing or invalid evidence."


def replay_evidence(agent, *, suffix="one", target="cancel-order", review_mutation=None,
                    mutate_call=None, failed_verification=False, interrupt_after_nudge=False,
                    stop_after_review=False):
    """No external I/O. Mutation inputs challenge policy/transport, not production permissions."""
    ids = {key: f"{key}-{suffix}" for key in ("candidate", "artifact", "tests", "review", "flaw", "trace", "verification", "second-review")}
    agent.valid_tool_names = set(FIELDS) | {"web_search"}  # admit the hostile decoy to the validating boundary
    agent.tools = deepcopy(SCHEMAS)
    requests, calls, accepted = [], [], []
    artifacts = {}
    previous = None
    premature_stop_sent = False
    trace_events = ["cancel_requested", "halted"]
    if failed_verification:
        trace_events.append("dispatch")

    def tool_io(name, args, *unused, **kwargs):
        nonlocal previous
        calls.append((name, deepcopy(args), kwargs["tool_call_id"]))
        # These contracts are keyed by the ACTUAL name, not an index-generated phase.
        contracts = {
            "inspect_candidate": (None, {"candidate_id": ids["candidate"]}),
            "read_test_report": ("inspect_candidate", {"artifact_id": ids["artifact"]}),
            "review_artifact": ("read_test_report", {"artifact_id": ids["artifact"], "test_report_id": ids["tests"]}),
            "compare_trace": ("review_artifact", {"artifact_id": ids["artifact"], "review_id": ids["review"],
                                                "flaw_id": ids["flaw"], "target_id": target}),
            "verify_trace": ("compare_trace", {"artifact_id": ids["artifact"], "trace_id": ids["trace"]}),
            "review_verification": ("verify_trace", {"artifact_id": ids["artifact"], "verification_id": ids["verification"]}),
        }
        contract = contracts.get(name)
        if (contract is None or previous != contract[0] or not isinstance(args, dict)
                or args != {**contract[1], "read_only": True} or args.get("read_only") is not True):
            return json.dumps({"kind": "blocked", "reason": "unknown, reordered or mismatched read-only call"})
        accepted.append(name)
        previous = name
        outputs = {
            "inspect_candidate": {"kind": "inspection", "artifact_id": ids["artifact"]},
            "read_test_report": {"kind": "test_report", "artifact_id": ids["artifact"], "test_report_id": ids["tests"]},
            "review_artifact": {"kind": "review", "artifact_id": ids["artifact"], "review_id": ids["review"],
                                "status": "REQUEST_CHANGES", "flaw_id": ids["flaw"], "target_id": target,
                                "trace_available": True},
            "compare_trace": {"kind": "trace", "artifact_id": ids["artifact"], "trace_id": ids["trace"],
                              "target_id": target, "events": trace_events},
            "verify_trace": {"kind": "verification", "artifact_id": ids["artifact"], "trace_id": ids["trace"],
                             "verification_id": ids["verification"]},
            "review_verification": {"kind": "verification_review", "artifact_id": ids["artifact"],
                                    "verification_id": ids["verification"], "review_id": ids["second-review"]},
        }
        output = outputs[name]
        if name == "verify_trace":
            events = artifacts[args["trace_id"]]["events"]
            output["status"] = "FAIL" if "dispatch" in events[events.index("cancel_requested") + 1:] else "PASS"
        if name == "review_verification":
            output["status"] = "APPROVE" if artifacts[args["verification_id"]]["status"] == "PASS" else "REJECT"
        if name == "compare_trace":
            artifacts[output["trace_id"]] = deepcopy(output)
        if name == "verify_trace":
            artifacts[output["verification_id"]] = deepcopy(output)
        if name == "inspect_candidate":
            agent.steer(gateway_steer("carry on with the investigation"))
        if name == "review_artifact" and review_mutation:
            output = review_mutation(deepcopy(output))
        if name == "review_artifact" and stop_after_review:
            agent.steer(gateway_steer("Stop now"))
        return json.dumps(output)

    def provider(**kwargs):
        nonlocal premature_stop_sent
        messages = kwargs["messages"]
        requests.append(deepcopy(messages))
        decision = evidence_policy(messages)
        # Only this deliberate premature stop is scripted. Even the injection requires
        # a valid review; unavailable/mismatched review takes the blocked branch directly.
        if isinstance(decision, tuple) and decision[0] == "compare_trace" and not premature_stop_sent:
            premature_stop_sent = True
            return _final(SUGGESTION)
        if isinstance(decision, str):
            return _final(decision)
        name, args = decision
        args = {**args, "read_only": True}
        if mutate_call:
            name, args = mutate_call(name, args)
        response = _mock_response(content="", finish_reason="tool_calls", tool_calls=[
            _mock_tool_call(name=name, arguments=json.dumps(args), call_id=f"call-{len(requests)}")])
        if interrupt_after_nudge and any(row.get("content") == CONTINUATION_RECONCILIATION_NUDGE for row in messages):
            agent.interrupt()
        return response

    agent.client.chat.completions.create.side_effect = provider
    with (
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", side_effect=tool_io),
    ):
        result = agent.run_conversation(REQUEST + "\nFixture target: " + json.dumps({"candidate_id": ids["candidate"]}))
    return result, calls, accepted, requests
