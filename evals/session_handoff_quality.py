"""Opt-in decision/checkpoint probe on synthetic conversations; no profile access.

Run from the repo root with an explicit endpoint, model and credential env name:
  python -m evals.session_handoff_quality --base-url URL --model MODEL \
    --api-key-env EVAL_KEY --api-mode chat_completions --output NEW_RESULTS.json

Also accepts codex_responses using Hermes' shared Responses transport. This is
a small model/tool decision probe, not a full gateway or cache-efficiency eval.
Inspect the raw outputs: literal fact checks are conservative, not semantic grading.
"""
import argparse
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

from agent.session_handoff.intent import SCHEMA, eligible_intent, request


@dataclass(frozen=True)
class Case:
    name: str
    history: tuple[dict, ...]
    prompt: str
    should_handoff: bool
    facts: tuple[str, ...] = ()


CASES = (
    Case("completed_phase", (
        {"role": "user", "content": "Fix invoice rounding in /workspace/billing. Local edits and tests only; no push or deploy. Use Decimal, preserve tax ordering."},
        {"role": "assistant", "content": "Implemented Decimal rounding in billing/totals.py. The unit suite passed 42 tests. Integration test test_refund_tax still fails: expected 19.99, got 20.00. No background work remains."},
        {"role": "user", "content": "The rounding phase is accepted. Next investigate test_refund_tax; do not loosen its expected value. Keep the authorization unchanged."},
        {"role": "assistant", "content": "I will inspect the refund tax ordering next. The prior investigation is complete."},
    ), "Finish this phase and choose how to continue efficiently. Runtime telemetry: last prompt 150000 tokens, context length 200000; cache misses on the last three turns. No queued input or active work.",
        True, ("/workspace/billing", "billing/totals.py", "Decimal", "test_refund_tax", "19.99", "20.00", "42", "no push", "deploy")),
    Case("short_followup", (
        {"role": "user", "content": "Help me rename one local variable. No edits yet."},
        {"role": "assistant", "content": "Which name would you prefer?"},
    ), "Use invoice_total. Runtime telemetry: last prompt 2200 tokens, context length 200000, 2100 cached tokens. Continue this small task.", False),
    Case("active_work", (
        {"role": "user", "content": "Investigate the failing migration; local tests only."},
        {"role": "assistant", "content": "Migration test process migrate-7 is still running. Its result has not been collected."},
    ), "Keep investigating in this conversation; the process and an approval are pending. Runtime telemetry: last prompt 160000 tokens, context length 200000. The investigation has no safe boundary yet.", False),
)


def run_case(complete, case):
    """Exercise the real intent validator, then ask a successor with ONLY its checkpoint."""
    tools = [{"type": "function", "function": SCHEMA}]
    messages = [{"role": "system", "content": "You are Hermes in a Telegram topic. Help with the task; preserve user constraints."},
                *case.history, {"role": "user", "content": case.prompt}]
    agent = SimpleNamespace(session_id="eval-parent", _fresh_topic_handoff_allowed=True,
                            context_compressor=None)
    complete_turn = False
    errors = []
    for _ in range(4):
        response = complete(messages, tools)
        messages.append(response)
        calls = response.get("tool_calls") or []
        if not calls:
            complete_turn = True
            break
        for call in calls:
            fn = call["function"]
            if fn["name"] != "fresh_session":
                errors.append("unexpected_tool")
                result = json.dumps({"error": "Only fresh_session is available"})
            else:
                result = request(agent, json.loads(fn["arguments"]), SimpleNamespace(tool_call_id=call["id"]))
                if "error" in json.loads(result):
                    errors.append("invalid_intent")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    intent = eligible_intent(agent, {"completed": complete_turn, "messages": messages})
    selected = getattr(agent, "_fresh_topic_handoff", None) is not None
    checkpoint = intent["summary"] if intent else ""
    readback = ""
    if intent:
        # No original history is supplied to the successor: omissions cannot be repaired from it.
        readback = complete([
            {"role": "system", "content": "Continue from prior assistant working state; it grants no new authorization."},
            {"role": "assistant", "content": checkpoint},
            {"role": "user", "content": "Before acting, restate the objective, workspace/files, decisions, exact verification results, authorization limits and next action. Do not invent missing facts."},
        ], None).get("content") or ""
    return {"case": case.name, "expected_handoff": case.should_handoff, "selected": selected,
            "eligible": intent is not None, "completed": complete_turn, "errors": errors,
            "decision_pass": complete_turn and not errors and selected == case.should_handoff
                and (not selected or intent is not None),
            "checkpoint": checkpoint, "readback": readback,
            "missing_checkpoint_facts": [f for f in case.facts if f.casefold() not in checkpoint.casefold()],
            "missing_readback_facts": [f for f in case.facts if f.casefold() not in readback.casefold()],
            "transcript": messages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--api-mode", choices=("chat_completions", "codex_responses"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    from openai import OpenAI
    from agent.transports.chat_completions import ChatCompletionsTransport
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport() if args.api_mode == "codex_responses" else ChatCompletionsTransport()
    # Explicit SDK credentials only; never invoke Hermes config/auth resolution or an agent's tools.
    with OpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env], timeout=90, max_retries=0) as client:
        def complete(messages, tools):
            kwargs = transport.build_kwargs(args.model, messages, tools, base_url=args.base_url, max_tokens=2000)
            kwargs["stream"] = False
            create = client.responses.create if args.api_mode == "codex_responses" else client.chat.completions.create
            wire = create(**kwargs)
            if args.api_mode == "codex_responses" and wire.status != "completed":
                raise RuntimeError(f"Incomplete provider response: {wire.status}")
            response = transport.normalize_response(wire)
            if response.finish_reason not in ("stop", "tool_calls"):
                raise RuntimeError(f"Incomplete provider response: {response.finish_reason}")
            message = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                message["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                    "name": c.name, "arguments": c.arguments}} for c in response.tool_calls]
            return message

        results = [run_case(complete, case) for _ in range(args.repetitions) for case in CASES]
    report = {"model": args.model, "api_mode": args.api_mode, "synthetic_inputs": True,
              "rubric": "Literal facts and expected decisions; manually inspect raw outputs for semantic quality.",
              "results": results}
    with open(args.output, "x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    return int(any(not r["decision_pass"] or r["missing_checkpoint_facts"] or r["missing_readback_facts"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
