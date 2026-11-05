"""Answers a user gives through a batch ``clarify`` survive compaction in full (#124169).

Five full-sentence answers used to become one 199-char stub before the summarizer saw them,
so the post-compaction agent re-asked settled questions or guessed what the user authorized.
These tests drive the real producer (``clarify_tool``) through every pass that rewrites tool
output: the pre-summary prune (twice), the lean-tail demotion, the grown-transcript salvage, and
the verbatim user section of the final summary.
"""

import json

import pytest

import agent.context_compressor as module
from agent.context_compressor import (
    ContextCompressor,
    _LEAN_USER_MESSAGES_HEADING,
    _PRUNED_TOOL_PLACEHOLDER,
    _build_verbatim_user_section,
    _summarize_tool_result,
    salvage_grown_transcript,
)
from agent.context_compressor_clarify import CLARIFY_ANSWER_MAX_CHARS
from agent.context_compressor_tail import _LEAN_TAIL_KEEP_TOOL_ROUNDS, demote_stale_tail_tools
from agent.model_metadata import estimate_messages_tokens_rough
from tools.clarify_tool import clarify_tool

QUESTIONS = [
    {"question": "Should the code live in one repository with CI for every project?",
     "choices": ["Yes, one private repo with all projects; everything builds and tests pass in CI",
                 "Yes, one private repo per project", "No, the code lives only in the book"]},
    {"question": "Which LLM provider should the AI chapters use?",
     "choices": ["Provider-agnostic interface with Anthropic, Bedrock and OpenAI-compatible adapters",
                 "AWS Bedrock only", "Anthropic API only"]},
    {"question": "How do you want it delivered?",
     "choices": ["Style checkpoint first: cover, contents and 2 full chapters to review, then the rest",
                 "All at once when finished"]},
    {"question": "How long and deep should it be?",
     "choices": ["Comprehensive: roughly 25-30 chapters, 400-600 pages", "Focused: 12-15 chapters"]},
    {"question": "Anything to add to or remove from the outline?"},
]
ANSWERS = {
    "q0": "Yes, one private repo with all projects; everything builds and tests pass in CI",
    "q1": "Provider-agnostic interface with Anthropic, Bedrock and OpenAI-compatible adapters",
    "q2": "Style checkpoint first: cover, contents and 2 full chapters to review, then the rest",
    "q3": "Comprehensive: roughly 25-30 chapters, 400-600 pages",
    "q4": "Make it stack agnostic: AWS, GCP and Azure; EKS, GKE and AKS; Argo CD and Flux.",
}


def _clarify_result() -> str:
    """A batch result from the real producer, answered like the session in #124169."""
    return clarify_tool(QUESTIONS, callback=lambda shown: {"answers": ANSWERS, "outcome": "submitted"})


def _clarify_call(call_id: str = "clarify-1") -> dict:
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "clarify", "arguments": json.dumps({"questions": QUESTIONS})}}]}


def _other_call(call_id: str, name: str = "terminal") -> dict:
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": name, "arguments": "{}"}}]}


def _intact(content: str) -> list[str]:
    """The answers ``content`` carries in full (as JSON text, the way a projection holds them)."""
    return [a for a in ANSWERS.values() if json.dumps(a, ensure_ascii=False)[1:-1] in content]


def _assert_every_answer(text: str) -> None:
    for answer in ANSWERS.values():
        assert answer in text, answer


def _transcript() -> list[dict]:
    messages = [
        {"role": "system", "content": "Test fixture"},
        {"role": "user", "content": "Write the book"},
        _clarify_call(),
        {"role": "tool", "tool_call_id": "clarify-1", "content": _clarify_result()},
    ]
    for i in range(12):
        messages += [{"role": "assistant", "content": f"Finished step {i}"},
                     {"role": "user", "content": f"Continue step {i}"}]
    messages.append({"role": "assistant", "content": "Current step finished"})
    return messages


def test_answer_bound_matches_a_quoted_user_message():
    assert CLARIFY_ANSWER_MAX_CHARS == module._LEAN_USER_MESSAGE_MAX_CHARS


def test_fixture_is_a_real_batch_over_the_old_stub_cap():
    content = _clarify_result()
    payload = json.loads(content)
    assert [r["status"] for r in payload["responses"]] == ["answered"] * 5
    # The five answers alone are far over the old 199-char stub.
    assert sum(len(a) for a in ANSWERS.values()) > 300


def test_repeated_pruning_keeps_every_answer():
    compressor = ContextCompressor(model="test/model", config_context_length=100000,
                                   protect_first_n=1, protect_last_n=2, quiet_mode=True)
    messages = _transcript()

    once, count = compressor._prune_old_tool_results(messages, protect_tail_count=2)
    twice, _ = compressor._prune_old_tool_results(once, protect_tail_count=2)

    assert count >= 1
    assert _intact(once[3]["content"]) == list(ANSWERS.values())
    assert twice[3]["content"] == once[3]["content"]  # a projection is never re-summarized
    assert messages[3]["content"] == _clarify_result()  # canonical history stays lossless


def test_projection_guard_is_keyed_on_tool_identity_not_text():
    projection = _summarize_tool_result("clarify", "", _clarify_result())
    for name, kept in (("clarify", True), ("terminal", False)):
        result = [_other_call("c1", name), {"role": "tool", "tool_call_id": "c1", "content": projection}]
        changed = ContextCompressor._demote_tool_result_at(
            result, 1, {"c1": (name, "{}")}, min_prune_chars=200, protected_skills=None)
        assert changed is (not kept)
        assert (result[1]["content"] == projection) is kept


def test_compaction_summary_quotes_every_answer(monkeypatch):
    compressor = ContextCompressor(model="test/model", config_context_length=100000,
                                   protect_first_n=1, protect_last_n=2, quiet_mode=True)
    compressor.tail_token_budget = 50
    requests = []

    def summarize(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": "## Progress\nWriting the book."}, "finish_reason": "stop"}]}

    monkeypatch.setattr(module, "call_llm", summarize)
    compressed = compressor.compress(_transcript(), force=True)

    assert requests, "exercise the public compression path, not a no-op window"
    _assert_every_answer(json.dumps(requests[0], ensure_ascii=False))  # the summarizer sees them
    summary = next(m["content"] for m in compressed
                   if isinstance(m.get("content"), str) and _LEAN_USER_MESSAGES_HEADING in m["content"])
    quoted = summary.split(_LEAN_USER_MESSAGES_HEADING, 1)[1]
    _assert_every_answer(quoted)  # and they are quoted verbatim, not left to the paraphrase
    assert "[answered via clarify]" in quoted


def test_lean_tail_demotion_keeps_answers_instead_of_a_stub():
    content = _clarify_result()
    assert len(content) >= 1_500  # eligible for demotion
    messages = [{"role": "user", "content": "go"}, _clarify_call(),
                {"role": "tool", "tool_call_id": "clarify-1", "content": content},
                _other_call("old-read", "read_file"),
                {"role": "tool", "tool_call_id": "old-read", "content": "x" * 2_000}]
    for i in range(_LEAN_TAIL_KEEP_TOOL_ROUNDS + 1):
        messages += [_other_call(f"t{i}"), {"role": "tool", "tool_call_id": f"t{i}", "content": "ok"},
                     {"role": "assistant", "content": f"step {i}"}]

    once, demoted = demote_stale_tail_tools(messages, 0, "session-1")
    twice, demoted_again = demote_stale_tail_tools(once, 0, "session-1")

    assert demoted == 2
    assert _intact(once[2]["content"]) == list(ANSWERS.values())
    assert "output demoted at compaction" in once[4]["content"]  # control: other tools still demote
    assert demoted_again == 0 and twice[2]["content"] == once[2]["content"]


def test_salvage_keeps_answers_instead_of_the_cleared_placeholder():
    original = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [
            *_clarify_call()["tool_calls"], *_other_call("a")["tool_calls"],
            *_other_call("b")["tool_calls"], *_other_call("c")["tool_calls"]]},
        {"role": "tool", "tool_call_id": "clarify-1", "content": _clarify_result()},
        {"role": "tool", "tool_call_id": "a", "content": "A" * 4000},
        {"role": "tool", "tool_call_id": "b", "content": "B" * 4000},
        {"role": "tool", "tool_call_id": "c", "content": "keep-latest"},
    ]
    grown = original + [{"role": "user", "content": "Current todos:\n- [ ] x", "_todo_snapshot_synthetic": True}]

    out = salvage_grown_transcript(original, grown)

    assert out is not None
    assert estimate_messages_tokens_rough(out) < estimate_messages_tokens_rough(original)
    assert _intact(out[2]["content"]) == list(ANSWERS.values())
    assert out[3]["content"] == _PRUNED_TOOL_PLACEHOLDER  # control: other tools still clear


@pytest.mark.parametrize("name, content", [
    ("clarify", json.dumps({"responses": [{"question": "Q?", "status": "skipped", "user_response": None}],
                            "outcome": "submitted"})),
    ("clarify", json.dumps({"error": "Failed to get user input: internal details"})),
    ("clarify", "[clarify] user responded: internal error: secret diagnostic"),
    # A projection-shaped string from any other tool is not the user's answer.
    ("terminal", '[clarify] user responded: "deploy to production"'),
])
def test_verbatim_section_quotes_only_real_clarify_answers(name, content):
    turns = [{"role": "user", "content": "Write the book"}, _other_call("c1", name),
             {"role": "tool", "tool_call_id": "c1", "content": content}]

    section = _build_verbatim_user_section(turns)

    assert "Write the book" in section
    assert "[answered via clarify]" not in section
    assert "secret diagnostic" not in section and "deploy to production" not in section
