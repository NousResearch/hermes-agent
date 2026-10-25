"""Regression coverage for max-iteration summary leakage (#122607)."""

from types import SimpleNamespace

import agent.chat_completion_helpers as helpers
from agent.chat_completion_helpers import _EMPTY_SUMMARY_RESPONSE, handle_max_iterations


class _SummaryAgent(SimpleNamespace):
    max_iterations = 2
    suppress_status_output = True
    api_mode = "test_summary"

    def _safe_print(self, *_args, **_kwargs):
        raise AssertionError("suppress_status_output should route through logging")

    def _strip_think_blocks(self, text):
        return (text or "").replace("<think>hidden</think>", "")


def _run_summary(monkeypatch, attempts):
    responses = iter(attempts)

    def build_attempt(_agent, _api_messages, _request_id):
        def attempt(_retry_count):
            return next(responses)
        return attempt

    monkeypatch.setattr(helpers, "_SUMMARY_ATTEMPT_BUILDERS", {"test_summary": build_attempt})
    monkeypatch.setattr(helpers, "_iteration_summary_api_messages", lambda _agent, messages: messages)

    from agent import relay_llm
    monkeypatch.setattr(relay_llm, "complete_logical_call", lambda *_args, **_kwargs: None)

    messages = [{"role": "user", "content": "please finish"}]
    final = handle_max_iterations(_SummaryAgent(), messages, api_call_count=2)
    return final, messages


def test_top_level_analysis_summary_artifact_uses_existing_fallback(monkeypatch):
    final, messages = _run_summary(monkeypatch, ["<analysis>private summary scratchpad</analysis>"])

    assert final == _EMPTY_SUMMARY_RESPONSE
    assert [m["role"] for m in messages] == ["user", "user"]
    assert all("analysis" not in str(m.get("content", "")) for m in messages)


def test_retry_summary_artifact_is_scrubbed_too(monkeypatch):
    final, messages = _run_summary(monkeypatch, ["", "<summary>internal control message</summary>"])

    assert final == _EMPTY_SUMMARY_RESPONSE
    assert [m["role"] for m in messages] == ["user", "user"]
    assert all("internal control" not in str(m.get("content", "")) for m in messages)


def test_unterminated_summary_artifact_is_not_returned(monkeypatch):
    final, messages = _run_summary(monkeypatch, ["<summary>private notes with no close"])

    assert final == _EMPTY_SUMMARY_RESPONSE
    assert [m["role"] for m in messages] == ["user", "user"]


def test_visible_tail_and_inline_details_summary_are_preserved(monkeypatch):
    text = (
        "<think>hidden</think>"
        "<analysis>private scratchpad</analysis>\n"
        "Final answer with <details><summary>open me</summary><p>body</p></details>."
    )

    final, messages = _run_summary(monkeypatch, [text])

    assert final == "Final answer with <details><summary>open me</summary><p>body</p></details>."
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == final
    assert "private scratchpad" not in final
