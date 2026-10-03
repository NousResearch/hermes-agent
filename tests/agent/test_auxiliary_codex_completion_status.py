"""Auxiliary Responses normalization must not certify a partial checkpoint."""

from types import SimpleNamespace

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter
from agent.context_compressor import ContextCompressor


def _message(text, *, phase: str | None = "final_answer", status="completed"):
    return SimpleNamespace(
        type="message", role="assistant", phase=phase, status=status,
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _reasoning(text):
    return SimpleNamespace(
        type="reasoning", id="rs_test", encrypted_content=None,
        summary=[SimpleNamespace(text=text)],
    )


def _adapter_response(final, *, streamed, base_url=""):
    class FakeStream:
        def __iter__(self):
            if getattr(final, "output_text", ""):
                yield SimpleNamespace(type="response.output_text.delta", delta=final.output_text)
            for item in final.output or []:
                yield SimpleNamespace(type="response.output_item.done", item=item)
            # Cancellation is a response status, not a Responses SSE event type.
            terminal = "failed" if final.status == "cancelled" else final.status
            yield SimpleNamespace(type=f"response.{terminal}", response=final)

        def close(self):
            pass

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return FakeStream() if streamed else final

    return _CodexCompletionsAdapter(
        SimpleNamespace(base_url=base_url, responses=FakeResponses()), "aux-model",
    ).create(messages=[{"role": "user", "content": "Summarize the task."}])


@pytest.mark.parametrize("fallback", [False, True], ids=["abort", "main-fallback"])
def test_compressor_rejects_actual_adapter_partial_summary(monkeypatch, fallback):
    partial = SimpleNamespace(
        status="incomplete", output=[_message("PARTIAL_CHECKPOINT")],
        incomplete_details={"reason": "max_output_tokens"}, error=None,
        usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14),
    )
    complete = SimpleNamespace(
        status="completed", output=[_message("COMPLETE_CHECKPOINT")],
        incomplete_details=None, error=None, usage=None,
    )
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs.get("model"))
        return _adapter_response(partial if len(calls) == 1 else complete, streamed=True)

    monkeypatch.setattr("agent.context_compressor.call_llm", fake_call_llm)
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **k: 100000)
    compressor = ContextCompressor(
        model="main-model", summary_model_override="aux-model" if fallback else "",
        quiet_mode=True, tail_mode="legacy", protect_first_n=2, protect_last_n=2,
        abort_on_summary_failure=False,
    )
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"task {i} " + "x" * 50}
        for i in range(12)
    ]
    original = [dict(message) for message in messages]
    result = compressor.compress(messages, current_tokens=999999, force=True)

    assert "PARTIAL_CHECKPOINT" not in (compressor._previous_summary or "")
    assert messages == original
    if fallback:
        assert calls == ["aux-model", None]
        assert "COMPLETE_CHECKPOINT" in (compressor._previous_summary or "")
        assert result != original
        assert compressor._last_compress_aborted is False
    else:
        assert calls == [None]
        assert result == original
        assert compressor._previous_summary is None
        assert compressor._last_summary_truncated_failure is True
        assert compressor._last_compress_aborted is True


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
@pytest.mark.parametrize(
    "base_url, reasoning_text, expected_content, expected_finish",
    [
        pytest.param(
            "https://chatgpt.com/backend-api/codex",
            "still thinking",
            None,
            "length",
            id="codex-reasoning-only",
        ),
        pytest.param(
            "https://api.x.ai/v1",
            "scratch\n<response>FINAL ANSWER</response>",
            "FINAL ANSWER",
            "stop",
            id="xai-reasoning-answer",
        ),
    ],
)
def test_actual_adapter_preserves_route_sensitive_reasoning(
    streamed, base_url, reasoning_text, expected_content, expected_finish,
):
    final = SimpleNamespace(
        status="completed", output=[_reasoning(reasoning_text)], output_text="",
        incomplete_details=None, error=None,
        usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14),
    )

    response = _adapter_response(final, streamed=streamed, base_url=base_url)

    choice = response.choices[0]
    assert choice.message.content == expected_content
    assert choice.finish_reason == expected_finish
    assert getattr(response, "status", None) == "completed"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (11, 3, 14)


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
@pytest.mark.parametrize(
    "status, output, reason, output_text, expected_content, expected_finish, error",
    [
        pytest.param("completed", [_message("FINAL")], None, "", "FINAL", "stop", None, id="final-answer"),
        pytest.param("completed", [_message("LEGACY", phase=None)], None, "", "LEGACY", "stop", None, id="unphased"),
        pytest.param("completed", [_message("COMMENTARY", phase="commentary")], None, "COMMENTARY", None, "length", None, id="commentary-only"),
        pytest.param("completed", [_message("ANALYSIS", phase="analysis")], None, "ANALYSIS", None, "length", None, id="analysis-only"),
        pytest.param("completed", [_message("COMMENTARY", phase="commentary"), _message("FINAL", phase="final")], None, "COMMENTARYFINAL", "FINAL", "stop", None, id="commentary-then-final"),
        pytest.param("completed", [SimpleNamespace(type="message", status="completed", content=[{"type": "refusal", "refusal": "DECLINED"}])], None, "", "DECLINED", "stop", None, id="refusal"),
        pytest.param("completed", [SimpleNamespace(type="function_call", status="completed", call_id="call_1", name="inspect", arguments='{"path":"x"}')], None, "", None, "tool_calls", None, id="tool-call"),
        pytest.param("incomplete", [_message("PARTIAL")], "max_output_tokens", "", "PARTIAL", "length", None, id="token-cap-final-phase"),
        pytest.param("incomplete", [_message("PARTIAL", phase=None)], "other", "", "PARTIAL", "length", None, id="other-incomplete"),
        pytest.param("completed", [_message("PARTIAL", status="incomplete")], None, "", "PARTIAL", "length", None, id="incomplete-item"),
        pytest.param("completed", [SimpleNamespace(type="function_call", status="in_progress", call_id="call_1", name="inspect", arguments='{"path":')], None, "", None, "length", None, id="partial-tool-call"),
        pytest.param("incomplete", None, "content_filter", "", None, "content_filter", None, id="content-filter"),
        pytest.param("failed", [_message("NOT_A_FINAL")], None, "", None, None, {"code": "provider_error", "message": "backend failed"}, id="failed"),
        pytest.param("cancelled", [_message("NOT_A_FINAL")], None, "", None, None, {"message": "request cancelled"}, id="cancelled"),
        pytest.param("completed", None, None, "DELTA_FINAL", "DELTA_FINAL", "stop", None, id="output-text-fallback"),
        pytest.param("completed", None, None, "", None, "stop", None, id="legacy-empty"),
    ],
)
def test_actual_adapter_preserves_completion_contract(
    streamed, status, output, reason, output_text, expected_content, expected_finish, error,
):
    details = {"reason": reason} if reason else None
    final = SimpleNamespace(
        status=status, output=output, output_text=output_text,
        incomplete_details=details, error=error,
        usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
    )
    if error:
        with pytest.raises(RuntimeError, match=error["message"]):
            _adapter_response(final, streamed=streamed)
        return

    response = _adapter_response(final, streamed=streamed)
    choice = response.choices[0]
    assert choice.message.content == expected_content
    assert choice.finish_reason == expected_finish
    assert getattr(response, "status", None) == status
    assert getattr(response, "incomplete_details", None) == details
    assert getattr(response, "error", None) is None
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (11, 3, 14)
    if expected_finish == "tool_calls":
        call, = choice.message.tool_calls
        assert (call.id, call.function.name, call.function.arguments) == ("call_1", "inspect", '{"path":"x"}')
    else:
        assert choice.message.tool_calls is None
