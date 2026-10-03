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


def _adapter_response(final, *, streamed, base_url=""):
    class FakeStream:
        def __iter__(self):
            for item in final.output or []:
                yield SimpleNamespace(type="response.output_item.done", item=item)
            if getattr(final, "output_text", None):
                yield SimpleNamespace(type="response.output_text.delta", delta=final.output_text)
            yield SimpleNamespace(type=f"response.{final.status}", response=final)

        def close(self):
            pass

    class FakeResponses:
        def create(self, **kwargs):
            assert kwargs["stream"] is True
            return FakeStream() if streamed else final

    return _CodexCompletionsAdapter(
        SimpleNamespace(base_url=base_url, responses=FakeResponses()), "aux-model",
    ).create(messages=[{"role": "user", "content": "Summarize the task."}])


def test_compressor_rejects_actual_adapter_partial_summary(monkeypatch):
    partial = SimpleNamespace(
        status="incomplete", output=[_message("PARTIAL_CHECKPOINT")],
        incomplete_details={"reason": "max_output_tokens"}, error=None,
        usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14),
    )
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs.get("model"))
        return _adapter_response(partial, streamed=True)

    monkeypatch.setattr("agent.context_compressor.call_llm", fake_call_llm)
    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **k: 100000)
    compressor = ContextCompressor(
        model="main-model", quiet_mode=True, tail_mode="legacy", protect_first_n=2, protect_last_n=2,
        abort_on_summary_failure=False,
    )
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"task {i} " + "x" * 50}
        for i in range(12)
    ]
    original = [dict(message) for message in messages]
    result = compressor.compress(messages, current_tokens=999999, force=True)

    assert calls == [None]
    assert result == original and messages == original
    assert compressor._previous_summary is None
    assert compressor._last_summary_truncated_failure is True
    assert compressor._last_compress_aborted is True


_TOOL_CALL = SimpleNamespace(
    type="function_call", status="completed", call_id="call_1", name="inspect", arguments='{"path":"x"}',
)
_LEAK_LIKE_ANSWER = 'Summary of work.\nRunning the tests next.\n{"cmd": "pytest -q"}'
_PARTIAL_TOOL_CALL = SimpleNamespace(
    type="function_call", status="in_progress", call_id="call_2", name="inspect", arguments='{"pa',
)
_CODEX_REASONING_ONLY = SimpleNamespace(
    type="reasoning", id="rs_test", encrypted_content=None, summary=[SimpleNamespace(text="still thinking")],
)


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
@pytest.mark.parametrize(
    "base_url, status, output, reason, expected_content, expected_finish",
    [
        pytest.param("", "completed", [_message("FINAL")], None, "FINAL", "stop", id="final-answer"),
        pytest.param("", "completed", [_message("COMMENTARY", phase="commentary")], None, None, "length", id="commentary-only"),
        pytest.param("", "completed", [_TOOL_CALL], None, None, "tool_calls", id="tool-call"),
        pytest.param("", "completed", [_message(_LEAK_LIKE_ANSWER, phase=None)], None, _LEAK_LIKE_ANSWER, "stop",
                     id="tool-call-like-answer"),
        pytest.param("", "completed", [_message(_LEAK_LIKE_ANSWER, phase=None), _PARTIAL_TOOL_CALL], None, None,
                     "length", id="tool-call-like-partial-item"),
        # A str ``output`` is delivered only via output_text (empty output list), as streamed Codex answers can be.
        pytest.param("", "completed", _LEAK_LIKE_ANSWER, None, _LEAK_LIKE_ANSWER, "stop", id="tool-call-like-output-text"),
        pytest.param("", "incomplete", [_message("PARTIAL")], "max_output_tokens", "PARTIAL", "length", id="token-cap"),
        pytest.param("", "incomplete", [_TOOL_CALL], "max_output_tokens", None, "tool_calls", id="token-cap-after-tool-call"),
        # Route-sensitive normalization: the issuer comes from the request's route classification.
        pytest.param("https://chatgpt.com/backend-api/codex", "completed", [_CODEX_REASONING_ONLY], None, None, "length",
                     id="codex-reasoning-only"),
    ],
)
def test_actual_adapter_preserves_completion_contract(
    streamed, base_url, status, output, reason, expected_content, expected_finish,
):
    final = SimpleNamespace(
        status=status, output=[] if isinstance(output, str) else output,
        output_text=output if isinstance(output, str) else "",
        incomplete_details={"reason": reason} if reason else None, error=None,
        usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
    )

    choice = _adapter_response(final, streamed=streamed, base_url=base_url).choices[0]

    assert (choice.message.content, choice.finish_reason) == (expected_content, expected_finish)
    if expected_finish == "tool_calls":
        call, = choice.message.tool_calls
        assert (call.id, call.function.name, call.function.arguments) == ("call_1", "inspect", '{"path":"x"}')
