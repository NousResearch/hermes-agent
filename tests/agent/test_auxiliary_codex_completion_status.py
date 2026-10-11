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


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter", "insufficient_system_resource"])
def test_compressor_rejects_actual_adapter_partial_summary(monkeypatch, reason):
    partial = SimpleNamespace(
        status="incomplete", output=[_message("PARTIAL_CHECKPOINT")],
        incomplete_details={"reason": reason}, error=None,
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
        pytest.param("", "completed", [{"type": "message", "role": "assistant", "status": "completed",
                                        "content": [{"type": "output_text", "text": "DICT"}]}], None, "DICT", "stop",
                     id="dict-items"),
        pytest.param("", "completed", [_message(_LEAK_LIKE_ANSWER, phase=None)], None, _LEAK_LIKE_ANSWER, "stop",
                     id="tool-call-like-answer"),
        pytest.param("", "completed", [_message(_LEAK_LIKE_ANSWER, phase=None), _PARTIAL_TOOL_CALL], None,
                     _LEAK_LIKE_ANSWER, "length", id="tool-call-like-partial-item"),
        # A str ``output`` is delivered only via output_text (empty output list), as streamed Codex answers can be.
        pytest.param("", "completed", _LEAK_LIKE_ANSWER, None, _LEAK_LIKE_ANSWER, "stop", id="tool-call-like-output-text"),
        # (items, text): output_text beside a commentary item is narration, never the rescued answer.
        pytest.param("", "completed", ([_message("", phase="commentary")], _LEAK_LIKE_ANSWER), None, None, "length",
                     id="tool-call-like-output-text-beside-commentary"),
        pytest.param("", "incomplete", [_message("PARTIAL")], "max_output_tokens", "PARTIAL", "length", id="token-cap"),
        pytest.param("", "incomplete", [_TOOL_CALL], "max_output_tokens", None, "tool_calls", id="token-cap-after-tool-call"),
        # Compressor only rejects "length": a content-filtered partial must not be committed as a summary.
        pytest.param("", "incomplete", [_message("CF")], "content_filter", "CF", "length", id="content-filter"),
        # Commentary + unphased text is incomplete whether or not the unphased text looks like a tool call.
        pytest.param("", "completed", [_message("NOTE", phase="commentary"), _message(_LEAK_LIKE_ANSWER, phase=None)],
                     None, _LEAK_LIKE_ANSWER, "length", id="tool-call-like-beside-commentary"),
        # Route-sensitive normalization: the issuer comes from the request's route classification.
        pytest.param("https://chatgpt.com/backend-api/codex", "completed", [_CODEX_REASONING_ONLY], None, None, "length",
                     id="codex-reasoning-only"),
    ],
)
def test_actual_adapter_preserves_completion_contract(
    streamed, base_url, status, output, reason, expected_content, expected_finish,
):
    items, output_text = output if isinstance(output, tuple) else ([], output) if isinstance(output, str) else (output, "")
    final = SimpleNamespace(
        status=status, output=items, output_text=output_text,
        incomplete_details={"reason": reason} if reason else None, error=None,
        usage={"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
    )

    choice = _adapter_response(final, streamed=streamed, base_url=base_url).choices[0]

    assert (choice.message.content, choice.finish_reason) == (expected_content, expected_finish)
    if expected_finish == "tool_calls":
        call, = choice.message.tool_calls
        assert (call.id, call.function.name, call.function.arguments) == ("call_1", "inspect", '{"path":"x"}')


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
@pytest.mark.parametrize("details_object", [False, True], ids=["dict-details", "object-details"])
@pytest.mark.parametrize("status, reason, tool_call, finish", [
    ("incomplete", "content_filter", False, "length"),
    ("incomplete", "max_output_tokens", False, "length"),
    ("incomplete", "insufficient_system_resource", False, "length"),
    ("incomplete", None, False, "length"),
    ("incomplete", "max_output_tokens", True, "tool_calls"),
    ("completed", None, False, "stop"),
])
def test_actual_adapter_retains_native_completion_cause(
    streamed, details_object, status, reason, tool_call, finish,
):
    details = None
    if reason is not None:
        details = SimpleNamespace(reason=reason) if details_object else {"reason": reason}
    final = SimpleNamespace(
        id="resp_native", model="served-model", status=status, incomplete_details=details, error=None,
        output=[_TOOL_CALL] if tool_call else [_message("PARTIAL")],
        usage=SimpleNamespace(input_tokens=11, output_tokens=3, total_tokens=14),
    )

    response = _adapter_response(final, streamed=streamed)
    choice, = response.choices

    assert {
        "id": response.id, "model": response.model,
        "status": getattr(response, "status", None),
        "incomplete_details": getattr(response, "incomplete_details", None),
        "content": choice.message.content, "finish_reason": choice.finish_reason,
        "tools": [(call.id, call.function.name, call.function.arguments) for call in choice.message.tool_calls or []],
        "usage": response.usage,
    } == {
        "id": final.id, "model": final.model, "status": status, "incomplete_details": details,
        "content": None if tool_call else "PARTIAL", "finish_reason": finish,
        "tools": [("call_1", "inspect", '{"path":"x"}')] if tool_call else [],
        "usage": SimpleNamespace(prompt_tokens=11, completion_tokens=3, total_tokens=14),
    }
