"""Regression contract: full tool facts reach the single summary call before pruning."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _summary_response() -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = "## Historical Task Snapshot\nThe user asked to retain tool findings."
    response.choices[0].finish_reason = "stop"
    return response


def test_full_structured_tool_result_reaches_summary_before_pruning(monkeypatch):
    """Head, middle and tail facts must survive until the sole summary request.

    The payload includes a correction, a structured error and a structured result.  It is
    deliberately bigger than the pruning floor, so the old order reduced it to a stub before
    ``_generate_summary`` built its prompt.
    """
    monkeypatch.setattr(
        ContextCompressor,
        "_is_synthetic_compression_user_turn",
        classmethod(lambda cls, message: False),
    )
    compressor = ContextCompressor(
        model="test/model",
        config_context_length=100_000,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
    )
    compressor.tail_token_budget = 50
    payload = (
        '{"head_fact":"FACT_HEAD: responsible=Lina",'
        '"padding_a":"' + "a" * 900 + '",'
        '"correction":"FACT_MIDDLE: quantity_final=37 replaces 42",'
        '"padding_b":"' + "b" * 900 + '",'
        '"error":{"code":"FACT_ERROR: validation_humaine_obligatoire"},'
        '"padding_c":"' + "c" * 900 + '",'
        '"result":{"destination":"FACT_TAIL: depot-safran","status":"complete"}}'
    )
    messages = [
        {"role": "system", "content": "Synthetic S0 test."},
        {"role": "user", "content": "Read the public fixture and retain its findings."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "read-1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path":"fixture.json"}'},
        }]},
        {"role": "tool", "tool_call_id": "read-1", "tool_name": "read_file", "content": payload},
        {"role": "assistant", "content": "The fixture was read."},
        {"role": "user", "content": "Continue with the public test."},
        {"role": "assistant", "content": "Continuing."},
        {"role": "user", "content": "Keep the newest task visible."},
        {"role": "assistant", "content": "Acknowledged."},
        {"role": "user", "content": "This is the latest task."},
        {"role": "assistant", "content": "Latest reply."},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_summary_response()) as call:
        compressor.compress(messages, current_tokens=90_000)

    assert call.call_count == 1
    prompt = call.call_args.kwargs["messages"][0]["content"]
    for fact in (
        "FACT_HEAD: responsible=Lina",
        "FACT_MIDDLE: quantity_final=37 replaces 42",
        "FACT_ERROR: validation_humaine_obligatoire",
        "FACT_TAIL: depot-safran",
    ):
        assert fact in prompt
    assert "[read_file] read fixture.json" not in prompt
