"""Regression coverage for provider hard message-count limits."""

from unittest.mock import Mock, patch

from agent.context_compressor import ContextCompressor


def test_compression_forces_boundary_below_gateway_message_limit():
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=272_000,
    ):
        compressor = ContextCompressor(
            model="subscriptions-quality",
            threshold_percent=0.55,
            summary_target_ratio=0.20,
            protect_first_n=3,
            protect_last_n=20,
            min_tail_user_messages=1,
            quiet_mode=True,
            config_context_length=272_000,
        )
    compressor._generate_summary = lambda *args, **kwargs: "Earlier work summary"

    messages = [{"role": "user", "content": "start"}]
    for index in range(401):
        messages.extend(
            [
                {"role": "assistant", "content": f"result {index}"},
                {"role": "user", "content": f"background event {index}"},
            ]
        )
    assert len(messages) == 803

    compressed = compressor.compress(messages, current_tokens=130_000)

    assert len(compressed) < 600
    assert any("Earlier work summary" in str(row.get("content")) for row in compressed)


def test_message_count_pressure_bypasses_feasibility_skip():
    """Hard row pressure must not be defeated by a token-light middle window."""
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=272_000,
    ):
        compressor = ContextCompressor(
            model="subscriptions-quality",
            threshold_percent=0.55,
            summary_target_ratio=0.20,
            protect_first_n=3,
            protect_last_n=20,
            min_tail_user_messages=1,
            quiet_mode=True,
            config_context_length=272_000,
        )

    # A prior ineffective token compression enables the feasibility gate. The
    # 805-row transcript is intentionally token-light, reproducing the review
    # case where that gate used to override the hard message-count boundary.
    compressor._ineffective_compression_count = 1
    generate_summary = Mock(return_value="Earlier work summary")
    compressor._generate_summary = generate_summary

    messages = [{"role": "user", "content": "start"}]
    for index in range(402):
        messages.extend(
            [
                {"role": "assistant", "content": f"ok {index}"},
                {"role": "user", "content": f"event {index}"},
            ]
        )
    assert len(messages) == 805

    compressed = compressor.compress(messages, current_tokens=155_913)

    assert len(compressed) < 600
    generate_summary.assert_called_once()
    assert compressor._last_feasibility_skip is False
