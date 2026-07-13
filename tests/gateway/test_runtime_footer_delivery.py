"""Regression coverage for runtime-footer delivery after streaming."""

from __future__ import annotations

import inspect

from gateway.run import GatewayRunner


def test_stream_delivery_state_is_finalized_inside_finally_before_return_cleanup():
    """Early returns must still publish `already_sent` to the footer caller.

    A return inside `_run_agent_inner` evaluates its value before `finally` and
    skips code after the finally block. Delivery state therefore has to be
    finalized inside `finally`, after the stream consumer flushes and before
    cleanup. Otherwise Discord receives the streamed body, while the footer is
    appended to a second response that the outer send path suppresses.
    """
    source = inspect.getsource(GatewayRunner._run_agent_inner)

    wait_marker = "# Wait for stream consumer to finish its final edit"
    finalize_marker = "await _finalize_stream_delivery_state(_delivery_candidate)"
    propagation_marker = '_held_agent_result["already_sent"] = True'
    cleanup_marker = "# Clean up tracking"

    assert source.count(finalize_marker) == 1
    assert source.index(wait_marker) < source.index(finalize_marker)
    assert source.index(finalize_marker) < source.index(propagation_marker)
    assert source.index(propagation_marker) < source.index(cleanup_marker)


def test_finally_mutation_reaches_an_already_evaluated_mutable_return():
    """Pin the Python return/finally contract relied on by early branches."""
    original_result: dict[str, object] = {"final_response": "done"}
    enriched_response: dict[str, object] = {"final_response": "done"}

    def early_return():
        try:
            return original_result
        finally:
            enriched_response["already_sent"] = True
            original_result["already_sent"] = enriched_response["already_sent"]

    returned = early_return()

    assert returned is original_result
    assert returned["already_sent"] is True


def test_runtime_footer_has_a_trailing_send_for_streamed_responses():
    source = inspect.getsource(GatewayRunner._handle_message_with_agent)

    already_sent_gate = (
        'if agent_result.get("already_sent") and not agent_result.get("failed"):'
    )
    footer_send = "await _foot_adapter.send("

    assert already_sent_gate in source
    gated_source = source[source.index(already_sent_gate):]
    assert footer_send in gated_source
    assert gated_source.index(footer_send) < gated_source.index("return None")
