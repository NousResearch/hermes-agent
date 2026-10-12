"""Compose gateway display before sealing the authoritative stream final."""

from gateway.media_repair import repair_explicit_computer_use_media_paths


def finish_stream_consumer(runner, ctx, result, agent_history, stream_consumer):
    # Canonicalize a model-emitted computer-use screenshot path at the common result boundary so
    # the streaming finalizer and the non-streaming delivery path see the same response.
    if isinstance(result, dict) and isinstance(result.get("final_response"), str):
        result["final_response"] = repair_explicit_computer_use_media_paths(
            result["final_response"], result.get("messages", []), history_offset=len(agent_history),
        )
    ctx.result_holder[0] = result
    if stream_consumer is None:
        return
    # Pass final_response as the authoritative finalize payload: it includes post-stream
    # augmentation (verifier footer, explainer) the accumulator never saw. Adopt ONLY a genuinely
    # completed final: interrupt paths return {interrupted: True, completed: False} with a
    # DIAGNOSTIC final_response — adopting it would seal the partial answer over with the
    # diagnostic AND suppress the gateway's own error delivery.
    _final_for_stream = None
    if (
        isinstance(result, dict) and not result.get("failed") and not result.get("interrupted")
        and result.get("completed") is not False
    ):
        fr = result.get("final_response")
        if isinstance(fr, str) and fr.strip() and fr != "(empty)":
            if result.get("last_reasoning"):
                rendered = runner._hmwa_prepend_reasoning(
                    result, fr, ctx.source, runner._is_intentional_silence(result, fr),
                )
                if rendered != fr:
                    # Keep the streamed answer as a prefix, including sealed split heads.
                    # Reasoning is a trailing display block on streaming transports.
                    reasoning_block = rendered[:-len(fr)].rstrip()
                    result["_gateway_reasoning_rendered"] = True
                    result["final_response"] = fr = f"{fr}\n\n{reasoning_block}"
            _final_for_stream = fr
    if _final_for_stream is None:
        stream_consumer.finish()
        return
    # Duck-type safe: test doubles / older consumers may expose a zero-arg finish().
    try:
        stream_consumer.finish(_final_for_stream)
    except TypeError:
        stream_consumer.finish()
