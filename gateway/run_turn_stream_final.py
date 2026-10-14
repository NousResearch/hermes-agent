"""Compose the gateway display before sealing the authoritative stream final."""
import logging

from agent.i18n import t
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths

logger = logging.getLogger("gateway.run")

_REASONING_QUOTE_STYLES = {
    "subtext": ("gateway.reasoning.quote_label_discord", "-# ", "-#"),
    "blockquote": ("gateway.reasoning.quote_label_md", "> ", ">"),
}


def render_reasoning(runner, agent_result, response, source, _intentional_silence):
    """Prepend the last reasoning block when show_reasoning is on for this platform. Mattermost
    requires an explicit per-platform opt-in (scratch text, not final-answer content)."""
    if agent_result.get("_gateway_reasoning_rendered"):
        return response
    from gateway.run import _load_gateway_config, _platform_config_key, _resolve_gateway_display_bool
    try:
        _show_reasoning_effective = _resolve_gateway_display_bool(
            _load_gateway_config(), _platform_config_key(source.platform), "show_reasoning",
            default=bool(getattr(runner, "_show_reasoning", False)), platform=source.platform,
            require_platform_override_for={Platform.MATTERMOST},
        )
    except Exception:
        logger.debug("Reasoning display configuration unavailable; using platform default", exc_info=True)
        _show_reasoning_effective = (
            False if source.platform == Platform.MATTERMOST else getattr(runner, "_show_reasoning", False)
        )
    last_reasoning = agent_result.get("last_reasoning")
    if not (_show_reasoning_effective and response and not _intentional_silence and last_reasoning):
        return response
    from gateway.stream_consumer_fences import escape_code_fences_for_display
    # Collapse long reasoning to keep messages readable
    lines = last_reasoning.strip().splitlines()
    if len(lines) > 15:
        display_reasoning = "\n".join(lines[:15]) + t("gateway.reasoning.more_lines", count=len(lines) - 15)
    else:
        display_reasoning = last_reasoning.strip()
    # Per-platform render style: Discord defaults to "-# " subtext, others keep the code block.
    try:
        from gateway.display_config import resolve_display_setting
        _reasoning_style = resolve_display_setting(
            _load_gateway_config(), _platform_config_key(source.platform), "reasoning_style", "code",
        )
    except Exception:
        logger.debug("Reasoning style configuration unavailable; using code style", exc_info=True)
        _reasoning_style = "code"
    _quote = _REASONING_QUOTE_STYLES.get(_reasoning_style)
    if _quote:
        header_key, prefix, empty = _quote
        _quoted = "\n".join(f"{prefix}{ln}" if ln else empty for ln in display_reasoning.splitlines())
        return f"{t(header_key)}\n{_quoted}\n\n{response}"
    # Escape ``` inside reasoning so inner fences don't break the outer code block.
    display_reasoning = escape_code_fences_for_display(display_reasoning)
    return t("gateway.reasoning.block", reasoning=display_reasoning, response=response)


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

