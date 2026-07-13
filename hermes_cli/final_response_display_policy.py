"""Classic-CLI final response display decisions.

The classic CLI has two independent answer surfaces during a turn:

* live token callbacks (draft/provisional text), and
* the canonical ``final_response`` returned by the agent finalizer.

This module keeps the policy small and testable so future rebases do not
accidentally show both as full assistant answers for one user turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.route_depth_bar import strip_route_depth_bar


@dataclass(frozen=True)
class FinalResponseDisplayDecision:
    """How the CLI should surface the canonical final response."""

    render_panel: bool
    close_streaming_tts_box: bool = False
    reason: str = "render_final_response"


def assistant_body_for_tts(response: str) -> str:
    """Return answer body text without the runtime-only route/status bar."""

    return strip_route_depth_bar(response or "")


def decide_final_response_display(
    *,
    response: str,
    response_previewed: bool,
    response_transformed: bool,
    stream_text_received: bool,
    stream_text_visible: bool,
    is_error_response: bool,
    use_streaming_tts: bool,
    streaming_tts_box_opened: bool,
) -> FinalResponseDisplayDecision:
    """Return the display action for one completed classic-CLI turn.

    ``stream_text_received`` is intentionally separate from
    ``stream_text_visible``: final-only UIs may receive draft assistant body
    deltas for partial-stream recovery while choosing not to render them.
    Only actually visible untransformed answer-body text can suppress the
    canonical final panel.
    """

    if not response:
        return FinalResponseDisplayDecision(False, reason="empty_response")

    # Runtime/plugin/finalizer transforms (including the route/depth bar)
    # make the returned final_response canonical. It must be rendered even if
    # raw text was previewed or spoken while streaming; otherwise terminal and
    # persisted final content diverge. The caller closes any open streaming-TTS
    # frame before rendering the canonical panel.
    if response_transformed:
        return FinalResponseDisplayDecision(
            True,
            close_streaming_tts_box=(
                use_streaming_tts
                and streaming_tts_box_opened
                and not is_error_response
            ),
            reason="transformed_final_response",
        )

    if response_previewed and stream_text_visible:
        return FinalResponseDisplayDecision(False, reason="previewed")

    if use_streaming_tts and streaming_tts_box_opened and not is_error_response:
        return FinalResponseDisplayDecision(
            False,
            close_streaming_tts_box=True,
            reason="streaming_tts_visible",
        )

    if is_error_response:
        return FinalResponseDisplayDecision(True, reason="error_response")

    if stream_text_visible:
        return FinalResponseDisplayDecision(False, reason="already_visible_stream")

    if stream_text_received:
        return FinalResponseDisplayDecision(True, reason="not_visible")

    return FinalResponseDisplayDecision(True, reason="render_final_response")
