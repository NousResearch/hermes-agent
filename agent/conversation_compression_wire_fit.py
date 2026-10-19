"""Send-time context-window fit checks, measured on the request as it goes on the wire.

The compaction trigger and the provider-overflow recovery both act on estimates of the
transcript. These helpers price the request the provider actually receives (``api_content``
replayed, persistence/trajectory keys and stale thinking dropped), reserve room to answer, and
scale the char/4 estimate by the provider's own ``prompt_tokens`` once it has priced a request.
They back three decisions that must agree: the preflight / post-tool forced compression
(``wire_overflow_reenters_compression``), the send-time guard in
``_ensure_request_fits_window`` (run by ``turn_api_request.build_api_request``), and the calibration read in
``turn_usage.record_response_usage``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Keys that live on in-memory/persisted rows but never reach the provider: ``reasoning`` is
# trajectory-only (build_api_messages pops it after copying it to ``reasoning_content``),
# ``finish_reason`` and the length-continuation marks are popped there too.
_NON_WIRE_MESSAGE_KEYS = frozenset(
    {"api_content", "reasoning", "finish_reason", "_length_continuation_fragment",
     "_length_continuation_nudge"}
)


def _wire_shape_for_estimate(agent: Any, messages: Any) -> list:
    """Shallow copies of ``messages`` reduced to what the provider receives, for sizing only.

    Accepts raw ``messages`` (persisted rows) or assembled ``api_messages``: replays the
    ``api_content`` sidecar as content (the exact bytes sent), drops persistence-only and
    trajectory-only keys, and — on routes that do not replay stale thinking — drops
    ``reasoning``/``reasoning_content`` of every assistant row but the newest, exactly like the
    compaction-trigger estimator. Without this a reasoning-heavy session prices every turn's
    thinking twice (``reasoning`` + ``reasoning_content``) and looks 2-5x its wire size."""
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS
    from agent.model_metadata import _strip_stale_thinking_for_estimate
    from agent.turn_context import _agent_stale_thinking_on_wire

    shaped = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            shaped.append(msg)
            continue
        out = {k: v for k, v in msg.items()
               if k not in _NON_WIRE_MESSAGE_KEYS and k not in PERSISTENCE_ONLY_MESSAGE_FIELDS}
        api_content = msg.get("api_content")
        if isinstance(api_content, str) and api_content and msg.get("role") in ("user", "assistant"):
            out["content"] = api_content
        if msg.get("role") == "assistant" and msg.get("reasoning") and not msg.get("reasoning_content"):
            out["reasoning_content"] = msg["reasoning"]  # mirrors _copy_reasoning_content_for_api
        shaped.append(out)
    if not _agent_stale_thinking_on_wire(agent):
        shaped = _strip_stale_thinking_for_estimate(shaped)
    return shaped


def wire_request_tokens(
    agent: Any, messages: Any, tools: Any = None, *, system_prompt: Any = None,
) -> int:
    """Estimated size of the request as it goes on the wire: the wire shape of ``messages``
    (see ``_wire_shape_for_estimate``), the system prompt when ``messages`` does not already
    lead with one, and the tool schemas. The single measure shared by the preflight / post-tool
    forced-compression decision and the assembly-time fallback, so they cannot disagree."""
    from agent.chat_completion_helpers import estimate_request_context_tokens

    shaped = _wire_shape_for_estimate(agent, messages)
    if (
        isinstance(system_prompt, str) and system_prompt
        and not (shaped and isinstance(shaped[0], dict) and shaped[0].get("role") == "system")
    ):
        shaped = [{"role": "system", "content": system_prompt}] + shaped
    return estimate_request_context_tokens({"messages": shaped, "tools": tools} if tools else shaped)


# Minimum room to answer the send-time fit check always leaves, even with no max_tokens
# configured. Without it a request using 130,902 of a 131,072 window "fit" (budget == window)
# and the provider stopped at once with "Context window full ... leaving no room to answer".
# 8K tokens is the same default output reservation MoA reference calls use
# (moa_loop._REFERENCE_DEFAULT_OUTPUT_RESERVE) and the native-compaction safety margin: room
# for a tool call or a short answer plus some thinking. Capped at 1/8 of tiny windows.
_MIN_ANSWER_RESERVE_TOKENS = 8_192
# The wire measure is a char/4 estimate; real tokenizers count dense code/JSON higher. Keep a
# window-proportional margin for that error (one observed session: rough 24,427 vs provider
# 24,131, i.e. char/4 within ~1-2% on that model; 5% leaves room for denser content).
_WIRE_ESTIMATE_SAFETY_RATIO = 0.05


def request_input_budget(agent: Any) -> Optional[int]:
    """Usable input budget for the wire fit check, ``None`` when the window is unknown:
    window minus the output reservation minus an estimate safety margin. The reservation is
    ``agent.max_tokens``, else the compressor's (which carries adapter defaults such as native
    Gemini's), but never less than ``_MIN_ANSWER_RESERVE_TOKENS`` (capped at 1/8 of the
    window), so an unset max_tokens can no longer mean "the whole window may be prompt"."""
    compressor = getattr(agent, "context_compressor", None)
    window = getattr(compressor, "context_length", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return None
    reserve = getattr(agent, "max_tokens", None)
    if reserve is None:
        reserve = getattr(compressor, "max_tokens", None)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        reserve = 0
    reserve = max(reserve, min(_MIN_ANSWER_RESERVE_TOKENS, window // 8))
    budget = window - reserve - int(window * _WIRE_ESTIMATE_SAFETY_RATIO)
    # A reservation that swallows the window (max_tokens >= window) must not make every
    # request "un-fittable": fall back to the minimum reserve, as _effective_input_window does.
    if budget <= 0:
        budget = window - min(_MIN_ANSWER_RESERVE_TOKENS, window // 8) - int(
            window * _WIRE_ESTIMATE_SAFETY_RATIO)
    return max(1, budget)


# Provider-calibrated wire estimate. The char/4 wire measure under-counts dense content: in
# one observed local-model session (source files with line-number gutters) every request was
# ~1.17x its estimate (est 100,376 -> provider 117,523; est 113,292 -> provider 131,808), so a
# 131,808-token request "fit" the 116,327 budget and was sent into a 400. The provider prices
# each request exactly, so the ratio of its prompt_tokens to the estimate of the request that
# was sent corrects the next estimate. Only ever scales UP (never trusts a lower count than
# char/4) and is bounded so one odd usage reading cannot make every request "un-fittable".
_WIRE_CALIBRATION_MAX = 2.0
# Ratios outside this band are not a tokenizer-density reading (MoA-folded usage, a stale
# estimate from another request, a provider that reports cached tokens separately ...).
_WIRE_CALIBRATION_SANE = (0.5, 3.0)


def _positive_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _calibration_key(agent: Any) -> tuple:
    # Tokenizer density is per model: a fallback to another model must not inherit the ratio.
    return (str(getattr(agent, "provider", "") or ""), str(getattr(agent, "model", "") or ""))


def wire_estimate_calibration(agent: Any) -> float:
    """Factor (>= 1.0) applied to ``wire_request_tokens`` by the fit checks: the last
    provider-reported prompt size divided by the estimate of the request it priced (same
    provider/model only)."""
    stored = getattr(agent, "_wire_estimate_ratio", None)
    if not isinstance(stored, tuple) or len(stored) != 2 or stored[0] != _calibration_key(agent):
        return 1.0
    ratio = _positive_number(stored[1])
    if ratio is None:
        return 1.0
    return min(max(ratio, 1.0), _WIRE_CALIBRATION_MAX)


def note_wire_estimate_sent(agent: Any, estimate: int) -> None:
    """Remember the (uncalibrated) estimate of the request about to be sent, so the usage of
    its response can calibrate the next fit check."""
    agent._wire_estimate_sent = int(estimate)


def calibrate_wire_estimate_from_usage(agent: Any, prompt_tokens: Any) -> Optional[float]:
    """Fold the provider's exact ``prompt_tokens`` for the last sent request into the wire
    estimate calibration. Returns the new raw ratio, or ``None`` when nothing was learned."""
    sent = _positive_number(getattr(agent, "_wire_estimate_sent", None))
    actual = _positive_number(prompt_tokens)
    if sent is None or actual is None:
        return None
    agent._wire_estimate_sent = None  # one reading per sent request
    ratio = actual / sent
    lo, hi = _WIRE_CALIBRATION_SANE
    if not lo <= ratio <= hi:
        return None
    agent._wire_estimate_ratio = (_calibration_key(agent), ratio)
    return ratio


def calibrated_wire_request_tokens(
    agent: Any, messages: Any, tools: Any = None, *, system_prompt: Any = None,
) -> int:
    """``wire_request_tokens`` scaled by the provider calibration (see above)."""
    raw = wire_request_tokens(agent, messages, tools, system_prompt=system_prompt)
    factor = wire_estimate_calibration(agent)
    return raw if factor <= 1.0 else math.ceil(raw * factor)


def request_send_budget(agent: Any) -> Optional[int]:
    """Input budget for the SEND-TIME window guard: window minus the minimum answer reserve
    (``_MIN_ANSWER_RESERVE_TOKENS``, capped at 1/8 of the window) minus the estimate margin;
    ``None`` when the window is unknown. Unlike ``request_input_budget`` it does not reserve a
    configured ``max_tokens``: a prompt that fits but leaves less than ``max_tokens`` is sendable,
    because the provider's output-cap error ("max_tokens > available_tokens") is recovered by
    shrinking max_tokens (turn_overflow._clamp_output_cap). With no max_tokens (llama.cpp /
    custom routes) both budgets are the same."""
    compressor = getattr(agent, "context_compressor", None)
    window = getattr(compressor, "context_length", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return None
    reserve = min(_MIN_ANSWER_RESERVE_TOKENS, window // 8)
    return max(1, window - reserve - int(window * _WIRE_ESTIMATE_SAFETY_RATIO))


def wire_request_cannot_fit(
    agent: Any, messages: Any, tools: Any = None, *, system_prompt: Any = None,
) -> Optional[bool]:
    """True when the wire request (``calibrated_wire_request_tokens``) exceeds the input
    budget (window minus output reservation) — the provider would 400 non-retryably ("prompt
    leaves no room to answer"). ``None`` when the window is unknown. Only this verdict may
    bypass the summary-failure cooldown; it measures what is actually sent, never stale
    thinking or bookkeeping fields the wire drops."""
    budget = request_input_budget(agent)
    if budget is None:
        return None
    return calibrated_wire_request_tokens(agent, messages, tools, system_prompt=system_prompt) > budget


def wire_overflow_reenters_compression(
    agent: Any, cooldown: Any, messages: Any, *, system_prompt: Any = None,
    api_messages: Any = None, api_tools: Any = None,
) -> tuple[Optional[bool], Any]:
    """``(wire_overflow, cooldown)`` for a compression gate.

    Measures the assembled request (``api_messages`` / ``api_tools``) when the loop supplies
    it, else the wire shape of ``messages`` plus ``system_prompt`` and the agent's tools. When
    the request provably cannot fit (``wire_request_cannot_fit`` is True) an armed
    summary-failure cooldown comes back as ``None``, so the gate re-enters compression
    (with ``bypass_cooldown_kwargs``) instead of sending a request the provider must reject.
    Unknown or False verdicts leave the cooldown as it was."""
    if isinstance(api_messages, list) and api_messages:
        wire_overflow = wire_request_cannot_fit(agent, api_messages, api_tools)
    else:
        wire_overflow = wire_request_cannot_fit(
            agent, messages, getattr(agent, "tools", None), system_prompt=system_prompt)
    if cooldown and wire_overflow is True:
        logger.info(
            "Compression cooldown active but the wire request cannot fit the model "
            "window: re-entering compression instead of sending a doomed request",
        )
        cooldown = None
    return wire_overflow, cooldown


def bypass_cooldown_kwargs(wire_overflow: Optional[bool]) -> dict[str, bool]:
    """``_compress_context`` kwargs for a gate that re-entered on a wire overflow: one real
    attempt while the summary-failure cooldown stays armed. Empty otherwise, so overrides and
    test doubles with the original signature keep working."""
    return {"bypass_cooldown": True} if wire_overflow is True else {}


class RequestExceedsContextWindowError(RuntimeError):
    """The assembled request cannot fit the model window even with every tool output
    cleared. Raised at send time so the request is never sent (the provider would answer
    HTTP 400 "prompt leaves no room to answer" non-retryably)."""

    def __init__(self, estimate: int, budget: int, window: Any):
        self.estimate, self.budget, self.window = estimate, budget, window
        super().__init__(
            f"Request does not fit the model context window: ~{estimate:,} tokens vs a "
            f"{budget:,}-token input budget (window {window:,} minus answer reserve and margin) "
            "even after compression and clearing every tool output; not sent."
            if isinstance(window, int) else
            f"Request does not fit the model context window: ~{estimate:,} tokens vs a "
            f"{budget:,}-token input budget; not sent."
        )


def _ensure_request_fits_window(agent: Any, api_messages: list, tools_for_api: Any) -> list:
    """Send-time window guard: never send a request the model cannot accept.

    Runs in ``build_api_request`` for EVERY attempt (first send and each retry), after the
    preflight gate had its chance to compress (it forces compression with the same measure,
    bypassing the summary-failure cooldown). Measures the assembled payload with the shared
    provider-calibrated measure (``calibrated_wire_request_tokens`` vs
    ``request_send_budget``: window minus answer reserve minus estimate margin). When it does
    not fit — compression unavailable, failed or aborted — clear the OLDEST tool outputs with
    ``_PRUNED_TOOL_PLACEHOLDER`` until it does. The system prompt and every user message are
    never touched; only role=="tool" rows shrink, oldest first, on copies (the transcript is
    not modified). Still over budget with every tool output cleared: raise
    :class:`RequestExceedsContextWindowError` instead of sending. Records the uncalibrated
    estimate of what is sent so the response usage can calibrate the next check.
    """
    from agent.context_compressor import _PRUNED_TOOL_PLACEHOLDER

    if not isinstance(api_messages, list):
        return api_messages
    factor = wire_estimate_calibration(agent)

    def _measure(msgs: list) -> tuple:
        raw = wire_request_tokens(agent, msgs, tools_for_api)
        return raw, (raw if factor <= 1.0 else math.ceil(raw * factor))

    raw, estimate = _measure(api_messages)
    budget = request_send_budget(agent)
    if budget is not None and estimate > budget:
        first_estimate = estimate
        out = list(api_messages)
        cleared = 0
        for i, msg in enumerate(out):
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not content or content == _PRUNED_TOOL_PLACEHOLDER or not isinstance(content, (str, list)):
                continue
            out[i] = {**msg, "content": _PRUNED_TOOL_PLACEHOLDER}
            cleared += 1
            raw, estimate = _measure(out)
            if estimate <= budget:
                break
        window = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        if estimate > budget:
            logger.error(
                "Request would not fit the model window (~%s tokens, calibration x%.3f, vs %s "
                "input budget of window %s) even after clearing %s tool outputs: NOT sending it",
                f"{estimate:,}", factor, f"{budget:,}", window, cleared,
            )
            raise RequestExceedsContextWindowError(estimate, budget, window)
        logger.warning(
            "Request would not fit the model window (~%s tokens, calibration x%.3f, vs %s input "
            "budget of window %s); cleared the %s oldest tool outputs with the pruned "
            "placeholder -> ~%s tokens (deterministic fallback; system prompt and user "
            "messages untouched)",
            f"{first_estimate:,}", factor, f"{budget:,}", window, cleared, f"{estimate:,}",
        )
        api_messages = out
    note_wire_estimate_sent(agent, raw)
    return api_messages
