"""Prompt-cache prewarm — pay the provider-side cache write before turn 1.

The first API call of a fresh session ingests the entire uncached prefix
(system prompt + tool schemas — commonly 50-70k tokens), which shows up as
10-20s of first-message latency on Anthropic-cached routes. Every later call
rides the prompt cache and drops to TTFT + generation.

``prewarm_prompt_cache`` issues one minimal, non-streaming request — same
tools, same system prompt, same thinking/effort/max_tokens (byte-parity
with the first real turn; the provider's cache key covers the rendered
request configuration, so ANY divergence risks warming the wrong prefix),
with a trivial trailing user message engineered for a few-token reply —
whose only purpose is the provider-side cache write at the injected
``cache_control`` breakpoints. The first real user turn then *reads* the
prefix cache instead of writing it cold.

Cost note: the cache write (1.25x input for the 5m TTL) is paid by the
first real call today anyway — prewarming only moves it earlier. The extra
spend is one cache *read* (0.1x) on the first real turn, plus the full
write being wasted when a session is opened but never used. That trade-off
is why the feature is config-gated (``agent.prewarm_prompt_cache``,
default off).

Everything here is fail-open: a prewarm failure must never surface to the
user or block the session — the worst case is the status quo (cold first
turn).

Pure helper — no threads. Callers (the TUI/desktop gateway) own scheduling.
"""

import logging
import time

logger = logging.getLogger(__name__)

# Transports whose request shape we can safely reproduce outside the main
# conversation loop. Codex/Responses, Bedrock Converse, ACP, and the MoA
# facade have bespoke client/stream ownership — and (except Bedrock) no
# Anthropic cache_control semantics — so they are excluded.
_PREWARM_API_MODES = {"chat_completions", "anthropic_messages"}


def prewarm_supported(agent) -> bool:
    """True when a prewarm request would actually warm a provider cache."""
    if not getattr(agent, "_use_prompt_caching", False):
        return False
    if getattr(agent, "api_mode", None) not in _PREWARM_API_MODES:
        return False
    if getattr(agent, "provider", None) == "moa":
        return False
    return True


def prewarm_prompt_cache(agent) -> bool:
    """Issue one minimal request so the provider writes the prompt-prefix cache.

    Builds the exact prefix the first real turn will send — same tool
    schemas (``agent.tools`` via ``_build_api_kwargs``), same system prompt
    with the same ``[static, volatile]`` cache_control layout — followed by
    a throwaway user message. The trailing user message differs from the
    real first message, but Anthropic caching is prefix-based: the
    breakpoints on the static system prefix and full system prompt land
    identically, which is where the tens of thousands of tokens live.

    Returns True when the prewarm request completed, False when skipped or
    failed. Never raises.
    """
    if not prewarm_supported(agent):
        return False

    try:
        system_prompt = getattr(agent, "_cached_system_prompt", None)
        if not system_prompt:
            # Same builder the first turn uses; also populates
            # ``_cached_system_prompt_static`` as a side effect. The real
            # turn rebuilds through ``_restore_or_build_system_prompt`` —
            # only the volatile tail (timestamp) can differ, and that sits
            # after the static-prefix breakpoint.
            system_prompt = agent._build_system_prompt()
        static_prefix = getattr(agent, "_cached_system_prompt_static", None)

        from agent.prompt_caching import apply_anthropic_cache_control

        api_messages = apply_anthropic_cache_control(
            [
                {"role": "system", "content": system_prompt},
                # Engineered for minimal output: adaptive thinking skips
                # reasoning on trivial prompts, so the throwaway completion
                # costs a handful of output tokens (observed: 4).
                {"role": "user", "content": "Reply with only the word: ok"},
            ],
            cache_ttl=getattr(agent, "_cache_ttl", "5m") or "5m",
            native_anthropic=getattr(agent, "_use_native_cache_layout", False),
            static_system_prefix=(
                static_prefix if isinstance(static_prefix, str) else None
            ),
        )

        api_kwargs = agent._build_api_kwargs(api_messages)
        api_kwargs.pop("stream", None)
        # ── Byte-parity with the FIRST REAL TURN is the whole game ──────
        # The provider's cache key covers the rendered request prefix, and
        # on Claude 4.6+/5-series the thinking configuration and resolved
        # effort are rendered into the prompt (per Anthropic's caching
        # docs, a thinking/effort change can invalidate the SYSTEM and
        # TOOLS breakpoints too). Empirically the rendered configuration
        # also incorporates ``max_tokens``: an A/B against the live nous /
        # claude-fable-5 route showed two byte-identical requests hit the
        # cache, while changing ONLY max_tokens (1 vs 128000) missed it
        # entirely. So — unlike the usual throwaway-request idiom — this
        # request must NOT cap max_tokens or strip thinking. Everything
        # except the trailing user message stays exactly as
        # ``_build_api_kwargs`` produced it, i.e. exactly what the first
        # real turn will send. Output cost is bounded by the trivial
        # prompt above, not by an output cap.
        #
        # One exception: MANUAL extended thinking (``thinking.type ==
        # "enabled"`` with ``budget_tokens``, legacy pre-4.6 models). There
        # the model ALWAYS burns thinking tokens — even on a trivial prompt
        # — which would make the prewarm cost real money in output. Strip
        # the thinking knobs and cap output to 1 token instead: legacy
        # manual-mode models are exactly the models whose system/tools
        # cache blocks are documented to survive thinking-parameter
        # changes, so the strip is cache-safe there (and only there).
        _thinking = api_kwargs.get("thinking")
        _manual_thinking = (
            isinstance(_thinking, dict) and _thinking.get("type") == "enabled"
        )
        if _manual_thinking:
            api_kwargs.pop("thinking", None)
            # Manual-mode kwargs also force temperature=1 and inflate
            # max_tokens past the budget (build_kwargs) — undo both so the
            # request stays a 1-token no-op.
            api_kwargs.pop("temperature", None)
            if "max_completion_tokens" in api_kwargs:
                api_kwargs["max_completion_tokens"] = 1
            else:
                api_kwargs["max_tokens"] = 1

        from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

        created: list[tuple[object, str]] = []

        def _make_client(reason: str, kind: str = "openai"):
            client = (
                agent._create_request_anthropic_client(reason=reason)
                if kind == "anthropic_messages"
                else agent._create_request_openai_client(
                    reason=reason, api_kwargs=api_kwargs
                )
            )
            created.append((client, kind))
            return client

        started = time.monotonic()
        try:
            _dispatch_nonstreaming_api_request(
                agent, api_kwargs, make_client=_make_client
            )
        finally:
            for client, kind in created:
                try:
                    if kind == "anthropic_messages":
                        close = getattr(client, "close", None)
                        if callable(close):
                            close()
                    else:
                        agent._close_request_openai_client(
                            client, reason="prompt_prewarm"
                        )
                except Exception:
                    logger.debug(
                        "prompt prewarm client close failed", exc_info=True
                    )

        logger.info(
            "Prompt-cache prewarm complete: model=%s provider=%s %.1fs",
            getattr(agent, "model", "") or "",
            getattr(agent, "provider", "") or "",
            time.monotonic() - started,
        )
        # Hand the exact sent bytes to the first real turn:
        # ``_restore_or_build_system_prompt`` adopts this instead of
        # rebuilding, so the volatile tail (timestamp) can't drift between
        # the prewarmed prefix and the first real request.
        agent._prewarmed_system_prompt = system_prompt
        return True
    except Exception:
        # Fail-open by contract: a failed prewarm just means the first real
        # turn pays the cache write itself (the status quo).
        logger.info("Prompt-cache prewarm failed (fail-open)", exc_info=True)
        return False
