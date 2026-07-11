"""Mixture-of-Agents runtime helpers for /moa turns.

The slash command is deliberately not a model tool. It marks one user turn as
MoA-enabled; the normal Hermes agent loop still owns tool calling and turn
termination, while this module gathers reference-model context before each model
iteration.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.auxiliary_client import call_llm
from agent.transports import get_transport

logger = logging.getLogger(__name__)

# Upper bound on concurrent reference-model calls. References are independent
# advisory calls (no tools, no inter-dependence), so we fan them out the same
# way delegate_task runs a batch: all in flight at once, results collected when
# every reference finishes. Presets rarely list more than a handful of
# references; this cap just protects against a pathologically large preset
# opening dozens of sockets at once.
_MAX_REFERENCE_WORKERS = 8


class _RefAccounting:
    """Per-reference token usage + estimated cost + full trace, carried as the
    third slot of a reference-output tuple.

    Kept as a tiny object (not a bare CanonicalUsage) because an advisor may
    run on a different model/provider than the aggregator, so its cost MUST be
    priced at its OWN model's rate — folding advisor tokens into the
    aggregator's usage and pricing the sum at the aggregator's rate would
    misprice every advisor. ``usage`` feeds accurate token counts;
    ``cost_usd`` feeds accurate cost.

    ``messages`` / ``output`` / ``model`` / ``provider`` / ``temperature``
    carry the FULL reference input and output for trace persistence (the
    display ``text`` is a truncated preview and is not enough to audit what an
    advisor actually saw). They are only populated when tracing is on; they add
    negligible cost otherwise.
    """

    __slots__ = (
        "usage",
        "cost_usd",
        "cost_status",
        "cost_source",
        "messages",
        "output",
        "model",
        "provider",
        "temperature",
        "succeeded",
    )

    def __init__(
        self,
        usage: Any,
        cost_usd: Any = None,
        cost_status: str | None = None,
        cost_source: str | None = None,
        *,
        messages: Any = None,
        output: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        temperature: Any = None,
        succeeded: bool = False,
    ):
        self.usage = usage
        self.cost_usd = cost_usd
        self.cost_status = cost_status
        self.cost_source = cost_source
        self.messages = messages
        self.output = output
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.succeeded = bool(succeeded)

# Per-tool-result character budget for the advisory reference view. Tool
# results can be huge (a full diff, a 5000-line file dump); replaying them
# verbatim per reference per tool-loop step would blow the reference model's
# context window and cost. We keep the agent's *actions* (tool calls) in full —
# they are cheap, high-signal, and tell the reference what the agent did — but
# preview each tool *result* head+tail so the reference still sees what came
# back without replaying megabytes. The acting aggregator always gets the full,
# untrimmed transcript; this budget only shapes the advisory copy.
_REFERENCE_TOOL_RESULT_BUDGET = 4000

# System prompt prepended to every reference-model call. References are
# advisory — they do NOT act, call tools, or own the task. Without this
# framing a reference receives the bare trimmed conversation and assumes it is
# the acting agent: it then refuses ("I can't access repositories / URLs from
# here") or tries to call tools it doesn't have. The prompt reframes the model
# as an analyst whose job is to reason about the presented state and hand its
# best thinking to the aggregator/orchestrator that will actually act.
_REFERENCE_SYSTEM_PROMPT = (
    "You are a reference advisor in a Mixture of Agents (MoA) process. You are "
    "NOT the acting agent and you do NOT execute anything: you cannot call "
    "tools, run commands, browse, or access files, repositories, or URLs, and "
    "you should not try to or apologize for being unable to. A separate "
    "aggregator/orchestrator model holds those capabilities and will take the "
    "actual actions.\n\n"
    "The conversation below is the current state of a task handled by that "
    "acting agent. Your job is to give your most intelligent analysis of that "
    "state: understand the goal, reason about the problem, and advise on what "
    "to do next. Surface the best approach, concrete next steps and tool-use "
    "strategy, likely pitfalls and risks, and anything the acting agent may "
    "have missed or gotten wrong. Assume any referenced files, URLs, or "
    "systems exist and reason about them from the context given rather than "
    "asking for access.\n\n"
    "Respond with your advice directly — no preamble, no disclaimers about "
    "tools or access. Your response is private guidance handed to the "
    "aggregator, not an answer shown to the user."
)



def _slot_label(slot: dict[str, str]) -> str:
    return f"{(slot.get('provider') or '').strip()}:{(slot.get('model') or '').strip()}"


def _slot_runtime(slot: dict[str, str]) -> dict[str, Any]:
    """Resolve a reference/aggregator slot to real runtime call kwargs.

    A MoA slot is just a model selection — it must be called the same way any
    model is called elsewhere, not through a bare ``call_llm(provider=...,
    model=...)`` that leaves base_url/api_key/api_mode unresolved and lets the
    auxiliary auto-detector guess. We route the slot's provider through
    ``resolve_runtime_provider`` (the canonical provider→api_mode/base_url/
    api_key resolver the CLI, gateway, and delegate_task all use), so the slot
    gets its provider's real API surface — e.g. MiniMax → anthropic_messages,
    GPT-5/o-series → max_completion_tokens, custom endpoints → their base_url.

    Returns the kwargs to pass through to ``call_llm`` (provider/model plus the
    resolved base_url/api_key when available). Falls back to the bare
    provider/model on any resolution error so a misconfigured slot still
    attempts the call rather than aborting the whole MoA turn.
    """
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    out: dict[str, Any] = {"provider": provider, "model": model}
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider(requested=provider, target_model=model)
        # Forward the resolved endpoint through to call_llm unconditionally.
        # call_llm's _resolve_task_provider_model() is the single chokepoint that
        # decides whether an explicit base_url collapses a call to the generic
        # ``custom`` route or keeps the provider's real identity: it preserves
        # identity for any first-class provider (via
        # _preserve_provider_with_base_url, a provider-catalog capability check),
        # so provider branches that add auth refresh / request metadata /
        # request-shape adapters — anthropic OAuth (Bearer + anthropic-beta),
        # openai-codex Responses wrapping + Cloudflare headers, xai-oauth,
        # bedrock SigV4 signing, nous Portal tags — still fire. Those branches
        # re-resolve their own credentials by name and ignore a forwarded
        # base_url/api_key, so forwarding is safe even for a placeholder key
        # (bedrock's "aws-sdk"). We used to maintain a name-preservation set here
        # too; that duplicated the chokepoint and drifted out of sync, so the
        # single source of truth now lives in call_llm.
        if rt.get("base_url"):
            out["base_url"] = rt["base_url"]
        if rt.get("api_key"):
            out["api_key"] = rt["api_key"]
        if rt.get("api_mode"):
            out["api_mode"] = rt["api_mode"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MoA slot runtime resolution failed for %s: %s", _slot_label(slot), exc)
    return out


def _maybe_apply_moa_cache_control(
    messages: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decorate an advisor or aggregator request with cache_control when its
    route honors it.

    Reuses the SAME policy function as the main agent loop
    (``anthropic_prompt_cache_policy``) resolved against the slot's own
    provider/base_url/api_mode/model, and the SAME breakpoint layout
    (``apply_anthropic_cache_control``, system_and_3). This keeps advisor and
    aggregator calls decorated exactly like an acting agent on that provider
    would be — no MoA-specific caching logic to drift.

    Returns the messages unchanged on any resolution error or when the
    policy says the route doesn't honor markers.
    """
    try:
        from types import SimpleNamespace

        from agent.agent_runtime_helpers import anthropic_prompt_cache_policy
        from agent.prompt_caching import apply_anthropic_cache_control

        # The policy function reads agent.* only as fallbacks for kwargs we
        # don't pass; provide a stub so the slot is judged purely on its own
        # resolved runtime.
        stub = SimpleNamespace(provider="", base_url="", api_mode="", model="")
        should_cache, native_layout = anthropic_prompt_cache_policy(
            stub,
            provider=runtime.get("provider") or "",
            base_url=runtime.get("base_url") or "",
            api_mode=runtime.get("api_mode") or "",
            model=runtime.get("model") or "",
        )
        if not should_cache:
            return messages
        return apply_anthropic_cache_control(
            messages, native_anthropic=native_layout
        )
    except Exception as exc:  # pragma: no cover - decoration must never break a call
        logger.debug("MoA cache_control decoration skipped: %s", exc)
        return messages


def _run_reference(
    slot: dict[str, str],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str, Any]:
    """Call one reference model and return ``(label, text, usage)``.

    The slot is resolved to its provider's real runtime (via ``_slot_runtime``)
    and called through the same ``call_llm`` request-building path any model
    uses, so per-model wire-format handling (anthropic_messages,
    max_completion_tokens, fixed/forbidden temperature) applies identically to
    a reference as it would if that model were the acting model. MoA imposes no
    cap of its own (``max_tokens`` defaults to ``None`` → omitted → the model's
    real maximum); ``temperature`` is only the user's configured preset value,
    which call_llm may still override per model.

    The reference's token usage is normalized with the slot's OWN resolved
    provider/api_mode (advisors may run on a different provider than the
    aggregator, with different usage wire shapes) and returned as a
    ``CanonicalUsage`` so the caller can fold advisor spend into session
    accounting. Without this, the entire reference fan-out — often the bulk of
    a MoA turn's token spend — is invisible to cost tracking, which only ever
    saw the aggregator's usage.

    Never raises: a failed reference becomes a labelled note so the aggregator
    can still act with partial context. Designed to run inside a thread pool —
    ``call_llm`` is synchronous/blocking, so threads (not asyncio) are the right
    concurrency primitive, mirroring ``delegate_task``'s batch fan-out.
    """
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

    label = _slot_label(slot)
    runtime = _slot_runtime(slot)
    try:
        # Prepend the advisory-role system prompt so the reference understands
        # it is analyzing state for an aggregator, not acting on the task. The
        # trimmed view (_reference_messages) already strips the agent's own
        # system prompt, so this is the only system message the reference sees.
        messages = [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages]
        # Apply the same Anthropic-style prompt-caching decoration the main
        # agent loop applies (system_and_3 breakpoints). The advisory view is
        # append-only across iterations (new turns append before the trailing
        # synthetic marker), so on cache-honoring routes (Claude via
        # OpenRouter/native, MiniMax, Qwen/DashScope) iteration N+1's prefix
        # replays iteration N's cached prefix. Without this, Claude advisors
        # served ZERO cache reads across an entire benchmark run (measured:
        # 0/1227 calls, 11.5M re-billed input tokens) because Anthropic
        # caching is opt-in per request. OpenAI-family advisors are untouched
        # (their caching is automatic; markers are ignored harmlessly, but we
        # only decorate when the policy says the route honors them).
        messages = _maybe_apply_moa_cache_control(messages, runtime)
        response = call_llm(
            task="moa_reference",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **runtime,
        )
        usage = CanonicalUsage()
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            try:
                usage = normalize_usage(
                    raw_usage,
                    provider=runtime.get("provider"),
                    api_mode=runtime.get("api_mode"),
                )
            except Exception:  # pragma: no cover - defensive
                usage = CanonicalUsage()
        # Price this advisor at ITS OWN model/provider rate (with correct
        # cache-read/cache-write split), not the aggregator's. This is why
        # advisor cost is summed as dollars rather than by folding tokens into
        # the aggregator's usage.
        cost_usd = None
        cost_status = None
        cost_source = None
        try:
            cost = estimate_usage_cost(
                slot.get("model") or "",
                usage,
                provider=runtime.get("provider"),
                base_url=runtime.get("base_url"),
                api_key=runtime.get("api_key"),
            )
            cost_usd = cost.amount_usd
            cost_status = cost.status
            cost_source = cost.source
        except Exception:  # pragma: no cover - defensive
            pass
        _output_text = _extract_text(response) or "(empty response)"
        acct = _RefAccounting(
            usage,
            cost_usd,
            cost_status,
            cost_source,
            messages=messages,
            output=_output_text,
            model=slot.get("model"),
            provider=runtime.get("provider") or slot.get("provider"),
            temperature=temperature,
            succeeded=True,
        )
        return label, _output_text, acct
    except Exception as exc:
        logger.warning("MoA reference model %s failed: %s", label, exc)
        return label, f"[failed: {exc}]", _RefAccounting(
            CanonicalUsage(),
            messages=[{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages],
            output=f"[failed: {exc}]",
            model=slot.get("model"),
            provider=runtime.get("provider") or slot.get("provider"),
            temperature=temperature,
            succeeded=False,
        )


def _run_references_parallel(
    reference_models: list[dict[str, str]],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[tuple[str, str, Any]]:
    """Fan out all reference models in parallel, returning outputs in order.

    Like ``delegate_task``'s batch mode, every reference is dispatched at once
    and we block until all of them finish before handing the joined results to
    the aggregator. Output order matches ``reference_models`` so the
    ``Reference {idx}`` labelling stays stable. MoA presets that reference
    another MoA preset are skipped here (recursion guard) with a labelled note.

    Each element is ``(label, text, usage)`` where usage is a
    ``CanonicalUsage`` (zeroed for skipped/failed references).
    """
    from agent.usage_pricing import CanonicalUsage

    if not reference_models:
        return []

    results: list[tuple[str, str, Any] | None] = [None] * len(reference_models)
    futures = {}
    workers = min(_MAX_REFERENCE_WORKERS, len(reference_models))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, slot in enumerate(reference_models):
            if slot.get("provider") == "moa":
                results[idx] = (
                    _slot_label(slot),
                    "[skipped: MoA presets cannot recursively reference MoA]",
                    _RefAccounting(CanonicalUsage(), succeeded=False),
                )
                continue
            futures[
                executor.submit(
                    _run_reference,
                    slot,
                    ref_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            ] = idx
        # Collect every reference before returning — the aggregator needs the
        # complete set, so there is no early-exit / first-completed path here.
        for future, idx in futures.items():
            results[idx] = future.result()

    return [r for r in results if r is not None]


def _truncate_tool_result(text: str, budget: int = _REFERENCE_TOOL_RESULT_BUDGET) -> str:
    """Head+tail preview of a tool result for the advisory view.

    Keeps the first and last halves of the budget with a ``[... N chars
    omitted ...]`` marker between them, so a reference sees both how the result
    started and how it ended without replaying the whole payload.
    """
    if not text or len(text) <= budget:
        return text
    half = budget // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} chars omitted ...]\n{text[-half:]}"


def _render_tool_calls(tool_calls: Any) -> str:
    """Render an assistant turn's tool_calls as readable text lines.

    The advisory view cannot carry real ``tool_calls`` payloads (strict
    providers reject tool_calls the reference never produced), so the agent's
    actions are flattened to text the reference can read and reason about.
    """
    lines: list[str] = []
    for tc in tool_calls or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        name = fn.get("name") or (tc.get("name") if isinstance(tc, dict) else "") or "tool"
        args = fn.get("arguments")
        if isinstance(args, str):
            args_text = args
        elif args is not None:
            try:
                import json

                args_text = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_text = str(args)
        else:
            args_text = ""
        lines.append(f"[called tool: {name}({args_text})]" if args_text else f"[called tool: {name}]")
    return "\n".join(lines)


_ADVISORY_INSTRUCTION = (
    "[The conversation above is the current state of the task. Give your "
    "most intelligent judgement: what is going on, what should happen next, "
    "what risks or mistakes you see, and how the acting agent should "
    "proceed.]"
)


def _reference_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an advisory view of the conversation for reference models.

    A reference gives an INFORMED judgement on the current state, so it must
    see what the agent actually did — its tool calls AND the tool results that
    came back — not just the agent's narration. We therefore preserve the whole
    conversation flow, but flatten it into clean user/assistant *text* turns:

      - system prompt: dropped (8K of Hermes boilerplate, not advisory signal).
      - assistant turns: kept; any ``tool_calls`` are rendered inline as
        ``[called tool: name(args)]`` text lines appended to the turn's text.
      - ``tool``-role results: NOT dropped. Each is folded (head+tail preview,
        see ``_truncate_tool_result``) into the *preceding* assistant turn as a
        ``[tool result: ...]`` block, so the reference sees what came back.

    This emits ZERO ``tool``-role messages and ZERO ``tool_calls`` arrays — only
    plain user/assistant text — so strict providers (Mistral, Fireworks) that
    reject orphan tool messages / unproduced tool_calls don't 400, while the
    reference still has the full picture.

    The view MUST end with a ``user`` turn. Anthropic (and OpenRouter→Anthropic)
    interpret a trailing assistant turn as an assistant *prefill* to continue,
    and no-prefill models (e.g. Claude Opus 4.8) reject it with
    ``400 ... must end with a user message``. Rather than DELETE the agent's
    latest context to satisfy that (which would blind the reference to the
    current state), we APPEND a synthetic user turn asking the reference to
    judge the state above. End-on-user is satisfied and no context is lost.

    The acting aggregator always receives the full, untrimmed transcript; this
    function only shapes the disposable advisory copy.
    """
    rendered: list[dict[str, Any]] = []
    last_user_content: str | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        text = content if isinstance(content, str) else ""

        if role == "system":
            continue
        if role == "user":
            if text.strip():
                last_user_content = text
            rendered.append({"role": "user", "content": text})
        elif role == "assistant":
            parts: list[str] = []
            if text.strip():
                parts.append(text.strip())
            calls_text = _render_tool_calls(msg.get("tool_calls"))
            if calls_text:
                parts.append(calls_text)
            # Empty assistant turns (no text, no calls) carry nothing advisory.
            if parts:
                rendered.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            # Fold the tool result into the preceding assistant turn as text so
            # the reference sees what came back, without emitting a tool-role
            # message a reference never produced.
            result_text = _truncate_tool_result(text)
            block = f"[tool result: {result_text}]"
            if rendered and rendered[-1].get("role") == "assistant":
                rendered[-1]["content"] = rendered[-1]["content"] + "\n" + block
            else:
                # No assistant turn to attach to (e.g. a leading tool result);
                # keep it as advisory context on its own assistant-role line.
                rendered.append({"role": "assistant", "content": block})
        # Any other role is ignored.

    # End on a user turn: append a synthetic advisory request rather than
    # deleting the agent's latest assistant context. This satisfies Anthropic's
    # no-trailing-assistant-prefill rule while preserving full state.
    if rendered and rendered[-1].get("role") == "assistant":
        rendered.append({"role": "user", "content": _ADVISORY_INSTRUCTION})
    elif rendered and rendered[-1].get("role") == "user":
        # Already ends on a user turn (fresh user prompt, no agent action yet).
        # Leave it — the reference answers that prompt directly.
        pass

    if not rendered:
        # Degenerate case: nothing rendered. Fall back to the latest user turn.
        if last_user_content is not None:
            return [{"role": "user", "content": last_user_content}]
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return [{"role": "user", "content": msg["content"]}]
    return rendered



def _extract_text(response: Any) -> str:
    try:
        transport = get_transport("chat_completions")
        if transport is None:
            raise RuntimeError("chat_completions transport unavailable")
        normalized = transport.normalize_response(response)
        text = (normalized.content or "").strip()
        if text:
            return text
    except Exception:
        pass
    try:
        message = response.choices[0].message
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", message)
        if not isinstance(content, str):
            content = str(content) if content else ""
        return content.strip()
    except Exception:
        return ""


def _preset_temperature(preset: dict[str, Any], key: str) -> float | None:
    """Read an optional temperature from a preset.

    Returns None when the key is absent, empty, or explicitly null — meaning
    "don't send temperature; let the provider default apply", exactly like a
    single-model Hermes agent (which never sends temperature unless
    configured). The old coercion ``float(preset.get(key, 0.6) or 0.6)``
    made unset impossible: absent, null, and even 0 all collapsed to the
    hardcoded default, so MoA advisors/aggregator always ran at 0.6/0.4
    while the same model running solo used the provider default.
    """
    value = preset.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("ignoring non-numeric %s=%r in MoA preset", key, value)
        return None


def aggregate_moa_context(
    *,
    user_prompt: str,
    api_messages: list[dict[str, Any]],
    reference_models: list[dict[str, str]],
    aggregator: dict[str, str],
    temperature: float | None = None,
    aggregator_temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Run configured reference models and synthesize their advice.

    Failures are returned as model-specific notes instead of aborting the normal
    agent loop; the main model can still act with partial context.

    ``max_tokens`` is ``None`` by default: MoA does not cap reference or
    aggregator output, so each model uses its own maximum. ``call_llm`` omits
    the parameter entirely when it is ``None`` (see its docstring), which also
    sidesteps providers that reject ``max_tokens`` outright. A hardcoded cap
    here previously truncated long aggregator syntheses.

    ``temperature`` / ``aggregator_temperature`` are ``None`` by default:
    like max_tokens, ``call_llm`` omits temperature when None so the
    provider default applies — matching single-model agent behavior. Presets
    may still pin explicit values.
    """
    reference_outputs: list[tuple[str, str, Any]] = []
    ref_messages = _reference_messages(api_messages)
    reference_outputs = _run_references_parallel(
        reference_models,
        ref_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    joined = "\n\n".join(
        f"Reference {idx} — {label}:\n{text}"
        for idx, (label, text, _usage) in enumerate(reference_outputs, start=1)
    )
    synth_prompt = (
        "You are the aggregator in a Mixture of Agents process. Synthesize the "
        "reference responses into concise, actionable guidance for the main "
        "Hermes agent. Focus on next steps, tool-use strategy, risks, and any "
        "disagreements. Do not answer the user directly unless that is all that "
        "is needed; produce context the main agent should use in its normal loop.\n\n"
        f"Original user prompt:\n{user_prompt}\n\n"
        f"Reference responses:\n{joined}"
    )

    agg_label = _slot_label(aggregator)
    agg_runtime = _slot_runtime(aggregator)
    try:
        # Same cache_control decoration as _run_reference's advisor calls
        # (see _maybe_apply_moa_cache_control) — this synthesis call is a
        # third, independent MoA call path that 22c5048d9 did not cover (it
        # only restored caching for the acting-aggregator turn in the
        # persistent `provider: moa` model and for advisor fan-out). Without
        # it, the one-shot `/moa <prompt>` command's synthesis call re-bills
        # its full input (system-less prompt containing every joined
        # reference output) on every invocation with zero cache_control
        # breakpoints, even when the resolved aggregator slot is a
        # cache-honoring route (e.g. Claude on OpenRouter/native Anthropic).
        agg_messages = _maybe_apply_moa_cache_control(
            [{"role": "user", "content": synth_prompt}], agg_runtime
        )
        response = call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            max_tokens=max_tokens,
            **agg_runtime,
        )
        synthesis = _extract_text(response)
    except Exception as exc:
        logger.warning("MoA aggregator model %s failed: %s", agg_label, exc)
        synthesis = ""

    if not synthesis:
        synthesis = joined

    return (
        "[Mixture of Agents context — use this as private guidance for the "
        "normal Hermes agent loop. You may call tools, continue reasoning, or "
        "finish normally.]\n"
        f"Aggregator: {agg_label}\n"
        f"References: {', '.join(_slot_label(slot) for slot in reference_models)}\n\n"
        f"{synthesis.strip()}"
    )


def _attach_reference_guidance(agg_messages: list[dict[str, Any]], guidance: str) -> None:
    """Attach the per-turn reference block at the END of the aggregator prompt.

    The reference text differs on every tool-loop iteration. In an agentic loop
    the most recent ``user`` message is the *original task* sitting near the TOP
    of the context (everything after it is assistant/tool turns), so merging the
    turn-varying reference block into it diverges the prompt prefix early — the
    server's KV cache cannot be reused and the entire conversation re-prefills on
    every step (full prefill each tool call, dominating latency on long contexts).

    Appending at the very end keeps the ``[system][task][tool-history]`` prefix
    stable and cache-reusable (only the new block re-prefills), and gives the
    aggregator the references with recency. Merge into the last message only when
    it is already a trailing string ``user`` turn (plain chat — still at the end).
    """
    last = agg_messages[-1] if agg_messages else None
    if last is not None and last.get("role") == "user" and isinstance(last.get("content"), str):
        last["content"] = last["content"] + "\n\n" + guidance
    else:
        agg_messages.append({"role": "user", "content": guidance})


class MoAChatCompletions:
    """OpenAI-chat-compatible facade where the aggregator is the acting model."""

    def __init__(self, preset_name: str, reference_callback: Any = None):
        self.preset_name = preset_name or "default"
        # Optional display hook. Called as reference outputs become available so
        # frontends can show each reference model's answer as a labelled block
        # before the aggregator acts. Signature:
        #   reference_callback(event, **kwargs)
        # where event is one of:
        #   "moa.reference"   kwargs: index, count, label, text
        #   "moa.aggregating" kwargs: aggregator (label), ref_count
        # Never raises into the model call — display is best-effort.
        self.reference_callback = reference_callback
        # State-scoped reference cache. The agent loop calls create() once per
        # tool-loop iteration; references should re-run whenever the task STATE
        # advances — i.e. on every new user message AND every new tool result —
        # so each reference judges the latest state. The advisory view
        # (_reference_messages) now renders tool calls + results as text, so its
        # signature changes on every new tool response; the cache key is that
        # signature, so a new tool result is a cache MISS (references re-run)
        # while a redundant create() call with identical state is a HIT (no
        # re-run, no re-emit). This gives "fire on every user/tool response"
        # for free, without re-firing on a pure no-op re-call.
        self._ref_cache_key: tuple | None = None
        self._ref_cache_outputs: list[tuple[str, str, Any]] = []
        # Token usage + estimated cost of the reference fan-out from the most
        # recent cache-MISS create() call, awaiting consumption by session
        # accounting. Set on every create() (zeroed on a cache HIT so per-turn
        # advisor spend is counted exactly once). Consumed via
        # ``consume_reference_usage``.
        from agent.usage_pricing import CanonicalUsage

        self._pending_reference_usage: Any = CanonicalUsage()
        self._pending_reference_cost: Any = None
        # Resolved aggregator slot ({provider, model, ...}) from the most recent
        # create(); read by session cost accounting to price the aggregator's
        # acting turn at its real model instead of the virtual preset name.
        self.last_aggregator_slot: Any = None
        # Full-turn trace parts stashed on a cache-MISS create(), awaiting the
        # caller to stitch in the live session_id + resolved aggregator output
        # and flush to the trace file (only when moa.save_traces is on).
        self._pending_trace: Any = None
        self._last_coordination_turn_facts: dict[str, Any] | None = None
        self._pending_coordination_turn_facts: dict[str, Any] | None = None

    def _coordination_facts_from_outputs(
        self,
        reference_outputs: list[tuple[str, str, Any]],
        aggregator: dict[str, str],
    ) -> dict[str, Any]:
        """Build private MoA completion facts from actual reference outcomes."""

        reference_labels: list[str] = []
        failed_count = 0
        for label, _text, accounting in reference_outputs:
            if isinstance(accounting, _RefAccounting) and accounting.succeeded is True:
                if isinstance(label, str):
                    reference_labels.append(label)
            else:
                failed_count += 1
        reference_total = len(reference_outputs)
        aggregator_model = str((aggregator or {}).get("model") or "").strip()
        aggregator_count = 1 if aggregator_model else 0
        return {
            "moa": {
                "observed": bool(reference_labels or failed_count or aggregator_count),
                "reference_models": reference_labels,
                "reference_count": len(reference_labels),
                "reference_total": reference_total,
                "failed_count": failed_count,
                "aggregator_model": aggregator_model,
                "aggregator_count": aggregator_count,
            }
        }

    def _publish_pending_coordination_turn_facts(self) -> None:
        pending = getattr(self, "_pending_coordination_turn_facts", None)
        self._pending_coordination_turn_facts = None
        if isinstance(pending, dict):
            self._last_coordination_turn_facts = pending

    def coordination_turn_facts(self) -> dict[str, Any]:
        """Return runtime facts for the most recently completed MoA create()."""

        facts = getattr(self, "_last_coordination_turn_facts", None)
        if not isinstance(facts, dict):
            return {}
        moa = facts.get("moa")
        if not isinstance(moa, dict):
            return {}
        reference_models = [str(label) for label in moa.get("reference_models") or [] if str(label)]
        aggregator_model = str(moa.get("aggregator_model") or "")
        if "reference_count" in moa:
            try:
                reference_count = int(moa.get("reference_count") or 0)
            except Exception:
                reference_count = len(reference_models)
        else:
            reference_count = len(reference_models)
        if "aggregator_count" in moa:
            try:
                aggregator_count = int(moa.get("aggregator_count") or 0)
            except Exception:
                aggregator_count = 1 if aggregator_model.strip() else 0
        else:
            aggregator_count = 1 if aggregator_model.strip() else 0
        if "failed_count" in moa:
            try:
                failed_count = int(moa.get("failed_count") or 0)
            except Exception:
                failed_count = 0
        else:
            failed_count = 0
        if "reference_total" in moa:
            try:
                reference_total = int(moa.get("reference_total") or 0)
            except Exception:
                reference_total = reference_count + failed_count
        else:
            reference_total = reference_count + failed_count
        return {
            "moa": {
                "observed": (
                    bool(moa.get("observed"))
                    or reference_count > 0
                    or failed_count > 0
                    or aggregator_count > 0
                ),
                "reference_models": reference_models,
                "reference_count": reference_count,
                "reference_total": reference_total,
                "failed_count": failed_count,
                "aggregator_model": aggregator_model,
                "aggregator_count": aggregator_count,
            }
        }

    def consume_reference_usage(self) -> tuple[Any, Any]:
        """Pop pending reference-fan-out usage + cost, resetting both to empty.

        Returns ``(CanonicalUsage, cost_usd_or_None)`` for the most recent
        ``create()`` and clears the pending values, so a subsequent read (e.g.
        a streaming retry re-entering accounting) cannot double-count. Usage is
        always a ``CanonicalUsage`` (zeroed if none); cost is a summed-dollars
        float or ``None`` when no advisor could be priced.
        """
        from agent.usage_pricing import CanonicalUsage

        usage = self._pending_reference_usage or CanonicalUsage()
        cost = self._pending_reference_cost
        self._pending_reference_usage = CanonicalUsage()
        self._pending_reference_cost = None
        return usage, cost

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn trace to disk, if one is pending.

        No-op when tracing is off (``save_moa_turn`` checks the config), when
        there is no pending trace (a cache-HIT iteration ran no references), or
        when the aggregator input was never recorded. Clears the pending trace
        so a repeat consume cannot double-write. Best-effort — never raises.

        ``aggregator_output_fallback`` is the aggregator's resolved acting text
        as the caller already holds it in memory (the streamed assistant text).
        On the streaming path the aggregator's output could not be captured
        inline at ``create()`` time (the raw token stream was handed to the live
        consumer), so ``pending["aggregator_output"]`` is None; we fold the
        caller's resolved text in here so the trace is self-contained in BOTH
        streaming and non-streaming modes. Non-streaming already has the inline
        output and ignores the fallback.
        """
        self._publish_pending_coordination_turn_facts()
        pending = self._pending_trace
        self._pending_trace = None
        if not pending or "aggregator_input_messages" not in pending:
            return
        try:
            from agent.moa_trace import save_moa_turn

            agg_slot = pending.get("aggregator_slot") or {}
            # Prefer the inline capture (non-streaming); fall back to the
            # caller's resolved streamed text when streaming left it None.
            agg_output = pending.get("aggregator_output")
            if agg_output is None and aggregator_output_fallback:
                agg_output = aggregator_output_fallback
            save_moa_turn(
                session_id=session_id,
                preset_name=pending.get("preset", ""),
                reference_outputs=pending.get("reference_outputs", []),
                aggregator_label=pending.get("aggregator_label", ""),
                aggregator_model=agg_slot.get("model"),
                aggregator_provider=agg_slot.get("provider"),
                aggregator_temperature=pending.get("aggregator_temperature"),
                aggregator_input_messages=pending.get("aggregator_input_messages"),
                aggregator_output=agg_output,
                aggregator_streamed=bool(pending.get("aggregator_streamed")),
            )
        except Exception as exc:  # pragma: no cover - tracing must never break a turn
            logger.debug("MoA trace flush failed: %s", exc)

    def _emit(self, event: str, **kwargs: Any) -> None:
        cb = self.reference_callback
        if cb is None:
            return
        try:
            cb(event, **kwargs)
        except Exception as exc:  # pragma: no cover - display must never break the turn
            logger.debug("MoA reference_callback failed for %s: %s", event, exc)

    def create(self, **api_kwargs: Any) -> Any:
        self._last_coordination_turn_facts = None
        self._pending_coordination_turn_facts = None

        from hermes_cli.config import load_config
        from hermes_cli.moa_config import resolve_moa_preset

        preset = resolve_moa_preset(load_config().get("moa") or {}, self.preset_name)
        messages = list(api_kwargs.get("messages") or [])
        reference_models = preset.get("reference_models") or []
        aggregator = preset.get("aggregator") or {}
        # Expose the resolved aggregator slot so session cost accounting can
        # price the aggregator's acting turn at its REAL model/provider. The
        # agent's model/provider on the MoA path are the virtual preset name
        # ("closed") and "moa", which have no pricing entry — without this the
        # aggregator's spend (often the bulk of the turn) is silently dropped
        # and the session cost reflects advisor fan-out only.
        self.last_aggregator_slot = dict(aggregator) if aggregator else None
        # By default MoA does not cap reference or aggregator output: each model
        # uses its own maximum (max_tokens=None → call_llm omits the parameter,
        # so a long aggregator synthesis is never truncated and providers that
        # reject max_tokens don't 400). A preset MAY set reference_max_tokens to
        # cap ADVISOR output only — advisor generation is the dominant MoA
        # latency (turn latency correlates ~0.88 with output tokens), and the
        # aggregator only needs the gist of each advisor's judgement, so a cap
        # (e.g. 600) measurably cuts per-turn wall time (~44% on a sample task).
        # The acting aggregator is never capped here (its output is the
        # user-visible answer).
        reference_max_tokens = preset.get("reference_max_tokens")
        # None (the default) = don't send temperature; provider default
        # applies, matching single-model agent behavior. Presets may pin
        # explicit values. See _preset_temperature.
        temperature = _preset_temperature(preset, "reference_temperature")
        aggregator_temperature = _preset_temperature(preset, "aggregator_temperature")
        if aggregator_temperature is None and api_kwargs.get("temperature") is not None:
            # The acting agent's own configured temperature (if any) still
            # applies to the aggregator, which IS the acting model.
            aggregator_temperature = api_kwargs.get("temperature")

        # When the preset is disabled, skip the reference fan-out and let the
        # configured aggregator act alone — it is the preset's acting model, so
        # a disabled MoA preset is simply "use the aggregator directly."
        if not preset.get("enabled", True):
            reference_models = []

        from agent.usage_pricing import CanonicalUsage

        reference_outputs: list[tuple[str, str, Any]] = []
        ref_messages = _reference_messages(messages)

        # Fan-out cadence. "per_iteration" (default): advisors re-run whenever
        # the advisory view changes — i.e. every tool iteration, since the
        # view grows with each tool result. "user_turn": advisors run ONCE per
        # user turn; subsequent tool iterations reuse that turn's advice and
        # the aggregator acts alone (the original MoA shape: synthesize at the
        # start, then let the acting model work). Implemented by hashing only
        # the prefix up to the LAST USER message so mid-turn growth doesn't
        # change the signature — iteration 2+ becomes a cache HIT.
        fanout_mode = str(preset.get("fanout") or "per_iteration").strip().lower()
        sig_messages = ref_messages
        if fanout_mode == "user_turn":
            # Find the last REAL user message. The advisory view appends a
            # synthetic user marker (_ADVISORY_INSTRUCTION) when it ends on an
            # assistant turn — i.e. on every tool iteration after the first —
            # so that marker must not count as a user turn or the prefix
            # would include the grown mid-turn context and the signature
            # would change every iteration (defeating the once-per-turn
            # cadence entirely).
            last_user_idx = None
            for _i in range(len(ref_messages) - 1, -1, -1):
                _m = ref_messages[_i]
                if _m.get("role") == "user" and _m.get("content") != _ADVISORY_INSTRUCTION:
                    last_user_idx = _i
                    break
            if last_user_idx is not None:
                sig_messages = ref_messages[: last_user_idx + 1]

        # Turn-scoped cache: only run + display references when the advisory
        # view changed (i.e. a new user turn). Within one turn the agent loop
        # calls create() once per tool iteration; in user_turn mode the
        # signature is stable across those iterations (prefix hash above), so
        # the fan-out runs once per user turn and iterations reuse the advice.
        _sig = hashlib.sha256(
            "\u0000".join(
                f"{m.get('role')}:{m.get('content')}" for m in sig_messages
            ).encode("utf-8", "replace")
        ).hexdigest()
        _cache_key = (self.preset_name, _sig, tuple(_slot_label(s) for s in reference_models))
        _refs_from_cache = _cache_key == self._ref_cache_key and bool(self._ref_cache_outputs)

        if _refs_from_cache:
            reference_outputs = list(self._ref_cache_outputs)
            # References already ran (and were accounted) earlier this turn;
            # this create() is a repeat tool-iteration reusing the cached
            # advice. Charging their tokens/cost again here would multiply
            # advisor spend by the tool-iteration count, so pending is zero.
            self._pending_reference_usage = CanonicalUsage()
            self._pending_reference_cost = None
            # Likewise no trace on a cache HIT — the full turn was already
            # traced on the MISS that ran the references. A repeat iteration is
            # not a new MoA turn.
            self._pending_trace = None
        else:
            reference_outputs = _run_references_parallel(
                reference_models,
                ref_messages,
                temperature=temperature,
                max_tokens=reference_max_tokens,
            )
            self._ref_cache_key = _cache_key
            self._ref_cache_outputs = list(reference_outputs)
            # Sum the advisor fan-out's token usage AND cost so the caller can
            # fold advisor spend into session accounting exactly once per turn.
            # Only the freshly run references (cache MISS) contribute; a cache
            # HIT above zeroes this. Token counts sum directly (each already
            # normalized per-advisor provider/api_mode); cost sums in dollars
            # because each advisor was priced at its OWN model rate — advisors
            # may be cheaper/pricier than the aggregator, so their tokens must
            # NOT be repriced at the aggregator's rate.
            _ref_usage = CanonicalUsage()
            _ref_cost: Any = None
            for _lbl, _txt, _acct in reference_outputs:
                if isinstance(_acct, _RefAccounting):
                    if isinstance(_acct.usage, CanonicalUsage):
                        _ref_usage = _ref_usage + _acct.usage
                    if _acct.cost_usd is not None:
                        _ref_cost = (_ref_cost or 0) + _acct.cost_usd
            self._pending_reference_usage = _ref_usage
            self._pending_reference_cost = _ref_cost
            # Stash the full reference fan-out for trace persistence. The
            # aggregator input/label are filled in below once agg_messages is
            # built; the aggregator OUTPUT is stitched in by the caller
            # (consume_and_save_trace) once the response resolves — the caller
            # holds the live session_id and the resolved aggregator response.
            self._pending_trace = {
                "preset": self.preset_name,
                "reference_outputs": list(reference_outputs),
                "aggregator_slot": aggregator,
                "aggregator_temperature": aggregator_temperature,
            }

            # Surface each reference model's answer to the display BEFORE the
            # aggregator acts — once per turn (only on the iteration that
            # actually ran them). The user sees one labelled block per
            # reference (rendered like a thinking block) so the MoA process is
            # visible rather than a silent pause. Best-effort: never blocks the
            # turn.
            _ref_count = len(reference_outputs)
            for _idx, (_label, _text, _usage) in enumerate(reference_outputs, start=1):
                self._emit(
                    "moa.reference",
                    index=_idx,
                    count=_ref_count,
                    label=_label,
                    text=_text,
                )
            if _ref_count:
                self._emit(
                    "moa.aggregating",
                    aggregator=_slot_label(aggregator),
                    ref_count=_ref_count,
                )

        self._pending_coordination_turn_facts = self._coordination_facts_from_outputs(
            reference_outputs,
            aggregator,
        )

        agg_messages = [dict(m) for m in messages]
        if reference_outputs:
            joined = "\n\n".join(
                f"Reference {idx} — {label}:\n{text}"
                for idx, (label, text, _usage) in enumerate(reference_outputs, start=1)
            )
            guidance = (
                "[Mixture of Agents reference context]\n"
                f"Preset: {self.preset_name}\n"
                f"Aggregator/acting model: {_slot_label(aggregator)}\n"
                f"References: {', '.join(label for label, _, _ in reference_outputs)}\n\n"
                "Use the reference responses below as private context. You are the aggregator and acting model: "
                "answer the user directly or call tools as needed.\n\n"
                f"{joined}"
            )
            _attach_reference_guidance(agg_messages, guidance)

        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_kwargs = dict(api_kwargs)
        agg_kwargs["messages"] = agg_messages
        # Record the exact aggregator INPUT (incl. the injected reference
        # context) into the pending trace so a trace captures what the
        # aggregator actually saw, not a reconstruction.
        if self._pending_trace is not None:
            self._pending_trace["aggregator_input_messages"] = agg_messages
            self._pending_trace["aggregator_label"] = _slot_label(aggregator)
        # The aggregator is the acting model. Resolve its slot to the provider's
        # real runtime (base_url/api_key/api_mode) and call it through the same
        # request-building path any model uses — so per-model wire-format
        # handling (anthropic_messages, max_completion_tokens, fixed/forbidden
        # temperature) applies identically to it. MoA imposes no output cap:
        # max_tokens is passed through from the caller (normally None → omitted
        # → the model's real maximum). The preset's old hardcoded 4096 default
        # is gone — it truncated long syntheses.
        # When the agent's streaming consumer calls us with stream=True, run the
        # references first (above) and then return the aggregator's RAW token
        # stream so the acting model's output reaches the user live. The consumer
        # reassembles chunks + tool_calls, runs stale-stream detection, and falls
        # back to a non-streaming retry on error. The non-streaming path
        # (stream=False) is unchanged — no stream/stream_options/timeout are
        # forwarded, so its behavior is byte-for-byte identical to before.
        stream = bool(api_kwargs.get("stream"))
        stream_kwargs: dict[str, Any] = {}
        if stream:
            stream_kwargs["stream"] = True
            stream_kwargs["stream_options"] = (
                api_kwargs.get("stream_options") or {"include_usage": True}
            )
            # Forward the consumer's per-request (stream read) timeout so it
            # actually governs the aggregator stream, not just call_llm's default.
            if api_kwargs.get("timeout") is not None:
                stream_kwargs["timeout"] = api_kwargs["timeout"]
        try:
            _agg_response = call_llm(
                task="moa_aggregator",
                messages=agg_messages,
                temperature=aggregator_temperature,
                max_tokens=agg_kwargs.get("max_tokens"),
                tools=agg_kwargs.get("tools"),
                extra_body=agg_kwargs.get("extra_body"),
                **stream_kwargs,
                **_slot_runtime(aggregator),
            )
        except Exception:
            self._pending_coordination_turn_facts = None
            raise
        # Non-streaming path (quiet mode / eval / subagents): the aggregator
        # output is available inline, so capture it into the pending trace now.
        # Streaming path: the aggregator's raw token stream is returned to the
        # consumer live and its acting output lands as the turn's assistant
        # message; the trace marks it streamed and points there.
        if self._pending_trace is not None:
            if stream:
                self._pending_trace["aggregator_streamed"] = True
                self._pending_trace["aggregator_output"] = None
            else:
                self._pending_trace["aggregator_streamed"] = False
                try:
                    self._pending_trace["aggregator_output"] = _extract_text(_agg_response)
                except Exception:  # pragma: no cover - defensive
                    self._pending_trace["aggregator_output"] = None
        if not stream:
            self._publish_pending_coordination_turn_facts()
        return _agg_response


class MoAClient:
    def __init__(self, preset_name: str, reference_callback: Any = None):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(preset_name, reference_callback=reference_callback)

    def consume_reference_usage(self) -> Any:
        """Pop the pending reference-fan-out usage from the completions facade.

        Lets session accounting fold the MoA advisor tokens into the turn's
        usage without reaching into ``.chat.completions`` internals.
        """
        return self.chat.completions.consume_reference_usage()

    def coordination_turn_facts(self) -> dict[str, Any]:
        """Return runtime facts for the most recently completed MoA create()."""

        return self.chat.completions.coordination_turn_facts()

    @property
    def last_aggregator_slot(self) -> Any:
        """Resolved aggregator slot ({provider, model, ...}) from the most
        recent create(), or None. Read by session cost accounting to price the
        aggregator's acting turn at its real model instead of the virtual
        preset name."""
        return getattr(self.chat.completions, "last_aggregator_slot", None)

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending full-turn MoA trace via the completions facade.

        No-op unless ``moa.save_traces`` is enabled and a turn is pending.
        ``aggregator_output_fallback`` supplies the resolved acting text so the
        streaming path's trace is self-contained (see the facade docstring).
        """
        return self.chat.completions.consume_and_save_trace(
            session_id, aggregator_output_fallback=aggregator_output_fallback
        )
