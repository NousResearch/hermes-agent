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

# Per-tool-result character budget for the advisory reference view. Tool
# results can be huge (a full diff, a 5000-line file dump); replaying them
# verbatim per reference per tool-loop step would blow the reference model's
# context window and cost. We keep the agent's *actions* (tool calls) in full —
# they are cheap, high-signal, and tell the reference what the agent did — but
# preview each tool *result* head+tail so the reference still sees what came
# back without replaying megabytes. The acting aggregator always gets the full,
# untrimmed transcript; this budget only shapes the advisory copy.
_REFERENCE_TOOL_RESULT_BUDGET = 4000
_REFERENCE_OUTPUT_RESERVE_TOKENS = 4096
_REFERENCE_MIN_INPUT_BUDGET_TOKENS = 512

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


class MoAAggregatorContextOverflow(RuntimeError):
    """Raised before calling the aggregator when the measured request is too large."""

    def __init__(
        self,
        *,
        estimated_tokens: int,
        context_length: int,
        aggregator_label: str,
    ) -> None:
        self.estimated_tokens = estimated_tokens
        self.context_length = context_length
        self.aggregator_label = aggregator_label
        super().__init__(
            "MoA aggregator input exceeds the context window of this model: "
            f"estimated request {estimated_tokens:,} tokens > context window "
            f"{context_length:,} tokens (aggregator={aggregator_label})."
        )



def _slot_label(slot: dict[str, str]) -> str:
    return f"{slot.get('provider', '').strip()}:{slot.get('model', '').strip()}"


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
        resolved_provider = str(rt.get("provider") or provider).strip().lower()
        # call_llm treats an explicit base_url as a custom endpoint. That is
        # correct for ordinary OpenAI-compatible targets, but wrong for OAuth /
        # provider-backed targets whose provider branch adds auth refresh,
        # request metadata, or request-shape adapters. Keep those providers
        # identified by name.
        if resolved_provider in {"nous", "openai-codex", "xai-oauth"}:
            return out
        # Pass the resolved endpoint through so call_llm builds the request for
        # the provider's actual API surface instead of auto-detecting. base_url
        # routes call_llm to the right adapter (incl. anthropic_messages mode);
        # api_key is the resolved credential for that provider.
        if rt.get("base_url"):
            out["base_url"] = rt["base_url"]
        if rt.get("api_key"):
            out["api_key"] = rt["api_key"]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MoA slot runtime resolution failed for %s: %s", _slot_label(slot), exc)
    return out


def _run_reference(
    slot: dict[str, str],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, str]:
    """Call one reference model and return ``(label, text)``.

    The slot is resolved to its provider's real runtime (via ``_slot_runtime``)
    and called through the same ``call_llm`` request-building path any model
    uses, so per-model wire-format handling (anthropic_messages,
    max_completion_tokens, fixed/forbidden temperature) applies identically to
    a reference as it would if that model were the acting model. MoA imposes no
    cap of its own (``max_tokens`` defaults to ``None`` → omitted → the model's
    real maximum); ``temperature`` is only the user's configured preset value,
    which call_llm may still override per model.

    Never raises: a failed reference becomes a labelled note so the aggregator
    can still act with partial context. Designed to run inside a thread pool —
    ``call_llm`` is synchronous/blocking, so threads (not asyncio) are the right
    concurrency primitive, mirroring ``delegate_task``'s batch fan-out.
    """
    label = _slot_label(slot)
    try:
        ref_messages = _fit_reference_messages_to_slot(slot, ref_messages)
        # Prepend the advisory-role system prompt so the reference understands
        # it is analyzing state for an aggregator, not acting on the task. The
        # trimmed view (_reference_messages) already strips the agent's own
        # system prompt, so this is the only system message the reference sees.
        messages = [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages]
        response = call_llm(
            task="moa_reference",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **_slot_runtime(slot),
        )
        return label, _extract_text(response) or "(empty response)"
    except Exception as exc:
        logger.warning("MoA reference model %s failed: %s", label, exc)
        return label, f"[failed: {exc}]"


def _run_references_parallel(
    reference_models: list[dict[str, str]],
    ref_messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> list[tuple[str, str]]:
    """Fan out all reference models in parallel, returning outputs in order.

    Like ``delegate_task``'s batch mode, every reference is dispatched at once
    and we block until all of them finish before handing the joined results to
    the aggregator. Output order matches ``reference_models`` so the
    ``Reference {idx}`` labelling stays stable. MoA presets that reference
    another MoA preset are skipped here (recursion guard) with a labelled note.
    """
    if not reference_models:
        return []

    results: list[tuple[str, str] | None] = [None] * len(reference_models)
    futures = {}
    workers = min(_MAX_REFERENCE_WORKERS, len(reference_models))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idx, slot in enumerate(reference_models):
            if slot.get("provider") == "moa":
                results[idx] = (
                    _slot_label(slot),
                    "[skipped: MoA presets cannot recursively reference MoA]",
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
    advisory_instruction = (
        "[The conversation above is the current state of the task. Give your "
        "most intelligent judgement: what is going on, what should happen next, "
        "what risks or mistakes you see, and how the acting agent should "
        "proceed.]"
    )

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
        rendered.append({"role": "user", "content": advisory_instruction})
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
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception:
        return ""


def _slot_context_length(slot: dict[str, str]) -> int | None:
    try:
        from agent.model_metadata import get_model_context_length

        runtime = _slot_runtime(slot)
        return get_model_context_length(
            str(runtime.get("model") or slot.get("model") or ""),
            base_url=str(runtime.get("base_url") or ""),
            api_key=str(runtime.get("api_key") or ""),
            provider=str(runtime.get("provider") or slot.get("provider") or ""),
        )
    except Exception:
        logger.debug("MoA slot context budget resolution failed for %s", _slot_label(slot), exc_info=True)
        return None


def _reference_input_target_tokens(context_length: int) -> int:
    if context_length <= _REFERENCE_MIN_INPUT_BUDGET_TOKENS:
        return max(1, int(context_length * 0.75))
    reserve = min(_REFERENCE_OUTPUT_RESERVE_TOKENS, max(0, context_length // 4))
    return max(_REFERENCE_MIN_INPUT_BUDGET_TOKENS, context_length - reserve)


def _reference_request_tokens(messages: list[dict[str, Any]]) -> int:
    from agent.model_metadata import estimate_request_tokens_rough

    return estimate_request_tokens_rough(
        [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *messages]
    )


def _fit_reference_messages_to_slot(
    slot: dict[str, str],
    ref_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    context_length = _slot_context_length(slot)
    if not context_length:
        return ref_messages

    target_tokens = _reference_input_target_tokens(context_length)
    if _reference_request_tokens(ref_messages) <= target_tokens:
        return ref_messages

    note = {
        "role": "assistant",
        "content": (
            "[Earlier MoA advisory context omitted to fit this reference "
            "model's context window.]"
        ),
    }
    kept_reversed: list[dict[str, Any]] = []
    for msg in reversed(ref_messages):
        candidate = [note, msg, *reversed(kept_reversed)]
        if _reference_request_tokens(candidate) <= target_tokens:
            kept_reversed.append(msg)

    fitted = [note, *reversed(kept_reversed)]
    if len(fitted) > 1:
        logger.info(
            "MoA reference context fitted for %s: %d -> %d messages "
            "(target ~%s tokens, ctx %s)",
            _slot_label(slot),
            len(ref_messages),
            len(fitted),
            f"{target_tokens:,}",
            f"{context_length:,}",
        )
        return fitted

    latest = ref_messages[-1] if ref_messages else {"role": "user", "content": ""}
    text = latest.get("content") if isinstance(latest, dict) else ""
    if not isinstance(text, str):
        text = str(text or "")
    trimmed = _trim_reference_text_to_fit(
        text,
        role=str(latest.get("role") or "user") if isinstance(latest, dict) else "user",
        note=note,
        target_tokens=target_tokens,
    )
    return [note, trimmed]


def _trim_reference_text_to_fit(
    text: str,
    *,
    role: str,
    note: dict[str, str],
    target_tokens: int,
) -> dict[str, str]:
    marker = "\n[... truncated to fit reference model context ...]\n"
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        keep = (low + high) // 2
        if keep >= len(text):
            candidate_text = text
        else:
            head = keep // 2
            tail = keep - head
            candidate_text = f"{text[:head]}{marker}{text[-tail:] if tail else ''}"
        candidate = {"role": role if role in {"user", "assistant"} else "user", "content": candidate_text}
        if _reference_request_tokens([note, candidate]) <= target_tokens:
            best = candidate_text
            low = keep + 1
        else:
            high = keep - 1
    return {"role": role if role in {"user", "assistant"} else "user", "content": best}


def _latest_user_content(messages: list[dict[str, Any]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            return msg["content"]
    return None


def _aggregator_context_length(aggregator: dict[str, str]) -> int | None:
    return _slot_context_length(aggregator)


def _raise_if_aggregator_context_overflow(
    *,
    messages: list[dict[str, Any]],
    aggregator: dict[str, str],
    tools: list[dict[str, Any]] | None,
) -> None:
    context_length = _aggregator_context_length(aggregator)
    if not context_length:
        return

    from agent.model_metadata import estimate_request_tokens_rough

    estimated_tokens = estimate_request_tokens_rough(messages, tools=tools)
    if estimated_tokens <= context_length:
        return

    raise MoAAggregatorContextOverflow(
        estimated_tokens=estimated_tokens,
        context_length=context_length,
        aggregator_label=_slot_label(aggregator),
    )


def aggregate_moa_context(
    *,
    user_prompt: str,
    api_messages: list[dict[str, Any]],
    reference_models: list[dict[str, str]],
    aggregator: dict[str, str],
    temperature: float = 0.6,
    aggregator_temperature: float = 0.4,
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
    """
    reference_outputs: list[tuple[str, str]] = []
    ref_messages = _reference_messages(api_messages)
    reference_outputs = _run_references_parallel(
        reference_models,
        ref_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    joined = "\n\n".join(
        f"Reference {idx} — {label}:\n{text}"
        for idx, (label, text) in enumerate(reference_outputs, start=1)
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
    try:
        response = call_llm(
            task="moa_aggregator",
            messages=[{"role": "user", "content": synth_prompt}],
            temperature=aggregator_temperature,
            max_tokens=max_tokens,
            **_slot_runtime(aggregator),
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
        self._ref_cache_outputs: list[tuple[str, str]] = []
        self._reuse_reference_outputs_after_overflow = False
        self._overflow_retry_user_content: str | None = None

    def _emit(self, event: str, **kwargs: Any) -> None:
        cb = self.reference_callback
        if cb is None:
            return
        try:
            cb(event, **kwargs)
        except Exception as exc:  # pragma: no cover - display must never break the turn
            logger.debug("MoA reference_callback failed for %s: %s", event, exc)

    def create(self, **api_kwargs: Any) -> Any:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import resolve_moa_preset

        preset = resolve_moa_preset(load_config().get("moa") or {}, self.preset_name)
        messages = list(api_kwargs.get("messages") or [])
        reference_models = preset.get("reference_models") or []
        aggregator = preset.get("aggregator") or {}
        # MoA does not cap reference or aggregator output: each model uses its
        # own maximum. Passing max_tokens=None makes call_llm omit the parameter
        # (it never caps by default), so a long aggregator synthesis is never
        # truncated and providers that reject max_tokens don't 400.
        temperature = float(preset.get("reference_temperature", 0.6) or 0.6)
        aggregator_temperature = float(preset.get("aggregator_temperature", api_kwargs.get("temperature") or 0.4) or 0.4)

        # When the preset is disabled, skip the reference fan-out and let the
        # configured aggregator act alone — it is the preset's acting model, so
        # a disabled MoA preset is simply "use the aggregator directly."
        if not preset.get("enabled", True):
            reference_models = []

        reference_outputs: list[tuple[str, str]] = []
        ref_messages = _reference_messages(messages)

        # Turn-scoped cache: only run + display references when the advisory
        # view changed (i.e. a new user turn). Within one turn the agent loop
        # calls create() once per tool iteration with the same advisory view;
        # reuse the cached outputs and skip both the re-run and the re-emit.
        _sig = hashlib.sha256(
            "\u0000".join(
                f"{m.get('role')}:{m.get('content')}" for m in ref_messages
            ).encode("utf-8", "replace")
        ).hexdigest()
        _cache_key = (self.preset_name, _sig, tuple(_slot_label(s) for s in reference_models))
        _refs_from_cache = _cache_key == self._ref_cache_key and bool(self._ref_cache_outputs)
        _latest_user = _latest_user_content(messages)
        _reuse_after_overflow = (
            self._reuse_reference_outputs_after_overflow
            and bool(self._ref_cache_outputs)
            and _latest_user is not None
            and _latest_user == self._overflow_retry_user_content
        )

        if _refs_from_cache or _reuse_after_overflow:
            reference_outputs = list(self._ref_cache_outputs)
            if _reuse_after_overflow:
                self._ref_cache_key = _cache_key
        else:
            reference_outputs = _run_references_parallel(
                reference_models,
                ref_messages,
                temperature=temperature,
                max_tokens=None,
            )
            self._ref_cache_key = _cache_key
            self._ref_cache_outputs = list(reference_outputs)

            # Surface each reference model's answer to the display BEFORE the
            # aggregator acts — once per turn (only on the iteration that
            # actually ran them). The user sees one labelled block per
            # reference (rendered like a thinking block) so the MoA process is
            # visible rather than a silent pause. Best-effort: never blocks the
            # turn.
            _ref_count = len(reference_outputs)
            for _idx, (_label, _text) in enumerate(reference_outputs, start=1):
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

        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_kwargs = dict(api_kwargs)
        agg_messages = [dict(m) for m in messages]
        if reference_outputs:
            joined = "\n\n".join(
                f"Reference {idx} — {label}:\n{text}"
                for idx, (label, text) in enumerate(reference_outputs, start=1)
            )
            guidance = (
                "[Mixture of Agents reference context]\n"
                f"Preset: {self.preset_name}\n"
                f"Aggregator/acting model: {_slot_label(aggregator)}\n"
                f"References: {', '.join(label for label, _ in reference_outputs)}\n\n"
                "Use the reference responses below as private context. You are the aggregator and acting model: "
                "answer the user directly or call tools as needed.\n\n"
                f"{joined}"
            )
            for msg in reversed(agg_messages):
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    msg["content"] = msg["content"] + "\n\n" + guidance
                    break
            else:
                agg_messages.append({"role": "user", "content": guidance})

        try:
            _raise_if_aggregator_context_overflow(
                messages=agg_messages,
                aggregator=aggregator,
                tools=agg_kwargs.get("tools"),
            )
        except MoAAggregatorContextOverflow:
            self._reuse_reference_outputs_after_overflow = True
            self._overflow_retry_user_content = _latest_user
            raise

        self._reuse_reference_outputs_after_overflow = False
        self._overflow_retry_user_content = None
        agg_kwargs["messages"] = agg_messages
        # The aggregator is the acting model. Resolve its slot to the provider's
        # real runtime (base_url/api_key/api_mode) and call it through the same
        # request-building path any model uses — so per-model wire-format
        # handling (anthropic_messages, max_completion_tokens, fixed/forbidden
        # temperature) applies identically to it. MoA imposes no output cap:
        # max_tokens is passed through from the caller (normally None → omitted
        # → the model's real maximum). The preset's old hardcoded 4096 default
        # is gone — it truncated long syntheses.
        return call_llm(
            task="moa_aggregator",
            messages=agg_messages,
            temperature=aggregator_temperature,
            max_tokens=agg_kwargs.get("max_tokens"),
            tools=agg_kwargs.get("tools"),
            extra_body=agg_kwargs.get("extra_body"),
            **_slot_runtime(aggregator),
        )


class MoAClient:
    def __init__(self, preset_name: str, reference_callback: Any = None):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(preset_name, reference_callback=reference_callback)
