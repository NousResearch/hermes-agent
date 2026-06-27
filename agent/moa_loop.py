"""Mixture-of-Agents runtime helpers for /moa turns.

The slash command is deliberately not a model tool. It marks one user turn as
MoA-enabled; the normal Hermes agent loop still owns tool calling and turn
termination, while this module gathers reference-model context before each model
iteration.
"""

from __future__ import annotations

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
        response = call_llm(
            task="moa_reference",
            messages=ref_messages,
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


def _reference_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build an advisory-safe view of the conversation for reference models.

    Reference calls are advisory: they never call tools and never emit the
    ``tool_calls`` the main model did. Replaying the full transcript verbatim
    (a) re-bills the ~8K-token Hermes system prompt per reference per
    iteration and (b) risks 400s from strict providers (Mistral, Fireworks)
    that reject orphan ``tool`` messages or ``tool_calls`` the reference never
    produced. We keep only the user/assistant *text* turns, dropping the
    system prompt, any ``tool``-role messages, and any ``tool_calls`` payloads.
    """
    trimmed: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            # Drop system prompt and tool-result messages.
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            # Skip non-text (multimodal/tool-call-only) assistant turns.
            if not content:
                continue
        text = content if isinstance(content, str) else ""
        if role == "assistant" and not text.strip():
            # Assistant turn that was purely tool calls — nothing advisory.
            continue
        trimmed.append({"role": role, "content": text})
    if not trimmed:
        # Degenerate case (e.g. first turn was stripped): fall back to a
        # minimal user turn so the reference still has something to answer.
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return [{"role": "user", "content": msg["content"]}]
    return trimmed



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

    def __init__(self, preset_name: str):
        self.preset_name = preset_name or "default"

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
        reference_outputs = _run_references_parallel(
            reference_models,
            ref_messages,
            temperature=temperature,
            max_tokens=None,
        )

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

        if aggregator.get("provider") == "moa":
            raise RuntimeError("MoA aggregator cannot be another MoA preset")
        agg_kwargs = dict(api_kwargs)
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
    def __init__(self, preset_name: str):
        self.chat = type("_MoAChat", (), {})()
        self.chat.completions = MoAChatCompletions(preset_name)
