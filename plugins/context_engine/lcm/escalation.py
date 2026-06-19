"""Three-level summarization escalation.

Level 1 (Normal):    LLM summary preserving details
Level 2 (Aggressive): LLM bullet-point summary at half the token budget
Level 3 (Fallback):   Deterministic truncation — no LLM, guaranteed convergence

Each level checks if Tokens(summary) < Tokens(source). If not, escalates.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .model_routing import apply_lcm_model_route
from .tokens import count_tokens

logger = logging.getLogger(__name__)


# Strip inline reasoning blocks emitted by thinking models (MiniMax-M2.7,
# GLM-5.1, Qwen QwQ, DeepSeek R1, etc.) before persisting summary text.
# Without this, the reasoning content — which often quotes the summarizer
# system prompt verbatim — gets stored as the summary and later confuses
# lcm_expand_query, which feeds the summary back to the model as context.
# Tags mirror the set handled in hermes-agent run_agent.py.
_THINK_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thinking|reasoning|thought|REASONING_SCRATCHPAD)\s*>"
    r".*?"
    r"</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)

_DEFAULT_ROUTE_KEY = "<task-default>"


@dataclass
class SummaryCircuitBreaker:
    """In-process circuit breaker for summary model routes.

    The breaker is intentionally small and process-local. It prevents a hot
    compression loop from repeatedly hitting a failing auxiliary route while
    preserving deterministic L3 truncation as the final convergence fallback.
    """

    failure_threshold: int = 2
    cooldown_seconds: int = 300
    _failures: dict[str, int] = field(default_factory=dict)
    _open_until: dict[str, float] = field(default_factory=dict)

    def _key(self, model: str | None) -> str:
        return (model or "").strip() or _DEFAULT_ROUTE_KEY

    def allows(self, model: str | None, *, now: float | None = None) -> bool:
        key = self._key(model)
        current_time = time.monotonic() if now is None else now
        opened_until = self._open_until.get(key, 0.0)
        if opened_until <= current_time:
            if key in self._open_until:
                self._open_until.pop(key, None)
            return True
        return False

    def record_success(self, model: str | None) -> None:
        key = self._key(model)
        self._failures.pop(key, None)
        self._open_until.pop(key, None)

    def record_failure(self, model: str | None, *, now: float | None = None) -> None:
        key = self._key(model)
        failures = self._failures.get(key, 0) + 1
        self._failures[key] = failures
        threshold = max(1, int(self.failure_threshold or 1))
        if failures >= threshold:
            current_time = time.monotonic() if now is None else now
            cooldown = max(0, int(self.cooldown_seconds or 0))
            self._open_until[key] = current_time + cooldown
            logger.warning(
                "LCM summary route circuit opened for %s after %d failure(s); cooldown=%ss",
                key,
                failures,
                cooldown,
            )


def _strip_reasoning_blocks(text: str) -> str:
    """Remove <think>/<thinking>/<reasoning>/<thought>/<REASONING_SCRATCHPAD>
    blocks from ``text``. Idempotent and safe on text without any tags."""
    if not text or "<" not in text:
        return text
    return _THINK_BLOCK_RE.sub("", text)


def _call_llm_for_summary(prompt: str, max_tokens: int,
                           model: str = "", timeout: float | None = None) -> Optional[str]:
    """Call the Hermes auxiliary LLM for summarization."""
    try:
        from agent.auxiliary_client import call_llm
        call_kwargs = {
            "task": "compression",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        apply_lcm_model_route(call_kwargs, model)
        if timeout is not None:
            call_kwargs["timeout"] = timeout
        response = call_llm(**call_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
        return _strip_reasoning_blocks(content).strip()
    except Exception as e:
        logger.warning("LLM summarization failed: %s", e)
        return None


def _invoke_summary_llm(prompt: str, max_tokens: int, model: str = "", timeout: float | None = None) -> Optional[str]:
    kwargs = {"model": model} if model else {}
    if timeout is not None:
        try:
            sig = inspect.signature(_call_llm_for_summary)
            if "timeout" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                kwargs["timeout"] = timeout
        except Exception:
            pass
    return _call_llm_for_summary(prompt, max_tokens, **kwargs)


def _normalized_focus_topic(focus_topic: str, max_chars: int = 160) -> str:
    """Return a single-line, bounded focus topic for prompt injection."""
    normalized = " ".join(str(focus_topic or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "…"


def _build_l1_focus_brief(focus_topic: str) -> str:
    topic = _normalized_focus_topic(focus_topic)
    if not topic:
        return ""
    return (
        "Focus brief:\n"
        f"Primary focus: {topic}\n"
        "Preserve concrete decisions, constraints, files, commands, identifiers, and current state for this focus.\n"
        "Spend roughly 60-70% of the summary budget on the focus when relevant.\n"
        "Do not discard unrelated blockers or active tasks just because they are off-focus.\n"
    )


def _build_l2_focus_brief(focus_topic: str) -> str:
    topic = _normalized_focus_topic(focus_topic)
    if not topic:
        return ""
    return (
        "Focus brief:\n"
        f"Primary focus: {topic}\n"
        "Prefer bullets that preserve decisions, blockers, files, commands, identifiers, and current state for this focus.\n"
        "Keep other active tasks only when they are current blockers or handoff state.\n"
    )


def _summary_model_chain(primary_model: str = "", fallback_models: list[str] | tuple[str, ...] | None = None) -> list[str]:
    chain: list[str] = []
    for model in [primary_model, *(fallback_models or [])]:
        normalized = (model or "").strip()
        if normalized not in chain:
            chain.append(normalized)
    if not chain:
        chain.append("")
    return chain


def _invoke_summary_llm_chain(
    prompt: str,
    max_tokens: int,
    *,
    model: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
    timeout: float | None = None,
    circuit_breaker: SummaryCircuitBreaker | None = None,
    accepts_result: Callable[[str], bool] | None = None,
) -> Optional[str]:
    chain = _summary_model_chain(model, fallback_models)
    skipped = 0
    for candidate_model in chain:
        if circuit_breaker is not None and not circuit_breaker.allows(candidate_model):
            skipped += 1
            logger.warning(
                "LCM summary route skipped by open circuit: %s",
                candidate_model or _DEFAULT_ROUTE_KEY,
            )
            continue
        try:
            result = _invoke_summary_llm(
                prompt,
                max_tokens,
                model=candidate_model,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning("LLM summarization failed: %s", exc)
            result = None
        if result and (accepts_result is None or accepts_result(result)):
            if circuit_breaker is not None:
                circuit_breaker.record_success(candidate_model)
            return result
        if circuit_breaker is not None:
            circuit_breaker.record_failure(candidate_model)
    if skipped == len(chain):
        logger.warning("LCM summary fallback chain exhausted: all routes are temporarily open")
    return None


# PRD-8.3 Prong A — identifier-fidelity instruction. The block is ALWAYS on in
# production (default). The ONLY reason it is toggleable is the AC-5
# baseline-repro arm of the disambiguation campaign: to prove the K=2 bug still
# reproduces, the baseline must run the PRE-FIX summarizer (fidelity OFF) on the
# SAME code. Gated by an internal env var the harness sets per-subprocess; it is
# NOT a user-facing setting. Default (unset) == on.
_L1_IDENTIFIER_FIDELITY = (
    "IDENTIFIER FIDELITY (do not violate): Never merge, group, range-collapse, or\n"
    "abbreviate distinct identifier->value mappings (recovery codes, IDs, keys, file\n"
    "paths, owner/person names). Each distinct identifier and its FULL value must\n"
    "survive verbatim and separately, even when many look similar — similar is NOT\n"
    "repetition. Never write a grouped/range line like \"1300/1600/1900 = Name\"; emit\n"
    "one line per distinct identifier. Never truncate a value mid-word.\n"
)
_L2_IDENTIFIER_FIDELITY = (
    "IDENTIFIER FIDELITY (do not violate): Never merge, group, range-collapse, or\n"
    "truncate distinct identifier->value mappings (codes, IDs, keys, paths, names) —\n"
    "one line per distinct identifier with its FULL value; \"similar\" is NOT a reason\n"
    "to combine. No grouped/range lines like \"1300/1600/1900 = Name\".\n"
)


def _identifier_fidelity_enabled() -> bool:
    """Default ON. Only the AC-5 baseline-repro arm sets this to a falsey value
    to reproduce pre-fix (merge-prone) summarization on identical code."""
    val = os.environ.get("LCM_IDENTIFIER_FIDELITY")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off")


def _build_l1_prompt(text: str, token_budget: int, depth: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    """Level 1: preserve details."""
    depth_guidance = {
        0: "Preserve decisions, rationale, constraints, active tasks, file paths, commands, and specific values.",
        1: "Distill into arc-level outcomes: what evolved, what was decided, current state. Drop per-turn detail.",
        2: "Capture durable narrative: decisions in effect, completed milestones, timeline. Drop process detail.",
    }
    guidance = depth_guidance.get(depth, depth_guidance[2])

    focus_guidance = _build_l1_focus_brief(focus_topic)

    custom_block = ""
    if custom_instructions:
        custom_block = f"\nAdditional instructions:\n{custom_instructions}\n"

    fidelity = _L1_IDENTIFIER_FIDELITY if _identifier_fidelity_enabled() else ""

    return f"""Summarize this conversation segment for future turns.
{guidance}
Remove repetition and conversational filler.
{fidelity}End with: "Expand for details about: <what was compressed>"
{focus_guidance}{custom_block}

Target ~{token_budget} tokens.

CONTENT:
{text}"""


def _build_l2_prompt(text: str, token_budget: int,
                     focus_topic: str = "", custom_instructions: str = "") -> str:
    """Level 2: aggressive bullet points."""

    focus_guidance = _build_l2_focus_brief(focus_topic)

    custom_block = ""
    if custom_instructions:
        custom_block = f"\nAdditional instructions:\n{custom_instructions}\n"

    fidelity = _L2_IDENTIFIER_FIDELITY if _identifier_fidelity_enabled() else ""

    return f"""Compress this into bullet points. Maximum {token_budget} tokens.
Keep only: decisions made, files changed, errors hit, current state.
Drop all reasoning, alternatives considered, and process detail.
{fidelity}{focus_guidance}{custom_block}

CONTENT:
{text}"""


def _deterministic_truncate(text: str, max_tokens: int) -> str:
    """Level 3: no LLM, just truncate deterministically.

    Takes the first and last portions to preserve start context and
    most recent state. Guaranteed to converge.
    """
    if count_tokens(text) <= max_tokens:
        return text

    # Rough char budget (4 chars/token)
    char_budget = max_tokens * 4
    if len(text) <= char_budget:
        return text

    head_budget = int(char_budget * 0.4)
    tail_budget = int(char_budget * 0.4)
    middle = "\n\n[...deterministic truncation — details available via lcm_expand...]\n\n"

    return text[:head_budget] + middle + text[-tail_budget:]


def summarize_with_escalation(
    text: str,
    source_tokens: int,
    token_budget: int,
    depth: int = 0,
    model: str = "",
    timeout: float | None = None,
    l2_budget_ratio: float = 0.50,
    l3_truncate_tokens: int = 512,
    focus_topic: str = "",
    custom_instructions: str = "",
    fallback_models: list[str] | tuple[str, ...] | None = None,
    circuit_breaker: SummaryCircuitBreaker | None = None,
) -> tuple[str, int]:
    """Run 3-level escalation. Returns (summary, level_used).

    Guarantees convergence: level 3 is deterministic and always produces
    output shorter than the source.
    """
    # Level 1: detailed summary
    l1_prompt = _build_l1_prompt(text, token_budget, depth,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l1_result = _invoke_summary_llm_chain(
        l1_prompt,
        token_budget * 2,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        circuit_breaker=circuit_breaker,
        accepts_result=lambda result: count_tokens(result) < source_tokens,
    )

    if l1_result:
        logger.debug("L1 summarization succeeded (%d tokens)", count_tokens(l1_result))
        return l1_result, 1

    # Level 2: aggressive bullets at reduced budget
    l2_budget = int(token_budget * l2_budget_ratio)
    l2_prompt = _build_l2_prompt(text, l2_budget,
                                 focus_topic=focus_topic,
                                 custom_instructions=custom_instructions)
    l2_result = _invoke_summary_llm_chain(
        l2_prompt,
        l2_budget * 2,
        model=model,
        fallback_models=fallback_models,
        timeout=timeout,
        circuit_breaker=circuit_breaker,
        accepts_result=lambda result: count_tokens(result) < source_tokens,
    )

    if l2_result:
        logger.debug("L2 summarization succeeded (%d tokens)", count_tokens(l2_result))
        return l2_result, 2

    # Level 3: deterministic truncation — guaranteed convergence
    l3_result = _deterministic_truncate(text, l3_truncate_tokens)
    logger.debug("L3 deterministic truncation (%d tokens)", count_tokens(l3_result))
    return l3_result, 3
