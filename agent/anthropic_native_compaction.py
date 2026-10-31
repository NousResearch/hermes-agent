"""Anthropic on-demand compaction (beta ``compact-2026-09-04``) as the compression summarizer.

Opt-in with ``compression.anthropic_native: true``. On the direct Anthropic Messages route, for
models that support compaction, the compression summary is written by the conversation's own
model over EXACTLY the messages of the last request it answered. That request just wrote the
prompt cache, so the summary call reads it back instead of re-sending the conversation to a
second model, and the summarizer sees the whole conversation, its earlier thinking included.

The readable text of the returned ``compaction`` block becomes the Hermes summary: head, verbatim
tail, persistence, redaction and every later request stay on the existing compressor pipeline,
so a session can still fall back to another provider. The signed block itself is not replayed.

Every ineligible state (other route or model, caching off, stale or foreign request, a tail that
would not keep the turns the request did not carry) and every failure (HTTP error, no block,
truncated summary) returns ``None``: the configured auxiliary summarizer then runs as before.
Docs: https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.anthropic_endpoints import _is_third_party_anthropic_endpoint

logger = logging.getLogger(__name__)

COMPACTION_BETA = "compact-2026-09-04"
# API limit for ``compaction.instructions``.
INSTRUCTIONS_MAX_CHARS = 16_384
# Output cap for the summary call, thinking included. Measured: a detailed checkpoint of a
# ~700K-token Opus conversation used up to ~14K output tokens at low effort.
COMPACTION_MAX_TOKENS = 20_000
# Effort for the summary call. Changing effort keeps the cache read (measured), and a checkpoint
# does not need the deep reasoning a coding turn may run at.
COMPACTION_EFFORT = "low"
# A captured request is reused only while the cache it wrote is still alive.
_CACHE_TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}
_TTL_SAFETY_MARGIN_SECONDS = 60.0
# Rejected on a compaction request (API docs): never forwarded.
_DROPPED_KWARGS = ("stream", "stop_sequences", "context_management")
_FORCED_TOOL_CHOICE_TYPES = frozenset({"any", "tool"})

# Minimum versions with on-demand compaction; fable/mythos ship it from their first release.
_MIN_VERSION = {"opus": (4, 6), "sonnet": (4, 6), "haiku": (5, 5), "fable": (5, 0), "mythos": (5, 0)}
_MODEL_RE = re.compile(
    # Semantic minors are short: claude-opus-4-20250514 stays 4.0, not 4.20250514.
    r"claude[-_.](opus|sonnet|haiku|fable|mythos)(?:[-_.](\d+)(?:[-_.](\d{1,2})(?=$|[-_.\[]))?)?",
    re.IGNORECASE,
)


def model_supports_compaction(model: Any) -> bool:
    """True for Claude models documented with on-demand compaction (unknown ids stay False)."""
    if not isinstance(model, str):
        return False
    match = _MODEL_RE.search(model.strip())
    if not match:
        return False
    family = match.group(1).lower()
    if match.group(2) is None:
        return family in ("fable", "mythos")  # e.g. claude-mythos-preview
    version = (int(match.group(2)), int(match.group(3) or 0))
    return version >= _MIN_VERSION[family]


def native_compaction_eligible(agent: Any) -> bool:
    """Per-call gate: opt-in, direct Anthropic Messages, supported model, prompt caching on."""
    if getattr(agent, "anthropic_native_compaction", False) is not True:
        return False
    if not getattr(agent, "compression_enabled", True) or getattr(agent, "_anthropic_native_compaction_rejected", False):
        return False
    if getattr(agent, "api_mode", None) != "anthropic_messages":
        return False
    if _is_third_party_anthropic_endpoint(getattr(agent, "base_url", None)):
        return False
    if not getattr(agent, "_use_prompt_caching", False) or getattr(agent, "_cache_ttl", None) not in _CACHE_TTL_SECONDS:
        return False
    return model_supports_compaction(getattr(agent, "model", None))


def _content_hash(value: Any) -> int:
    if isinstance(value, str):
        return hash(value)  # cached on the str object: cheap on every later request
    return hash(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False))


def message_fingerprint(message: Any) -> tuple:
    """What a request carries from one history row: role, content, tool calls and tool result id."""
    if not isinstance(message, dict):
        return ("?", _content_hash(repr(message)))
    tool_calls = message.get("tool_calls")
    return (
        message.get("role"), _content_hash(message.get("content")), message.get("tool_call_id"),
        _content_hash(tool_calls) if tool_calls else 0,
    )


@dataclass
class CompactionSeed:
    """The last request the main model answered on this agent, and the history rows it was built from."""

    kwargs: dict[str, Any]
    fingerprints: tuple
    session_id: str
    model: str
    captured_at: float = field(default_factory=time.monotonic)


def record_compaction_seed(agent: Any, api_kwargs: Any, messages: Any) -> None:
    """Remember an ANSWERED request (call after a valid response). Never raises."""
    try:
        if not native_compaction_eligible(agent) or not isinstance(api_kwargs, dict) or not isinstance(messages, list):
            return
        if not isinstance(api_kwargs.get("messages"), list) or not api_kwargs["messages"]:
            return
        agent._anthropic_compaction_seed = CompactionSeed(
            kwargs=api_kwargs, fingerprints=tuple(message_fingerprint(m) for m in messages),
            session_id=str(getattr(agent, "session_id", "") or ""), model=str(getattr(agent, "model", "") or ""),
        )
    except Exception:
        logger.debug("Recording the Anthropic compaction seed failed (non-fatal)", exc_info=True)


def _seed_max_age(agent: Any) -> float:
    return _CACHE_TTL_SECONDS.get(str(getattr(agent, "_cache_ttl", "") or ""), 0.0) - _TTL_SAFETY_MARGIN_SECONDS


def _merged_betas(client: Any, kwargs: dict[str, Any]) -> str:
    """The request's beta list (its own header overrides the client's) plus the compaction beta."""
    headers = kwargs.get("extra_headers")
    headers = headers if isinstance(headers, dict) else {}
    raw = headers.get("anthropic-beta")
    if raw is None:
        raw = (getattr(client, "_custom_headers", None) or {}).get("anthropic-beta") or ""
    betas = [b.strip() for b in str(raw).split(",") if b.strip()]
    return ",".join(betas + ([] if COMPACTION_BETA in betas else [COMPACTION_BETA]))


def build_compaction_kwargs(seed_kwargs: dict[str, Any], instructions: str, *, client: Any = None) -> dict[str, Any]:
    """The answered request, unchanged up to its last message, turned into a compaction request.

    Prefix-affecting fields (``system``, ``tools``, ``messages``, ``thinking``) are kept so the cache
    the request wrote is read back; only fields the API rejects with ``compaction`` are removed.
    """
    kwargs = {k: v for k, v in seed_kwargs.items() if k not in _DROPPED_KWARGS}
    tool_choice = kwargs.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") in _FORCED_TOOL_CHOICE_TYPES:
        kwargs.pop("tool_choice")
    output_config = kwargs.get("output_config")
    if isinstance(output_config, dict):
        output_config = {k: v for k, v in output_config.items() if k != "format"}
        if "effort" in output_config:
            output_config["effort"] = COMPACTION_EFFORT
        budget = output_config.get("task_budget")
        if isinstance(budget, dict) and "remaining" in budget:
            output_config["task_budget"] = {k: v for k, v in budget.items() if k != "remaining"}
        if output_config:
            kwargs["output_config"] = output_config
        else:
            kwargs.pop("output_config")
    max_tokens = COMPACTION_MAX_TOKENS
    thinking = kwargs.get("thinking")
    if isinstance(thinking, dict) and isinstance(thinking.get("budget_tokens"), int):
        max_tokens = max(max_tokens, thinking["budget_tokens"] + COMPACTION_MAX_TOKENS)  # max_tokens must exceed it
    kwargs["max_tokens"] = max_tokens
    kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), "anthropic-beta": _merged_betas(client, seed_kwargs)}
    kwargs["extra_body"] = {
        **(kwargs.get("extra_body") or {}), "compaction": {"type": "summarize", "instructions": instructions},
    }
    return kwargs


def compaction_usage(data: dict[str, Any]) -> dict[str, int]:
    """Billed usage: the top-level counters are zero, the call is reported in ``usage.iterations``."""
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    iterations = [it for it in (usage.get("iterations") or []) if isinstance(it, dict)] or [usage]
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return {key: sum(int(it.get(key) or 0) for it in iterations) for key in keys}


def compaction_summary_text(data: dict[str, Any]) -> Optional[str]:
    """The readable summary, only when the call ended normally with a non-empty block."""
    if data.get("stop_reason") != "compaction":
        return None
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "compaction":
            text = block.get("content")
            return text if isinstance(text, str) and text.strip() else None
    return None


@dataclass
class NativeSummary:
    text: str
    model: str
    duration_ms: int
    usage: dict[str, int]


class AnthropicNativeSummary:
    """Summary source handed to ``ContextCompressor.compress(native_summary=...)`` for one attempt."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent  # lives for one compression attempt only

    def covered_prefix(self, messages: list[dict[str, Any]]) -> Optional[int]:
        """How many leading rows of ``messages`` the captured request carried, or None when it cannot be used."""
        agent = self._agent
        seed = getattr(agent, "_anthropic_compaction_seed", None) if agent is not None else None
        if agent is None or not isinstance(seed, CompactionSeed) or not native_compaction_eligible(agent):
            return None
        reason = None
        if seed.session_id != str(getattr(agent, "session_id", "") or "") or seed.model != str(agent.model or ""):
            reason = "captured for another session or model"
        elif time.monotonic() - seed.captured_at > _seed_max_age(agent):
            reason = "older than the prompt-cache lifetime"
        elif len(seed.fingerprints) > len(messages) or any(
            message_fingerprint(m) != fp for m, fp in zip(messages, seed.fingerprints)
        ):
            reason = "history changed since the request"
        if reason:
            logger.info("Anthropic native compaction skipped: last request %s; using the auxiliary summarizer", reason)
            return None
        return len(seed.fingerprints)

    def summarize(self, instructions: str) -> Optional[NativeSummary]:
        """One compaction request over the captured request; None on any failure (caller falls back)."""
        agent = self._agent
        seed = getattr(agent, "_anthropic_compaction_seed", None) if agent is not None else None
        if agent is None or not isinstance(seed, CompactionSeed):
            return None
        if not instructions.strip() or len(instructions) > INSTRUCTIONS_MAX_CHARS:
            logger.info("Anthropic native compaction skipped: instructions are %d chars (limit %d)",
                        len(instructions), INSTRUCTIONS_MAX_CHARS)
            return None
        agent._anthropic_compaction_seed = None  # one use: the history is about to be rewritten
        client = getattr(agent, "_anthropic_client", None)
        if client is None:
            return None
        kwargs = build_compaction_kwargs(seed.kwargs, instructions, client=client)
        import anthropic
        from agent.auxiliary_client import _effective_aux_timeout
        started = time.monotonic()
        try:
            raw = client.messages.with_raw_response.create(**kwargs, timeout=_effective_aux_timeout("compression", None))
            data = raw.http_response.json()
        except (anthropic.APIError, OSError, ValueError) as exc:  # HTTP/transport errors, unreadable JSON body
            text = str(exc).lower()
            if getattr(exc, "status_code", None) == 400 and ("compaction" in text or "compact-2026" in text):
                agent._anthropic_native_compaction_rejected = True  # structural: stop trying this session
                logger.warning("Anthropic rejected native compaction (%s); disabled for this session", exc)
            else:
                logger.warning("Anthropic native compaction failed (%s); using the auxiliary summarizer", exc)
            return None
        duration_ms = int((time.monotonic() - started) * 1000)
        usage = compaction_usage(data if isinstance(data, dict) else {})
        model = str((data or {}).get("model") or seed.model)
        _record_usage(agent, model, usage)
        summary = compaction_summary_text(data if isinstance(data, dict) else {})
        if summary is None:
            logger.warning("Anthropic native compaction returned no summary (stop_reason=%s, stop_details=%s); using "
                           "the auxiliary summarizer", (data or {}).get("stop_reason"), (data or {}).get("stop_details"))
            return None
        logger.info(
            "Anthropic native compaction summary: model=%s seconds=%.1f input=%d cache_read=%d cache_write=%d "
            "output=%d chars=%d", model, duration_ms / 1000, usage["input_tokens"], usage["cache_read_input_tokens"],
            usage["cache_creation_input_tokens"], usage["output_tokens"], len(summary),
        )
        return NativeSummary(text=summary, model=model, duration_ms=duration_ms, usage=usage)


def _record_usage(agent: Any, model: str, usage: dict[str, int]) -> None:
    """Bill the summary call to the session like any auxiliary compression call (best-effort)."""
    try:
        from types import SimpleNamespace
        from agent.aux_accounting import record_aux_usage
        record_aux_usage(SimpleNamespace(model=model, usage=usage), "compression", provider="anthropic",
                         base_url=getattr(agent, "base_url", None))
    except Exception:
        logger.debug("Recording Anthropic compaction usage failed (non-fatal)", exc_info=True)


def native_summary_for(agent: Any) -> Optional[AnthropicNativeSummary]:
    """The summary source for this attempt, or None when native compaction cannot apply."""
    if not native_compaction_eligible(agent) or not isinstance(getattr(agent, "_anthropic_compaction_seed", None),
                                                                 CompactionSeed):
        return None
    return AnthropicNativeSummary(agent)
