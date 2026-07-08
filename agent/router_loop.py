"""Model Router runtime: classifier-routed acting model per user turn.

The inverse of Mixture-of-Agents (``agent/moa_loop.py``): instead of fanning a
prompt out to several reference models and letting an aggregator act, a tiny
*classifier* call decides which execution tier the prompt needs ("simple" or
"complex") and the WHOLE turn then runs on that tier's model. Retryable
failures walk the preset's fallback chain (local-first by default), so a dead
local server degrades to the next candidate instead of failing the turn.

Like MoA, the router is a virtual provider: the normal Hermes agent loop still
owns tool calling and turn termination; this module only decides *which* real
model serves each ``create()`` call and forwards the request unchanged.

Routing is sticky per user turn: the classifier runs once when a new user
message arrives, and every tool-loop iteration of that turn reuses the
decision — a task never swaps models mid-flight. Failed candidates are also
remembered per turn, so after the simple tier fails once the remaining
iterations go straight to the surviving candidate.

Mid-stream failures cannot be caught here (the raw token stream is handed to
the consumer); they surface via the consumer's stale-stream retry, which
re-enters ``create()`` — the sticky decision plus the per-turn failed-slot set
then route the retry to the next candidate.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from agent.auxiliary_client import call_llm
from agent.moa_loop import (
    _extract_text,
    _maybe_apply_moa_cache_control,
    _slot_label,
    _slot_runtime,
)

logger = logging.getLogger(__name__)

# System prompt for the classification side-call. The verdict contract is a
# single word so the call stays tiny (a few hundred input tokens, ~1 output
# token) even on a metered classifier provider.
_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a model-routing classifier. Read the conversation snippet and "
    "decide which execution tier the NEXT reply needs. Answer with exactly "
    'one word: "simple" or "complex". No punctuation, no explanation.\n\n'
    "simple = casual conversation, chit-chat, quick questions, short "
    "texting-style replies, small factual lookups, image descriptions, "
    "greetings, reminders about trivial things.\n"
    "complex = coding, debugging, multi-step tasks, heavy tool use (files, "
    "terminal, browser, scheduling, web research), long documents, planning, "
    "anything likely to take many steps or careful reasoning."
)

# Per-message character budget for the classifier's context snippet. The
# classifier only needs the gist of recent turns to judge the tier — replaying
# full messages would multiply classifier cost for no routing benefit.
_CLASSIFIER_CONTEXT_CHAR_BUDGET = 240

# Head+tail budget for the new user message in the classifier prompt. Both
# ends are kept so intent expressed at either end (a leading question or a
# trailing "…can you code that up?") survives truncation.
_CLASSIFIER_MESSAGE_CHAR_BUDGET = 2000

# Markers that disqualify a short message from the experimental
# short-circuit: code fences and URLs signal a technical task regardless of
# message length.
_SHORT_CIRCUIT_DISQUALIFIERS = re.compile(r"```|https?://", re.IGNORECASE)

# Top-level chat-completions request kwargs the conversation loop (or its
# request_overrides config) may set that call_llm's signature doesn't carry.
# They're folded into extra_body — the OpenAI SDK merges extra_body into the
# JSON request body, which is wire-equivalent to passing the kwarg — so
# routed turns keep tool_choice / response_format / service_tier ("fast"
# tier!) / sampling params instead of silently dropping them.
_PASSTHROUGH_REQUEST_KEYS = (
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "stop",
    "top_p",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "metadata",
    "service_tier",
    "prediction",
    "modalities",
    "reasoning_effort",
    "verbosity",
)

# Floor for the acting call's timeout when neither the caller nor the slot
# provider's config supplies one. The acting call is a full agent turn: a
# small local model prefilling a ~20K-token agent prompt can take minutes on
# consumer hardware before the first token, so call_llm's 30s auxiliary
# default is far too tight. Sized to cover local prefill + generation while
# still bounding a genuinely hung endpoint.
_ACTING_TIMEOUT_FLOOR_SECONDS = 600.0


def _message_text(message: Any) -> str:
    """Best-effort plain text of one chat message (str or parts list)."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif part.get("type") in {"image_url", "image"}:
                    parts.append("[image]")
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(p for p in parts if p)
    return "" if content is None else str(content)


def _truncate_head_tail(text: str, budget: int) -> str:
    """Head+tail preview with an omission marker (same shape as MoA's
    ``_truncate_tool_result``)."""
    if not text or len(text) <= budget:
        return text
    half = budget // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n[... {omitted} chars omitted ...]\n{text[-half:]}"


def _last_user_index(messages: list[dict[str, Any]]) -> int | None:
    """Index of the last real user message, or None."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return None


def make_agent_decision_relay(agent: Any):
    """Decision-callback that relays router events to an agent's
    ``tool_progress_callback`` — every surface that consumes the tool
    lifecycle (CLI, TUI, desktop, gateway) can render the routing decision
    inline. Display-only, best-effort, never touches history.

    Shared by initial client construction (``agent_init``) and the
    mid-session ``/model`` switch (``switch_model``) so routing notes don't
    silently disappear after a live switch onto the router.
    """

    def _relay(event: str, **kwargs: Any) -> None:
        cb = getattr(agent, "tool_progress_callback", None)
        if cb is None:
            return
        try:
            if event == "router.decision":
                cb(
                    "router.decision",
                    str(kwargs.get("label") or ""),
                    str(kwargs.get("tier") or ""),
                    None,
                    router_classifier=kwargs.get("classifier_label"),
                    router_cached=kwargs.get("cached"),
                )
            elif event == "router.fallback":
                cb(
                    "router.fallback",
                    str(kwargs.get("from_label") or ""),
                    str(kwargs.get("to_label") or ""),
                    None,
                    router_error=kwargs.get("error"),
                )
        except Exception:
            pass

    return _relay


class _FailureMarkingStream:
    """Iterator proxy that marks the acting candidate failed on stream death.

    The streaming path hands the acting model's RAW token stream to the
    consumer, so a provider that dies mid-stream raises during the CALLER's
    iteration — after ``create()`` already returned. Without this proxy that
    failure never reaches ``_failed_labels``, and the consumer's stale-stream
    retry re-enters ``create()`` on the SAME dead candidate instead of the
    next one in the chain.

    ``__getattr__`` forwards everything else (``close()``, SDK internals) to
    the underlying stream. Note it forwards ``choices`` lookups too, so the
    consumer's "complete response pretending to be a stream" sniff
    (``hasattr(stream, "choices")``) keeps working for adapters that return a
    whole response despite ``stream=True``.

    A consumer-initiated cancel (user interrupt force-closing the socket)
    also raises through iteration and marks the candidate failed; that only
    biases the remainder of the SAME user turn toward the fallback chain and
    resets on the next turn's fresh classification — accepted over trying to
    distinguish cancel from death here.
    """

    def __init__(self, inner: Any, on_error: Any) -> None:
        self._inner = inner
        self._on_error = on_error

    def __iter__(self) -> Any:
        try:
            for chunk in self._inner:
                yield chunk
        except Exception:
            try:
                self._on_error()
            except Exception:  # pragma: no cover - marking must never mask
                pass
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _build_classifier_messages(
    messages: list[dict[str, Any]],
    *,
    platform: str | None,
    channel_hints: dict[str, str],
    context_messages: int,
) -> list[dict[str, Any]]:
    """Build the tiny classification prompt from the live conversation.

    The classifier sees: an optional channel-bias hint, the last N
    conversational turns as truncated one-liners, and the new user message
    (head+tail truncated). The agent's own system prompt is deliberately NOT
    included — it is large, cache-hostile for the classifier provider, and
    irrelevant to the simple/complex judgement.
    """
    last_user = _last_user_index(messages)
    convo = [m for m in messages if m.get("role") in {"user", "assistant"}]
    current = convo[-1] if convo and last_user is not None and messages[last_user] is convo[-1] else None
    if current is not None:
        recent = convo[:-1][-max(context_messages, 0):] if context_messages > 0 else []
    else:
        recent = convo[-max(context_messages, 0):] if context_messages > 0 else []

    lines: list[str] = []
    hint = (channel_hints or {}).get((platform or "").strip().lower())
    if platform and hint:
        lines.append(
            f"Channel: {platform} (hint: bias toward \"{hint}\" unless the "
            "message is clearly the other tier)"
        )
        lines.append("")
    if recent:
        lines.append("Recent context:")
        for m in recent:
            text = _truncate_head_tail(
                _message_text(m).strip(), _CLASSIFIER_CONTEXT_CHAR_BUDGET
            )
            if text:
                lines.append(f"[{m.get('role')}] {text}")
        lines.append("")
    lines.append("New message:")
    new_text = _message_text(messages[last_user]) if last_user is not None else ""
    lines.append(_truncate_head_tail(new_text.strip(), _CLASSIFIER_MESSAGE_CHAR_BUDGET) or "(empty)")

    return [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _parse_verdict(raw: str, default_route: str) -> str:
    """Leniently parse the classifier's output into a tier name.

    Exact-word scan: "complex" wins over "simple" when both appear (a model
    that hedges with "simple... actually complex" meant the stronger tier);
    anything unrecognizable falls open to ``default_route``.
    """
    text = (raw or "").strip().lower()
    if not text:
        return default_route
    if re.search(r"\bcomplex\b", text):
        return "complex"
    if re.search(r"\bsimple\b", text):
        return "simple"
    return default_route


class RouterChatCompletions:
    """OpenAI-chat-compatible facade where a classifier picks the acting model."""

    def __init__(
        self,
        preset_name: str,
        decision_callback: Any = None,
        platform: str | None = None,
    ):
        self.preset_name = preset_name or "default"
        # Optional display hook, mirroring MoA's reference_callback. Called as
        # routing events happen so frontends can show the decision inline.
        # Signature: decision_callback(event, **kwargs) where event is one of:
        #   "router.decision" kwargs: tier, label, classifier_label, cached
        #   "router.fallback" kwargs: from_label, to_label, error
        # Never raises into the model call — display is best-effort.
        self.decision_callback = decision_callback
        # Gateway platform of the session ("whatsapp", "telegram", ...) when
        # known; feeds the preset's channel_hints bias into the classifier.
        self.platform = platform
        # Sticky-per-user-turn decision cache. Keyed by a hash of the message
        # prefix up to the LAST USER message (same technique as MoA's
        # user_turn fanout), so tool-loop iterations 2..N of one turn reuse
        # the decision and never reclassify or swap models mid-task.
        self._decision_key: tuple | None = None
        self._decision: dict[str, Any] | None = None
        # Labels of candidates that already failed during the current turn.
        # Reset when the decision key changes (a new user turn). Makes the
        # fallback walk sticky too: after the simple tier fails once, the
        # remaining iterations go straight to the surviving candidate.
        self._failed_labels: set[str] = set()
        # Classifier usage/cost from the most recent cache-MISS create(),
        # awaiting consumption by session accounting. The attribute names
        # (consume_reference_usage / last_aggregator_slot /
        # consume_and_save_trace) intentionally match the MoA client protocol
        # duck-typed by agent/conversation_loop.py, so the router needs no
        # conversation-loop changes: the "reference" usage here is the
        # classifier side-call, and the "aggregator" slot is the acting slot.
        from agent.usage_pricing import CanonicalUsage

        self._pending_classifier_usage: Any = CanonicalUsage()
        self._pending_classifier_cost: Any = None
        # Resolved acting slot ({provider, model}) actually used by the most
        # recent create() — post-fallback, so session cost accounting prices
        # the turn at the real model that served it.
        self.last_aggregator_slot: Any = None
        # Trace parts stashed on a decision cache MISS, flushed by the caller
        # via consume_and_save_trace (only when router.save_traces is on).
        self._pending_trace: Any = None

    # ── MoA-protocol members (duck-typed by conversation_loop.py) ──────────

    def consume_reference_usage(self) -> tuple[Any, Any]:
        """Pop pending classifier usage + cost, resetting both to empty.

        Same contract as ``MoAChatCompletions.consume_reference_usage`` (the
        conversation loop consumes either client identically): returns
        ``(CanonicalUsage, cost_usd_or_None)`` and clears the pending values
        so a repeat read cannot double-count.
        """
        from agent.usage_pricing import CanonicalUsage

        usage = self._pending_classifier_usage or CanonicalUsage()
        cost = self._pending_classifier_cost
        self._pending_classifier_usage = CanonicalUsage()
        self._pending_classifier_cost = None
        return usage, cost

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending routed-turn trace to disk, if one is pending.

        No-op when tracing is off, or when there is no pending trace (a
        cache-HIT iteration made no new decision). ``aggregator_output_fallback``
        carries the resolved streamed acting text, exactly like the MoA
        client's flush (see ``MoAChatCompletions.consume_and_save_trace``).
        """
        pending = self._pending_trace
        self._pending_trace = None
        if not pending:
            return
        try:
            from agent.router_trace import save_router_turn

            acting_slot = pending.get("acting_slot") or {}
            acting_output = pending.get("acting_output")
            if acting_output is None and aggregator_output_fallback:
                acting_output = aggregator_output_fallback
            save_router_turn(
                session_id=session_id,
                preset_name=pending.get("preset", ""),
                platform=pending.get("platform"),
                classifier=pending.get("classifier") or {},
                route=pending.get("route") or {},
                fallback_events=pending.get("fallback_events") or [],
                acting_label=_slot_label(acting_slot) if acting_slot else "",
                acting_model=acting_slot.get("model"),
                acting_provider=acting_slot.get("provider"),
                acting_output=acting_output,
                acting_streamed=bool(pending.get("acting_streamed")),
            )
        except Exception as exc:  # pragma: no cover - tracing must never break a turn
            logger.debug("Router trace flush failed: %s", exc)

    def _emit(self, event: str, **kwargs: Any) -> None:
        cb = self.decision_callback
        if cb is None:
            return
        try:
            cb(event, **kwargs)
        except Exception as exc:  # pragma: no cover - display must never break the turn
            logger.debug("Router decision_callback failed for %s: %s", event, exc)

    # ── Classification ──────────────────────────────────────────────────────

    def _classify(
        self,
        preset: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        """Run the classification side-call; returns ``(tier, trace_record)``.

        Fail-open by design: a classifier error, timeout, or unparseable
        verdict routes to the preset's ``default_route`` (local "simple" by
        default) so chat keeps working when the classifier provider is down.
        The failure is recorded in the trace record for offline audit.
        """
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, normalize_usage

        default_route = preset.get("default_route") or "simple"
        classifier_slot = preset.get("classifier") or {}
        record: dict[str, Any] = {
            "provider": classifier_slot.get("provider"),
            "model": classifier_slot.get("model"),
            "failed": False,
            "error": None,
            "usage": CanonicalUsage(),
            "cost_usd": None,
        }

        if not preset.get("enabled", True):
            record["verdict"] = default_route
            record["raw_output"] = None
            record["skipped"] = "preset_disabled"
            return default_route, record

        # Experimental short-circuit: trivially short messages without code
        # fences / URLs skip the metered classifier call entirely.
        short_circuit = int(preset.get("short_circuit_chars") or 0)
        if short_circuit > 0:
            last_user = _last_user_index(messages)
            text = (_message_text(messages[last_user]) if last_user is not None else "").strip()
            if text and len(text) <= short_circuit and not _SHORT_CIRCUIT_DISQUALIFIERS.search(text):
                record["verdict"] = "simple"
                record["raw_output"] = None
                record["skipped"] = "short_circuit"
                return "simple", record

        classifier_messages = _build_classifier_messages(
            messages,
            platform=self.platform,
            channel_hints=preset.get("channel_hints") or {},
            context_messages=int(preset.get("classifier_context_messages") or 4),
        )
        record["input_messages"] = classifier_messages
        runtime = _slot_runtime(classifier_slot)
        try:
            response = call_llm(
                task="router_classifier",
                messages=classifier_messages,
                temperature=0,
                max_tokens=int(preset.get("classifier_max_tokens") or 16),
                **runtime,
            )
            raw = _extract_text(response)
            record["raw_output"] = raw
            tier = _parse_verdict(raw, default_route)
            record["verdict"] = tier
            raw_usage = getattr(response, "usage", None)
            if raw_usage:
                try:
                    usage = normalize_usage(
                        raw_usage,
                        provider=runtime.get("provider"),
                        api_mode=runtime.get("api_mode"),
                    )
                    record["usage"] = usage
                    # Price the classifier at ITS OWN model rate — it may run
                    # on a different provider than the acting model.
                    cost = estimate_usage_cost(
                        classifier_slot.get("model") or "",
                        usage,
                        provider=runtime.get("provider"),
                        base_url=runtime.get("base_url"),
                        api_key=runtime.get("api_key"),
                    )
                    record["cost_usd"] = cost.amount_usd
                except Exception:  # pragma: no cover - defensive
                    pass
            return tier, record
        except Exception as exc:
            logger.warning(
                "Router classifier %s failed (%s); failing open to '%s'",
                _slot_label(classifier_slot),
                exc,
                default_route,
            )
            record["failed"] = True
            record["error"] = str(exc)
            record["raw_output"] = None
            record["verdict"] = default_route
            return default_route, record

    # ── Candidate execution ──────────────────────────────────────────────────

    def _maybe_preload_lmstudio(self, slot: dict[str, Any], runtime: dict[str, Any]) -> None:
        """Best-effort LM Studio context guard for a routed slot.

        The agent's own ``_ensure_lmstudio_runtime_loaded`` only fires for
        ``provider == "lmstudio"`` agents, which a router agent is not — so a
        routed lmstudio slot is ensured here instead.
        ``ensure_lmstudio_model_loaded`` internally caps the requested context
        at the model's own maximum, so asking for the 64K floor loads a
        smaller model at its real max.

        Runs on EVERY acting call, not once per process: LM Studio evicts
        idle models under memory pressure and JIT-reloads them at its own
        small defaults (e.g. 8K + TTL), which silently breaks the local tier
        mid-session. The ensure call is a cheap local probe when the model is
        already loaded with sufficient context, and a corrective reload when
        it isn't. Never blocks a turn on failure.
        """
        if (slot.get("provider") or "").strip().lower() != "lmstudio":
            return
        label = _slot_label(slot)
        try:
            from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
            from hermes_cli.models import ensure_lmstudio_model_loaded

            ensure_lmstudio_model_loaded(
                slot.get("model") or "",
                runtime.get("base_url"),
                runtime.get("api_key"),
                MINIMUM_CONTEXT_LENGTH,
            )
        except Exception as exc:  # pragma: no cover - guard must never break a turn
            logger.debug("Router LM Studio context guard skipped for %s: %s", label, exc)

    def create(self, **api_kwargs: Any) -> Any:
        from hermes_cli.config import load_config
        from hermes_cli.router_config import resolve_router_preset

        from agent.error_classifier import classify_api_error
        from agent.usage_pricing import CanonicalUsage

        preset = resolve_router_preset(load_config().get("router") or {}, self.preset_name)
        messages = list(api_kwargs.get("messages") or [])

        # Sticky-per-user-turn decision. Hash the message prefix up to the
        # last user message (the same technique MoA's user_turn fanout uses):
        # tool-loop iterations grow the tail with assistant/tool messages but
        # leave that prefix unchanged, so iteration 2..N is a cache HIT and
        # the turn stays on the model that started it.
        last_user = _last_user_index(messages)
        sig_messages = messages[: last_user + 1] if last_user is not None else messages
        sig = hashlib.sha256(
            "\u0000".join(
                f"{m.get('role')}:{_message_text(m)}" for m in sig_messages
            ).encode("utf-8", "replace")
        ).hexdigest()
        cache_key = (self.preset_name, sig)

        if cache_key == self._decision_key and self._decision is not None:
            decision = self._decision
            # The classifier already ran (and was accounted/traced) on the
            # MISS earlier this turn; a repeat iteration must not re-charge
            # or re-trace it.
            self._pending_classifier_usage = CanonicalUsage()
            self._pending_classifier_cost = None
            self._pending_trace = None
            cached = True
        else:
            tier, classifier_record = self._classify(preset, messages)
            decision = {
                "tier": tier,
                "classifier_record": classifier_record,
                "hint": (preset.get("channel_hints") or {}).get(
                    (self.platform or "").strip().lower()
                ),
            }
            self._decision_key = cache_key
            self._decision = decision
            self._failed_labels = set()
            usage = classifier_record.get("usage")
            self._pending_classifier_usage = usage if isinstance(usage, CanonicalUsage) else CanonicalUsage()
            self._pending_classifier_cost = classifier_record.get("cost_usd")
            cached = False

        tier = decision["tier"]
        routes = preset.get("routes") or {}
        primary = routes.get(tier) or routes.get(preset.get("default_route") or "simple") or {}

        # Candidate chain: the routed tier first, then the preset's fallbacks,
        # de-duplicated by label, minus candidates that already failed this
        # turn. If everything failed already, retry the full chain rather
        # than erroring outright — a provider may have recovered between
        # iterations.
        chain: list[dict[str, str]] = []
        seen: set[str] = set()
        for slot in [primary, *(preset.get("fallbacks") or [])]:
            if not slot or not slot.get("model"):
                continue
            label = _slot_label(slot)
            if label in seen:
                continue
            seen.add(label)
            chain.append(dict(slot))
        live_chain = [s for s in chain if _slot_label(s) not in self._failed_labels] or chain

        if not cached:
            self._emit(
                "router.decision",
                tier=tier,
                label=_slot_label(live_chain[0]) if live_chain else "",
                classifier_label=_slot_label(preset.get("classifier") or {}),
                cached=False,
            )
            self._pending_trace = {
                "preset": self.preset_name,
                "platform": self.platform,
                "classifier": decision["classifier_record"],
                "route": {
                    "tier": tier,
                    "provider": primary.get("provider"),
                    "model": primary.get("model"),
                    "hint": decision.get("hint"),
                    "short_circuited": decision["classifier_record"].get("skipped") == "short_circuit",
                },
                "fallback_events": [],
            }

        # Streaming passthrough — same contract as the MoA aggregator call:
        # when the consumer asks for a stream, return the acting model's RAW
        # token stream; the consumer reassembles chunks + tool_calls and owns
        # stale-stream retries. Non-streaming is byte-for-byte a plain call.
        stream = bool(api_kwargs.get("stream"))
        stream_kwargs: dict[str, Any] = {}
        if stream:
            stream_kwargs["stream"] = True
            stream_kwargs["stream_options"] = (
                api_kwargs.get("stream_options") or {"include_usage": True}
            )
            # NOTE: the caller's timeout is folded into acting_timeout below
            # (passed explicitly to call_llm), not into stream_kwargs.

        # Preserve request kwargs call_llm's signature doesn't carry by
        # folding them into extra_body (see _PASSTHROUGH_REQUEST_KEYS). An
        # explicit extra_body key from the caller wins over a passthrough.
        passthrough = {
            key: api_kwargs[key]
            for key in _PASSTHROUGH_REQUEST_KEYS
            if api_kwargs.get(key) is not None
        }
        caller_extra_body = api_kwargs.get("extra_body")
        acting_extra_body = (
            {**passthrough, **caller_extra_body}
            if isinstance(caller_extra_body, dict)
            else (passthrough or caller_extra_body)
        )

        last_exc: Exception | None = None
        for idx, slot in enumerate(live_chain):
            label = _slot_label(slot)
            runtime = _slot_runtime(slot)
            self._maybe_preload_lmstudio(slot, runtime)
            acting_messages = _maybe_apply_moa_cache_control(
                [dict(m) for m in messages], runtime
            )
            # The acting call IS the whole agent turn — call_llm's 30s
            # auxiliary default would cut off a local model prefilling a
            # 20K-token agent prompt. Use the caller's timeout when given,
            # else the slot provider's configured request timeout, else a
            # generous floor sized for local prefill + generation.
            acting_timeout = api_kwargs.get("timeout")
            if acting_timeout is None:
                try:
                    from hermes_cli.timeouts import get_provider_request_timeout

                    acting_timeout = get_provider_request_timeout(
                        slot.get("provider") or "", slot.get("model")
                    )
                except Exception:  # pragma: no cover - defensive
                    acting_timeout = None
            if acting_timeout is None:
                acting_timeout = _ACTING_TIMEOUT_FLOOR_SECONDS
            # Expose the slot BEFORE the call so accounting prices a
            # partially-streamed failure at the model that actually ran.
            self.last_aggregator_slot = dict(slot)
            if self._pending_trace is not None:
                self._pending_trace["acting_slot"] = dict(slot)
            try:
                response = call_llm(
                    task="router_acting",
                    messages=acting_messages,
                    temperature=api_kwargs.get("temperature"),
                    # The transport may have emitted the GPT-5-style
                    # max_completion_tokens spelling; call_llm re-derives the
                    # right per-provider spelling from a plain max_tokens.
                    max_tokens=api_kwargs.get("max_tokens") or api_kwargs.get("max_completion_tokens"),
                    tools=api_kwargs.get("tools"),
                    extra_body=acting_extra_body,
                    timeout=acting_timeout,
                    **stream_kwargs,
                    **runtime,
                )
            except Exception as exc:
                last_exc = exc
                self._failed_labels.add(label)
                verdict = None
                try:
                    verdict = classify_api_error(
                        exc, provider=slot.get("provider") or "", model=slot.get("model") or ""
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
                error_class = getattr(getattr(verdict, "reason", None), "value", None) or type(exc).__name__
                next_slot = live_chain[idx + 1] if idx + 1 < len(live_chain) else None
                if next_slot is None:
                    # Chain exhausted — re-raise and let the agent's own
                    # fallback_providers / credential-pool machinery own it.
                    raise
                # Any call-time failure walks the chain: even "non-retryable"
                # classes (auth on that provider, billing exhaustion, context
                # overflow on a small local model) are exactly the situations
                # a different candidate can survive — that resilience is the
                # point of the local-first fallback order.
                logger.warning(
                    "Router acting model %s failed (%s); falling back to %s",
                    label,
                    exc,
                    _slot_label(next_slot),
                )
                self._emit(
                    "router.fallback",
                    from_label=label,
                    to_label=_slot_label(next_slot),
                    error=str(exc),
                )
                if self._pending_trace is not None:
                    self._pending_trace.setdefault("fallback_events", []).append(
                        {
                            "from": label,
                            "to": _slot_label(next_slot),
                            "error_class": error_class,
                            "error": str(exc),
                        }
                    )
                continue

            if self._pending_trace is not None:
                if stream:
                    self._pending_trace["acting_streamed"] = True
                    self._pending_trace["acting_output"] = None
                else:
                    self._pending_trace["acting_streamed"] = False
                    try:
                        self._pending_trace["acting_output"] = _extract_text(response)
                    except Exception:  # pragma: no cover - defensive
                        self._pending_trace["acting_output"] = None
            if stream and not hasattr(response, "choices"):
                # Genuine token stream: proxy it so a mid-stream death marks
                # this candidate failed before the consumer's stale-stream
                # retry re-enters create() — the retry then starts at the
                # next candidate instead of the one that just died.
                failed_label = label

                def _mark_failed(_label: str = failed_label) -> None:
                    self._failed_labels.add(_label)

                return _FailureMarkingStream(response, _mark_failed)
            return response

        # Defensive: live_chain is never empty (routes always exist), but if a
        # hand-broken preset produced no candidates, surface a clear error.
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"Router preset '{self.preset_name}' has no usable route candidates"
        )


class RouterClient:
    """Thin wrapper exposing the router facade as ``client.chat.completions``,
    mirroring ``MoAClient``. Implements the same accounting/trace protocol the
    conversation loop duck-types, so router turns are priced at the acting
    model and classifier spend is folded into session accounting."""

    def __init__(
        self,
        preset_name: str,
        decision_callback: Any = None,
        platform: str | None = None,
    ):
        self.chat = type("_RouterChat", (), {})()
        self.chat.completions = RouterChatCompletions(
            preset_name, decision_callback=decision_callback, platform=platform
        )

    def consume_reference_usage(self) -> Any:
        """Pop the pending classifier usage from the completions facade."""
        return self.chat.completions.consume_reference_usage()

    @property
    def last_aggregator_slot(self) -> Any:
        """Acting slot ({provider, model}) actually used by the most recent
        create() — post-fallback — so session cost accounting prices the turn
        at the real model instead of the virtual preset name."""
        return getattr(self.chat.completions, "last_aggregator_slot", None)

    def consume_and_save_trace(
        self, session_id: Any = None, aggregator_output_fallback: Any = None
    ) -> None:
        """Flush the pending routed-turn trace via the completions facade."""
        return self.chat.completions.consume_and_save_trace(
            session_id, aggregator_output_fallback=aggregator_output_fallback
        )
