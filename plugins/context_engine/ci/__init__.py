"""Context Intelligence plugin: ``plugins/context_engine/ci``.

Two layers, same hard-budget authority:

* Selection: the ``bounded`` engine's deterministic subset (T0 system prompt, T1
  knowledge in the system prompt, T2 recent-window conversation; current request and
  newest pending tool chain always preserved; persisted history never mutated).
* Context intelligence on top, as explicit, budget-bounded, evidence-only tiers
  injected immediately BEFORE the current request:
  - T3 durable memory (built-in MemoryStore, MEMORY.md / USER.md), integrated only
    when the memory config enables the target and the target is not already baked
    into the system prompt (dedup; host memory stays the single source of truth);
  - T4 selective historical-session retrieval (the existing ``session_search``
    tool), fired AT MOST ONCE per request and only on a deterministic recall-intent
    signal (never automatically for every request).

Authority semantics: Knowledge (T0/T1) always precedes supplements and wins on
conflict; every supplement is labelled "[Contextual evidence only — not
authoritative; Knowledge wins.]". T3 is admitted before T4 (higher priority), both
only into headroom under the same hard budget the bounded engine guarantees, capped
by ``supplement_max_ratio``. A second re-estimation stage verifies the final request
against BOTH the canonical rough message estimate and a serialized-prompt-size
estimate and trims the lowest-priority supplements if needed — the returned request
is never over the hard budget by the canonical estimate.

Failure behavior never disturbs the current task: a memory or history-source failure
skips only its tier, a stage error returns the bounded selection unchanged, and the
engine itself fails open (returns ``None``) exactly like ``bounded``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agent.context_engine import sanitize_memory_context
from plugins.context_engine.bounded import (
    BoundedContextEngine,
    _CLAMP_MARKER,
    _current_request_estimator,
    _current_request_index,
    _message_text,
    _token_overlap,
    _word_tokens,
)
from tools.memory_tool_store import ENTRY_DELIMITER, MEMORY_BLOCK_HEADERS

logger = logging.getLogger("plugins.context_engine.ci")

__all__ = ["ContextIntelligenceEngine", "register"]

#: Authority semantics: supplemental tiers are contextual evidence only. Authoritative
#: Knowledge (the preserved system prompt) always precedes them and wins on conflict.
_AUTHORITY_NOTE = "[Contextual evidence only - not authoritative; Knowledge wins.] "

_T3_LABEL = "[durable memory] "
_T4_LABEL = "[historical recall] "

#: Deterministic recall-intent signal that unlocks T4 — the recall phrases a user
#: reaches for ("what did we", "earlier", "last time", ...). Extra terms can be added
#: per instance via ``history_recall_intent_terms``.
_RECALL_INTENT_RE = re.compile(
    r"\b(remember|recall|recalling|remembered|earlier|previous|previously|before|"
    r"last time|what did we|where did we|when did we|we talked|we discussed|we "
    r"worked|remind me|bring back|picked up where|in past sessions|from a past "
    r"session|back to that)\b",
    re.IGNORECASE,
)

#: Capped size below which a retrieved text is eligible for exact-substring dedup
#: against already-selected content (T2 window / T3). Larger texts rely on the digest.
_DEDUP_TEXT_MAX_CHARS = 4000


class ContextIntelligenceEngine(BoundedContextEngine):
    """Bounded selection plus gated, budget-bounded T3 (memory) / T4 (history) tiers."""

    def __init__(self, **kwargs: Any) -> None:
        ci_keys = frozenset({
            "intelligence_enabled",
            "memory_integration_enabled",
            "history_retrieval_enabled",
            "history_retrieval_limit",
            "history_retrieval_max_messages",
            "history_retrieval_max_chars_per_message",
            "history_retrieval_min_relevance",
            "history_retrieval_query_max_words",
            "history_recall_intent_terms",
            "memory_max_entry_chars",
            "supplement_max_ratio",
            "_memory_store_loader",
            "_session_search_fn",
        })
        ci = {k: kwargs.pop(k) for k in list(ci_keys) if k in kwargs}
        super().__init__(**kwargs)
        self.intelligence_enabled = bool(ci.get("intelligence_enabled", True))
        # T3 engine-level switch; the runtime memory-config flags are still respected
        # (production memory stays disabled).
        self.memory_integration_enabled = bool(ci.get("memory_integration_enabled", True))
        # T4 selective history: OFF by default AND gated on a recall-intent signal.
        self.history_retrieval_enabled = bool(ci.get("history_retrieval_enabled", False))
        self.history_retrieval_limit = max(1, min(10, int(ci.get("history_retrieval_limit", 3))))
        self.history_retrieval_max_messages = max(0, int(ci.get("history_retrieval_max_messages", 12)))
        self.history_retrieval_max_chars_per_message = max(
            128, int(ci.get("history_retrieval_max_chars_per_message", 2000))
        )
        self.history_retrieval_min_relevance = max(0, int(ci.get("history_retrieval_min_relevance", 2)))
        self.history_retrieval_query_max_words = max(1, int(ci.get("history_retrieval_query_max_words", 8)))
        self.history_recall_intent_terms = tuple(ci.get("history_recall_intent_terms") or ())
        self.memory_max_entry_chars = max(128, int(ci.get("memory_max_entry_chars", 800)))
        self.supplement_max_ratio = max(0.0, min(1.0, float(ci.get("supplement_max_ratio", 0.15))))
        # Test/config seams: left None they resolve the real implementations lazily.
        self._memory_store_loader = ci.get("_memory_store_loader")
        self._session_search_fn = ci.get("_session_search_fn")
        self._ci_injection_count = 0

    @property
    def name(self) -> str:
        return "ci"

    # -- selection ------------------------------------------------------------------
    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        incoming_message: Optional[Dict[str, Any]] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        if not self.enabled:
            return None
        if not isinstance(request_messages, list) or not request_messages:
            return None
        context_length = budget_tokens or self.context_length or 0
        if context_length <= 0:
            return None
        hard_budget = self._hard_budget_for(context_length)
        if hard_budget <= 0:
            return None

        # Stage 1 (authority): the bounded selection guarantees the hard budget,
        # preserves system/current/pending, and fails open on any error.
        base_out = super().select_context(
            request_messages,
            conversation_messages=conversation_messages,
            incoming_message=incoming_message,
            budget_tokens=budget_tokens,
        )
        if base_out is None:
            return None

        # Stage 2: gated context intelligence. Any failure leaves the bounded
        # selection intact — the request pipeline is never disturbed.
        try:
            final = self._apply_intelligence(base_out, request_messages, hard_budget)
        except Exception as exc:
            logger.warning(
                "ci: context intelligence failed, returning bounded selection unchanged: %s", exc
            )
            return base_out
        return final

    def _apply_intelligence(
        self,
        base_out: List[Dict[str, Any]],
        request_messages: List[Dict[str, Any]],
        hard_budget: int,
    ) -> List[Dict[str, Any]]:
        if not self.intelligence_enabled:
            return base_out
        estimate = _current_request_estimator()
        current_est = estimate(base_out)
        if current_est > hard_budget:
            return base_out  # base already at/over budget; nothing can be added
        cap = max(0, min(hard_budget - current_est, int(hard_budget * self.supplement_max_ratio)))
        if cap <= 0:
            return base_out

        current_idx = _current_request_index(base_out)
        if current_idx is None:
            return base_out

        base_text = "\n".join(_message_text(m) for m in base_out)
        system_text = "\n".join(
            _message_text(m) for m in base_out if isinstance(m, dict) and m.get("role") == "system"
        )

        supplements: List[Tuple[int, int, Dict[str, Any]]] = []  # (priority, est, message)
        used = 0
        memory_count = 0
        for item in self._memory_tier(system_text, base_text):
            est = estimate([item])
            if est <= 0 or used + est > cap:
                continue
            supplements.append((0, est, item))
            used += est
            memory_count += 1

        history_count = 0
        if self.history_retrieval_enabled:
            for item in self._history_tier(request_messages, base_text):
                est = estimate([item])
                if est <= 0 or used + est > cap:
                    continue
                supplements.append((1, est, item))
                used += est
                history_count += 1

        if not supplements:
            return base_out

        # Highest priority first (T3 before T4); insertion order within a tier.
        ordered = sorted(supplements, key=lambda t: t[0])
        block = [m for _, _, m in ordered]
        final = list(base_out[:current_idx]) + block + list(base_out[current_idx:])
        supp_indices = list(range(current_idx, current_idx + len(block)))

        # Second-stage re-estimation: on overflow vs the canonical OR serialized
        # estimate, drop the LOWEST-priority supplement (end of the block) until the
        # final fits. Never trims anything but our own supplements.
        final_est = estimate(final)
        serial_est = _serialized_estimate(final)
        dropped = 0
        while (final_est > hard_budget or serial_est > hard_budget) and supp_indices:
            drop = supp_indices.pop()
            final.pop(drop)
            dropped += 1
            for j in range(len(supp_indices)):
                if supp_indices[j] > drop:
                    supp_indices[j] -= 1
            final_est = estimate(final)
            serial_est = _serialized_estimate(final)
        if final_est > hard_budget or serial_est > hard_budget:
            # Only reachable when the bounded base itself overflows the serialized
            # estimate; the canonical bound (the operational budget) still holds.
            logger.warning("ci: selection over budget after reduction; using bounded selection")
            return base_out

        self._ci_injection_count += 1
        logger.info("ci context intelligence: %s", {
            "session_id": self._session_id,
            "intelligence_enabled": self.intelligence_enabled,
            "memory_messages": memory_count,
            "history_messages": history_count,
            "supplement_messages": len(block) - dropped,
            "dropped_supplements": dropped,
            "base_tokens_est": current_est,
            "selected_tokens_est": final_est,
            "hard_budget": hard_budget,
        })
        return final

    # -- T3: durable memory ----------------------------------------------------------
    def _memory_tier(self, system_text: str, base_text: str) -> List[Dict[str, Any]]:
        if not self.memory_integration_enabled:
            return []
        store = self._resolve_memory_store()
        if store is None:
            return []

        items: List[Dict[str, Any]] = []
        for target in ("memory", "user"):
            try:
                enabled = store.target_enabled(target)
            except Exception:
                enabled = False
            if not enabled:
                continue
            try:
                block = store.format_for_system_prompt(target) or ""
            except Exception as exc:
                logger.warning("ci: memory target %r unreadable, skipping: %s", target, exc)
                continue
            if not block:
                continue
            if MEMORY_BLOCK_HEADERS[target] in system_text:
                continue  # host already baked this target into the system prompt (dedup)
            block = sanitize_memory_context(block)
            for chunk in block.split(ENTRY_DELIMITER):
                chunk = _clamp_text(chunk, self.memory_max_entry_chars)
                if not chunk:
                    continue
                if chunk in base_text:
                    continue
                items.append({
                    "role": "user",
                    "content": _AUTHORITY_NOTE + _T3_LABEL + chunk,
                })
        return items

    def _resolve_memory_store(self) -> Any:
        loader = self._memory_store_loader
        if loader is None:
            from tools.memory_tool import load_on_disk_store as loader
        try:
            return loader()
        except Exception as exc:
            logger.warning("ci: memory unavailable, skipping T3: %s", exc)
            return None

    # -- T4: selective historical retrieval ------------------------------------------
    def _history_tier(self, request_messages: List[Dict[str, Any]], base_text: str) -> List[Dict[str, Any]]:
        if not self.history_retrieval_enabled:
            return []
        current_idx = _current_request_index(request_messages)
        if current_idx is None:
            return []
        request_text = _message_text(request_messages[current_idx])
        if not self._has_recall_intent(request_text):
            return []
        query = _recall_query(request_text, self.history_retrieval_query_max_words)
        if not query:
            return []
        payload = self._run_session_search(query)
        if not payload or not payload.get("success"):
            return []

        items: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        emitted = 0
        for entry in payload.get("results") or []:
            if not isinstance(entry, dict):
                continue
            for m in entry.get("messages") or []:
                if emitted >= self.history_retrieval_max_messages:
                    return items
                if not isinstance(m, dict):
                    continue
                content = _message_text(m)
                if not content:
                    continue
                digest = _digest(content)
                if digest in seen:
                    continue
                seen.add(digest)
                if len(content) <= _DEDUP_TEXT_MAX_CHARS and content in base_text:
                    continue  # already selected in T2/T3 — no amplification
                if _token_overlap(content, request_text) < self.history_retrieval_min_relevance:
                    continue  # relevance floor: sparse topical overlap is excluded
                content = _clamp_text(content, self.history_retrieval_max_chars_per_message)
                items.append({
                    "role": "user",
                    "content": _AUTHORITY_NOTE + _T4_LABEL + content,
                })
                emitted += 1
        return items

    def _has_recall_intent(self, text: str) -> bool:
        if not text:
            return False
        if _RECALL_INTENT_RE.search(text):
            return True
        lowered = text.lower()
        return any(term.strip().lower() in lowered for term in self.history_recall_intent_terms if term)

    def _run_session_search(self, query: str) -> Optional[Dict[str, Any]]:
        fn = self._session_search_fn
        if fn is None:
            from tools.session_search_tool import session_search as fn
        try:
            raw = fn(
                query=query,
                limit=self.history_retrieval_limit,
                detail="adaptive",
                current_session_id=self._session_id,
            )
        except Exception as exc:
            logger.warning("ci: session_search failed, skipping T4: %s", exc)
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                payload: Any = json.loads(raw)
            except (ValueError, TypeError) as exc:
                logger.warning("ci: session_search returned invalid JSON, skipping T4: %s", exc)
                return None
            return payload if isinstance(payload, dict) else None
        return None

    # -- telemetry -------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "engine": self.name,
            "intelligence_enabled": self.intelligence_enabled,
            "memory_integration_enabled": self.memory_integration_enabled,
            "history_retrieval_enabled": self.history_retrieval_enabled,
            "supplement_injections": self._ci_injection_count,
        })
        return status


def register(ctx: Any) -> None:
    """Plugin-loader entry: capture this engine via the discovery collector."""
    ctx.register_context_engine(ContextIntelligenceEngine())


# ---- deterministic helpers ----------------------------------------------------------
def _recall_query(text: str, max_words: int) -> str:
    """FTS5-safe OR query from the request's content words (deterministic, bounded)."""
    words = _word_tokens(text)[: max(1, max_words)]
    return " OR ".join(words) or ""


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()


def _clamp_text(text: str, max_chars: int) -> str:
    """Clamp *text* to *max_chars*, preserving head/tail context plus the marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(_CLAMP_MARKER))
    head = int(keep * 0.7)
    tail = keep - head
    return text[:head] + _CLAMP_MARKER + (text[-tail:] if tail else "")


def _serialized_estimate(messages: List[Dict[str, Any]]) -> int:
    """Serialized-prompt-size estimate: what the wire would actually carry."""
    from agent.model_metadata import estimate_tokens_rough

    return sum(
        estimate_tokens_rough(json.dumps(m, ensure_ascii=False, default=str)) for m in messages
    )