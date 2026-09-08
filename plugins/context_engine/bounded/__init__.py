"""Bounded context selection plugin: ``plugins/context_engine/bounded``.

Replaces the pre-generation context for every provider request with a deterministic
subset that (a) preserves the system prompt, the current user request and the newest
pending tool chain, (b) stays within a *hard budget* derived from the model context
window (``budget_ratio`` of the window minus an output/tool-schema reserve), and (c)
never mutates persisted history or the original request message list.

This is "selection", not "compression": ``compress()`` shrinks an over-long transcript
via the wrapped compressor lifecycle, while ``select_context`` bounds the request that
goes on the wire each turn. Selection is deterministic, uses no LLM embeddings, and
fails open (returns ``None``) on any error so the host's normal preflight/compression
path still governs.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from agent.context_engine import ContextEngine

if TYPE_CHECKING:
    from agent.context_compressor import ContextCompressor

logger = logging.getLogger("plugins.context_engine.bounded")

__all__ = ["BoundedContextEngine", "register"]

#: Truncation marker inserted where an oversized retained tool result is clamped.
_CLAMP_MARKER = "\n...[clamped by hermes bounded context]...\n"

SelectResult = Tuple[List[Dict[str, Any]], int, int, int, int, int, str]


class BoundedContextEngine(ContextEngine):
    """Plugin context engine: bounded, deterministic, replay-safe request selection."""

    #: Attributes the host may write on the plugin that govern the wrapped compressor's
    #: compaction policy; mirrored to it so the plugin stays the single interface.
    _MIRRORED_ATTRS = frozenset({
        "model_thresholds",
        "_micro_compact_enabled",
        "_micro_compact_every_n_turns",
        "_micro_compact_defrag_threshold_tokens",
    })

    def __init__(
        self,
        *,
        budget_ratio: float = 0.5,
        reserve_cap_ratio: float = 0.20,
        output_reserve_tokens: int = 4096,
        tool_schema_reserve_tokens: int = 8192,
        recent_window_messages: int = 40,
        min_recent_tail_messages: int = 8,
        head_anchor_messages: int = 3,
        max_relevant_messages: int = 24,
        relevance_floor_score: int = 2,
        relevance_min_tokens: int = 1024,
        relevance_max_tokens: int = 8192,
        max_tool_result_chars: int = 12000,
        clamp_head_ratio: float = 0.7,
        clamp_tail_ratio: float = 0.2,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_compressor", None)
        object.__setattr__(self, "model", "")
        object.__setattr__(self, "_session_id", None)
        self.enabled = bool(enabled)
        self.budget_ratio = float(budget_ratio)
        self.reserve_cap_ratio = float(reserve_cap_ratio)
        self.output_reserve_tokens = int(output_reserve_tokens)
        self.tool_schema_reserve_tokens = int(tool_schema_reserve_tokens)
        self.recent_window_messages = max(1, int(recent_window_messages))
        self.min_recent_tail_messages = max(1, int(min_recent_tail_messages))
        self.head_anchor_messages = max(0, int(head_anchor_messages))
        self.max_relevant_messages = max(0, int(max_relevant_messages))
        self.relevance_floor_score = max(0, int(relevance_floor_score))
        self.relevance_min_tokens = max(0, int(relevance_min_tokens))
        self.relevance_max_tokens = max(0, int(relevance_max_tokens))
        self.max_tool_result_chars = max(256, int(max_tool_result_chars))
        self.clamp_head_ratio = float(clamp_head_ratio)
        self.clamp_tail_ratio = float(clamp_tail_ratio)
        self._selection_count = 0

    @property
    def name(self) -> str:
        return "bounded"

    # -- attribute mirroring to the wrapped compressor ---------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._MIRRORED_ATTRS:
            inner = self.__dict__.get("_compressor")
            if inner is not None:
                setattr(inner, name, value)
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        inner = self.__dict__.get("_compressor")
        if inner is not None and hasattr(inner, name):
            return getattr(inner, name)
        raise AttributeError(name)

    # -- compressor lifecycle (delegated; owns compaction policy) ---------------------
    def _ensure_compressor(self) -> "ContextCompressor":
        inner = self.__dict__.get("_compressor")
        if inner is not None:
            return inner
        from agent.context_compressor import ContextCompressor

        inner = ContextCompressor(
            model=getattr(self, "model", "") or "unknown",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=6,
            quiet_mode=True,
        )
        object.__setattr__(self, "_compressor", inner)
        return inner

    def _inner(self) -> Optional["ContextCompressor"]:
        return self.__dict__.get("_compressor")

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        object.__setattr__(self, "model", model)
        super().update_model(
            model,
            context_length,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
        )
        try:
            inner = self._ensure_compressor()
            inner.update_model(
                model,
                context_length,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                api_mode=api_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("bounded: compressor update_model failed: %s", exc)

    def bind_session_state(self, session_state: Any) -> None:
        try:
            inner = self._ensure_compressor()
            return inner.bind_session_state(session_state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("bounded: bind_session_state failed: %s", exc)
            return None

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        object.__setattr__(self, "_session_id", session_id)
        inner = self._inner()
        if inner is not None and hasattr(inner, "on_session_start"):
            try:
                return inner.on_session_start(session_id, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("bounded: on_session_start failed: %s", exc)
                return None
        return None

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        inner = self._inner()
        if inner is not None and hasattr(inner, "on_session_end"):
            try:
                return inner.on_session_end(session_id, messages)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("bounded: on_session_end failed: %s", exc)
                return None
        return None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        inner = self._inner()
        if inner is not None and hasattr(inner, "on_session_reset"):
            try:
                inner.on_session_reset()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("bounded: on_session_reset failed: %s", exc)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        inner = self._inner()
        if inner is not None:
            return inner.update_from_response(usage)
        pt = (usage or {}).get("prompt_tokens") or 0
        ct = (usage or {}).get("completion_tokens") or 0
        self.last_prompt_tokens = int(pt)
        self.last_completion_tokens = int(ct)
        self.last_total_tokens = int(pt) + int(ct)

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        inner = self._inner()
        if inner is None:
            return False
        return inner.should_compress(prompt_tokens)  # type: ignore[invalid-argument-type]

    def should_compress_info(
        self, prompt_tokens: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        inner = self._inner()
        if inner is None or not hasattr(inner, "should_compress_info"):
            return self.should_compress(prompt_tokens), None
        return inner.should_compress_info(prompt_tokens)  # type: ignore[invalid-argument-type]

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        inner = self._inner()
        if inner is None:
            return messages
        return inner.compress(
            messages,
            current_tokens=current_tokens,
            focus_topic=focus_topic,
            force=force,
            memory_context=memory_context,
        )

    def prune_tool_results_only(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        inner = self._inner()
        if inner is None:
            return messages, 0
        return inner.prune_tool_results_only(messages, current_tokens=current_tokens)

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        inner = self._inner()
        if inner is None:
            return True
        return inner.has_content_to_compress(messages)

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "engine": self.name,
            "selection_enabled": self.enabled,
            "selection_count": self._selection_count,
        })
        inner = self._inner()
        if inner is not None and hasattr(inner, "get_status"):
            try:
                inner_status = inner.get_status()
                if isinstance(inner_status, dict):
                    status.update(inner_status)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("bounded: get_status() failed: %s", exc)
        return status

    # ---- hard budget ----------------------------------------------------------------
    def _hard_budget_for(self, context_length: int) -> int:
        def _capped(value: int) -> int:
            return min(int(value), int(context_length * 0.10))

        reserve = min(
            int(context_length * self.reserve_cap_ratio),
            _capped(self.output_reserve_tokens)
            + _capped(self.tool_schema_reserve_tokens),
        )
        return max(0, int(context_length * self.budget_ratio) - reserve)

    # ---- selection ------------------------------------------------------------------
    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        incoming_message: Optional[Dict[str, Any]] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:  # type: ignore[invalid-method-override]
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
        try:
            result = self._select(request_messages, hard_budget)
        except Exception as exc:  # fail open: never break the request pipeline
            logger.warning(
                "bounded: selection failed, leaving request unchanged: %s", exc
            )
            return None
        self._selection_count += 1
        if result is None:
            return None
        selected, in_tokens, out_tokens, omitted, pruned, clamped, mode = result
        safe = {
            "session_id": self._session_id,
            "mode": mode,
            "input_messages": len(request_messages),
            "output_messages": len(selected),
            "input_tokens_est": in_tokens,
            "selected_tokens_est": out_tokens,
            "budget_tokens": context_length,
            "hard_budget": hard_budget,
            "omitted_messages": omitted,
            "pruned_tool_results": pruned,
            "clamped_tool_results": clamped,
        }
        logger.info("bounded selection: %s", safe)
        return selected

    def _select(
        self, request_messages: List[Dict[str, Any]], hard_budget: int
    ) -> Optional[SelectResult]:
        estimate = _current_request_estimator()
        in_tokens = estimate(request_messages)
        if in_tokens <= hard_budget:
            return request_messages, in_tokens, in_tokens, 0, 0, 0, "short"

        n = len(request_messages)
        system_idx = {i for i, m in enumerate(request_messages) if _is_system(m)}
        systems_msgs = [request_messages[i] for i in sorted(system_idx)]
        first_non_system = next((i for i in range(n) if i not in system_idx), 0)
        current_idx = _current_request_index(request_messages)
        if current_idx is None:
            current_idx = first_non_system
        pending = _newest_tool_group_indices(request_messages, after_index=current_idx)

        mandatory = set(system_idx) | {current_idx} | pending
        mandatory_est = sum(estimate([request_messages[i]]) for i in mandatory)
        if mandatory_est > hard_budget:
            return None
        remaining = hard_budget - mandatory_est

        window_ids: List[int] = []
        est_window = 0
        limit_front = max(0, current_idx - self.recent_window_messages)
        j = current_idx - 1
        while j >= limit_front:
            m = request_messages[j]
            if len(window_ids) < self.min_recent_tail_messages:
                window_ids.append(j)
                est_window += estimate([m])
                j -= 1
                continue
            if est_window + estimate([m]) > remaining:
                break
            window_ids.append(j)
            est_window += estimate([m])
            j -= 1
        window_ids.reverse()

        mandatory_ids = set(mandatory)
        keep_ids = sorted(set(window_ids) | (mandatory_ids - set(system_idx)))
        trimmable_identities = {
            id(request_messages[i]) for i in set(window_ids) if i not in mandatory_ids
        }
        original_identities = {id(request_messages[i]) for i in keep_ids}
        protected_positions = {pos for pos, i in enumerate(keep_ids) if i in pending}

        window_msgs = [request_messages[i] for i in keep_ids]
        window_msgs = self._clamp_window(
            window_msgs, protected_positions, hard_budget, estimate
        )
        clamped = sum(1 for m in window_msgs if id(m) not in original_identities)
        window_msgs, pruned = _enforce_pair_validity(window_msgs)

        final = systems_msgs + window_msgs
        final_est = estimate(final)
        if final_est > hard_budget:
            final, final_est = _trim_from_front_until_fit(
                final, hard_budget, estimate, trimmable_identities=trimmable_identities
            )
            final, pruned_extra = _enforce_pair_validity(final)
            pruned += pruned_extra
            final_est = estimate(final)
        if final_est > hard_budget:
            return None

        room = hard_budget - final_est
        if room >= self.relevance_min_tokens and (
            self.head_anchor_messages > 0 or self.max_relevant_messages > 0
        ):
            preamble = self._build_preamble(
                request_messages,
                current_idx,
                system_idx,
                current_idx,
                room,
                estimate,
                exclude_indices=set(keep_ids),
            )
            if preamble:
                candidate = systems_msgs + preamble + window_msgs
                candidate_est = estimate(candidate)
                if candidate_est <= hard_budget:
                    final, final_est = candidate, candidate_est

        omitted = n - len(final)
        mode = "selected_pruned" if (clamped or pruned) else "selected"
        return final, in_tokens, final_est, omitted, pruned, clamped, mode

    def _clamp_window(
        self,
        window_msgs: List[Dict[str, Any]],
        protected: Set[int],
        hard_budget: int,
        estimate: Callable[[List[Dict[str, Any]]], int],
    ) -> List[Dict[str, Any]]:
        if estimate(window_msgs) <= hard_budget:
            return window_msgs
        for k, m in enumerate(window_msgs):
            if k in protected:
                continue
            if not _is_tool_result(m):
                continue
            text = _message_text(m)
            if len(text) <= self.max_tool_result_chars:
                continue
            head = min(
                int(len(text) * self.clamp_head_ratio),
                int(self.max_tool_result_chars * 0.8),
            )
            tail = min(
                int(len(text) * self.clamp_tail_ratio),
                int(self.max_tool_result_chars * 0.2),
            )
            clone = dict(m)
            clone["content"] = (
                text[:head] + _CLAMP_MARKER + (text[-tail:] if tail else "")
            )
            window_msgs[k] = clone
            if estimate(window_msgs) <= hard_budget:
                break
        return window_msgs

    def _build_preamble(
        self,
        request_messages: List[Dict[str, Any]],
        prefix_end: int,
        system_idx: Set[int],
        current_idx: int,
        room: int,
        estimate: Callable[[List[Dict[str, Any]]], int],
        exclude_indices: Set[int],
    ) -> List[Dict[str, Any]]:
        if room < self.relevance_min_tokens and self.head_anchor_messages <= 0:
            return []
        picked: List[Tuple[int, Dict[str, Any]]] = []
        picked_idx: Set[int] = set(exclude_indices)
        picked_tokens = 0
        current_text = _message_text(request_messages[current_idx])
        need_anchor = self.head_anchor_messages
        prefix_end = min(prefix_end, len(request_messages))

        def _consider(i: int, m: Dict[str, Any]) -> None:
            nonlocal picked_tokens
            tok = estimate([m])
            if tok <= 0 or picked_tokens + tok > room:
                return
            picked.append((i, m))
            picked_tokens += tok

        for i in range(prefix_end):
            if need_anchor <= 0:
                break
            if i in picked_idx or i in system_idx:
                continue
            if not _is_standalone(request_messages[i]):
                continue
            if not _message_text(request_messages[i]):
                continue
            _consider(i, request_messages[i])
            need_anchor -= 1

        available = self.max_relevant_messages + self.head_anchor_messages
        if (
            room - picked_tokens >= self.relevance_min_tokens
            and len(picked) < available
        ):
            allowed = min(self.relevance_max_tokens, room - picked_tokens)
            for i in range(prefix_end):
                if len(picked) >= available:
                    break
                if i in picked_idx or i in system_idx:
                    continue
                m = request_messages[i]
                if not _is_standalone(m):
                    continue
                text = _message_text(m)
                if not text:
                    continue
                if _token_overlap(text, current_text) < self.relevance_floor_score:
                    continue
                tok = estimate([m])
                if tok <= 0 or tok > 4096:
                    continue
                if picked_tokens + tok > allowed:
                    continue
                _consider(i, m)

        picked.sort(key=lambda item: item[0])
        return [m for _, m in picked]


def register(ctx: Any) -> None:
    """Plugin-loader entry: capture this engine via the discovery collector."""
    ctx.register_context_engine(BoundedContextEngine())


# ---- message classification ---------------------------------------------------------
def _is_system(m: Any) -> bool:
    return isinstance(m, dict) and m.get("role") == "system"


def _is_assistant(m: Any) -> bool:
    return isinstance(m, dict) and m.get("role") == "assistant"


def _is_tool_call(m: Any) -> bool:
    return _is_assistant(m) and bool(m.get("tool_calls"))


def _is_tool_result(m: Any) -> bool:
    return (
        isinstance(m, dict) and m.get("role") == "tool" and bool(m.get("tool_call_id"))
    )


def _is_standalone(m: Any) -> bool:
    return not (_is_system(m) or _is_tool_call(m) or _is_tool_result(m))


def _message_text(m: Any) -> str:
    content = m.get("content") if isinstance(m, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                for key in ("text", "content"):
                    value = block.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return "\n".join(parts)
    return ""


def _current_request_index(request_messages: List[Dict[str, Any]]) -> Optional[int]:
    """Index of the live user request (last user message without a tool_call_id).

    When the trailing user message carries list-content (multimodal attachment/MoA
    epilogue), the actual text request is the previous user message.
    """
    last_user = None
    for i in range(len(request_messages) - 1, -1, -1):
        m = request_messages[i]
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and not m.get("tool_call_id")
        ):
            last_user = i
            break
    if last_user is None:
        return None
    content = request_messages[last_user].get("content")
    if isinstance(content, list):
        for i in range(last_user - 1, -1, -1):
            m = request_messages[i]
            if (
                isinstance(m, dict)
                and m.get("role") == "user"
                and not m.get("tool_call_id")
            ):
                return i
    return last_user


def _newest_tool_group_indices(
    request_messages: List[Dict[str, Any]],
    after_index: int,
) -> Set[int]:
    """Indices of the in-flight tool chain strictly newer than ``after_index``.

    Only an assistant ``tool_calls`` message that appears AFTER the live user request
    (plus its contiguous results) counts as the pending group; a completed tool group
    below the current request is ordinary history and is trimmable.
    """
    n = len(request_messages)
    for i in range(n - 1, after_index - 1, -1):
        if _is_tool_call(request_messages[i]):
            indices = {i}
            j = i + 1
            while j < n and _is_tool_result(request_messages[j]):
                indices.add(j)
                j += 1
            return indices
    return set()


def _enforce_pair_validity(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop tool results whose tool_calls pair is not in the selection (pair-validity).

    Runs after any message is trimmed so a retained tool result never dangles without
    its initiating assistant message.
    """
    created_ids = set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                created_ids.add(tc["id"])
    kept: List[Dict[str, Any]] = []
    pruned = 0
    for m in messages:
        if _is_tool_result(m) and m.get("tool_call_id") not in created_ids:
            pruned += 1
            continue
        kept.append(m)
    return kept, pruned


def _trim_from_front_until_fit(
    final: List[Dict[str, Any]],
    hard_budget: int,
    estimate: Callable[[List[Dict[str, Any]]], int],
    trimmable_identities: Set[int],
) -> Tuple[List[Dict[str, Any]], int]:
    """Safety net: drop the oldest trimmable message until the selection fits the budget.

    ``trimmable_identities`` covers the verbatim older window messages only — never the
    system prompt, the current request, the pending tool chain, or messages already
    replaced by a clamped copy — so the request invariants cannot be violated here.
    """
    removable = [
        i
        for i, m in enumerate(final)
        if not _is_system(m) and id(m) in trimmable_identities
    ]
    while estimate(final) > hard_budget and removable:
        i = removable.pop(0)
        del final[i]
        removable = [j - 1 if j > i else j for j in removable]
    return final, estimate(final)


# ---- deterministic relevance --------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_'-]{2,}", re.UNICODE)

_STOPWORDS = frozenset(
    """
a about after again all also am an and any are as at be because been before being between
both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me
more most my myself no nor not now of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves
""".split()
)


def _word_tokens(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _token_overlap(a: str, b: str) -> int:
    wa = _word_tokens(a)
    wb = _word_tokens(b)
    if not wa or not wb:
        return 0
    return len(set(wa) & set(wb))


# ---- rough token estimator (host-canonical, imported lazily) ------------------------
_COMMON_ESTIMATOR = None


def _current_request_estimator() -> Callable[[List[Dict[str, Any]]], int]:
    global _COMMON_ESTIMATOR
    if _COMMON_ESTIMATOR is None:
        from agent.model_metadata import estimate_messages_tokens_rough

        _COMMON_ESTIMATOR = estimate_messages_tokens_rough
    return _COMMON_ESTIMATOR
