"""Lightweight recall routing for external memory providers.

The router decides whether the current turn should pay the cost/noise of
external-memory prefetch.  It is intentionally fed a *small, clean* context:
current user text, a bounded recent-turn window, and lightweight platform/session
metadata.  It never receives tool outputs, injected memory-context blocks, or the
full transcript.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agent.memory_manager import sanitize_context

logger = logging.getLogger(__name__)

_SHORT_ACK_RE = re.compile(
    r"^\s*(ok|okay|好的?|嗯+|唔+|收到|了解|继续|行|可以|是的|对|哈哈+|233+|草|[👍👌🙏😂🤣❤️💜~～。！？!?.\s]+)\s*$",
    re.IGNORECASE,
)
_PAST_REFERENCE_RE = re.compile(
    r"(上次|之前|以前|刚刚|前面|当时|继续(那个|上次)?|我们(讨论|说|决定|聊)过|remember|last time|previously|earlier)",
    re.IGNORECASE,
)
_EVIDENCE_RE = re.compile(
    r"(原文|证据|记录|消息|transcript|quote|verbatim|怎么说的|当时.*为什么|查一下历史|找一下)",
    re.IGNORECASE,
)
_LONG_TERM_RE = re.compile(
    r"(记忆|图记忆|Graphiti|memory|recall|Hermes|配置|认证|权限|gateway|APISIX|SpiceDB|session_search|provider|prefetch|MCP|服务器|偏好|身份|关系|决策|架构)",
    re.IGNORECASE,
)

_DEPTH_ORDER = {"none": 0, "light": 1, "standard": 2, "deep": 3, "evidence": 4}
_BUDGET_ORDER = {"tiny": 0, "small": 1, "medium": 2, "large": 3}
_SENSITIVE_METADATA_KEY_RE = re.compile(r"(user|chat|thread|session|gateway).*(_?id|key)$|^.*_id$", re.IGNORECASE)


def _sanitize_gate_metadata(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return low-risk platform/session metadata for the optional LLM gate.

    The gate only needs coarse routing hints.  Preserve labels such as platform,
    but hash stable identifiers so auxiliary providers never see raw Discord /
    Telegram / gateway IDs.
    """
    safe: Dict[str, Any] = {}
    for key, value in dict(metadata or {}).items():
        key_str = str(key)
        if value is None:
            safe[key_str] = ""
            continue
        value_str = str(value)
        if _SENSITIVE_METADATA_KEY_RE.search(key_str):
            if value_str:
                digest = hashlib.sha256(value_str.encode("utf-8", "ignore")).hexdigest()[:12]
                safe[key_str] = f"sha256:{digest}"
            else:
                safe[key_str] = ""
        else:
            safe[key_str] = value
    return safe


@dataclass
class RecallDecision:
    should_recall: bool
    query: str = ""
    depth: str = "none"
    sources: List[str] = field(default_factory=list)
    budget: str = "tiny"
    provenance: str = "ids"
    reason: str = ""
    mode: str = "auto"
    decision_source: str = "heuristic"

    def normalized(self, *, max_depth: str = "standard", max_budget: str = "small") -> "RecallDecision":
        depth = self.depth if self.depth in _DEPTH_ORDER else "light"
        budget = self.budget if self.budget in _BUDGET_ORDER else "tiny"
        if _DEPTH_ORDER[depth] > _DEPTH_ORDER.get(max_depth, 2):
            depth = max_depth
        if _BUDGET_ORDER[budget] > _BUDGET_ORDER.get(max_budget, 1):
            budget = max_budget
        sources = [s for s in (self.sources or []) if isinstance(s, str)]
        if not sources and self.should_recall:
            sources = ["graph"]
        return RecallDecision(
            should_recall=bool(self.should_recall),
            query=(self.query or "").strip(),
            depth=depth,
            sources=sources,
            budget=budget,
            provenance=self.provenance or "ids",
            reason=(self.reason or "").strip(),
            mode=self.mode or "auto",
            decision_source=self.decision_source or "heuristic",
        )


def stable_recall_key(query: str, *, mode: str = "auto", depth: str = "light", sources: Optional[Iterable[str]] = None) -> str:
    """Return a compact key for duplicate-suppression of equivalent recall work."""
    norm_query = re.sub(r"\s+", " ", sanitize_context(query or "").strip().lower())[:500]
    norm_sources = ",".join(sorted(str(s).strip().lower() for s in (sources or []) if str(s).strip()))
    payload = f"{mode}|{depth}|{norm_sources}|{norm_query}"
    return hashlib.sha256(payload.encode("utf-8", "ignore")).hexdigest()[:16]


def build_recall_gate_context(
    current_message: str,
    recent_messages: Iterable[Mapping[str, Any]] = (),
    *,
    platform_context: Optional[Mapping[str, Any]] = None,
    session_metadata: Optional[Mapping[str, Any]] = None,
    max_turns: int = 6,
    max_chars: int = 4000,
) -> Dict[str, Any]:
    """Build the small, clean context used by the recall gate.

    Only user/assistant text snippets are included; tool messages and injected
    memory-context blocks are stripped.  This keeps the judge from seeing stale
    recall output and prevents recursive recall amplification.
    """
    clean_current = sanitize_context(current_message or "").strip()
    turns: List[Dict[str, str]] = []
    for msg in list(recent_messages or [])[-max_turns:]:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        content = sanitize_context(content).strip()
        if not content:
            continue
        turns.append({"role": str(role), "text": content[:1000]})

    ctx: Dict[str, Any] = {
        "current_message": clean_current[:2000],
        "recent_turns": turns,
        "platform_context": _sanitize_gate_metadata(platform_context),
        "session_metadata": _sanitize_gate_metadata(session_metadata),
    }
    encoded = json.dumps(ctx, ensure_ascii=False)
    if len(encoded) > max_chars:
        # Keep current message and metadata, shrink recent turns first.
        while turns and len(json.dumps(ctx, ensure_ascii=False)) > max_chars:
            turns.pop(0)
        ctx["recent_turns"] = turns
    if len(json.dumps(ctx, ensure_ascii=False)) > max_chars:
        # Very long current turns can exceed the cap even with no history.
        # Preserve the beginning of the user's message and leave room for
        # metadata/JSON framing rather than leaking an unbounded gate prompt.
        overhead_ctx = dict(ctx)
        overhead_ctx["current_message"] = ""
        overhead = len(json.dumps(overhead_ctx, ensure_ascii=False))
        budget = max(0, max_chars - overhead - 32)
        ctx["current_message"] = ctx["current_message"][:budget]
    return ctx


def heuristic_recall_decision(current_message: str) -> RecallDecision:
    """Cheap local fallback/first pass for recall routing."""
    text = sanitize_context(current_message or "").strip()
    if not text or _SHORT_ACK_RE.match(text):
        return RecallDecision(False, reason="short acknowledgement/no recall signal")

    sources = ["graph"]
    depth = "none"
    budget = "tiny"
    reason = ""

    if _EVIDENCE_RE.search(text):
        depth = "standard"  # auto gate is capped; manual/user recall can go deeper.
        sources = ["graph", "session_fts"]
        budget = "small"
        reason = "explicit historical/evidence reference"
    elif _PAST_REFERENCE_RE.search(text):
        depth = "standard"
        sources = ["graph", "session_fts"]
        budget = "small"
        reason = "past-reference cue"
    elif _LONG_TERM_RE.search(text):
        depth = "light"
        sources = ["graph"]
        reason = "long-term project/memory cue"

    if depth == "none":
        return RecallDecision(False, reason="no recall cue")
    return RecallDecision(
        True,
        query=text,
        depth=depth,
        sources=sources,
        budget=budget,
        provenance="ids",
        reason=reason,
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def llm_recall_decision(gate_context: Mapping[str, Any], *, timeout: float = 8.0) -> Optional[RecallDecision]:
    """Ask the configured cheap auxiliary model to classify recall need.

    The prompt is intentionally narrow and JSON-only.  Failures return None so
    callers can fall back to heuristics without delaying the main turn.
    """
    try:
        from agent.auxiliary_client import call_llm

        system = (
            "You are a lightweight recall gate for an AI agent. Decide whether "
            "the current user turn needs external memory recall. Use only the "
            "provided clean gate context. Do not answer the user. Return compact "
            "JSON with keys: should_recall boolean, query string, depth one of "
            "none/light/standard, sources array using graph or session_fts, "
            "budget tiny or small, provenance ids, reason string. Automatic recall "
            "must stay cheap: never choose deep/evidence/verbatim."
        )
        user = json.dumps(gate_context, ensure_ascii=False)
        response = call_llm(
            task="memory_recall",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=220,
            timeout=timeout,
        )
        content = response.choices[0].message.content
        data = _parse_json_object(content)
        if not data:
            return None
        return RecallDecision(
            should_recall=bool(data.get("should_recall")),
            query=str(data.get("query") or gate_context.get("current_message") or ""),
            depth=str(data.get("depth") or "light"),
            sources=list(data.get("sources") or []),
            budget=str(data.get("budget") or "tiny"),
            provenance=str(data.get("provenance") or "ids"),
            reason=str(data.get("reason") or ""),
            decision_source="llm",
        )
    except Exception as exc:
        logger.debug("memory recall LLM gate failed; falling back to heuristic: %s", exc)
        return None


def decide_recall(
    gate_context: Mapping[str, Any],
    *,
    strategy: str = "heuristic",
    max_depth: str = "standard",
    max_budget: str = "small",
    timeout: float = 8.0,
) -> RecallDecision:
    """Return a normalized auto-recall decision.

    strategy:
      - heuristic: local rules only
      - llm: cheap auxiliary model, falling back to heuristic on failure
      - hybrid: heuristic first; if it says no, ask the cheap model
    """
    current = str(gate_context.get("current_message") or "")
    heuristic = heuristic_recall_decision(current)
    strategy = (strategy or "heuristic").strip().lower()

    if strategy == "heuristic":
        return heuristic.normalized(max_depth=max_depth, max_budget=max_budget)

    if strategy == "hybrid" and heuristic.should_recall:
        return heuristic.normalized(max_depth=max_depth, max_budget=max_budget)

    llm_decision = llm_recall_decision(gate_context, timeout=timeout)
    if llm_decision is not None:
        return llm_decision.normalized(max_depth=max_depth, max_budget=max_budget)
    return heuristic.normalized(max_depth=max_depth, max_budget=max_budget)
