#!/usr/bin/env python3
"""Hybrid retrieval efficiency helpers for XE-203.

Implements deterministic top-k selective recall with optional live OpenClaw
memory search, local reranking, abstain-safe thresholds, and token-savings
telemetry for knowledge-task routing.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
OBSIDIAN_OPS_PATH = REPO_ROOT / "ops" / "obsidian"
if str(OBSIDIAN_OPS_PATH) not in sys.path:
    sys.path.insert(0, str(OBSIDIAN_OPS_PATH))

from retrieval_rerank import (  # type: ignore  # noqa: E402
    apply_rerank,
    build_memory_registry_lookup,
    detect_doc_intent,
    extract_results,
    parse_json_payload,
)


HYBRID_RETRIEVAL_SCHEMA = "clawd.session_topology.hybrid_retrieval_efficiency.v1"
DEFAULT_MAX_RESULTS = 12
DEFAULT_TOP_K = 4
DEFAULT_HIGH_CONFIDENCE_TOP_K = 2
DEFAULT_RERANK_TOP_N = 12
DEFAULT_MIN_TOP_SCORE = 0.70
DEFAULT_MIN_MARGIN = 0.02
DEFAULT_HIGH_CONFIDENCE_TOP_SCORE = 0.85
DEFAULT_HIGH_CONFIDENCE_MARGIN = 0.08
DEFAULT_SEARCH_TIMEOUT_SEC = 30
DEFAULT_ABSTAIN_MESSAGE = "No sufficiently relevant knowledge was retrieved; abstaining instead of stuffing weak context."
DEFAULT_MEMORY_REGISTRY_REL = Path("state/continuity/latest/xk_obsidian_memory_registry_latest.json")
DEFAULT_TOKEN_MATCH_WEIGHT = 0.05
DEFAULT_COLLISION_WEIGHT = 0.03


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _estimate_tokens(text: str) -> int:
    token = str(text or "")
    if not token:
        return 0
    return max(1, (len(token) + 3) // 4)


def _candidate_text(row: Mapping[str, Any]) -> str:
    for key in ("snippet", "content", "text", "body", "excerpt", "summary"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    path = str(row.get("source_path") or row.get("path") or "")
    title = str(row.get("title") or "").strip()
    if title and path:
        return f"{title}\n{path}"
    return title or path


def _preview(text: str, limit: int = 280) -> str:
    token = str(text or "")
    if len(token) <= limit:
        return token
    return token[: limit - 1] + "…"


def normalize_hybrid_retrieval_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    raw = request.get("knowledge_retrieval") if isinstance(request.get("knowledge_retrieval"), Mapping) else {}
    enabled = bool(raw.get("enabled") is True) if raw else False
    query = str(raw.get("query") or request.get("invocation_prompt") or "").strip()
    doc_intent = str(raw.get("doc_intent") or "auto").strip().lower() or "auto"

    payload: Dict[str, Any] = {
        "enabled": enabled,
        "required": bool(raw.get("required") is True),
        "query": query,
        "doc_intent": doc_intent,
        "max_results": max(1, _parse_int(raw.get("max_results"), DEFAULT_MAX_RESULTS)),
        "top_k": max(1, _parse_int(raw.get("top_k"), DEFAULT_TOP_K)),
        "high_confidence_top_k": max(1, _parse_int(raw.get("high_confidence_top_k"), DEFAULT_HIGH_CONFIDENCE_TOP_K)),
        "rerank_top_n": max(0, _parse_int(raw.get("rerank_top_n"), DEFAULT_RERANK_TOP_N)),
        "min_top_score": _parse_float(raw.get("min_top_score"), DEFAULT_MIN_TOP_SCORE),
        "min_margin": _parse_float(raw.get("min_margin"), DEFAULT_MIN_MARGIN),
        "high_confidence_top_score": _parse_float(
            raw.get("high_confidence_top_score"), DEFAULT_HIGH_CONFIDENCE_TOP_SCORE
        ),
        "high_confidence_margin": _parse_float(raw.get("high_confidence_margin"), DEFAULT_HIGH_CONFIDENCE_MARGIN),
        "search_timeout_sec": max(1, _parse_int(raw.get("search_timeout_sec"), DEFAULT_SEARCH_TIMEOUT_SEC)),
        "abstain_message": str(raw.get("abstain_message") or DEFAULT_ABSTAIN_MESSAGE),
        "candidate_results": raw.get("candidate_results"),
        "live_search": bool(raw.get("live_search") is True),
        "memory_registry_path": str(raw.get("memory_registry_path") or "").strip(),
        "memory_registry_surface": raw.get("memory_registry_surface"),
        "token_match_weight": max(0.0, _parse_float(raw.get("token_match_weight"), DEFAULT_TOKEN_MATCH_WEIGHT)),
        "collision_weight": max(0.0, _parse_float(raw.get("collision_weight"), DEFAULT_COLLISION_WEIGHT)),
    }
    return payload


def validate_hybrid_retrieval_request(request: Mapping[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    raw = request.get("knowledge_retrieval")
    if raw is None:
        return True, None, {"present": False, "enabled": False}
    if not isinstance(raw, Mapping):
        return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": "expected_object"}

    norm = normalize_hybrid_retrieval_request(request)

    for key in ("max_results", "top_k", "high_confidence_top_k", "rerank_top_n", "search_timeout_sec"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except Exception:
            return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": f"{key}_must_be_int"}
        if parsed < 0:
            return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": f"{key}_must_be_non_negative"}

    for key in (
        "min_top_score",
        "min_margin",
        "high_confidence_top_score",
        "high_confidence_margin",
        "token_match_weight",
        "collision_weight",
    ):
        value = raw.get(key)
        if value is None:
            continue
        try:
            float(value)
        except Exception:
            return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": f"{key}_must_be_number"}

    candidate_results = raw.get("candidate_results")
    if candidate_results is not None and not isinstance(candidate_results, (list, dict)):
        return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": "candidate_results_must_be_array_or_object"}

    memory_registry_surface = raw.get("memory_registry_surface")
    if memory_registry_surface is not None and not isinstance(memory_registry_surface, Mapping):
        return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": "memory_registry_surface_must_be_object"}

    if norm.get("enabled") and not str(norm.get("query") or "").strip():
        return False, "routing_request_invalid", {"error": "knowledge_retrieval_invalid", "detail": "query_required_when_enabled"}

    return True, None, {
        "present": True,
        "enabled": bool(norm.get("enabled") is True),
        "required": bool(norm.get("required") is True),
        "query": norm.get("query"),
        "doc_intent": norm.get("doc_intent"),
    }


def _run_openclaw_memory_search(query: str, max_results: int, timeout_sec: int) -> Tuple[bool, Optional[str], Dict[str, Any], List[Dict[str, Any]]]:
    proc = subprocess.run(
        [
            "openclaw",
            "memory",
            "search",
            "--query",
            query,
            "--max-results",
            str(max_results),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=max(1, int(timeout_sec)),
    )

    if proc.returncode != 0:
        return (
            False,
            "retrieval_search_error",
            {
                "command": "openclaw memory search",
                "returncode": proc.returncode,
                "stderr": proc.stderr.strip()[:500],
            },
            [],
        )

    try:
        payload = parse_json_payload(proc.stdout)
    except Exception as exc:
        return False, "retrieval_search_error", {"error": "invalid_search_json", "detail": str(exc)}, []

    results = [dict(x) if isinstance(x, Mapping) else {"value": x} for x in extract_results(payload)]
    return True, None, {"command": "openclaw memory search", "result_count": len(results)}, results


def _load_candidates(config: Mapping[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any], List[Dict[str, Any]]]:
    candidate_results = config.get("candidate_results")
    if candidate_results is not None:
        rows = [dict(x) if isinstance(x, Mapping) else {"value": x} for x in extract_results(candidate_results)]
        return True, None, {"source": "injected_candidate_results", "result_count": len(rows)}, rows

    if not bool(config.get("live_search") is True):
        return True, None, {"source": "not_requested", "result_count": 0}, []

    return _run_openclaw_memory_search(
        str(config.get("query") or ""),
        max_results=max(1, int(config.get("max_results") or DEFAULT_MAX_RESULTS)),
        timeout_sec=max(1, int(config.get("search_timeout_sec") or DEFAULT_SEARCH_TIMEOUT_SEC)),
    )


def _resolve_registry_path(raw: str) -> Path:
    token = str(raw or "").strip()
    if not token:
        return (REPO_ROOT / DEFAULT_MEMORY_REGISTRY_REL).resolve()
    path = Path(token).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _load_registry_lookup(config: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    surface = config.get("memory_registry_surface")
    if isinstance(surface, Mapping):
        lookup = build_memory_registry_lookup(surface)
        return lookup, {
            "source": "injected_memory_registry_surface",
            "loaded": True,
            "registry_fingerprint": surface.get("registry_fingerprint"),
        }

    registry_path = _resolve_registry_path(str(config.get("memory_registry_path") or ""))
    if not registry_path.exists() or not registry_path.is_file():
        return None, {
            "source": "file",
            "loaded": False,
            "path": str(registry_path),
            "error": "memory_registry_missing",
        }

    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, {
            "source": "file",
            "loaded": False,
            "path": str(registry_path),
            "error": f"memory_registry_invalid_json:{exc}",
        }

    if not isinstance(payload, Mapping):
        return None, {
            "source": "file",
            "loaded": False,
            "path": str(registry_path),
            "error": "memory_registry_payload_not_object",
        }

    lookup = build_memory_registry_lookup(payload)
    return lookup, {
        "source": "file",
        "loaded": True,
        "path": str(registry_path),
        "registry_fingerprint": payload.get("registry_fingerprint"),
    }


def _select_confidence_tier(*, top_score: float, margin: float, config: Mapping[str, Any]) -> Tuple[str, int]:
    high_top = float(config.get("high_confidence_top_score") or DEFAULT_HIGH_CONFIDENCE_TOP_SCORE)
    high_margin = float(config.get("high_confidence_margin") or DEFAULT_HIGH_CONFIDENCE_MARGIN)
    base_top = float(config.get("min_top_score") or DEFAULT_MIN_TOP_SCORE)
    base_margin = float(config.get("min_margin") or DEFAULT_MIN_MARGIN)

    if top_score >= high_top and margin >= high_margin:
        return "high", max(1, int(config.get("high_confidence_top_k") or DEFAULT_HIGH_CONFIDENCE_TOP_K))
    if top_score >= base_top and margin >= base_margin:
        return "pass", max(1, int(config.get("top_k") or DEFAULT_TOP_K))
    return "abstain", 0


def _round6(value: float) -> float:
    return float(f"{float(value):.6f}")


def _selected_context_slices(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    slices: List[Dict[str, Any]] = []
    for idx, row in enumerate(results, start=1):
        path = str(row.get("source_path") or row.get("path") or f"candidate_{idx}")
        text = _candidate_text(row)
        slices.append(
            {
                "slice_id": f"retrieval:{idx}:{path}",
                "source_path": path,
                "content": text,
                "estimated_tokens": _estimate_tokens(text),
                "rerank_score": _round6(_parse_float(row.get("rerank_score"), _parse_float(row.get("score"), 0.0))),
                "original_score": _round6(_parse_float(row.get("score"), 0.0)),
                "doc_type": row.get("doc_type"),
                "registry_token_match_score": _round6(_parse_float(row.get("registry_token_match_score"), 0.0)),
                "registry_collision_score": _round6(_parse_float(row.get("registry_collision_score"), 0.0)),
            }
        )
    return slices


def evaluate_hybrid_retrieval_efficiency(request: Mapping[str, Any]) -> Dict[str, Any]:
    config = normalize_hybrid_retrieval_request(request)
    if not bool(config.get("enabled") is True):
        return {
            "schema": HYBRID_RETRIEVAL_SCHEMA,
            "status": "not_requested",
            "required": bool(config.get("required") is True),
            "enabled": False,
            "mode": "disabled",
            "query": config.get("query"),
            "context_policy": {"mode": "full_context_unchanged", "stuffing_blocked_by_default": False},
        }

    query = str(config.get("query") or "").strip()
    requested_intent = str(config.get("doc_intent") or "auto").strip().lower() or "auto"
    effective_intent = detect_doc_intent(query) if requested_intent == "auto" else requested_intent
    if requested_intent == "auto" and not effective_intent:
        effective_intent = None

    registry_lookup, registry_meta = _load_registry_lookup(config)
    token_match_weight = max(0.0, _parse_float(config.get("token_match_weight"), DEFAULT_TOKEN_MATCH_WEIGHT))
    collision_weight = max(0.0, _parse_float(config.get("collision_weight"), DEFAULT_COLLISION_WEIGHT))
    registry_enabled = bool(registry_lookup) and (token_match_weight > 0.0 or collision_weight > 0.0)

    load_ok, load_reason, load_meta, candidates = _load_candidates(config)
    if not load_ok:
        packet = {
            "schema": HYBRID_RETRIEVAL_SCHEMA,
            "status": "search_error",
            "required": bool(config.get("required") is True),
            "enabled": True,
            "mode": "hybrid_vector_keyword_rerank",
            "query": query,
            "doc_intent": effective_intent,
            "source": load_meta.get("source") or "openclaw_memory_search",
            "error": load_meta,
            "registry_boost": {
                "enabled": registry_enabled,
                "token_match_weight": _round6(token_match_weight),
                "collision_weight": _round6(collision_weight),
                **registry_meta,
            },
            "context_policy": {
                "mode": "top_k_only",
                "stuffing_blocked_by_default": True,
                "candidate_count": 0,
                "selected_count": 0,
                "saved_tokens": 0,
                "saved_pct": 0.0,
            },
        }
        if bool(config.get("required") is True):
            packet["block"] = True
            packet["block_reason"] = load_reason or "retrieval_search_error"
        return packet

    rerank_top_n = max(0, int(config.get("rerank_top_n") or DEFAULT_RERANK_TOP_N))
    now_utc = dt.datetime.now(dt.timezone.utc)
    reranked = apply_rerank(
        candidates,
        query=query,
        top_n=rerank_top_n,
        doc_intent=effective_intent,
        now_utc=now_utc,
        base_weight=0.72,
        trust_weight=0.14,
        recency_weight=0.10,
        doc_type_weight=0.04,
        registry_lookup=registry_lookup,
        token_match_weight=token_match_weight if registry_enabled else 0.0,
        collision_weight=collision_weight if registry_enabled else 0.0,
    )

    result_count = len(reranked)
    top_score = _parse_float((reranked[0] if result_count else {}).get("rerank_score"), 0.0) if result_count else 0.0
    second_score = _parse_float((reranked[1] if result_count > 1 else {}).get("rerank_score"), 0.0) if result_count > 1 else None
    margin = top_score if second_score is None else max(0.0, top_score - second_score)

    confidence_tier, selected_k = _select_confidence_tier(top_score=top_score, margin=margin, config=config)
    selected = reranked[:selected_k] if selected_k > 0 else []

    candidate_tokens = sum(_estimate_tokens(_candidate_text(row)) for row in reranked)
    selected_tokens = sum(_estimate_tokens(_candidate_text(row)) for row in selected)
    saved_tokens = max(0, candidate_tokens - selected_tokens)
    saved_pct = 0.0 if candidate_tokens <= 0 else (saved_tokens / candidate_tokens) * 100.0

    top_candidates: List[Dict[str, Any]] = []
    for idx, row in enumerate(reranked[: max(5, selected_k)], start=1):
        text = _candidate_text(row)
        top_candidates.append(
            {
                "rank": idx,
                "path": row.get("source_path") or row.get("path"),
                "doc_type": row.get("doc_type"),
                "selected": idx <= selected_k,
                "original_score": _round6(_parse_float(row.get("score"), 0.0)),
                "rerank_score": _round6(_parse_float(row.get("rerank_score"), 0.0)),
                "registry_token_match_score": _round6(_parse_float(row.get("registry_token_match_score"), 0.0)),
                "registry_collision_score": _round6(_parse_float(row.get("registry_collision_score"), 0.0)),
                "preview": _preview(text),
            }
        )

    packet: Dict[str, Any] = {
        "schema": HYBRID_RETRIEVAL_SCHEMA,
        "status": "pass" if selected_k > 0 else "abstain",
        "required": bool(config.get("required") is True),
        "enabled": True,
        "mode": "hybrid_vector_keyword_rerank",
        "query": query,
        "doc_intent": effective_intent,
        "source": load_meta.get("source") or "openclaw_memory_search",
        "registry_boost": {
            "enabled": registry_enabled,
            "token_match_weight": _round6(token_match_weight),
            "collision_weight": _round6(collision_weight),
            **registry_meta,
        },
        "result_count": result_count,
        "thresholds": {
            "min_top_score": _round6(float(config.get("min_top_score") or DEFAULT_MIN_TOP_SCORE)),
            "min_margin": _round6(float(config.get("min_margin") or DEFAULT_MIN_MARGIN)),
            "high_confidence_top_score": _round6(
                float(config.get("high_confidence_top_score") or DEFAULT_HIGH_CONFIDENCE_TOP_SCORE)
            ),
            "high_confidence_margin": _round6(float(config.get("high_confidence_margin") or DEFAULT_HIGH_CONFIDENCE_MARGIN)),
        },
        "scores": {
            "top_score": _round6(top_score),
            "second_score": _round6(second_score) if second_score is not None else None,
            "margin": _round6(margin),
        },
        "confidence_tier": confidence_tier,
        "selected_top_k": selected_k,
        "selected_recall": {
            "slice_count": len(selected),
            "slices": _selected_context_slices(selected),
        },
        "top_candidates": top_candidates,
        "context_policy": {
            "mode": "top_k_only",
            "stuffing_blocked_by_default": True,
            "candidate_count": result_count,
            "selected_count": len(selected),
            "dropped_count": max(0, result_count - len(selected)),
            "full_context_tokens": candidate_tokens,
            "selected_context_tokens": selected_tokens,
            "saved_tokens": saved_tokens,
            "saved_pct": _round6(saved_pct),
        },
    }

    if selected_k <= 0:
        packet["abstain_reason"] = "below_threshold"
        packet["message"] = str(config.get("abstain_message") or DEFAULT_ABSTAIN_MESSAGE)
        if bool(config.get("required") is True):
            packet["block"] = True
            packet["block_reason"] = "hybrid_retrieval_abstain"

    return packet
