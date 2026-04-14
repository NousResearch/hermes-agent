from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _estimate_tokens(text: str) -> int:
    compact = str(text or "")
    if not compact:
        return 0
    return max(1, (len(compact) + 3) // 4)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": "clawd.session_topology.context_delta_cache.v1",
            "updated_at": now_iso(),
            "entries": {},
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("context_delta_cache_not_object")
    entries = payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {}
    return {
        "schema": str(payload.get("schema") or "clawd.session_topology.context_delta_cache.v1"),
        "updated_at": str(payload.get("updated_at") or now_iso()),
        "entries": dict(entries),
    }


def _save_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "clawd.session_topology.context_delta_cache.v1",
        "updated_at": now_iso(),
        "entries": payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {},
    }
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_context_slices(request: Mapping[str, Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    raw_slices = request.get("context_slices") if isinstance(request.get("context_slices"), list) else []
    for idx, item in enumerate(raw_slices):
        if not isinstance(item, Mapping):
            continue
        slice_id = str(item.get("slice_id") or item.get("id") or f"slice_{idx + 1}").strip() or f"slice_{idx + 1}"
        content = str(item.get("content") or "")
        if not content.strip():
            continue
        normalized.append(
            {
                "slice_id": slice_id,
                "content": content,
                "tokens": _estimate_tokens(content),
                "sha256": _sha256_text(content),
            }
        )

    if normalized:
        return normalized

    prompt = request.get("invocation_prompt")
    if isinstance(prompt, str) and prompt.strip():
        normalized.append(
            {
                "slice_id": "invocation_prompt",
                "content": prompt,
                "tokens": _estimate_tokens(prompt),
                "sha256": _sha256_text(prompt),
            }
        )

    return normalized


def _binding_from_transport_decision(transport_decision: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(transport_decision, Mapping):
        return {}

    payload: Mapping[str, Any] = transport_decision
    if str(payload.get("schema") or "") == "clawd.session_topology_transport_routing.decision.v1":
        if str(payload.get("decision") or "").strip().upper() == "BLOCK":
            return {}
        route_payload = payload.get("route")
        if isinstance(route_payload, Mapping):
            payload = route_payload

    if str(payload.get("schema_version") or "") != "session.topology.routing_decision.v1":
        return {}

    routing_basis = payload.get("routing_basis") if isinstance(payload.get("routing_basis"), Mapping) else {}
    lane = payload.get("lane") if isinstance(payload.get("lane"), Mapping) else {}
    session = payload.get("session") if isinstance(payload.get("session"), Mapping) else {}

    binding = {
        "transport_key": str(routing_basis.get("transport_key") or "").strip(),
        "lane_name": str(lane.get("name") or "").strip(),
        "agent_id": str(lane.get("agent_id") or "").strip(),
        "session_key": str(session.get("session_key") or "").strip(),
    }
    return {k: v for k, v in binding.items() if v}


def _binding_from_request(request: Mapping[str, Any]) -> Dict[str, Any]:
    route = request.get("transport_route") if isinstance(request.get("transport_route"), Mapping) else {}
    binding = {
        "transport_key": str(route.get("transport_key") or request.get("transport_key") or "").strip(),
        "lane_name": str(route.get("lane_name") or request.get("lane_name") or "").strip(),
        "agent_id": str(route.get("agent_id") or request.get("agent_id") or request.get("requested_agent_id") or "").strip(),
        "session_key": str(route.get("session_key") or request.get("session_key") or request.get("requested_session_key") or "").strip(),
    }
    return {k: v for k, v in binding.items() if v}


def _flow_key(binding: Mapping[str, Any]) -> str:
    # session_key is primary; fallback to tuple if legacy surface is sparse.
    session_key = str(binding.get("session_key") or "").strip()
    if session_key:
        return f"session:{session_key}"

    agent_id = str(binding.get("agent_id") or "").strip()
    transport_key = str(binding.get("transport_key") or "").strip()
    lane_name = str(binding.get("lane_name") or "").strip()
    if agent_id and transport_key:
        return f"tuple:{transport_key}|{lane_name}|{agent_id}"
    return ""


def _snapshot_hash(slices: List[Dict[str, Any]]) -> str:
    payload = [{"slice_id": row.get("slice_id"), "sha256": row.get("sha256")} for row in slices]
    return _sha256_text(_stable_json(payload))


def _materialize_slice_map(slices: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in slices:
        sid = str(row.get("slice_id") or "").strip()
        if not sid:
            continue
        out[sid] = {
            "slice_id": sid,
            "content": str(row.get("content") or ""),
            "tokens": int(row.get("tokens") or 0),
            "sha256": str(row.get("sha256") or _sha256_text(str(row.get("content") or ""))),
        }
    return out


def evaluate_context_delta_transport(
    *,
    request: Mapping[str, Any],
    transport_decision: Optional[Mapping[str, Any]],
    cache_path: Path,
    enabled: bool,
    flush: bool,
) -> Dict[str, Any]:
    slices = _normalize_context_slices(request)
    full_tokens = int(sum(int(row.get("tokens") or 0) for row in slices))
    binding = _binding_from_transport_decision(transport_decision) or _binding_from_request(request)
    flow_key = _flow_key(binding)

    out: Dict[str, Any] = {
        "schema": "clawd.session_topology.context_delta_transport.v1",
        "evaluated_at": now_iso(),
        "enabled": bool(enabled),
        "flow": {
            "supported": bool(flow_key),
            "flow_key": flow_key or None,
            "binding": binding,
        },
        "mode": "full",
        "fallback_reason": None,
        "slices": {
            "slice_count": len(slices),
            "changed_count": len(slices),
            "unchanged_count": 0,
            "removed_slice_ids": [],
        },
        "tokens": {
            "full_context_tokens": full_tokens,
            "transmitted_tokens": full_tokens,
            "saved_tokens": 0,
            "saved_pct": 0.0,
        },
        "integrity": {
            "baseline_snapshot_hash": None,
            "current_snapshot_hash": _snapshot_hash(slices),
            "reconstructed_snapshot_hash": _snapshot_hash(slices),
            "reconstruction_ok": True,
        },
        "cache": {
            "path": str(cache_path),
            "status": "disabled" if not enabled else "miss",
        },
    }

    if not enabled:
        out["fallback_reason"] = "delta_disabled"
        return out

    if not slices:
        out["fallback_reason"] = "context_slices_missing"
        out["cache"]["status"] = "bypass"
        return out

    if not flow_key:
        out["fallback_reason"] = "transport_binding_missing"
        out["cache"]["status"] = "bypass"
        return out

    if flush and cache_path.exists():
        cache_path.unlink()

    cache_corrupt = False
    try:
        cache = _load_cache(cache_path)
    except Exception:
        cache = {
            "schema": "clawd.session_topology.context_delta_cache.v1",
            "updated_at": now_iso(),
            "entries": {},
        }
        cache_corrupt = True

    entries = cache.get("entries") if isinstance(cache.get("entries"), Mapping) else {}
    entries_obj: Dict[str, Any] = dict(entries)

    prev_entry = entries_obj.get(flow_key) if isinstance(entries_obj.get(flow_key), Mapping) else None
    prev_slices = prev_entry.get("slices") if isinstance(prev_entry, Mapping) and isinstance(prev_entry.get("slices"), list) else []
    prev_slice_map = _materialize_slice_map([dict(x) for x in prev_slices if isinstance(x, Mapping)])
    curr_slice_map = _materialize_slice_map(slices)

    changed: List[Dict[str, Any]] = []
    unchanged_count = 0
    for row in slices:
        sid = str(row.get("slice_id") or "")
        prev = prev_slice_map.get(sid)
        if prev is None or str(prev.get("sha256") or "") != str(row.get("sha256") or ""):
            changed.append(dict(row))
        else:
            unchanged_count += 1

    removed_slice_ids = sorted([sid for sid in prev_slice_map.keys() if sid not in curr_slice_map])

    baseline_hash = _snapshot_hash([dict(x) for x in prev_slices if isinstance(x, Mapping)]) if prev_slices else None
    current_hash = _snapshot_hash(slices)

    if prev_entry is None:
        out["mode"] = "full"
        out["fallback_reason"] = "first_turn_no_baseline"
        transmitted_tokens = full_tokens
        cache_status = "miss"
    else:
        out["mode"] = "delta"
        out["fallback_reason"] = None
        transmitted_tokens = int(sum(int(row.get("tokens") or 0) for row in changed))
        cache_status = "hit"

    reconstructed_map = dict(prev_slice_map)
    for sid in removed_slice_ids:
        reconstructed_map.pop(sid, None)
    for row in changed:
        reconstructed_map[str(row.get("slice_id") or "")] = {
            "slice_id": str(row.get("slice_id") or ""),
            "content": str(row.get("content") or ""),
            "tokens": int(row.get("tokens") or 0),
            "sha256": str(row.get("sha256") or ""),
        }

    ordered_reconstructed = [reconstructed_map[sid] for sid in sorted(reconstructed_map.keys())]
    reconstructed_hash = _snapshot_hash(ordered_reconstructed)
    reconstruction_ok = reconstructed_hash == current_hash

    saved_tokens = max(0, full_tokens - transmitted_tokens)
    saved_pct = round((saved_tokens / full_tokens) * 100.0, 2) if full_tokens > 0 else 0.0

    out["slices"] = {
        "slice_count": len(slices),
        "changed_count": len(changed),
        "unchanged_count": int(unchanged_count),
        "removed_slice_ids": removed_slice_ids,
        "changed_slice_ids": [str(row.get("slice_id") or "") for row in changed],
    }
    out["tokens"] = {
        "full_context_tokens": full_tokens,
        "transmitted_tokens": transmitted_tokens,
        "saved_tokens": saved_tokens,
        "saved_pct": saved_pct,
    }
    out["integrity"] = {
        "baseline_snapshot_hash": baseline_hash,
        "current_snapshot_hash": current_hash,
        "reconstructed_snapshot_hash": reconstructed_hash,
        "reconstruction_ok": reconstruction_ok,
    }
    out["cache"] = {
        "path": str(cache_path),
        "status": cache_status,
        "cache_corrupt_recovered": cache_corrupt,
    }

    entries_obj[flow_key] = {
        "updated_at": now_iso(),
        "flow_binding": binding,
        "snapshot_hash": current_hash,
        "slices": [
            {
                "slice_id": str(row.get("slice_id") or ""),
                "content": str(row.get("content") or ""),
                "tokens": int(row.get("tokens") or 0),
                "sha256": str(row.get("sha256") or ""),
            }
            for row in slices
        ],
    }

    cache["entries"] = entries_obj
    _save_cache(cache_path, cache)

    return out


ANCHOR_ID_HINTS = (
    "directive",
    "acceptance",
    "criteria",
    "blocker",
    "constraint",
    "guardrail",
    "requirement",
    "objective",
    "scope",
)

TRADEOFF_KEYWORDS = (
    "tradeoff",
    "trade-off",
    "versus",
    " vs ",
    "however",
    "but",
    "cost",
    "benefit",
)

CONTRADICTION_KEYWORDS = (
    "contradict",
    "conflict",
    "mismatch",
    "inconsistent",
    "but",
    "however",
    "yet",
)

DECISION_KEYWORDS = (
    "decision",
    "decide",
    "decided",
    "choose",
    "chosen",
    "selected",
    "will ",
    "final",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _load_compaction_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "schema": "clawd.session_topology.context_compaction_cache.v1",
            "updated_at": now_iso(),
            "entries": {},
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("context_compaction_cache_not_object")
    entries = payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {}
    return {
        "schema": str(payload.get("schema") or "clawd.session_topology.context_compaction_cache.v1"),
        "updated_at": str(payload.get("updated_at") or now_iso()),
        "entries": dict(entries),
    }


def _save_compaction_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema": "clawd.session_topology.context_compaction_cache.v1",
        "updated_at": now_iso(),
        "entries": payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {},
    }
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _compact_rollup_items(values: List[str], *, max_items: int, max_chars: int = 140) -> List[str]:
    out: List[str] = []
    for text in _dedupe_keep_order(values):
        compact = " ".join(str(text or "").split())
        if len(compact) > max_chars:
            compact = compact[: max(1, max_chars - 1)].rstrip() + "…"
        if compact:
            out.append(compact)
        if len(out) >= max_items:
            break
    return out


def _extract_deliberation_primitives(content: str) -> Dict[str, List[str]]:
    text = str(content or "").strip()
    if not text:
        return {
            "claims": [],
            "tradeoffs": [],
            "contradictions": [],
            "decisions": [],
        }

    parts = [str(x or "").strip() for x in SENTENCE_SPLIT_RE.split(text) if str(x or "").strip()]
    claims: List[str] = []
    tradeoffs: List[str] = []
    contradictions: List[str] = []
    decisions: List[str] = []

    for part in parts:
        lowered = part.lower()
        claims.append(part)
        if any(token in lowered for token in TRADEOFF_KEYWORDS):
            tradeoffs.append(part)
        if any(token in lowered for token in CONTRADICTION_KEYWORDS):
            contradictions.append(part)
        if any(token in lowered for token in DECISION_KEYWORDS):
            decisions.append(part)

    claims = _dedupe_keep_order(claims)
    if not claims and text:
        claims = [text]

    return {
        "claims": claims,
        "tradeoffs": _dedupe_keep_order(tradeoffs),
        "contradictions": _dedupe_keep_order(contradictions),
        "decisions": _dedupe_keep_order(decisions),
    }


def _build_anchor_selection(
    *,
    request: Mapping[str, Any],
    slices: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], str]:
    explicit_anchor_ids: List[str] = []
    raw_explicit = request.get("immutable_anchor_slice_ids") if isinstance(request.get("immutable_anchor_slice_ids"), list) else []
    for raw in raw_explicit:
        sid = str(raw or "").strip()
        if sid:
            explicit_anchor_ids.append(sid)
    explicit_anchor_ids = _dedupe_keep_order(explicit_anchor_ids)

    heuristic_anchor_ids: List[str] = []
    for row in slices:
        sid = str(row.get("slice_id") or "").strip()
        lowered_id = sid.lower()
        lowered_content = str(row.get("content") or "").lower()
        if any(token in lowered_id for token in ANCHOR_ID_HINTS) or any(token in lowered_content for token in ANCHOR_ID_HINTS):
            heuristic_anchor_ids.append(sid)
    heuristic_anchor_ids = _dedupe_keep_order(heuristic_anchor_ids)

    if explicit_anchor_ids:
        return explicit_anchor_ids, explicit_anchor_ids, "explicit_request"

    if heuristic_anchor_ids:
        return [], heuristic_anchor_ids, "heuristic_detection"

    if slices:
        implicit_anchor = str((slices[0] or {}).get("slice_id") or "").strip()
        if implicit_anchor:
            return [], [implicit_anchor], "implicit_primary_anchor"

    return [], [], "no_anchor_candidate"


def _build_deliberation_capsule(
    *,
    row: Mapping[str, Any],
    flow_key: str,
    anchor_refs: List[str],
) -> Dict[str, Any]:
    sid = str(row.get("slice_id") or "").strip()
    content = str(row.get("content") or "")
    sha = str(row.get("sha256") or _sha256_text(content))
    primitives = _extract_deliberation_primitives(content)
    capsule_id = f"cap_{_sha256_text(f'{flow_key}|{sid}|{sha}')[:16]}"

    provenance_refs = [f"context_slices/{sid}#sha256:{sha}"]
    provenance_refs.extend([f"anchor_ref:{ref}" for ref in anchor_refs])

    return {
        "capsule_id": capsule_id,
        "slice_id": sid,
        "source_sha256": sha,
        "token_estimate": int(row.get("tokens") or 0),
        "anchor_refs": list(anchor_refs),
        "provenance_refs": provenance_refs,
        "primitives": primitives,
    }


def _rollup_capsules(capsules: List[Mapping[str, Any]]) -> Dict[str, Any]:
    claims: List[str] = []
    tradeoffs: List[str] = []
    contradictions: List[str] = []
    decisions: List[str] = []

    for cap in capsules:
        primitives = cap.get("primitives") if isinstance(cap.get("primitives"), Mapping) else {}
        claims.extend([str(x) for x in (primitives.get("claims") or []) if str(x or "").strip()])
        tradeoffs.extend([str(x) for x in (primitives.get("tradeoffs") or []) if str(x or "").strip()])
        contradictions.extend([str(x) for x in (primitives.get("contradictions") or []) if str(x or "").strip()])
        decisions.extend([str(x) for x in (primitives.get("decisions") or []) if str(x or "").strip()])

    rollup_primitives = {
        "claims": _compact_rollup_items(claims, max_items=8),
        "tradeoffs": _compact_rollup_items(tradeoffs, max_items=6),
        "contradictions": _compact_rollup_items(contradictions, max_items=6),
        "decisions": _compact_rollup_items(decisions, max_items=6),
    }

    rollup_hash = _sha256_text(_stable_json(rollup_primitives))
    return {
        "rollup_hash": rollup_hash,
        "primitives": rollup_primitives,
    }


def evaluate_anchor_preserving_summary_compaction(
    *,
    request: Mapping[str, Any],
    context_transport: Optional[Mapping[str, Any]],
    transport_decision: Optional[Mapping[str, Any]],
    cache_path: Path,
    enabled: bool,
    flush: bool,
) -> Dict[str, Any]:
    slices = _normalize_context_slices(request)
    full_tokens = int(sum(int(row.get("tokens") or 0) for row in slices))

    transport_flow = context_transport.get("flow") if isinstance(context_transport, Mapping) and isinstance(context_transport.get("flow"), Mapping) else {}
    binding = (
        transport_flow.get("binding")
        if isinstance(transport_flow.get("binding"), Mapping)
        else _binding_from_transport_decision(transport_decision) or _binding_from_request(request)
    )
    binding = dict(binding) if isinstance(binding, Mapping) else {}
    flow_key = str(transport_flow.get("flow_key") or _flow_key(binding) or "").strip()

    explicit_anchor_ids, required_anchor_ids, anchor_resolution = _build_anchor_selection(request=request, slices=slices)

    current_snapshot_hash = _snapshot_hash(slices)
    out: Dict[str, Any] = {
        "schema": "clawd.session_topology.anchor_preserving_summary_compaction.v1",
        "evaluated_at": now_iso(),
        "enabled": bool(enabled),
        "flow": {
            "supported": bool(flow_key),
            "flow_key": flow_key or None,
            "binding": binding,
        },
        "mode": "full_passthrough",
        "status": "disabled" if not enabled else "bypass",
        "fallback_reason": None if enabled else "compaction_disabled",
        "anchors": {
            "explicit_anchor_slice_ids": explicit_anchor_ids,
            "required_anchor_slice_ids": required_anchor_ids,
            "preserved_anchor_slice_ids": [],
            "missing_anchor_slice_ids": [],
            "preserved": False,
            "resolution": anchor_resolution,
        },
        "tokens": {
            "full_context_tokens": full_tokens,
            "compacted_tokens": full_tokens,
            "saved_tokens": 0,
            "saved_pct": 0.0,
        },
        "deliberation_capsules": {
            "capsule_count": 0,
            "changed_capsule_count": 0,
            "changed_capsule_ids": [],
            "primitive_counts": {
                "claims": 0,
                "tradeoffs": 0,
                "contradictions": 0,
                "decisions": 0,
            },
            "capsules": [],
        },
        "hierarchy": {
            "level_count": 0,
            "levels": [],
            "drift": {
                "status": "not_evaluated",
                "previous_rollup_hash": None,
                "current_rollup_hash": None,
            },
        },
        "integrity": {
            "baseline_snapshot_hash": None,
            "current_snapshot_hash": current_snapshot_hash,
            "reconstructed_snapshot_hash": current_snapshot_hash,
            "reconstruction_ok": True,
            "semantic_loss_check_ok": True,
            "semantic_loss_reasons": [],
        },
        "cache": {
            "path": str(cache_path),
            "status": "disabled" if not enabled else "bypass",
            "cache_corrupt_recovered": False,
        },
        "summary_artifact": {
            "anchor_refs": [],
            "capsule_provenance_refs": [],
            "deliberation_rollup": {
                "claims": [],
                "tradeoffs": [],
                "contradictions": [],
                "decisions": [],
            },
        },
    }

    if not enabled:
        return out

    if not slices:
        out["status"] = "bypass"
        out["fallback_reason"] = "context_slices_missing"
        return out

    if not flow_key:
        out["status"] = "fail_closed"
        out["fallback_reason"] = "transport_binding_missing"
        return out

    if flush and cache_path.exists():
        cache_path.unlink()

    cache_corrupt = False
    try:
        cache = _load_compaction_cache(cache_path)
    except Exception:
        cache = {
            "schema": "clawd.session_topology.context_compaction_cache.v1",
            "updated_at": now_iso(),
            "entries": {},
        }
        cache_corrupt = True

    entries = cache.get("entries") if isinstance(cache.get("entries"), Mapping) else {}
    entries_obj: Dict[str, Any] = dict(entries)
    prev_entry = entries_obj.get(flow_key) if isinstance(entries_obj.get(flow_key), Mapping) else None

    baseline_snapshot_hash = (
        str(prev_entry.get("snapshot_hash") or "").strip() if isinstance(prev_entry, Mapping) else ""
    ) or None
    previous_rollup_hash = (
        str(prev_entry.get("rollup_hash") or "").strip() if isinstance(prev_entry, Mapping) else ""
    ) or None

    current_slice_map = _materialize_slice_map(slices)
    preserved_anchor_slice_ids = [sid for sid in required_anchor_ids if sid in current_slice_map]
    missing_anchor_slice_ids = [sid for sid in required_anchor_ids if sid not in current_slice_map]

    anchor_rows: List[Dict[str, Any]] = [dict(current_slice_map[sid]) for sid in required_anchor_ids if sid in current_slice_map]
    anchor_refs = [f"{row.get('slice_id')}@{str(row.get('sha256') or '')[:12]}" for row in anchor_rows]

    changed_slice_ids = []
    if isinstance(context_transport, Mapping):
        slices_row = context_transport.get("slices") if isinstance(context_transport.get("slices"), Mapping) else {}
        changed_slice_ids = [str(x) for x in (slices_row.get("changed_slice_ids") or []) if str(x or "").strip()]

    capsules: List[Dict[str, Any]] = []
    changed_capsule_ids: List[str] = []
    for row in slices:
        sid = str(row.get("slice_id") or "").strip()
        if not sid or sid in required_anchor_ids:
            continue
        capsule = _build_deliberation_capsule(row=row, flow_key=flow_key, anchor_refs=anchor_refs)
        capsules.append(capsule)
        if not changed_slice_ids or sid in changed_slice_ids:
            changed_capsule_ids.append(str(capsule.get("capsule_id") or ""))

    rollup = _rollup_capsules(capsules)
    rollup_primitives = rollup.get("primitives") if isinstance(rollup.get("primitives"), Mapping) else {}
    rollup_hash = str(rollup.get("rollup_hash") or "").strip() or None

    compacted_payload = {
        "anchors": [
            {
                "slice_id": row.get("slice_id"),
                "sha256": row.get("sha256"),
            }
            for row in anchor_rows
        ],
        "capsule_refs": [
            {
                "capsule_id": cap.get("capsule_id"),
                "slice_id": cap.get("slice_id"),
                "source_sha256": cap.get("source_sha256"),
            }
            for cap in capsules
        ],
        "rollup": rollup_primitives,
    }
    compacted_fragments: List[str] = []
    compacted_fragments.extend([str(row.get("content") or "") for row in anchor_rows])
    for key in ("claims", "tradeoffs", "contradictions", "decisions"):
        compacted_fragments.extend([str(x or "") for x in (rollup_primitives.get(key) or [])])
    compacted_fragments.extend([f"{cap.get('slice_id')}:{cap.get('source_sha256')}" for cap in capsules])
    compacted_tokens = _estimate_tokens("\n".join([x for x in compacted_fragments if str(x or "").strip()]))

    covered_slice_sha: Dict[str, str] = {
        str(row.get("slice_id") or ""): str(row.get("sha256") or "")
        for row in anchor_rows
    }
    for cap in capsules:
        sid = str(cap.get("slice_id") or "")
        sha = str(cap.get("source_sha256") or "")
        if sid and sha:
            covered_slice_sha[sid] = sha

    reconstructed_pairs: List[Dict[str, Any]] = []
    for row in slices:
        sid = str(row.get("slice_id") or "").strip()
        if sid in covered_slice_sha:
            reconstructed_pairs.append({"slice_id": sid, "sha256": covered_slice_sha[sid]})
    reconstructed_hash = _snapshot_hash(reconstructed_pairs)
    reconstruction_ok = reconstructed_hash == current_snapshot_hash and len(reconstructed_pairs) == len(slices)

    source_profile = {
        "claims": 0,
        "tradeoffs": 0,
        "contradictions": 0,
        "decisions": 0,
    }
    for row in slices:
        sid = str(row.get("slice_id") or "").strip()
        if sid in required_anchor_ids:
            continue
        primitives = _extract_deliberation_primitives(str(row.get("content") or ""))
        for key in source_profile.keys():
            source_profile[key] += int(len(primitives.get(key) or []))

    capsule_profile = {
        "claims": int(len(rollup_primitives.get("claims") or [])),
        "tradeoffs": int(len(rollup_primitives.get("tradeoffs") or [])),
        "contradictions": int(len(rollup_primitives.get("contradictions") or [])),
        "decisions": int(len(rollup_primitives.get("decisions") or [])),
    }

    semantic_loss_reasons: List[str] = []
    for key in source_profile.keys():
        if source_profile[key] > 0 and capsule_profile[key] == 0:
            semantic_loss_reasons.append(f"missing_{key}")
    semantic_loss_check_ok = len(semantic_loss_reasons) == 0

    fail_reasons: List[str] = []
    if missing_anchor_slice_ids:
        fail_reasons.append("required_anchor_missing")
    if not reconstruction_ok:
        fail_reasons.append("anchor_roundtrip_reconstruction_failed")
    if not semantic_loss_check_ok:
        fail_reasons.append("semantic_loss_detected")

    saved_tokens = max(0, full_tokens - compacted_tokens)
    saved_pct = round((saved_tokens / full_tokens) * 100.0, 2) if full_tokens > 0 else 0.0

    out["anchors"] = {
        "explicit_anchor_slice_ids": explicit_anchor_ids,
        "required_anchor_slice_ids": required_anchor_ids,
        "preserved_anchor_slice_ids": preserved_anchor_slice_ids,
        "missing_anchor_slice_ids": missing_anchor_slice_ids,
        "preserved": len(missing_anchor_slice_ids) == 0,
        "resolution": anchor_resolution,
    }
    out["tokens"] = {
        "full_context_tokens": full_tokens,
        "compacted_tokens": compacted_tokens,
        "saved_tokens": saved_tokens,
        "saved_pct": saved_pct,
    }
    out["deliberation_capsules"] = {
        "capsule_count": len(capsules),
        "changed_capsule_count": len(changed_capsule_ids),
        "changed_capsule_ids": changed_capsule_ids,
        "primitive_counts": capsule_profile,
        "capsules": capsules,
    }
    out["hierarchy"] = {
        "level_count": 3,
        "levels": [
            {
                "level": 0,
                "name": "immutable_anchor_layer",
                "item_count": len(anchor_rows),
                "anchor_refs": [f"context_slices/{row.get('slice_id')}#sha256:{row.get('sha256')}" for row in anchor_rows],
            },
            {
                "level": 1,
                "name": "deliberation_capsule_layer",
                "item_count": len(capsules),
                "capsule_ids": [str(cap.get("capsule_id") or "") for cap in capsules],
            },
            {
                "level": 2,
                "name": "rollup_layer",
                "item_count": sum(int(len(v or [])) for v in rollup_primitives.values()),
                "rollup_hash": rollup_hash,
            },
        ],
        "drift": {
            "status": (
                "baseline_initialized"
                if previous_rollup_hash is None
                else ("stable" if previous_rollup_hash == rollup_hash else "changed")
            ),
            "previous_rollup_hash": previous_rollup_hash,
            "current_rollup_hash": rollup_hash,
        },
    }
    out["integrity"] = {
        "baseline_snapshot_hash": baseline_snapshot_hash,
        "current_snapshot_hash": current_snapshot_hash,
        "reconstructed_snapshot_hash": reconstructed_hash,
        "reconstruction_ok": reconstruction_ok,
        "semantic_loss_check_ok": semantic_loss_check_ok,
        "semantic_loss_reasons": semantic_loss_reasons,
    }
    out["summary_artifact"] = {
        "anchor_refs": [f"context_slices/{row.get('slice_id')}#sha256:{row.get('sha256')}" for row in anchor_rows],
        "capsule_provenance_refs": [
            str(((cap.get("provenance_refs") or [None])[0]) or "")
            for cap in capsules
            if str(((cap.get("provenance_refs") or [None])[0]) or "").strip()
        ],
        "deliberation_rollup": rollup_primitives,
    }

    if fail_reasons:
        out["status"] = "fail_closed"
        out["mode"] = "full_passthrough"
        out["fallback_reason"] = fail_reasons[0]
        out["tokens"] = {
            "full_context_tokens": full_tokens,
            "compacted_tokens": full_tokens,
            "saved_tokens": 0,
            "saved_pct": 0.0,
        }
        out["cache"] = {
            "path": str(cache_path),
            "status": "skip_fail_closed",
            "cache_corrupt_recovered": cache_corrupt,
        }
        return out

    out["status"] = "pass"
    out["mode"] = "bootstrap_compact" if prev_entry is None else "rolling_compact"
    out["fallback_reason"] = None
    out["cache"] = {
        "path": str(cache_path),
        "status": "miss" if prev_entry is None else "hit",
        "cache_corrupt_recovered": cache_corrupt,
    }

    entries_obj[flow_key] = {
        "updated_at": now_iso(),
        "flow_binding": binding,
        "snapshot_hash": current_snapshot_hash,
        "rollup_hash": rollup_hash,
        "anchors": [
            {
                "slice_id": row.get("slice_id"),
                "sha256": row.get("sha256"),
            }
            for row in anchor_rows
        ],
        "capsules": [
            {
                "capsule_id": cap.get("capsule_id"),
                "slice_id": cap.get("slice_id"),
                "source_sha256": cap.get("source_sha256"),
                "anchor_refs": cap.get("anchor_refs"),
                "primitives": cap.get("primitives"),
            }
            for cap in capsules
        ],
    }
    cache["entries"] = entries_obj
    _save_compaction_cache(cache_path, cache)

    return out
