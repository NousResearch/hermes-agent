"""Durable prior-work-first helpers for repository task execution.

The helpers in this module let a fresh agent recover an accepted execution
boundary from repository artifacts instead of re-running broad discovery on
every session.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROGRAM_ID = "HERMES-PRIOR-WORK-FIRST-EXECUTION-MEMORY-V1"
CANONICAL_STATE_ROOT = Path(".task-state") / "prior-work-first"
PACKET_FILENAME = "active.packet.json"
TRANSCRIPT_KEYS = frozenset(
    {
        "MESSAGES",
        "TRANSCRIPT",
        "TRANSCRIPTS",
        "CHAT_HISTORY",
        "CONVERSATION",
        "CONVERSATION_HISTORY",
        "RAG_TRANSCRIPT",
    }
)
PACKET_FIELDS = (
    "PROGRAM_ID",
    "GOAL",
    "CAPABILITY_ID",
    "PREFLIGHT_STATUS",
    "ACCEPTED_HEAD",
    "ACCEPTED_TREE",
    "FIRST_UNPROVEN_BOUNDARY",
    "KNOWN_FILES",
    "KNOWN_SYMBOLS",
    "KNOWN_TESTS",
    "KNOWN_DEPENDENCIES",
    "KNOWN_HANDOFFS",
    "KNOWN_RECEIPTS",
    "KNOWN_WORKTREES",
    "KNOWN_BRANCHES",
    "KNOWN_DRAFT_BUILDS",
    "KNOWN_BASELINE_EXCEPTIONS",
    "RELEVANT_CONTRACT_SLICES",
    "RELEVANT_SDD_SLICES",
    "PROVEN_INVARIANTS",
    "INVALIDATE_IF",
    "LAST_MACHINE_VERIFIED_AT",
    "PRIMARY_WRITER",
    "PARALLEL_SCOUTS",
)
LIST_FIELDS = {
    "KNOWN_FILES",
    "KNOWN_SYMBOLS",
    "KNOWN_TESTS",
    "KNOWN_DEPENDENCIES",
    "KNOWN_HANDOFFS",
    "KNOWN_RECEIPTS",
    "KNOWN_WORKTREES",
    "KNOWN_BRANCHES",
    "KNOWN_DRAFT_BUILDS",
    "KNOWN_BASELINE_EXCEPTIONS",
    "RELEVANT_CONTRACT_SLICES",
    "RELEVANT_SDD_SLICES",
    "PROVEN_INVARIANTS",
    "INVALIDATE_IF",
    "PARALLEL_SCOUTS",
}
PATH_LIST_FIELDS = {
    "KNOWN_FILES",
    "KNOWN_TESTS",
    "KNOWN_HANDOFFS",
    "KNOWN_RECEIPTS",
    "KNOWN_WORKTREES",
    "KNOWN_BRANCHES",
    "KNOWN_DRAFT_BUILDS",
    "RELEVANT_CONTRACT_SLICES",
    "RELEVANT_SDD_SLICES",
}
TARGETED_INVALIDATOR_KINDS = {
    "contract_change": "RELEVANT_CONTRACT_SLICES",
    "sdd_change": "RELEVANT_SDD_SLICES",
    "symbol_change": "KNOWN_SYMBOLS",
    "dependency_change": "KNOWN_DEPENDENCIES",
    "test_semantics_change": "KNOWN_TESTS",
    "file_change": "KNOWN_FILES",
    "architecture_owner_change": "RELEVANT_SDD_SLICES",
}
GLOBAL_INVALIDATOR_KINDS = {
    "accepted_baseline_invalidation",
    "contradictory_machine_evidence",
}
AUTHORITY_ORDER = {
    "locator": [
        "preflight_packet",
        "receipt",
        "handoff",
        "rag",
        "conversation",
    ],
    "semantic": [
        "contract",
        "source",
        "tests",
        "sdd",
        "machine_evidence",
        "receipt",
        "handoff",
        "rag",
        "conversation",
    ],
    "implementation": [
        "source",
        "tests",
        "contract",
        "sdd",
        "machine_evidence",
        "receipt",
        "handoff",
        "rag",
        "conversation",
    ],
    "verification": [
        "machine_evidence",
        "tests",
        "source",
        "contract",
        "sdd",
        "receipt",
        "handoff",
        "rag",
        "conversation",
    ],
}


def capability_slug(capability_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", capability_id.strip().lower()).strip("-")
    return slug or "unnamed-capability"


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_symbol(path: Path, symbol: str) -> str | None:
    source = _read_text(path)
    if source is None:
        return None
    segment = _extract_symbol_segment(source, symbol)
    if segment is None:
        return None
    return digest_text(segment)


def canonical_capability_dir(repo_root: Path, capability_id: str) -> Path:
    return repo_root / CANONICAL_STATE_ROOT / capability_slug(capability_id)


def canonical_packet_path(repo_root: Path, capability_id: str) -> Path:
    return canonical_capability_dir(repo_root, capability_id) / PACKET_FILENAME


def normalize_packet(payload: Mapping[str, Any]) -> dict[str, Any]:
    packet: dict[str, Any] = {"PROGRAM_ID": PROGRAM_ID}
    for key in PACKET_FIELDS:
        if key not in payload:
            continue
        if key in TRANSCRIPT_KEYS:
            continue
        value = payload[key]
        if key in LIST_FIELDS:
            packet[key] = _normalize_list_field(key, value)
        else:
            packet[key] = value
    for key in TRANSCRIPT_KEYS:
        packet.pop(key, None)
    packet.setdefault("PROGRAM_ID", PROGRAM_ID)
    packet.setdefault("PREFLIGHT_STATUS", "unknown")
    packet.setdefault("INVALIDATE_IF", [])
    packet.setdefault("PROVEN_INVARIANTS", [])
    packet.setdefault("PRIMARY_WRITER", None)
    packet.setdefault("PARALLEL_SCOUTS", [])
    return packet


def discover_prior_work(
    repo_root: Path | str, capability_id: str, *, limit: int = 64
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    slug = capability_slug(capability_id)
    state_root = repo_root / ".task-state"
    capability_root = canonical_capability_dir(repo_root, capability_id)
    packet_path = capability_root / PACKET_FILENAME
    artifacts = {
        "preflight_packets": [_relpath(packet_path, repo_root)] if packet_path.is_file() else [],
        "handoffs": _limited_file_listing(capability_root / "handoffs", repo_root=repo_root, limit=limit),
        "receipts": _limited_file_listing(capability_root / "receipts", repo_root=repo_root, limit=limit),
        "draft_builds": _limited_file_listing(capability_root / "drafts", repo_root=repo_root, limit=limit),
        "rag": _limited_file_listing(capability_root / "rag", repo_root=repo_root, limit=limit),
        "ledgers": _limited_file_listing(capability_root / "ledgers", repo_root=repo_root, limit=limit),
    }
    fallback_hits: list[str] = []
    if state_root.exists():
        pattern = f"*{slug}*"
        for path in state_root.rglob(pattern):
            if len(fallback_hits) >= limit:
                break
            if path.is_file():
                fallback_hits.append(_relpath(path, repo_root))
    git_state = collect_git_state(repo_root)
    return {
        "GOAL": None,
        "CAPABILITY_ID": capability_id,
        "SEARCH_MODE": "targeted_read_only",
        "FULL_HISTORY_LOADED": False,
        "SEARCHED_ROOTS": [
            _relpath(state_root, repo_root),
            _relpath(capability_root, repo_root),
        ],
        "ARTIFACTS": artifacts,
        "FALLBACK_MATCHES": fallback_hits,
        "KNOWN_WORKTREES": git_state["KNOWN_WORKTREES"],
        "KNOWN_BRANCHES": git_state["KNOWN_BRANCHES"],
        "ACCEPTED_HEAD": git_state["HEAD"],
        "ACCEPTED_TREE": git_state["TREE"],
        "HANDOFF_DISCOVERY_REQUIRED": True,
        "DRAFT_BUILD_DISCOVERY_REQUIRED": True,
        "WORKTREE_DISCOVERY_REQUIRED": True,
        "RAG_AS_LOCATOR_REQUIRED": True,
    }


def collect_git_state(repo_root: Path | str) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    head = _git_output(repo_root, "rev-parse", "HEAD")
    tree = _git_output(repo_root, "rev-parse", "HEAD^{tree}")
    worktrees = _parse_worktree_list(_git_output(repo_root, "worktree", "list", "--porcelain"))
    branches = _parse_branch_list(
        _git_output(
            repo_root,
            "for-each-ref",
            "--format=%(refname:short)\t%(objectname)\t%(upstream:short)",
            "refs/heads",
            "refs/remotes",
        )
    )
    return {
        "HEAD": head,
        "TREE": tree,
        "KNOWN_WORKTREES": worktrees,
        "KNOWN_BRANCHES": branches,
    }


def capture_repo_state(
    repo_root: Path | str,
    packet: Mapping[str, Any],
    *,
    machine_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    packet = normalize_packet(packet)
    git_state = collect_git_state(repo_root)
    state: dict[str, Any] = {
        "HEAD": git_state["HEAD"],
        "TREE": git_state["TREE"],
        "KNOWN_WORKTREES": git_state["KNOWN_WORKTREES"],
        "KNOWN_BRANCHES": git_state["KNOWN_BRANCHES"],
        "KNOWN_FILES": _capture_path_records(repo_root, packet.get("KNOWN_FILES", [])),
        "KNOWN_TESTS": _capture_path_records(repo_root, packet.get("KNOWN_TESTS", [])),
        "KNOWN_HANDOFFS": _capture_path_records(repo_root, packet.get("KNOWN_HANDOFFS", [])),
        "KNOWN_RECEIPTS": _capture_path_records(repo_root, packet.get("KNOWN_RECEIPTS", [])),
        "KNOWN_DRAFT_BUILDS": _capture_path_records(
            repo_root, packet.get("KNOWN_DRAFT_BUILDS", [])
        ),
        "RELEVANT_CONTRACT_SLICES": _capture_path_records(
            repo_root, packet.get("RELEVANT_CONTRACT_SLICES", [])
        ),
        "RELEVANT_SDD_SLICES": _capture_path_records(
            repo_root, packet.get("RELEVANT_SDD_SLICES", [])
        ),
        "KNOWN_SYMBOLS": _capture_symbol_records(repo_root, packet.get("KNOWN_SYMBOLS", [])),
        "KNOWN_DEPENDENCIES": _capture_dependency_records(
            repo_root, packet.get("KNOWN_DEPENDENCIES", [])
        ),
        "MACHINE_EVIDENCE": [dict(item) for item in (machine_evidence or [])],
    }
    return state


def evaluate_preflight(
    packet: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    attempted_full_rediscovery: bool = False,
    attempted_full_preflight_reexecution: bool = False,
    attempted_replan: bool = False,
) -> dict[str, Any]:
    packet = normalize_packet(packet)
    writer_check = validate_writer_invariants(packet)
    invalidations = detect_invalidations(packet, state)
    targeted_only = invalidations and all(
        item["kind"] not in GLOBAL_INVALIDATOR_KINDS for item in invalidations
    )
    resume_from_boundary = bool(packet.get("FIRST_UNPROVEN_BOUNDARY"))
    accepted = str(packet.get("PREFLIGHT_STATUS") or "").lower() in {
        "accepted",
        "accepted_reuse",
        "pass",
        "passed",
        "verified",
    }
    no_invalidators = len(invalidations) == 0
    decision = {
        "PROGRAM_ID": packet.get("PROGRAM_ID", PROGRAM_ID),
        "CAPABILITY_ID": packet.get("CAPABILITY_ID"),
        "GOAL": packet.get("GOAL"),
        "PREFLIGHT_STATUS": (
            "accepted_reuse"
            if accepted and no_invalidators
            else "targeted_refresh_required"
            if targeted_only
            else "global_refresh_required"
            if invalidations
            else packet.get("PREFLIGHT_STATUS", "unknown")
        ),
        "ACCEPTED_HEAD": packet.get("ACCEPTED_HEAD"),
        "ACCEPTED_TREE": packet.get("ACCEPTED_TREE"),
        "CURRENT_HEAD": state.get("HEAD"),
        "CURRENT_TREE": state.get("TREE"),
        "FIRST_UNPROVEN_BOUNDARY": packet.get("FIRST_UNPROVEN_BOUNDARY"),
        "INVALIDATOR_COUNT": len(invalidations),
        "INVALIDATORS_TRIGGERED": invalidations,
        "REFRESH_SCOPE": _refresh_scope(invalidations),
        "RESUME_FROM_FIRST_UNPROVEN_BOUNDARY": resume_from_boundary,
        "FULL_DISCOVERY_FORBIDDEN": no_invalidators or targeted_only,
        "FULL_PREFLIGHT_REEXECUTION_FORBIDDEN": no_invalidators or targeted_only,
        "REPLAN_FORBIDDEN": no_invalidators or targeted_only,
        "REDUNDANT_PREFLIGHT_ATTEMPT": no_invalidators
        and (
            attempted_full_rediscovery
            or attempted_full_preflight_reexecution
            or attempted_replan
        ),
        "PREFLIGHT_REUSE_REQUIRED": accepted and no_invalidators,
        "REOPEN_REQUIRED": not (accepted and no_invalidators),
        "HANDOFF_DISCOVERY_REQUIRED": True,
        "DRAFT_BUILD_DISCOVERY_REQUIRED": True,
        "WORKTREE_DISCOVERY_REQUIRED": True,
        "RAG_AS_LOCATOR_REQUIRED": True,
        "ONE_WRITER_INVARIANT_PRESERVED": writer_check["ok"],
        "CANONICAL_PRIMARY_WRITER_COUNT": writer_check["primary_writer_count"],
        "PARALLEL_SCOUT_VIOLATIONS": writer_check["violations"],
    }
    return decision


def detect_invalidations(
    packet: Mapping[str, Any], state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    invalidations: list[dict[str, Any]] = []
    record_maps = _record_maps(state)
    for raw_rule in packet.get("INVALIDATE_IF", []):
        rule = dict(raw_rule) if isinstance(raw_rule, Mapping) else {"kind": str(raw_rule)}
        kind = str(rule.get("kind") or "").strip()
        if kind in TARGETED_INVALIDATOR_KINDS:
            field = TARGETED_INVALIDATOR_KINDS[kind]
            current = _resolve_record_for_rule(field, rule, record_maps)
            expected = _expected_digest(rule, current)
            current_digest = None if current is None else current.get("current_digest")
            if expected and current_digest and expected != current_digest:
                invalidations.append(
                    _make_invalidation(
                        kind,
                        rule,
                        scope="targeted",
                        current=current,
                        reason=f"{kind} changed",
                    )
                )
            elif expected and current is None:
                invalidations.append(
                    _make_invalidation(
                        kind,
                        rule,
                        scope="targeted",
                        current=None,
                        reason=f"{kind} target missing",
                    )
                )
            continue
        if kind == "accepted_baseline_invalidation":
            if _baseline_invalidated(rule, packet, state):
                invalidations.append(
                    _make_invalidation(
                        kind,
                        rule,
                        scope="global",
                        current=None,
                        reason="accepted baseline marked invalid",
                    )
                )
            continue
        if kind == "contradictory_machine_evidence":
            if _machine_evidence_contradicts(rule, state):
                invalidations.append(
                    _make_invalidation(
                        kind,
                        rule,
                        scope="global",
                        current=None,
                        reason="fresh machine evidence contradicts accepted state",
                    )
                )
            continue
    return invalidations


def validate_writer_invariants(packet: Mapping[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    primary = packet.get("PRIMARY_WRITER")
    primary_count = 0
    if isinstance(primary, (list, tuple, set)):
        primary_count = len({str(item) for item in primary if str(item).strip()})
    elif str(primary or "").strip():
        primary_count = 1
    if primary_count > 1:
        violations.append("multiple canonical writers declared")
    for scout in packet.get("PARALLEL_SCOUTS", []):
        data = dict(scout) if isinstance(scout, Mapping) else {"name": str(scout)}
        read_only = data.get("read_only")
        mode = str(data.get("mode") or "").strip().lower()
        if read_only is False or mode not in {"", "read_only", "locator"}:
            violations.append(f"parallel scout not read-only: {data.get('name') or data}")
    return {
        "ok": not violations and primary_count <= 1,
        "primary_writer_count": primary_count,
        "violations": violations,
    }


def resolve_authority(
    evidence: Sequence[Mapping[str, Any]], *, intent: str = "implementation"
) -> Mapping[str, Any] | None:
    order = AUTHORITY_ORDER.get(intent, AUTHORITY_ORDER["implementation"])
    ranked = {kind: index for index, kind in enumerate(order)}
    selected: Mapping[str, Any] | None = None
    selected_rank = len(order) + 1
    for item in evidence:
        kind = str(item.get("kind") or "").strip()
        rank = ranked.get(kind, len(order) + 1)
        if selected is None or rank < selected_rank:
            selected = item
            selected_rank = rank
    return selected


def reconstruct_resume_context(
    repo_root: Path | str,
    packet: Mapping[str, Any],
    *,
    machine_evidence: Sequence[Mapping[str, Any]] | None = None,
    attempted_full_rediscovery: bool = False,
    attempted_full_preflight_reexecution: bool = False,
    attempted_replan: bool = False,
) -> dict[str, Any]:
    packet = normalize_packet(packet)
    capability_id = str(packet.get("CAPABILITY_ID") or "")
    discovery = discover_prior_work(repo_root, capability_id)
    state = capture_repo_state(repo_root, packet, machine_evidence=machine_evidence)
    decision = evaluate_preflight(
        packet,
        state,
        attempted_full_rediscovery=attempted_full_rediscovery,
        attempted_full_preflight_reexecution=attempted_full_preflight_reexecution,
        attempted_replan=attempted_replan,
    )
    return {
        "GOAL": packet.get("GOAL"),
        "CAPABILITY_ID": capability_id,
        "PRIOR_WORK_LOCATE": discovery,
        "ACCEPTED_BASELINE": {
            "HEAD": packet.get("ACCEPTED_HEAD"),
            "TREE": packet.get("ACCEPTED_TREE"),
        },
        "FIRST_UNPROVEN_BOUNDARY": packet.get("FIRST_UNPROVEN_BOUNDARY"),
        "INVALIDATOR_CHECK": decision,
        "LOAD_RELEVANT_CONTRACT_SLICES": packet.get("RELEVANT_CONTRACT_SLICES", []),
        "LOAD_RELEVANT_SDD_SLICES": packet.get("RELEVANT_SDD_SLICES", []),
        "LOAD_KNOWN_FILES": packet.get("KNOWN_FILES", []),
        "LOAD_KNOWN_SYMBOLS": packet.get("KNOWN_SYMBOLS", []),
        "LOAD_KNOWN_TESTS": packet.get("KNOWN_TESTS", []),
        "NEXT_ACTION": (
            "RESUME_FROM_FIRST_UNPROVEN_BOUNDARY"
            if decision["PREFLIGHT_REUSE_REQUIRED"]
            else "REFRESH_ONLY_AFFECTED_SLICES"
            if decision["PREFLIGHT_STATUS"] == "targeted_refresh_required"
            else "REBUILD_ACCEPTED_BASELINE"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check and recover prior-work-first execution packets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    locate = subparsers.add_parser("locate", help="Read-only targeted prior-work discovery.")
    locate.add_argument("--repo-root", default=".")
    locate.add_argument("--capability-id", required=True)

    check = subparsers.add_parser("check", help="Evaluate a prior-work packet.")
    check.add_argument("--repo-root", default=".")
    check.add_argument("--packet", required=True)
    check.add_argument("--machine-evidence")
    check.add_argument("--attempt-full-rediscovery", action="store_true")
    check.add_argument("--attempt-full-preflight-reexecution", action="store_true")
    check.add_argument("--attempt-replan", action="store_true")

    resume = subparsers.add_parser(
        "resume",
        help="Reconstruct the startup flow from packet + repository artifacts.",
    )
    resume.add_argument("--repo-root", default=".")
    resume.add_argument("--packet", required=True)
    resume.add_argument("--machine-evidence")
    resume.add_argument("--attempt-full-rediscovery", action="store_true")
    resume.add_argument("--attempt-full-preflight-reexecution", action="store_true")
    resume.add_argument("--attempt-replan", action="store_true")

    compact = subparsers.add_parser(
        "compact-packet", help="Normalize a packet into the compact durable shape."
    )
    compact.add_argument("--packet", required=True)
    compact.add_argument("--output")

    args = parser.parse_args(argv)
    if args.command == "locate":
        print(
            json.dumps(
                discover_prior_work(Path(args.repo_root), args.capability_id),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "compact-packet":
        payload = json.loads(Path(args.packet).read_text(encoding="utf-8"))
        compact_packet = normalize_packet(payload)
        rendered = json.dumps(compact_packet, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0
    packet = json.loads(Path(args.packet).read_text(encoding="utf-8"))
    machine_evidence = _load_machine_evidence(getattr(args, "machine_evidence", None))
    repo_root = Path(args.repo_root)
    if args.command == "check":
        state = capture_repo_state(repo_root, packet, machine_evidence=machine_evidence)
        decision = evaluate_preflight(
            packet,
            state,
            attempted_full_rediscovery=args.attempt_full_rediscovery,
            attempted_full_preflight_reexecution=args.attempt_full_preflight_reexecution,
            attempted_replan=args.attempt_replan,
        )
        print(json.dumps(decision, indent=2, sort_keys=True))
        return 0
    context = reconstruct_resume_context(
        repo_root,
        packet,
        machine_evidence=machine_evidence,
        attempted_full_rediscovery=args.attempt_full_rediscovery,
        attempted_full_preflight_reexecution=args.attempt_full_preflight_reexecution,
        attempted_replan=args.attempt_replan,
    )
    print(json.dumps(context, indent=2, sort_keys=True))
    return 0


def _normalize_list_field(key: str, value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    if key in PATH_LIST_FIELDS:
        return [_coerce_path_like(item) for item in value]
    if key == "KNOWN_SYMBOLS":
        return [_coerce_symbol_like(item) for item in value]
    if key == "KNOWN_DEPENDENCIES":
        return [_coerce_dependency_like(item) for item in value]
    if key == "INVALIDATE_IF":
        return [dict(item) if isinstance(item, Mapping) else {"kind": str(item)} for item in value]
    if key == "PARALLEL_SCOUTS":
        normalized = []
        for item in value:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            else:
                normalized.append({"name": str(item), "mode": "read_only", "read_only": True})
        return normalized
    return list(value)


def _coerce_path_like(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    return {"path": str(item)}


def _coerce_symbol_like(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    raw = str(item)
    if "::" in raw:
        path, symbol = raw.split("::", 1)
        return {"path": path, "symbol": symbol}
    return {"symbol": raw}


def _coerce_dependency_like(item: Any) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    return {"name": str(item)}


def _capture_path_records(repo_root: Path, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    captured = []
    for raw in records:
        entry = dict(raw)
        path = entry.get("path")
        if not path:
            continue
        abs_path = repo_root / str(path)
        entry["path"] = _relpath(abs_path, repo_root)
        entry["exists"] = abs_path.exists()
        entry["current_digest"] = digest_path(abs_path)
        captured.append(entry)
    return captured


def _capture_symbol_records(
    repo_root: Path, records: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    captured = []
    for raw in records:
        entry = dict(raw)
        path = entry.get("path")
        symbol = entry.get("symbol")
        if not path or not symbol:
            continue
        abs_path = repo_root / str(path)
        entry["path"] = _relpath(abs_path, repo_root)
        entry["exists"] = abs_path.exists()
        entry["current_digest"] = digest_symbol(abs_path, str(symbol))
        captured.append(entry)
    return captured


def _capture_dependency_records(
    repo_root: Path, records: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    captured = []
    for raw in records:
        entry = dict(raw)
        path = entry.get("path")
        if path:
            abs_path = repo_root / str(path)
            entry["path"] = _relpath(abs_path, repo_root)
            entry["exists"] = abs_path.exists()
            entry["current_digest"] = digest_path(abs_path)
        else:
            entry["exists"] = False
            entry["current_digest"] = None
        captured.append(entry)
    return captured


def _load_machine_evidence(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    return []


def _resolve_record_for_rule(
    field: str, rule: Mapping[str, Any], record_maps: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> Mapping[str, Any] | None:
    if field == "KNOWN_SYMBOLS":
        key = f"{_normalize_path_key(rule.get('path'))}::{rule.get('symbol')}"
        return record_maps.get(field, {}).get(key)
    path = rule.get("path")
    if path:
        return record_maps.get(field, {}).get(_normalize_path_key(path))
    return None


def _expected_digest(rule: Mapping[str, Any], current: Mapping[str, Any] | None) -> str | None:
    for key in ("expected_digest", "digest", "accepted_digest", "baseline_digest"):
        value = rule.get(key)
        if value:
            return str(value)
    if current is not None:
        for key in ("digest", "accepted_digest", "baseline_digest"):
            value = current.get(key)
            if value:
                return str(value)
    return None


def _baseline_invalidated(
    rule: Mapping[str, Any], packet: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    marker = rule.get("baseline_valid")
    if marker is False:
        return True
    expected_tree = rule.get("expected_tree")
    if expected_tree and expected_tree != state.get("TREE"):
        return True
    expected_head = rule.get("expected_head")
    if expected_head and expected_head != state.get("HEAD"):
        return True
    mismatch = rule.get("mismatch_requires_invalidation")
    if mismatch and packet.get("ACCEPTED_TREE") != state.get("TREE"):
        return True
    return False


def _machine_evidence_contradicts(
    rule: Mapping[str, Any], state: Mapping[str, Any]
) -> bool:
    subjects = set()
    if rule.get("test"):
        subjects.add(str(rule["test"]))
    if rule.get("subject"):
        subjects.add(str(rule["subject"]))
    for item in state.get("MACHINE_EVIDENCE", []):
        status = str(item.get("status") or "").upper()
        if status not in {"FAIL", "FAILED", "ERROR"}:
            continue
        if not subjects:
            return True
        if str(item.get("test") or item.get("subject") or "") in subjects:
            return True
    return False


def _make_invalidation(
    kind: str,
    rule: Mapping[str, Any],
    *,
    scope: str,
    current: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    entry = {
        "kind": kind,
        "scope": scope,
        "reason": reason,
    }
    for key in ("path", "symbol", "test", "subject"):
        if rule.get(key):
            entry[key] = rule[key]
    if current is not None:
        entry["current_digest"] = current.get("current_digest")
    expected = _expected_digest(rule, current)
    if expected:
        entry["expected_digest"] = expected
    return entry


def _refresh_scope(invalidations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refresh = []
    seen = set()
    for item in invalidations:
        target = {
            "kind": item.get("kind"),
            "scope": item.get("scope"),
        }
        for key in ("path", "symbol", "test", "subject"):
            if item.get(key):
                target[key] = item[key]
        marker = tuple(sorted(target.items()))
        if marker in seen:
            continue
        seen.add(marker)
        refresh.append(target)
    return refresh


def _record_maps(state: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    maps: dict[str, dict[str, Mapping[str, Any]]] = {}
    for field in TARGETED_INVALIDATOR_KINDS.values():
        lookup: dict[str, Mapping[str, Any]] = {}
        for record in state.get(field, []):
            if field == "KNOWN_SYMBOLS":
                key = f"{_normalize_path_key(record.get('path'))}::{record.get('symbol')}"
            else:
                key = _normalize_path_key(record.get("path"))
            lookup[key] = record
        maps[field] = lookup
    return maps


def _limited_file_listing(
    directory: Path,
    *,
    repo_root: Path | None = None,
    suffixes: set[str] | None = None,
    limit: int = 64,
) -> list[str]:
    if not directory.exists():
        return []
    found: list[str] = []
    for path in sorted(directory.rglob("*")):
        if len(found) >= limit:
            break
        if not path.is_file():
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        found.append(_relpath(path, repo_root) if repo_root is not None else str(path))
    return found


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _normalize_path_key(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _parse_worktree_list(output: str | None) -> list[dict[str, str]]:
    if not output:
        return []
    entries: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in output.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree ") :].strip()
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD ") :].strip()
        elif line.startswith("branch "):
            branch = line[len("branch ") :].strip()
            current["branch"] = branch.removeprefix("refs/heads/")
    if current:
        entries.append(current)
    return entries


def _parse_branch_list(output: str | None) -> list[dict[str, str]]:
    if not output:
        return []
    branches = []
    for line in output.splitlines():
        if not line.strip():
            continue
        name, _, remainder = line.partition("\t")
        commit, _, upstream = remainder.partition("\t")
        branches.append(
            {
                "branch": name.strip(),
                "head": commit.strip(),
                "upstream": upstream.strip(),
            }
        )
    return branches


def _read_text(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _extract_symbol_segment(source: str, symbol: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_fallback_symbol_segment(source, symbol)
    lines = source.splitlines()
    for qualname, node in _iter_symbol_nodes(tree):
        if qualname != symbol:
            continue
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if not start or not end:
            return None
        return "\n".join(lines[start - 1 : end])
    return _extract_fallback_symbol_segment(source, symbol)


def _iter_symbol_nodes(tree: ast.AST) -> Iterable[tuple[str, ast.AST]]:
    def walk(body: list[ast.stmt], prefix: str = "") -> Iterable[tuple[str, ast.AST]]:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                yield name, node
                child_body = getattr(node, "body", None)
                if isinstance(child_body, list):
                    yield from walk(child_body, name)

    return walk(getattr(tree, "body", []))


def _extract_fallback_symbol_segment(source: str, symbol: str) -> str | None:
    for line in source.splitlines():
        if symbol in line:
            return line
    return None


if __name__ == "__main__":
    raise SystemExit(main())
