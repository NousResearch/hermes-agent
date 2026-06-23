"""Policy layer for background self-modification quarantine.

Background self-improvement review is allowed to observe and propose, but not
mutate canonical skills, profile/memory, rules, MCP config, or other
personalization state.  This module is intentionally tool-layer usable so a
prompt cannot bypass the boundary by calling the mutating tool directly.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover
    def get_hermes_home() -> Path:  # type: ignore
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))

PROPOSAL_SCHEMA = "loaw.system-improvement-proposal.v1"
MEMORY_CANDIDATE_SCHEMA = "loaw.governed-memory-candidate.v1"
IMMUTABLE_RECEIPT_SCHEMA = "loaw.immutable-capture-receipt.v1"

PROTECTED_PATH_HINTS = (
    "SKILL.md",
    "/references/",
    "/templates/",
    "/scripts/",
    "/profiles/",
    "/memories/",
    "/memory",
    "/rules/",
    ".cursorrules",
    "AGENTS.md",
    "CLAUDE.md",
    "mcp.json",
    "config.yaml",
    "tool-permissions",
)

PROHIBITED_MEMORY_PATTERNS = {
    "secret": re.compile(r"(?i)(api[_-]?key|token|password|secret|bearer\s+[a-z0-9])"),
    "client_or_matter_fact": re.compile(r"(?i)(client|matter|case\s*(no\.?|number)|clio|opposing counsel)"),
    "privileged_or_email": re.compile(r"(?i)(attorney[- ]client|privileged|raw email|gmail|inbox|from:|subject:)"),
    "medical_filing_iolta": re.compile(r"(?i)(medical|diagnos|treatment|filing|served|service|iolta|trust account)"),
    "protected_trait_or_appearance": re.compile(r"(?i)(race|ethnicity|religion|disability|pregnan|appearance|looks like)"),
    "unsupported_inference": re.compile(r"(?i)(must be|obviously|clearly)\s+(lying|guilty|fraud|disabled|illegal)"),
}

@dataclass(frozen=True)
class QuarantineDecision:
    allowed: bool
    response: Optional[str] = None


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _home() -> Path:
    return Path(get_hermes_home()).expanduser()


def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _hash_file(path: Path) -> Optional[str]:
    try:
        return _sha256_bytes(path.read_bytes())
    except FileNotFoundError:
        return None


def is_background_review_context() -> bool:
    try:
        from tools.skill_provenance import is_background_review
        return bool(is_background_review())
    except Exception:
        return False


def _skill_path(name: str, category: Optional[str], file_path: Optional[str]) -> Path:
    base = _home() / "skills"
    parts = [p for p in (category or "").split("/") if p]
    parts.append(name)
    if file_path:
        parts.append(file_path)
    else:
        parts.append("SKILL.md")
    return base.joinpath(*parts)


def _safe_unified_diff(path: Path, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="",
    ))


def _proposed_skill_diff(action: str, name: str, payload: Dict[str, Any]) -> tuple[List[str], str, Dict[str, Optional[str]]]:
    category = payload.get("category")
    file_path = payload.get("file_path")
    target = _skill_path(name, category, file_path)
    before = _read_file(target)
    after = before
    if action in {"create", "edit"}:
        after = payload.get("content") or ""
    elif action == "patch":
        old = payload.get("old_string") or ""
        new = payload.get("new_string") or ""
        if payload.get("replace_all"):
            after = before.replace(old, new)
        elif old in before:
            after = before.replace(old, new, 1)
        else:
            after = before + "\n# Proposed patch could not be applied exactly in quarantine.\n"
    elif action == "write_file":
        after = payload.get("file_content") or ""
    elif action in {"delete", "remove_file"}:
        after = ""
    diff = _safe_unified_diff(target, before, after)
    return [str(target)], diff, {str(target): _hash_file(target)}


def _proposal_dir() -> Path:
    return _home() / "system-improvement-proposals"


def build_system_improvement_proposal(
    *,
    problem: str,
    proposed_change_type: str,
    proposed_paths: List[str],
    current_file_hashes: Dict[str, Optional[str]],
    proposed_diff: str,
    triggering_run_session: Optional[str] = None,
    risk_tier: str = "yellow",
    tests_required: Optional[List[str]] = None,
    rollback: Optional[str] = None,
    source_anchors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    diff_hash = _sha256_text(proposed_diff)
    return {
        "schema": PROPOSAL_SCHEMA,
        "proposal_id": f"sip-{_utc().replace(':','').replace('-','')}-{uuid.uuid4().hex[:12]}",
        "created_at": _utc(),
        "created_by": "hermes-background-review-quarantine" if is_background_review_context() else "hermes-policy-layer",
        "triggering_run_session": triggering_run_session or os.environ.get("HERMES_SESSION_ID", "unknown"),
        "problem": problem,
        "proposed_change_type": proposed_change_type,
        "proposed_paths": proposed_paths,
        "source_anchors": source_anchors or {},
        "current_file_hashes": current_file_hashes,
        "proposed_diff": proposed_diff,
        "proposed_diff_sha256": diff_hash,
        "risk_tier": risk_tier,
        "tests_required": tests_required or ["focused unit tests", "protected file hash sentinel"],
        "mutation_boundary": "background review may propose only; direct canonical mutation is forbidden",
        "rollback": rollback or "drop proposal; no canonical mutation occurred",
        "proof_gates": ["proposal schema valid", "diff hash matches", "execute_approved is false", "human approval before mutation"],
        "approval_requirement": "explicit foreground approval through separate change control",
        "execute_approved": False,
        "state": "awaiting_approval",
    }


def quarantine_skill_mutation(action: str, name: str, payload: Dict[str, Any]) -> QuarantineDecision:
    if action not in {"create", "edit", "patch", "delete", "write_file", "remove_file"}:
        return QuarantineDecision(True)
    if not is_background_review_context():
        return QuarantineDecision(True)
    proposed_paths, diff, hashes = _proposed_skill_diff(action, name, payload)
    proposal = build_system_improvement_proposal(
        problem="Background review attempted a canonical skill/library mutation.",
        proposed_change_type=f"skill_manage.{action}",
        proposed_paths=proposed_paths,
        current_file_hashes=hashes,
        proposed_diff=diff,
        tests_required=["quarantine proposal test", "rollback required", "protected file byte-for-byte sentinel"],
        rollback="No rollback needed unless later approved; current operation performs no mutation.",
    )
    path = _proposal_dir() / f"{proposal['proposal_id']}.json"
    _atomic_write(path, proposal)
    return QuarantineDecision(False, json.dumps({
        "success": True,
        "quarantined": True,
        "proposal_path": str(path),
        "proposal_id": proposal["proposal_id"],
        "execute_approved": False,
        "message": "Background self-modification was converted to a system-improvement proposal; no protected file was changed.",
    }, ensure_ascii=False))


def _classify_memory_candidate(content: str, target: str) -> Dict[str, Any]:
    violations = [name for name, rx in PROHIBITED_MEMORY_PATTERNS.items() if rx.search(content or "")]
    if violations:
        return {"allowed": False, "violations": violations, "data_classification": "prohibited"}
    lowered = (content or "").lower()
    if any(k in lowered for k in ["prefers", "preference", "likes", "dislikes", "call me", "wants"]):
        route = "governed_profile_candidate" if target == "user" else "honcho_candidate"
        classification = "low_risk_personalization_candidate"
    elif any(k in lowered for k in ["project", "repo", "branch", "service", "port"]):
        route = "letta_candidate"
        classification = "project_or_operational_state_candidate"
    else:
        route = "honcho_candidate"
        classification = "low_risk_operational_map_candidate"
    return {"allowed": True, "violations": [], "target_routing": route, "data_classification": classification}


def quarantine_memory_mutation(action: str, target: str, content: Optional[str], old_text: Optional[str]) -> QuarantineDecision:
    if action not in {"add", "replace", "remove"}:
        return QuarantineDecision(True)
    if not is_background_review_context():
        return QuarantineDecision(True)
    text = content if action != "remove" else old_text
    classification = _classify_memory_candidate(text or "", target)
    receipt = {
        "schema": IMMUTABLE_RECEIPT_SCHEMA,
        "receipt_id": f"memory-capture-{uuid.uuid4().hex[:12]}",
        "created_at": _utc(),
        "action": action,
        "target": target,
        "content_sha256": _sha256_text(text or ""),
        "classification": classification,
        "source_receipt": {"origin": "background_review", "session": os.environ.get("HERMES_SESSION_ID", "unknown")},
        "idempotency_key": _sha256_text(json.dumps({"action": action, "target": target, "text": text}, sort_keys=True)),
        "promotion_policy": "candidate-only; separate approval required for canonical profile/memory writes",
        "execute_approved": False,
    }
    receipts_dir = _home() / "memory-capture-receipts"
    receipt_path = receipts_dir / f"{receipt['receipt_id']}.json"
    _atomic_write(receipt_path, receipt)
    if not classification["allowed"]:
        return QuarantineDecision(False, json.dumps({
            "success": False,
            "rejected": True,
            "violations": classification["violations"],
            "receipt_path": str(receipt_path),
            "message": "Background memory capture rejected by data-classification policy.",
        }, ensure_ascii=False))
    candidate = {
        "schema": MEMORY_CANDIDATE_SCHEMA,
        "candidate_id": f"memcand-{uuid.uuid4().hex[:12]}",
        "created_at": _utc(),
        "action": action,
        "target": target,
        "target_routing": classification["target_routing"],
        "data_classification": classification["data_classification"],
        "source_receipt": str(receipt_path),
        "conflict_check": "pending_human_review",
        "idempotency_key": receipt["idempotency_key"],
        "promotion_policy": receipt["promotion_policy"],
        "execute_approved": False,
        "state": "awaiting_approval",
        "content": content if action != "remove" else None,
        "old_text": old_text,
    }
    cand_path = _home() / "governed-memory-candidates" / f"{candidate['candidate_id']}.json"
    _atomic_write(cand_path, candidate)
    return QuarantineDecision(False, json.dumps({
        "success": True,
        "candidate_created": True,
        "candidate_path": str(cand_path),
        "receipt_path": str(receipt_path),
        "execute_approved": False,
        "message": "Background memory write captured as governed candidate; canonical profile/memory unchanged.",
    }, ensure_ascii=False))
