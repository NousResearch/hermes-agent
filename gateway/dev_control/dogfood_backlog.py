"""Harness-scoped dogfood backlog for Hermes Lab observe loops."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional


ALLOWED_TARGET_PREFIXES = (
    "gateway/dev_control/",
    "apps/oryn-workspace/",
    "scripts/",
    "docs/",
    "hermes-ops/",
    "tests/",
    "bootstrap/",
)
FORBIDDEN_TARGET_PREFIXES = (
    "agent/",
    "agent/transports/",
    "gateway/platforms/",
)
FORBIDDEN_TARGET_FILES = {
    "agent/conversation_loop.py",
    "agent/transports/chat_completions.py",
}
GUARDRAIL_TERMS = ("verify", "verification", "ci", "review", "merge", "gate", "policy")


def normalize_target_path(path: Any) -> str:
    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def dogfood_scope_check(target_paths: Iterable[Any]) -> dict[str, Any]:
    """Validate harness scope. Engine/core paths are skipped, never failed."""

    normalized = [normalize_target_path(path) for path in target_paths or [] if normalize_target_path(path)]
    rejected: list[str] = []
    for path in normalized:
        if path in FORBIDDEN_TARGET_FILES or any(path.startswith(prefix) for prefix in FORBIDDEN_TARGET_PREFIXES):
            rejected.append(path)
            continue
        if not any(path.startswith(prefix) or path == prefix.rstrip("/") for prefix in ALLOWED_TARGET_PREFIXES):
            rejected.append(path)
    return {
        "ok": bool(normalized) and not rejected,
        "target_paths": normalized,
        "rejected_paths": rejected,
        "status": "in_scope" if normalized and not rejected else "out_of_scope",
    }


def is_guardrail_touching(candidate: dict[str, Any]) -> bool:
    text = " ".join([
        str(candidate.get("profile_id") or ""),
        str(candidate.get("risk_level") or ""),
        str(candidate.get("source") or ""),
        str(candidate.get("prompt") or ""),
        " ".join(str(path) for path in candidate.get("target_paths") or []),
    ]).lower()
    return any(term in text for term in GUARDRAIL_TERMS)


def candidate_key(candidate: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        "\n".join([
            str(candidate.get("source") or ""),
            str(candidate.get("prompt") or ""),
            "\n".join(normalize_target_path(path) for path in candidate.get("target_paths") or []),
        ]).encode("utf-8")
    ).hexdigest()[:16]
    return f"dogfood:{digest}"


def normalize_candidate(candidate: dict[str, Any], *, approved: Optional[bool] = None, now: Optional[float] = None) -> dict[str, Any]:
    now_value = float(now or time.time())
    payload = dict(candidate or {})
    target_paths = [normalize_target_path(path) for path in payload.get("target_paths") or [] if normalize_target_path(path)]
    scope = dogfood_scope_check(target_paths)
    normalized = {
        "candidate_id": str(payload.get("candidate_id") or candidate_key({**payload, "target_paths": target_paths})),
        "prompt": str(payload.get("prompt") or "").strip(),
        "profile_id": str(payload.get("profile_id") or "platform.implement").strip(),
        "risk_level": str(payload.get("risk_level") or "low").strip().lower(),
        "target_paths": target_paths,
        "source": str(payload.get("source") or "manual").strip().lower(),
        "approved": bool(payload.get("approved") if approved is None else approved),
        "status": str(payload.get("status") or "candidate").strip().lower(),
        "scope_status": scope["status"],
        "scope_warnings": [f"Out-of-scope path: {path}" for path in scope["rejected_paths"]],
        "guardrail_touching": bool(payload.get("guardrail_touching")) or is_guardrail_touching(payload),
        "created_at": float(payload.get("created_at") or now_value),
        "updated_at": float(payload.get("updated_at") or now_value),
        "payload": payload.get("payload") or {},
    }
    if not normalized["prompt"]:
        normalized["prompt"] = "Improve Hermes harness reliability for the listed target paths."
    return normalized


def discover_todo_candidates(*, repo_roots: Iterable[Path | str], limit: int = 20, now: Optional[float] = None) -> list[dict[str, Any]]:
    """Find small TODO/FIXME-backed dogfood tasks inside allowlisted harness paths."""

    candidates: list[dict[str, Any]] = []
    pattern = re.compile(r"\b(TODO|FIXME)\b[:\s-]*(.+)?", re.IGNORECASE)
    for root_value in repo_roots or []:
        root = Path(root_value).expanduser()
        if not root.exists():
            continue
        for prefix in ALLOWED_TARGET_PREFIXES:
            base = root / prefix
            if not base.exists() or not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if len(candidates) >= limit:
                    return candidates
                if not path.is_file() or path.stat().st_size > 64_000:
                    continue
                rel = path.relative_to(root).as_posix()
                scope = dogfood_scope_check([rel])
                if not scope["ok"]:
                    continue
                try:
                    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for line_number, line in enumerate(lines, start=1):
                    match = pattern.search(line)
                    if not match:
                        continue
                    detail = (match.group(2) or line.strip())[:220]
                    candidates.append(normalize_candidate({
                        "candidate_id": f"dogfood:todo:{hashlib.sha1(f'{rel}:{line_number}:{detail}'.encode()).hexdigest()[:12]}",
                        "prompt": f"Address this harness TODO in {rel}:{line_number}: {detail}",
                        "profile_id": _profile_for_path(rel),
                        "risk_level": "low",
                        "target_paths": [rel],
                        "source": "todo",
                        "payload": {"line": line_number, "snippet": line.strip()},
                    }, now=now))
                    break
    return candidates[:limit]


def candidates_from_weakest_categories(categories: Iterable[dict[str, Any]], *, limit: int = 5, now: Optional[float] = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in list(categories or [])[: max(1, limit)]:
        category = str(row.get("category") or "").strip()
        if not category:
            continue
        prompt = (
            f"Improve reliability for {category}. Use the scorecard evidence to identify a small harness-scoped "
            "test, docs, or dev_control improvement. Do not touch Hermes engine files."
        )
        candidates.append(normalize_candidate({
            "prompt": prompt,
            "profile_id": "platform.implement",
            "risk_level": "medium" if row.get("escape_count") else "low",
            "target_paths": ["gateway/dev_control/", "tests/"],
            "source": "reliability",
            "guardrail_touching": is_guardrail_touching({"prompt": prompt, "target_paths": ["gateway/dev_control/", "tests/"]}),
            "payload": {"category": category, "scorecard": row},
        }, now=now))
    return candidates


def preapproval_sources_from_env() -> set[str]:
    raw = os.getenv("HERMES_DEV_LAB_PREAPPROVED_SOURCES", "docs,test,todo")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def preapproval_allows(candidate: dict[str, Any]) -> bool:
    source = str(candidate.get("source") or "").lower()
    if source in preapproval_sources_from_env():
        return not is_guardrail_touching(candidate)
    target_paths = [normalize_target_path(path) for path in candidate.get("target_paths") or []]
    if target_paths and all(path.startswith(("docs/", "tests/")) for path in target_paths):
        return not is_guardrail_touching(candidate)
    return False


def _profile_for_path(path: str) -> str:
    if path.startswith("apps/oryn-workspace/"):
        return "workspace.implement"
    return "platform.implement"
