"""Deterministic local skill routing/proof helpers.

This module is intentionally local and provider-free.  It ranks visible skills
from the active Hermes skill roots so complex tasks can prove which skills were
candidate matches before the model decides what to load with ``skill_view``.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from agent.skill_utils import (
    extract_skill_conditions,
    get_all_skills_dirs,
    get_disabled_skill_names,
    iter_skill_index_files,
    parse_frontmatter,
    skill_matches_platform,
)

ROUTER_NAME = "deterministic-local-skill-router"
DEFAULT_TOP_K = 8
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._-]*", re.IGNORECASE)
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LOW_SIGNAL_TASK_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "help",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "the",
    "this",
    "to",
    "use",
    "using",
    "when",
    "with",
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        lowered = raw.lower()
        tokens.append(lowered)
        if "-" in lowered or "_" in lowered:
            tokens.extend(part for part in re.split(r"[-_]", lowered) if part)
    return tokens


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,\n]", value) if part.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, dict)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _metadata_hermes(frontmatter: dict[str, Any]) -> dict[str, Any]:
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    hermes_meta = metadata.get("hermes")
    return hermes_meta if isinstance(hermes_meta, dict) else {}


def _skill_should_show(
    conditions: dict[str, list[str]],
    available_tools: set[str] | None,
    available_toolsets: set[str] | None,
) -> bool:
    if available_tools is None and available_toolsets is None:
        return True
    tools = available_tools or set()
    toolsets = available_toolsets or set()
    for toolset in conditions.get("fallback_for_toolsets", []):
        if toolset in toolsets:
            return False
    for tool in conditions.get("fallback_for_tools", []):
        if tool in tools:
            return False
    for toolset in conditions.get("requires_toolsets", []):
        if toolset not in toolsets:
            return False
    for tool in conditions.get("requires_tools", []):
        if tool not in tools:
            return False
    return True


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _required_environment_names(frontmatter: dict[str, Any]) -> list[str]:
    required_raw = frontmatter.get("required_environment_variables")
    if isinstance(required_raw, (str, dict)):
        required_raw = [required_raw]
    if not isinstance(required_raw, list):
        required_raw = []

    names: list[str] = []
    seen: set[str] = set()

    def add_name(value: Any) -> None:
        name = str(value or "").strip()
        if not name or name in seen or not _ENV_VAR_NAME_RE.match(name):
            return
        seen.add(name)
        names.append(name)

    for item in required_raw:
        if isinstance(item, str):
            add_name(item)
        elif isinstance(item, dict):
            add_name(item.get("name") or item.get("env_var"))

    prerequisites = frontmatter.get("prerequisites")
    if isinstance(prerequisites, dict):
        legacy_envs = prerequisites.get("env_vars")
        if isinstance(legacy_envs, str):
            legacy_envs = [legacy_envs]
        if isinstance(legacy_envs, list):
            for item in legacy_envs:
                add_name(item)

    setup = frontmatter.get("setup")
    if isinstance(setup, dict):
        collect_secrets = setup.get("collect_secrets")
        if isinstance(collect_secrets, dict):
            collect_secrets = [collect_secrets]
        if isinstance(collect_secrets, list):
            for item in collect_secrets:
                if isinstance(item, dict):
                    add_name(item.get("env_var"))

    return names


def _readiness_metadata(frontmatter: dict[str, Any]) -> dict[str, Any]:
    required_names = _required_environment_names(frontmatter)
    missing_names = [name for name in required_names if not os.getenv(name)]
    setup_needed = bool(missing_names)
    return {
        "readiness_status": "setup_needed" if setup_needed else "available",
        "setup_needed": setup_needed,
        "missing_required_environment_variables": missing_names,
    }


def _skill_record(skill_file: Path, root: Path) -> dict[str, Any] | None:
    try:
        raw = skill_file.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw)
    except Exception:
        return None
    if not skill_matches_platform(frontmatter):
        return None

    rel = skill_file.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2:
        fallback_name = parts[-2]
        category = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
    else:
        fallback_name = skill_file.parent.name
        category = "general"

    name = str(frontmatter.get("name") or fallback_name)
    desc = str(frontmatter.get("description") or "")
    hermes_meta = _metadata_hermes(frontmatter)
    tags = _listify(hermes_meta.get("tags") or frontmatter.get("tags"))
    related = _listify(
        hermes_meta.get("related_skills") or frontmatter.get("related_skills")
    )
    aliases = _listify(hermes_meta.get("aliases") or frontmatter.get("aliases"))
    conditions = extract_skill_conditions(frontmatter)

    return {
        "name": name,
        "category": category,
        "description": desc,
        "tags": tags,
        "related_skills": related,
        "aliases": aliases,
        "conditions": conditions,
        "path": _safe_relative(skill_file, root),
        "root": str(root),
        "body_tokens": _tokenize(body),
        "readiness": _readiness_metadata(frontmatter),
    }


def visible_skill_records(
    *,
    available_tools: set[str] | None = None,
    available_toolsets: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return skill records visible under platform/disabled/toolset filters."""
    disabled = get_disabled_skill_names()
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for root in get_all_skills_dirs():
        if not root.exists():
            continue
        for skill_file in iter_skill_index_files(root, "SKILL.md"):
            record = _skill_record(skill_file, root)
            if not record:
                continue
            name = str(record["name"])
            fallback_name = Path(record["path"]).parent.name
            if name in disabled or fallback_name in disabled:
                continue
            if name in seen_names:
                continue
            if not _skill_should_show(record["conditions"], available_tools, available_toolsets):
                continue
            seen_names.add(name)
            records.append(record)
    return records


def _score_record(task_tokens: Counter[str], task_text: str, record: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    name = str(record["name"]).lower()
    category = str(record["category"]).lower()
    desc = str(record.get("description") or "").lower()
    tags = [str(t).lower() for t in record.get("tags") or []]
    aliases = [str(a).lower() for a in record.get("aliases") or []]
    related = [str(r).lower() for r in record.get("related_skills") or []]

    weighted_terms: dict[str, float] = {}
    for term in _tokenize(name):
        weighted_terms[term] = weighted_terms.get(term, 0.0) + 8.0
    for term in _tokenize(category):
        weighted_terms[term] = weighted_terms.get(term, 0.0) + 5.0
    for tag in tags + aliases + related:
        for term in _tokenize(tag):
            weighted_terms[term] = weighted_terms.get(term, 0.0) + 6.0
    for term in _tokenize(desc):
        weighted_terms[term] = weighted_terms.get(term, 0.0) + 2.0
    body_term_counts = Counter(record.get("body_tokens") or [])
    for term, count in body_term_counts.items():
        weighted_terms[term] = weighted_terms.get(term, 0.0) + min(count, 4) * 0.25

    score = 0.0
    matched_terms: list[str] = []
    for term, count in task_tokens.items():
        weight = weighted_terms.get(term, 0.0)
        if weight:
            score += min(count, 3) * weight
            matched_terms.append(term)

    reasons: list[str] = []
    if name and name in task_text:
        score += 40
        reasons.append("exact skill name champion gate matched task")
    if category and category in task_text:
        score += 8
        reasons.append("category appears in task")
    for alias in aliases:
        if alias and alias in task_text:
            score += 12
            reasons.append(f"alias '{alias}' appears in task")
    for tag in tags:
        if tag and tag in task_text:
            score += 10
            reasons.append(f"tag '{tag}' appears in task")

    if matched_terms:
        reasons.append("matched terms: " + ", ".join(sorted(set(matched_terms))[:12]))
    return score, sorted(set(matched_terms)), reasons


def route_skills(
    task: str,
    *,
    available_tools: set[str] | None = None,
    available_toolsets: set[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Rank visible skills for *task* and return an auditable proof envelope."""
    top_k = max(1, min(int(top_k or DEFAULT_TOP_K), 25))
    task_text = (task or "").lower()
    task_terms = _tokenize(task_text)
    meaningful_task_terms = [
        term for term in task_terms if term not in _LOW_SIGNAL_TASK_TERMS
    ]
    task_tokens = Counter(meaningful_task_terms)
    records = visible_skill_records(
        available_tools=available_tools,
        available_toolsets=available_toolsets,
    )

    candidates: list[dict[str, Any]] = []
    for record in records:
        score, matched_terms, reasons = _score_record(task_tokens, task_text, record)
        if score <= 0:
            continue
        readiness = record.get("readiness") or {}
        candidates.append(
            {
                "name": record["name"],
                "category": record["category"],
                "score": round(score, 3),
                "matched_terms": matched_terms,
                "why": reasons or ["deterministic lexical match"],
                "path": record["path"],
                "root": record["root"],
                "conditions": record["conditions"],
                "readiness_status": readiness.get("readiness_status", "available"),
                "setup_needed": bool(readiness.get("setup_needed")),
                "missing_required_environment_variables": readiness.get(
                    "missing_required_environment_variables", []
                ),
            }
        )

    candidates.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    selected = candidates[:top_k]
    skipped = candidates[top_k : top_k + 10]
    if candidates:
        warning = None
    elif task_terms and not meaningful_task_terms:
        warning = "SKILL_ROUTER_ABSTAINED_LOW_CONFIDENCE_USE_MANUAL_SELECTION"
    else:
        warning = "SKILL_ROUTER_NO_MATCHES_USE_MANUAL_SKILLS_LIST_AND_EXPLAIN_SELECTION"
    return {
        "success": True,
        "router": ROUTER_NAME,
        "router_available": True,
        "task": task,
        "visible_skill_count": len(records),
        "candidate_count": len(candidates),
        "selected_skills": selected,
        "candidate_skills": candidates[: max(top_k, 12)],
        "skipped_better_looking_skills": skipped,
        "selection_source": "deterministic local lexical router; model must still load selected skills with skill_view before relying on them",
        "warning": warning,
    }
