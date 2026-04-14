#!/usr/bin/env python3
"""Bounded read-only Virtual Artifact Filesystem (VAFS) resolver.

VAFS-R0 intentionally exposes only a small, explicit OpenClaw artifact surface.
It resolves logical paths such as:
- vafs://continuity/current
- vafs://continuity/latest/*
- vafs://handover/latest
- vafs://truth/ground-truth/latest
- vafs://truth/gtc/gateboard
- vafs://reports/latest/<topic>
- vafs://memory/memory-md
- vafs://memory/obsidian-registry/latest
- vafs://docs/ops/*

Every successful resolution emits a machine-readable metadata envelope with:
- requested/logical path
- repo-relative source path
- absolute resolved path
- existence + sha256
- canonical/support classification
- freshness/truth/lifecycle projection
- alias metadata where applicable

Out-of-scope namespaces fail closed with deterministic denial reasons.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
SCHEMA = "clawd.vafs.resolver.v0"
SEARCH_SCHEMA = "clawd.vafs.search.v0"
LIST_SCHEMA = "clawd.vafs.list.v0"
DENIAL_EXIT_CODE = 2
MAX_SUGGESTIONS = 5

ALLOWED_NAMESPACES = ("continuity", "handover", "truth", "reports", "memory", "docs")
PATH_SEP_RE = re.compile(r"/+")
REPORT_DATE_RE = re.compile(r"^(?P<topic>.+)_(?P<date>\d{4}-\d{2}-\d{2})$")

TRUTH_LATEST_ALIAS_TARGETS = {
    "truth/latest/current": "continuity/current",
    "truth/latest/handover": "handover/latest.json",
    "truth/latest/ground-truth": "truth/ground-truth/latest",
    "truth/latest/continuity-read-pointer": "truth/continuity-read-pointer",
    "truth/latest/latest-pointer": "truth/latest-pointer",
    "truth/latest/continuity-now": "continuity/latest/continuity_now_latest.json",
    "truth/latest/operator-mission-control": "continuity/latest/operator_mission_control.json",
    "truth/latest/operator-triage-console": "continuity/latest/operator_triage_console.json",
    "truth/latest/autonomous-execution-intent": "continuity/latest/autonomous_execution_intent_latest.json",
    "truth/latest/execution-frontier": "continuity/latest/no_nudge_execution_frontier_controller_tick_latest.json",
    "truth/latest/idle-lane-autospawn": "continuity/latest/no_nudge_idle_lane_autospawn_latest.json",
}

CONTINUITY_LATEST_ALIAS_TARGETS = {
    "continuity/latest/current": "continuity/current",
    "continuity/latest/current.json": "continuity/current",
}


class VAFSDenial(RuntimeError):
    def __init__(self, reason: str, *, detail: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class VAFSError(RuntimeError):
    pass


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_z(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_iso8601(value: Any) -> Optional[dt.datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def normalize_logical_path(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("vafs://"):
        text = text[len("vafs://") :]
    text = text.replace("\\", "/")
    text = PATH_SEP_RE.sub("/", text)
    text = text.strip("/")
    if not text:
        return ""
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise VAFSDenial("invalid_path_segment", detail=text)
    return "/".join(parts)


def normalize_search_text(raw: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", raw.casefold()).split())


def search_tokens(raw: str) -> Tuple[str, Tuple[str, ...]]:
    normalized = normalize_search_text(raw)
    return normalized, tuple(token for token in normalized.split(" ") if token)


def load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def nested_get(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def classification_for_logical(logical_path: str, *, alias: bool) -> Tuple[str, str, str]:
    if logical_path.startswith("continuity/"):
        return "canonical", "authoritative", "high"
    if logical_path.startswith("handover/"):
        return "canonical", "authoritative", "high"
    if logical_path.startswith("truth/"):
        return "canonical", "authoritative", "high"
    if logical_path.startswith("docs/"):
        return "canonical", "authoritative", "high"
    if logical_path == "memory/obsidian-registry/latest":
        return "support", "derived", "medium"
    if logical_path.startswith("memory/"):
        return "support", "authoritative", "medium"
    if logical_path.startswith("reports/"):
        return "support", "derived" if alias else "authoritative", "medium"
    return "unknown", "unknown", "low"


def freshness_ttl_seconds(logical_path: str) -> int:
    if logical_path.startswith(("continuity/", "handover/", "truth/")):
        return 900
    if logical_path == "memory/obsidian-registry/latest":
        return 7 * 24 * 3600
    if logical_path.startswith("memory/"):
        return 30 * 24 * 3600
    if logical_path.startswith(("reports/", "docs/")):
        return 30 * 24 * 3600
    return 24 * 3600


def extract_source_schema(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    value = payload.get("schema_version") or payload.get("schema")
    return value if isinstance(value, str) and value.strip() else None


def extract_freshness(payload: Optional[Dict[str, Any]], *, logical_path: str, observed_at: Optional[dt.datetime]) -> Dict[str, Any]:
    candidates: List[Tuple[str, Optional[dt.datetime]]] = []
    valid_until: Optional[dt.datetime] = None

    if isinstance(payload, dict):
        valid_until = (
            parse_iso8601(payload.get("valid_until"))
            or parse_iso8601(nested_get(payload, "coherence", "valid_until"))
            or parse_iso8601(payload.get("coherence_valid_until"))
        )
        candidates.extend(
            [
                ("generated_at", parse_iso8601(payload.get("generated_at"))),
                ("updated_at", parse_iso8601(payload.get("updated_at"))),
                ("snapshot_ts_utc", parse_iso8601(payload.get("snapshot_ts_utc"))),
                ("generated_at_utc", parse_iso8601(payload.get("generated_at_utc"))),
                ("recorded_at", parse_iso8601(payload.get("recorded_at"))),
                ("published_at", parse_iso8601(nested_get(payload, "coherence", "published_at"))),
                ("coherence_generated_at", parse_iso8601(nested_get(payload, "coherence", "generated_at"))),
            ]
        )

    if observed_at is not None:
        candidates.append(("mtime", observed_at))

    as_of_source = "none"
    as_of_value: Optional[dt.datetime] = None
    for source_name, candidate in candidates:
        if candidate is not None:
            as_of_source = source_name
            as_of_value = candidate
            break

    now = now_utc()
    age_sec: Optional[int] = None
    if as_of_value is not None:
        age_sec = max(0, int((now - as_of_value).total_seconds()))

    if valid_until is not None:
        freshness_state = "fresh" if now <= valid_until else "expired"
        freshness_source = "valid_until"
    elif as_of_value is not None:
        freshness_state = "fresh" if age_sec is not None and age_sec <= freshness_ttl_seconds(logical_path) else "stale"
        freshness_source = as_of_source
    else:
        freshness_state = "unknown"
        freshness_source = "none"

    return {
        "state": freshness_state,
        "as_of": iso_z(as_of_value) if as_of_value is not None else None,
        "valid_until": iso_z(valid_until) if valid_until is not None else None,
        "age_sec": age_sec,
        "source": freshness_source,
    }


def extract_truth_refs(
    payload: Optional[Dict[str, Any]],
    *,
    content_sha256: Optional[str],
    latest_alias_of: Optional[str],
) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    if content_sha256:
        refs["sha256"] = content_sha256

    if isinstance(payload, dict):
        candidates = {
            "checkpoint_id": payload.get("checkpoint_id") or nested_get(payload, "metadata", "checkpoint_id"),
            "snapshot_id": payload.get("snapshot_id"),
            "journal_offset": payload.get("journal_offset"),
            "pointer_hash": payload.get("pointer_hash"),
            "coherence_tuple_hash": payload.get("coherence_tuple_hash") or nested_get(payload, "coherence", "tuple_hash"),
            "coherence_build_generation_id": payload.get("coherence_build_generation_id")
            or nested_get(payload, "coherence", "build_generation_id")
            or nested_get(payload, "base_coherence_guard", "coherence_build_generation_id"),
            "build_generation_id": payload.get("build_generation_id"),
            "policy_signature": payload.get("policy_signature")
            or nested_get(payload, "coherence", "policy", "signature")
            or nested_get(payload, "base_coherence_guard", "policy_signature"),
            "source_schema_version": extract_source_schema(payload),
            "manifest_auth_key_id": nested_get(payload, "manifest_auth", "key_id"),
        }
        for key, value in candidates.items():
            if value not in (None, "", []):
                refs[key] = value

    if latest_alias_of:
        refs["latest_alias_of"] = latest_alias_of

    return refs


def lifecycle_state_for(logical_path: str, *, latest_alias_of: Optional[str]) -> Dict[str, Any]:
    if "/history/" in logical_path:
        state = "historical"
    else:
        state = "active"
    return {
        "state": state,
        "superseded_by": None,
        "latest_alias_of": latest_alias_of,
    }


def continuity_pointer_projection(root: Path, *, content_sha256: Optional[str]) -> Optional[Dict[str, Any]]:
    pointer_path = root / "state" / "continuity" / "latest" / "continuity_read_pointer.json"
    payload = load_json_object(pointer_path)
    if not payload:
        return None
    expected_sha = nested_get(payload, "continuity_read_contract", "continuity_current_sha256") or nested_get(
        payload, "source_current", "sha256"
    )
    return {
        "pointer_path": rel_to_root(pointer_path, root),
        "pointer_schema": payload.get("schema") or payload.get("schema_version"),
        "pointer_current_sha256": expected_sha,
        "pointer_sha_match": bool(expected_sha and content_sha256 and expected_sha == content_sha256),
    }


def report_alias_key(raw: str) -> str:
    text = raw.strip().removesuffix(".md")
    return text.casefold()


def dated_report_entries(root: Path) -> List[Dict[str, Any]]:
    reports_dir = root / "reports"
    out: List[Dict[str, Any]] = []
    if not reports_dir.exists():
        return out

    for path in sorted(reports_dir.glob("*.md")):
        match = REPORT_DATE_RE.match(path.stem)
        if not match:
            continue
        topic = match.group("topic")
        date_text = match.group("date")
        stat = path.stat()
        out.append(
            {
                "topic": topic,
                "date": date_text,
                "path": path,
                "filename": path.name,
                "stem": path.stem,
                "sort_key": (date_text, stat.st_mtime, path.name),
            }
        )

    return out


def suggestion_logical_path(logical_path: str, *, reason: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kind": "logical_path",
        "logical_path": f"vafs://{logical_path}",
        "reason": reason,
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def suggestion_command(command: str, *, reason: str, logical_path: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "kind": "command",
        "command": command,
        "reason": reason,
    }
    if logical_path:
        payload["logical_path"] = f"vafs://{logical_path}"
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def dedupe_suggestions(suggestions: Iterable[Dict[str, Any]], *, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for suggestion in suggestions:
        key = stable_json(suggestion)
        if key in seen:
            continue
        seen.add(key)
        out.append(suggestion)
        if len(out) >= limit:
            break
    return out


def rank_text_candidates(query: str, candidates: Iterable[str], *, limit: int = MAX_SUGGESTIONS) -> List[str]:
    normalized_query, query_tokens = search_tokens(query)
    query_set = set(query_tokens)
    ranked: List[Tuple[Tuple[int, int, int, str], str]] = []

    for candidate in candidates:
        normalized_candidate, candidate_tokens = search_tokens(candidate)
        candidate_set = set(candidate_tokens)
        overlap = len(query_set & candidate_set) if query_set else 0
        all_token_prefix = bool(query_tokens) and all(
            any(token.startswith(query_token) for token in candidate_tokens)
            for query_token in query_tokens
        )

        if candidate.casefold() == query.casefold() or normalized_candidate == normalized_query:
            rank = (0, 0, 0, candidate)
        elif candidate.casefold().startswith(query.casefold()) or normalized_candidate.startswith(normalized_query):
            rank = (1, 0, 0, candidate)
        elif query.casefold() in candidate.casefold() or (normalized_query and normalized_query in normalized_candidate):
            rank = (2, 0, 0, candidate)
        elif all_token_prefix:
            rank = (3, 0, -overlap, candidate)
        elif overlap and (len(query_set) <= 1 or overlap == len(query_set)):
            rank = (4, len(candidate_set - query_set), -overlap, candidate)
        else:
            ratio = difflib.SequenceMatcher(a=normalized_query, b=normalized_candidate).ratio() if normalized_query else 0.0
            if ratio < 0.45:
                continue
            rank = (5, int((1.0 - ratio) * 1000), 0, candidate)

        ranked.append((rank, candidate))

    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[: max(1, limit)]]


def listable_prefix_children(logical_prefix: str) -> List[str]:
    if logical_prefix == "continuity":
        return ["continuity/current", "continuity/latest", "continuity/contracts"]
    if logical_prefix == "continuity/contracts":
        return ["continuity/contracts/continuity_now"]
    if logical_prefix == "handover":
        return ["handover/latest", "handover/latest.json"]
    if logical_prefix == "truth":
        return [
            "truth/ground-truth/latest",
            "truth/gtc/gateboard",
            "truth/gtc/publish-manifest",
            "truth/continuity-read-pointer",
            "truth/latest-pointer",
            "truth/latest",
        ]
    if logical_prefix == "truth/latest":
        return sorted(TRUTH_LATEST_ALIAS_TARGETS)
    if logical_prefix == "reports":
        return ["reports/latest", "reports/by-date", "reports/by-topic"]
    if logical_prefix == "memory":
        return ["memory/memory-md", "memory/obsidian-registry/latest", "memory/daily"]
    if logical_prefix == "docs":
        return ["docs/ops", "docs/schemas", "docs/templates"]
    return []


def latest_report_aliases(root: Path) -> Dict[str, Dict[str, Any]]:
    aliases: Dict[str, Dict[str, Any]] = {}
    for entry in dated_report_entries(root):
        key = report_alias_key(entry["topic"])
        current = aliases.get(key)
        if current is None or entry["sort_key"] > current["sort_key"]:
            aliases[key] = entry
    return aliases


def reports_by_date(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in dated_report_entries(root):
        grouped.setdefault(entry["date"], []).append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda row: row["sort_key"], reverse=True)
    return grouped


def reports_by_topic(root: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in dated_report_entries(root):
        grouped.setdefault(report_alias_key(entry["topic"]), []).append(entry)
    for entries in grouped.values():
        entries.sort(key=lambda row: row["sort_key"], reverse=True)
    return grouped


def resolve_report_bucket_entry(entries: Sequence[Dict[str, Any]], leaf: str) -> Path:
    normalized_leaf = leaf.strip()
    if normalized_leaf.endswith(".md"):
        filename = normalized_leaf
        stem = normalized_leaf.removesuffix(".md")
    else:
        filename = f"{normalized_leaf}.md"
        stem = normalized_leaf

    for entry in entries:
        if entry["filename"] == filename or entry["stem"] == stem:
            return entry["path"]

    raise VAFSDenial("unknown_reports_bucket_entry", detail=leaf)


def report_alias_suggestions(root: Path, raw_query: str, *, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    aliases = latest_report_aliases(root)
    ranked_topics = rank_text_candidates(raw_query, [entry["topic"] for entry in aliases.values()], limit=limit)
    suggestions: List[Dict[str, Any]] = []
    for topic in ranked_topics:
        entry = aliases.get(report_alias_key(topic))
        if entry is None:
            continue
        suggestions.append(
            suggestion_logical_path(
                f"reports/latest/{entry['topic']}",
                reason="nearest_reports_latest_alias",
                latest_alias_of=f"reports/{entry['filename']}",
            )
        )
    return dedupe_suggestions(suggestions, limit=limit)


def bounded_prefix_suggestions(root: Path, logical_path: str, *, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    namespace = logical_path.split("/", 1)[0] if logical_path else ""
    suggestions: List[Dict[str, Any]] = []

    if namespace == "reports" and logical_path.startswith("reports/latest/"):
        suffix = logical_path[len("reports/latest/") :]
        suggestions.extend(report_alias_suggestions(root, suffix, limit=limit))
        suggestions.append(suggestion_command("list", reason="list_supported_prefix", logical_path="reports/latest"))

    for child in listable_prefix_children(namespace):
        suggestions.append(suggestion_logical_path(child, reason="supported_prefix"))

    if "/" in logical_path:
        parent = logical_path.rsplit("/", 1)[0]
        suggestions.append(suggestion_command("list", reason="list_parent_prefix", logical_path=parent))
        suggestions.append(suggestion_logical_path(parent, reason="parent_prefix"))

    return dedupe_suggestions(suggestions, limit=limit)


def missing_path_suggestions(root: Path, logical_path: str, *, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    if logical_path.startswith("reports/latest/"):
        return report_alias_suggestions(root, logical_path[len("reports/latest/") :], limit=limit)
    return bounded_prefix_suggestions(root, logical_path, limit=limit)


def denial_suggestions(
    root: Path,
    command: str,
    logical_path: str,
    denial: VAFSDenial,
    *,
    limit: int = MAX_SUGGESTIONS,
) -> List[Dict[str, Any]]:
    suggestions: List[Dict[str, Any]] = []

    if denial.reason == "root_requires_list":
        suggestions.append(suggestion_command("list", reason="root_is_listable", logical_path=""))
        for namespace in ALLOWED_NAMESPACES:
            suggestions.append(suggestion_logical_path(namespace, reason="supported_namespace"))
        return dedupe_suggestions(suggestions, limit=limit)

    if denial.reason == "directory_requires_list":
        prefix = logical_path or (denial.detail or "")
        prefix = prefix.rstrip("/")
        if prefix:
            suggestions.append(suggestion_command("list", reason="directory_requires_list", logical_path=prefix))
            suggestions.append(suggestion_logical_path(prefix, reason="directory_prefix"))
        return dedupe_suggestions(suggestions, limit=limit)

    if denial.reason == "unsupported_namespace":
        ranked_namespaces = rank_text_candidates(denial.detail or logical_path or "", ALLOWED_NAMESPACES)
        for namespace in ranked_namespaces or list(ALLOWED_NAMESPACES):
            suggestions.append(suggestion_logical_path(namespace, reason="supported_namespace"))
        return dedupe_suggestions(suggestions, limit=limit)

    if denial.reason == "unknown_reports_latest_alias":
        suggestions.extend(report_alias_suggestions(root, denial.detail or logical_path, limit=limit))
        suggestions.append(suggestion_command("list", reason="list_supported_prefix", logical_path="reports/latest"))
        return dedupe_suggestions(suggestions, limit=limit)

    if denial.reason == "unknown_reports_bucket_entry":
        parent = logical_path.rsplit("/", 1)[0] if "/" in logical_path else "reports"
        suggestions.append(suggestion_command("list", reason="list_parent_prefix", logical_path=parent))
        suggestions.append(suggestion_logical_path(parent, reason="parent_prefix"))
        return dedupe_suggestions(suggestions, limit=limit)

    if denial.reason in {"unsupported_logical_path", "unsupported_list_prefix", "missing_directory"}:
        suggestions.extend(bounded_prefix_suggestions(root, denial.detail or logical_path, limit=limit))
        return dedupe_suggestions(suggestions)

    return []


def resolve_reports_latest(root: Path, alias_topic: str) -> Tuple[Path, str]:
    aliases = latest_report_aliases(root)
    key = report_alias_key(alias_topic)
    entry = aliases.get(key)
    if entry is None:
        raise VAFSDenial("unknown_reports_latest_alias", detail=alias_topic)
    latest_alias_of = f"reports/{entry['filename']}"
    return entry["path"], latest_alias_of


def resolve_logical_path(root: Path, logical_path: str) -> Tuple[Path, Optional[str]]:
    if not logical_path:
        raise VAFSDenial("root_requires_list")

    alias_target = CONTINUITY_LATEST_ALIAS_TARGETS.get(logical_path) or TRUTH_LATEST_ALIAS_TARGETS.get(logical_path)
    if alias_target:
        source_path, _ = resolve_logical_path(root, alias_target)
        return source_path, alias_target

    if logical_path == "continuity/current":
        return root / "state" / "continuity" / "current.json", None
    if logical_path.startswith("continuity/latest/"):
        suffix = logical_path[len("continuity/latest/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        return root / "state" / "continuity" / "latest" / suffix, None
    if logical_path.startswith("continuity/contracts/continuity_now/"):
        suffix = logical_path[len("continuity/contracts/continuity_now/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        return root / "state" / "continuity" / "contracts" / "continuity_now" / suffix, None
    if logical_path == "handover/latest" or logical_path == "handover/latest.md":
        return root / "state" / "handover" / "latest.md", None
    if logical_path == "handover/latest.json":
        return root / "state" / "handover" / "latest.json", None
    if logical_path == "truth/ground-truth/latest":
        return root / "state" / "ground_truth" / "latest.json", None
    if logical_path == "truth/gtc/gateboard":
        return root / "state" / "gtc-v2" / "latest" / "gateboard.json", None
    if logical_path == "truth/gtc/publish-manifest":
        return root / "state" / "gtc-v2" / "latest" / "publish_manifest.json", None
    if logical_path == "truth/continuity-read-pointer":
        return root / "state" / "continuity" / "latest" / "continuity_read_pointer.json", None
    if logical_path == "truth/latest-pointer":
        return root / "state" / "continuity" / "latest" / "latest_pointer.json", None
    if logical_path == "memory/memory-md":
        return root / "MEMORY.md", None
    if logical_path == "memory/obsidian-registry/latest":
        return root / "state" / "continuity" / "latest" / "xk_obsidian_memory_registry_latest.json", None
    if logical_path.startswith("memory/daily/"):
        suffix = logical_path[len("memory/daily/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        if not suffix.endswith(".md"):
            suffix = f"{suffix}.md"
        return root / "memory" / suffix, None
    if logical_path.startswith("docs/ops/"):
        suffix = logical_path[len("docs/ops/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        return root / "docs" / "ops" / suffix, None
    if logical_path.startswith("docs/schemas/"):
        suffix = logical_path[len("docs/schemas/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        return root / "docs" / "ops" / "schemas" / suffix, None
    if logical_path.startswith("docs/templates/"):
        suffix = logical_path[len("docs/templates/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        return root / "docs" / "ops" / "templates" / suffix, None
    if logical_path.startswith("reports/latest/"):
        suffix = logical_path[len("reports/latest/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        return resolve_reports_latest(root, suffix)
    if logical_path.startswith("reports/by-date/"):
        suffix = logical_path[len("reports/by-date/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        date_text, sep, leaf = suffix.partition("/")
        if not sep or not leaf:
            raise VAFSDenial("directory_requires_list", detail=logical_path)
        entries = reports_by_date(root).get(date_text)
        if not entries:
            raise VAFSDenial("unsupported_logical_path", detail=logical_path)
        return resolve_report_bucket_entry(entries, leaf), None
    if logical_path.startswith("reports/by-topic/"):
        suffix = logical_path[len("reports/by-topic/") :]
        if not suffix:
            raise VAFSDenial("directory_requires_list")
        topic_text, sep, leaf = suffix.partition("/")
        if not sep or not leaf:
            raise VAFSDenial("directory_requires_list", detail=logical_path)
        entries = reports_by_topic(root).get(report_alias_key(topic_text))
        if not entries:
            raise VAFSDenial("unknown_reports_latest_alias", detail=topic_text)
        return resolve_report_bucket_entry(entries, leaf), None

    namespace = logical_path.split("/", 1)[0]
    if namespace not in ALLOWED_NAMESPACES:
        raise VAFSDenial("unsupported_namespace", detail=namespace)
    raise VAFSDenial("unsupported_logical_path", detail=logical_path)


def resolve_node(root: Path, requested_path: str, logical_path: str) -> Dict[str, Any]:
    source_path, latest_alias_of = resolve_logical_path(root, logical_path)
    if source_path.exists() and source_path.is_dir():
        raise VAFSDenial("directory_requires_list", detail=logical_path)

    payload = load_json_object(source_path)
    exists = source_path.exists() and source_path.is_file()
    observed_at = (
        dt.datetime.fromtimestamp(source_path.stat().st_mtime, tz=dt.timezone.utc)
        if source_path.exists()
        else None
    )
    content_sha256 = sha256_file(source_path) if exists else None
    authority_class, truth_class, truth_confidence = classification_for_logical(logical_path, alias=latest_alias_of is not None)
    node: Dict[str, Any] = {
        "requested_path": requested_path,
        "logical_path": f"vafs://{logical_path}",
        "source_path": rel_to_root(source_path, root),
        "real_path": str(source_path.resolve()),
        "exists": exists,
        "resolution_state": "resolved" if exists else "missing",
        "kind": "file",
        "authority_class": authority_class,
        "content_sha256": content_sha256,
        "source_schema_version": extract_source_schema(payload),
        "latest_alias_of": latest_alias_of,
        "freshness": extract_freshness(payload, logical_path=logical_path, observed_at=observed_at),
        "truth": {
            "class": truth_class,
            "confidence": truth_confidence,
            "refs": extract_truth_refs(payload, content_sha256=content_sha256, latest_alias_of=latest_alias_of),
        },
        "lifecycle": lifecycle_state_for(logical_path, latest_alias_of=latest_alias_of),
    }

    if logical_path in {"continuity/current", "continuity/latest/current", "continuity/latest/current.json"}:
        node["pointer_parity"] = continuity_pointer_projection(root, content_sha256=content_sha256)

    if not exists:
        suggestions = missing_path_suggestions(root, logical_path)
        if suggestions:
            node["suggestions"] = suggestions

    return node


def namespace_entry(namespace: str) -> Dict[str, Any]:
    return {
        "logical_path": f"vafs://{namespace}",
        "kind": "namespace",
        "authority_class": "canonical" if namespace in {"continuity", "handover", "truth", "docs"} else "support",
    }


def directory_entry(logical_path: str, authority_class: str) -> Dict[str, Any]:
    return {
        "logical_path": f"vafs://{logical_path}",
        "kind": "directory",
        "authority_class": authority_class,
    }


def list_directory(root: Path, *, logical_prefix: str, directory: Path, authority_class: str) -> List[Dict[str, Any]]:
    if not directory.exists():
        raise VAFSDenial("missing_directory", detail=logical_prefix)
    if not directory.is_dir():
        return [resolve_node(root, logical_prefix, logical_prefix)]

    out: List[Dict[str, Any]] = []
    for child in sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
        child_logical = f"{logical_prefix}/{child.name}"
        if child.is_dir():
            out.append(directory_entry(child_logical, authority_class))
        else:
            try:
                out.append(resolve_node(root, child_logical, child_logical))
            except VAFSDenial:
                out.append(directory_entry(child_logical, authority_class))
    return out


def list_nodes(root: Path, requested_prefix: str, logical_prefix: str) -> List[Dict[str, Any]]:
    if logical_prefix == "":
        return [namespace_entry(namespace) for namespace in ALLOWED_NAMESPACES]
    if logical_prefix == "continuity":
        return [
            resolve_node(root, requested_prefix, "continuity/current"),
            directory_entry("continuity/latest", "canonical"),
            directory_entry("continuity/contracts", "canonical"),
        ]
    if logical_prefix == "continuity/latest":
        return list_directory(
            root,
            logical_prefix="continuity/latest",
            directory=root / "state" / "continuity" / "latest",
            authority_class="canonical",
        )
    if logical_prefix.startswith("continuity/latest/"):
        suffix = logical_prefix[len("continuity/latest/") :]
        return list_directory(
            root,
            logical_prefix=logical_prefix,
            directory=root / "state" / "continuity" / "latest" / suffix,
            authority_class="canonical",
        )
    if logical_prefix == "continuity/contracts":
        return [directory_entry("continuity/contracts/continuity_now", "canonical")]
    if logical_prefix == "continuity/contracts/continuity_now":
        return list_directory(
            root,
            logical_prefix="continuity/contracts/continuity_now",
            directory=root / "state" / "continuity" / "contracts" / "continuity_now",
            authority_class="canonical",
        )
    if logical_prefix.startswith("continuity/contracts/continuity_now/"):
        suffix = logical_prefix[len("continuity/contracts/continuity_now/") :]
        return list_directory(
            root,
            logical_prefix=logical_prefix,
            directory=root / "state" / "continuity" / "contracts" / "continuity_now" / suffix,
            authority_class="canonical",
        )
    if logical_prefix == "handover":
        return [resolve_node(root, requested_prefix, "handover/latest"), resolve_node(root, requested_prefix, "handover/latest.json")]
    if logical_prefix == "truth":
        return [
            resolve_node(root, requested_prefix, "truth/ground-truth/latest"),
            resolve_node(root, requested_prefix, "truth/gtc/gateboard"),
            resolve_node(root, requested_prefix, "truth/gtc/publish-manifest"),
            resolve_node(root, requested_prefix, "truth/continuity-read-pointer"),
            resolve_node(root, requested_prefix, "truth/latest-pointer"),
            directory_entry("truth/latest", "canonical"),
        ]
    if logical_prefix == "truth/latest":
        return [resolve_node(root, requested_prefix, alias_path) for alias_path in sorted(TRUTH_LATEST_ALIAS_TARGETS)]
    if logical_prefix == "memory":
        return [
            resolve_node(root, requested_prefix, "memory/memory-md"),
            resolve_node(root, requested_prefix, "memory/obsidian-registry/latest"),
            directory_entry("memory/daily", "support"),
        ]
    if logical_prefix == "memory/daily":
        daily_dir = root / "memory"
        out: List[Dict[str, Any]] = []
        for child in sorted(daily_dir.glob("*.md")):
            logical = f"memory/daily/{child.name}"
            out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix == "docs":
        return [
            directory_entry("docs/ops", "canonical"),
            directory_entry("docs/schemas", "canonical"),
            directory_entry("docs/templates", "canonical"),
        ]
    if logical_prefix == "docs/ops":
        return list_directory(root, logical_prefix="docs/ops", directory=root / "docs" / "ops", authority_class="canonical")
    if logical_prefix.startswith("docs/ops/"):
        suffix = logical_prefix[len("docs/ops/") :]
        return list_directory(
            root,
            logical_prefix=logical_prefix,
            directory=root / "docs" / "ops" / suffix,
            authority_class="canonical",
        )
    if logical_prefix == "docs/schemas":
        return list_directory(
            root,
            logical_prefix="docs/schemas",
            directory=root / "docs" / "ops" / "schemas",
            authority_class="canonical",
        )
    if logical_prefix.startswith("docs/schemas/"):
        suffix = logical_prefix[len("docs/schemas/") :]
        return list_directory(
            root,
            logical_prefix=logical_prefix,
            directory=root / "docs" / "ops" / "schemas" / suffix,
            authority_class="canonical",
        )
    if logical_prefix == "docs/templates":
        return list_directory(
            root,
            logical_prefix="docs/templates",
            directory=root / "docs" / "ops" / "templates",
            authority_class="canonical",
        )
    if logical_prefix.startswith("docs/templates/"):
        suffix = logical_prefix[len("docs/templates/") :]
        return list_directory(
            root,
            logical_prefix=logical_prefix,
            directory=root / "docs" / "ops" / "templates" / suffix,
            authority_class="canonical",
        )
    if logical_prefix == "reports":
        return [
            directory_entry("reports/latest", "support"),
            directory_entry("reports/by-date", "support"),
            directory_entry("reports/by-topic", "support"),
        ]
    if logical_prefix == "reports/latest":
        out: List[Dict[str, Any]] = []
        for entry in sorted(latest_report_aliases(root).values(), key=lambda row: row["topic"]):
            logical = f"reports/latest/{entry['topic']}"
            out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix == "reports/by-date":
        return [directory_entry(f"reports/by-date/{date_text}", "support") for date_text in sorted(reports_by_date(root), reverse=True)]
    if logical_prefix.startswith("reports/by-date/"):
        suffix = logical_prefix[len("reports/by-date/") :]
        if "/" in suffix:
            raise VAFSDenial("directory_requires_list", detail=logical_prefix)
        entries = reports_by_date(root).get(suffix)
        if not entries:
            raise VAFSDenial("unsupported_list_prefix", detail=logical_prefix)
        out: List[Dict[str, Any]] = []
        for entry in entries:
            logical = f"reports/by-date/{suffix}/{entry['filename']}"
            out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix == "reports/by-topic":
        return [directory_entry(f"reports/by-topic/{entry['topic']}", "support") for entry in sorted(latest_report_aliases(root).values(), key=lambda row: row["topic"])]
    if logical_prefix.startswith("reports/by-topic/"):
        suffix = logical_prefix[len("reports/by-topic/") :]
        if "/" in suffix:
            raise VAFSDenial("directory_requires_list", detail=logical_prefix)
        entries = reports_by_topic(root).get(report_alias_key(suffix))
        if not entries:
            raise VAFSDenial("unsupported_list_prefix", detail=logical_prefix)
        out: List[Dict[str, Any]] = []
        for entry in entries:
            logical = f"reports/by-topic/{entry['topic']}/{entry['filename']}"
            out.append(resolve_node(root, logical, logical))
        return out

    namespace = logical_prefix.split("/", 1)[0]
    if namespace not in ALLOWED_NAMESPACES:
        raise VAFSDenial("unsupported_namespace", detail=namespace)
    raise VAFSDenial("unsupported_list_prefix", detail=logical_prefix)


def collect_recursive_nodes(root: Path, logical_prefix: str) -> List[Dict[str, Any]]:
    if logical_prefix in {"", "continuity", "handover", "truth", "reports", "memory", "docs"}:
        prefixes = [
            "continuity/current",
            "continuity/latest",
            "continuity/contracts/continuity_now",
            "handover",
            "truth",
            "truth/latest",
            "reports/latest",
            "reports/by-date",
            "reports/by-topic",
            "memory",
            "memory/daily",
            "docs/ops",
            "docs/schemas",
            "docs/templates",
        ]
        if logical_prefix:
            prefixes = [logical_prefix]
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for prefix in prefixes:
            for node in collect_recursive_nodes(root, prefix):
                key = node["logical_path"]
                if key not in seen:
                    seen.add(key)
                    out.append(node)
        return out

    if logical_prefix == "continuity/current":
        return [resolve_node(root, logical_prefix, logical_prefix)]
    if logical_prefix.startswith("continuity/latest"):
        base_dir = root / "state" / "continuity" / "latest"
        suffix = logical_prefix[len("continuity/latest") :].strip("/")
        if suffix:
            base_dir = base_dir / suffix
        if base_dir.is_file():
            return [resolve_node(root, logical_prefix, logical_prefix)]
        out: List[Dict[str, Any]] = []
        if base_dir.exists():
            for child in sorted(base_dir.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(root / "state" / "continuity" / "latest").as_posix()
                    logical = f"continuity/latest/{rel}"
                    out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix.startswith("continuity/contracts/continuity_now"):
        base_dir = root / "state" / "continuity" / "contracts" / "continuity_now"
        suffix = logical_prefix[len("continuity/contracts/continuity_now") :].strip("/")
        if suffix:
            base_dir = base_dir / suffix
        if base_dir.is_file():
            return [resolve_node(root, logical_prefix, logical_prefix)]
        out: List[Dict[str, Any]] = []
        if base_dir.exists():
            for child in sorted(base_dir.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(root / "state" / "continuity" / "contracts" / "continuity_now").as_posix()
                    logical = f"continuity/contracts/continuity_now/{rel}"
                    out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix == "handover":
        return list_nodes(root, logical_prefix, logical_prefix)
    if logical_prefix == "truth":
        return list_nodes(root, logical_prefix, logical_prefix)
    if logical_prefix == "truth/latest":
        return list_nodes(root, logical_prefix, logical_prefix)
    if logical_prefix == "reports/latest":
        return list_nodes(root, logical_prefix, logical_prefix)
    if logical_prefix == "reports/by-date":
        out: List[Dict[str, Any]] = []
        for date_text, entries in sorted(reports_by_date(root).items(), reverse=True):
            for entry in entries:
                logical = f"reports/by-date/{date_text}/{entry['filename']}"
                out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix.startswith("reports/by-date/"):
        suffix = logical_prefix[len("reports/by-date/") :]
        entries = reports_by_date(root).get(suffix)
        if not entries:
            return []
        out: List[Dict[str, Any]] = []
        for entry in entries:
            logical = f"reports/by-date/{suffix}/{entry['filename']}"
            out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix == "reports/by-topic":
        out: List[Dict[str, Any]] = []
        for topic_key, entries in sorted(reports_by_topic(root).items()):
            for entry in entries:
                logical = f"reports/by-topic/{entry['topic']}/{entry['filename']}"
                out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix.startswith("reports/by-topic/"):
        suffix = logical_prefix[len("reports/by-topic/") :]
        entries = reports_by_topic(root).get(report_alias_key(suffix))
        if not entries:
            return []
        out: List[Dict[str, Any]] = []
        for entry in entries:
            logical = f"reports/by-topic/{entry['topic']}/{entry['filename']}"
            out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix == "memory":
        return [
            resolve_node(root, "memory/memory-md", "memory/memory-md"),
            resolve_node(root, "memory/obsidian-registry/latest", "memory/obsidian-registry/latest"),
            *collect_recursive_nodes(root, "memory/daily"),
        ]
    if logical_prefix == "memory/daily":
        return list_nodes(root, logical_prefix, logical_prefix)
    if logical_prefix.startswith("docs/ops"):
        base_dir = root / "docs" / "ops"
        suffix = logical_prefix[len("docs/ops") :].strip("/")
        if suffix:
            base_dir = base_dir / suffix
        if base_dir.is_file():
            return [resolve_node(root, logical_prefix, logical_prefix)]
        out = []
        if base_dir.exists():
            for child in sorted(base_dir.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(root / "docs" / "ops").as_posix()
                    logical = f"docs/ops/{rel}"
                    out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix.startswith("docs/schemas"):
        base_dir = root / "docs" / "ops" / "schemas"
        suffix = logical_prefix[len("docs/schemas") :].strip("/")
        if suffix:
            base_dir = base_dir / suffix
        if base_dir.is_file():
            return [resolve_node(root, logical_prefix, logical_prefix)]
        out = []
        if base_dir.exists():
            for child in sorted(base_dir.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(root / "docs" / "ops" / "schemas").as_posix()
                    logical = f"docs/schemas/{rel}"
                    out.append(resolve_node(root, logical, logical))
        return out
    if logical_prefix.startswith("docs/templates"):
        base_dir = root / "docs" / "ops" / "templates"
        suffix = logical_prefix[len("docs/templates") :].strip("/")
        if suffix:
            base_dir = base_dir / suffix
        if base_dir.is_file():
            return [resolve_node(root, logical_prefix, logical_prefix)]
        out = []
        if base_dir.exists():
            for child in sorted(base_dir.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(root / "docs" / "ops" / "templates").as_posix()
                    logical = f"docs/templates/{rel}"
                    out.append(resolve_node(root, logical, logical))
        return out

    return [resolve_node(root, logical_prefix, logical_prefix)]


def search_nodes(root: Path, *, query: str, prefix: str, limit: int) -> List[Dict[str, Any]]:
    if not query.strip():
        raise VAFSError("search query must not be empty")

    logical_prefix = normalize_logical_path(prefix)
    nodes = collect_recursive_nodes(root, logical_prefix)
    ranked: List[Tuple[Tuple[int, str], Dict[str, Any]]] = []
    for node in nodes:
        haystacks = [
            node.get("logical_path", ""),
            node.get("source_path", ""),
            node.get("real_path", ""),
            node.get("latest_alias_of", "") or "",
        ]
        best = rank_text_candidates(query, [str(x) for x in haystacks if x], limit=1)
        if not best:
            continue
        logical = str(node.get("logical_path", ""))
        best_text = best[0]
        normalized_query, query_tokens = search_tokens(query)
        normalized_best, best_tokens = search_tokens(best_text)
        if best_text.casefold() == query.casefold() or normalized_best == normalized_query:
            rank = 0
        elif best_text.casefold().startswith(query.casefold()) or normalized_best.startswith(normalized_query):
            rank = 1
        elif query.casefold() in best_text.casefold() or (normalized_query and normalized_query in normalized_best):
            rank = 2
        elif query_tokens and all(
            any(token.startswith(query_token) for token in best_tokens)
            for query_token in query_tokens
        ):
            rank = 3
        else:
            rank = 4
        ranked.append(((rank, logical), node))

    ranked.sort(key=lambda item: item[0])
    return [node for _, node in ranked[: max(1, limit)]]


def search_suggestions(root: Path, *, query: str, logical_prefix: str, limit: int) -> List[Dict[str, Any]]:
    if logical_prefix.startswith("reports/latest") or logical_prefix.startswith("reports/by-topic") or logical_prefix == "reports":
        return report_alias_suggestions(root, query, limit=limit)

    if logical_prefix in ALLOWED_NAMESPACES:
        return bounded_prefix_suggestions(root, logical_prefix, limit=limit)

    if not logical_prefix:
        ranked_namespaces = rank_text_candidates(query, ALLOWED_NAMESPACES, limit=limit)
        return dedupe_suggestions(
            [suggestion_logical_path(namespace, reason="supported_namespace") for namespace in ranked_namespaces],
            limit=limit,
        )

    return []


def print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


def denial_payload(root: Path, command: str, requested_path: str, logical_path: str, denial: VAFSDenial) -> Dict[str, Any]:
    payload = {
        "ok": False,
        "schema": SCHEMA,
        "command": command,
        "requested_path": requested_path,
        "logical_path": f"vafs://{logical_path}" if logical_path else "vafs://",
        "denied": True,
        "denial_reason": denial.reason,
        "detail": denial.detail,
        "allowed_namespaces": list(ALLOWED_NAMESPACES),
    }
    suggestions = denial_suggestions(root, command, logical_path, denial)
    if suggestions:
        payload["suggestions"] = suggestions
    return payload


def resolve_search_query(args: argparse.Namespace) -> str:
    exact = (args.exact or "").strip()
    positional = (args.query or "").strip()
    if exact and positional:
        return exact if exact == positional else exact
    return exact or positional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve bounded OpenClaw VAFS-R0 logical paths.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="OpenClaw repo root (default: current workspace root)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve one logical path")
    resolve_parser.add_argument("logical_path")
    resolve_parser.add_argument("--json", action="store_true", help="Compatibility flag; output is JSON either way")

    list_parser = subparsers.add_parser("list", help="List a bounded logical prefix")
    list_parser.add_argument("logical_prefix")
    list_parser.add_argument("--json", action="store_true", help="Compatibility flag; output is JSON either way")

    search_parser = subparsers.add_parser("search", help="Search bounded logical nodes")
    search_parser.add_argument("query", nargs="?", help="Single-string search query")
    search_parser.add_argument("--exact", help="Exact/substring search query (legacy alias)")
    search_parser.add_argument("--prefix", default="", help="Logical prefix to search under")
    search_parser.add_argument("--limit", type=int, default=25)
    search_parser.add_argument("--json", action="store_true", help="Compatibility flag; output is JSON either way")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.repo_root).expanduser().resolve()

    try:
        if args.command == "resolve":
            requested = args.logical_path
            logical = normalize_logical_path(requested)
            node = resolve_node(root, requested, logical)
            print_json({
                "ok": True,
                "schema": SCHEMA,
                "command": "resolve",
                "requested_path": requested,
                "logical_path": f"vafs://{logical}",
                "result": node,
            })
            return 0

        if args.command == "list":
            requested = args.logical_prefix
            logical = normalize_logical_path(requested)
            entries = list_nodes(root, requested, logical)
            print_json({
                "ok": True,
                "schema": LIST_SCHEMA,
                "command": "list",
                "requested_path": requested,
                "logical_prefix": f"vafs://{logical}" if logical else "vafs://",
                "count": len(entries),
                "entries": entries,
            })
            return 0

        if args.command == "search":
            prefix = args.prefix or ""
            logical_prefix = normalize_logical_path(prefix)
            query = resolve_search_query(args)
            if not query:
                raise VAFSError("search query must not be empty")
            entries = search_nodes(root, query=query, prefix=prefix, limit=max(1, args.limit))
            payload = {
                "ok": True,
                "schema": SEARCH_SCHEMA,
                "command": "search",
                "query": query,
                "logical_prefix": f"vafs://{logical_prefix}" if logical_prefix else "vafs://",
                "count": len(entries),
                "entries": entries,
            }
            if not entries:
                suggestions = search_suggestions(root, query=query, logical_prefix=logical_prefix, limit=max(1, min(args.limit, MAX_SUGGESTIONS)))
                if suggestions:
                    payload["suggestions"] = suggestions
            print_json(payload)
            return 0

        raise VAFSError(f"unsupported command: {args.command}")

    except VAFSDenial as denial:
        requested = ""
        if args.command == "resolve":
            requested = args.logical_path
        elif args.command == "list":
            requested = args.logical_prefix
        elif args.command == "search":
            requested = args.prefix or ""
        logical = ""
        try:
            logical = normalize_logical_path(requested)
        except Exception:
            logical = ""
        print_json(denial_payload(root, args.command, requested, logical, denial))
        return DENIAL_EXIT_CODE
    except VAFSError as exc:
        print_json({"ok": False, "schema": SCHEMA, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
