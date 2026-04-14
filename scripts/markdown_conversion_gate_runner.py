#!/usr/bin/env python3
"""Deterministic markdown conversion quality gate runner (v1).

Evaluates a long-form conversion packet against bounded fail-closed gates:
1) schema
2) file_set
3) source_map
4) structure
5) markdown_profile

Design goals:
- deterministic machine-readable decisions
- strict fail-closed behavior
- bounded validation scope (conversion-quality only)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover (environment wiring)
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "markdown_conversion_candidate.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "markdown_conversion_gate" / "decisions.jsonl"

DEFAULT_EXPECTATIONS: Dict[str, Any] = {
    "min_chunks": 1,
    "max_chunks": 2000,
    "require_source_map_chunk_cover": True,
    "require_chunk_heading_alignment": True,
}

DEFAULT_PROFILE: Dict[str, Any] = {
    "markdown_flavor": "gfm",
    "require_frontmatter": False,
    "required_frontmatter_keys": [],
    "disallow_raw_html": True,
    "disallow_javascript_links": True,
    "max_line_length": 1200,
    "require_heading_hierarchy": True,
}

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$", re.MULTILINE)
SCRIPT_TAG_RE = re.compile(r"<\s*(script|iframe|object|embed)\b", re.IGNORECASE)
JS_LINK_RE = re.compile(r"\]\(\s*javascript:\s*", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def normalize_sha256(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def _resolve_path(*, root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _merge_dict(base: Dict[str, Any], override: Any) -> Dict[str, Any]:
    merged = dict(base)
    if isinstance(override, dict):
        for k, v in override.items():
            merged[k] = v
    return merged


def _extract_headings(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for m in HEADING_RE.finditer(text):
        lvl = len(m.group(1))
        title = _normalize_heading(m.group(2))
        if title:
            out.append((lvl, title))
    return out


def _normalize_heading(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[`*_~]", "", text)
    text = re.sub(r"\[[^\]]+\]\([^\)]+\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" #")


def _frontmatter_keys(text: str) -> Optional[Dict[str, str]]:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    body = text[4:end]
    keys: Dict[str, str] = {}
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        k, v = stripped.split(":", 1)
        keys[k.strip()] = v.strip()
    return keys


def gate_schema(candidate: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(candidate),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        },
    )


def _load_effective_packet(
    candidate: Dict[str, Any],
    repo_root: Path,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []

    package = candidate.get("package") if isinstance(candidate.get("package"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}

    raw_package_root = package.get("package_root")
    raw_book_path = package.get("book_path")
    raw_chunks_dir = package.get("chunks_dir")
    raw_source_map_path = package.get("source_map_path")

    if not isinstance(raw_package_root, str) or not raw_package_root.strip():
        issues.append({"reason": "package_root_missing"})
        return None, issues

    package_root = _resolve_path(root=repo_root, raw_path=raw_package_root)
    if not is_within(repo_root, package_root):
        issues.append({"reason": "package_root_outside_repo", "package_root": str(package_root)})
        return None, issues

    if not package_root.exists() or not package_root.is_dir():
        issues.append({"reason": "package_root_missing", "package_root": str(package_root)})
        return None, issues

    if not isinstance(raw_book_path, str) or not raw_book_path.strip():
        issues.append({"reason": "book_path_missing"})
    if not isinstance(raw_chunks_dir, str) or not raw_chunks_dir.strip():
        issues.append({"reason": "chunks_dir_missing"})
    if not isinstance(raw_source_map_path, str) or not raw_source_map_path.strip():
        issues.append({"reason": "source_map_path_missing"})

    if issues:
        return None, issues

    book_path = _resolve_path(root=package_root, raw_path=raw_book_path)
    chunks_dir = _resolve_path(root=package_root, raw_path=raw_chunks_dir)
    source_map_path = _resolve_path(root=package_root, raw_path=raw_source_map_path)

    if not is_within(package_root, book_path):
        issues.append({"reason": "book_path_outside_package", "book_path": str(book_path)})
    if not is_within(package_root, chunks_dir):
        issues.append({"reason": "chunks_dir_outside_package", "chunks_dir": str(chunks_dir)})
    if not is_within(package_root, source_map_path):
        issues.append({"reason": "source_map_outside_package", "source_map_path": str(source_map_path)})

    if not book_path.exists() or not book_path.is_file():
        issues.append({"reason": "book_missing", "book_path": str(book_path)})
    if not chunks_dir.exists() or not chunks_dir.is_dir():
        issues.append({"reason": "chunks_dir_missing", "chunks_dir": str(chunks_dir)})
    if not source_map_path.exists() or not source_map_path.is_file():
        issues.append({"reason": "source_map_missing", "source_map_path": str(source_map_path)})

    source_path = None
    if isinstance(source.get("source_path"), str) and source.get("source_path").strip():
        source_path = _resolve_path(root=repo_root, raw_path=str(source.get("source_path")))
        if not source_path.exists() or not source_path.is_file():
            issues.append({"reason": "source_path_missing", "source_path": str(source_path)})

    if source_path is not None and isinstance(source.get("source_sha256"), str) and source.get("source_sha256").strip():
        declared = normalize_sha256(str(source.get("source_sha256")))
        try:
            actual = file_sha256(source_path)
        except Exception as exc:
            issues.append({"reason": "source_hash_compute_failed", "detail": str(exc)})
        else:
            if declared != actual:
                issues.append(
                    {
                        "reason": "source_hash_mismatch",
                        "declared": declared,
                        "actual": actual,
                        "source_path": str(source_path),
                    }
                )

    if issues:
        return None, issues

    expectations = _merge_dict(DEFAULT_EXPECTATIONS, candidate.get("expectations"))
    profile = _merge_dict(DEFAULT_PROFILE, candidate.get("profile"))

    packet = {
        "package_root": package_root,
        "book_path": book_path,
        "chunks_dir": chunks_dir,
        "source_map_path": source_map_path,
        "source_path": source_path,
        "expectations": expectations,
        "profile": profile,
    }
    return packet, issues


def gate_file_set(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    packet, issues = _load_effective_packet(candidate, repo_root)
    if packet is None:
        return False, "fileset_unready", {"issues": issues}

    chunks_dir: Path = packet["chunks_dir"]
    chunk_files = sorted(p for p in chunks_dir.rglob("*.md") if p.is_file())

    expectations = packet["expectations"]
    min_chunks = int(expectations.get("min_chunks", DEFAULT_EXPECTATIONS["min_chunks"]))
    max_chunks = int(expectations.get("max_chunks", DEFAULT_EXPECTATIONS["max_chunks"]))

    if min_chunks < 1:
        return False, "fileset_unready", {"issues": [{"reason": "min_chunks_invalid", "value": min_chunks}]}
    if max_chunks < min_chunks:
        return (
            False,
            "fileset_unready",
            {"issues": [{"reason": "max_chunks_invalid", "min_chunks": min_chunks, "max_chunks": max_chunks}]},
        )

    count = len(chunk_files)
    if count < min_chunks or count > max_chunks:
        return (
            False,
            "fileset_unready",
            {
                "issues": [
                    {
                        "reason": "chunk_count_out_of_bounds",
                        "count": count,
                        "min_chunks": min_chunks,
                        "max_chunks": max_chunks,
                    }
                ]
            },
        )

    return (
        True,
        None,
        {
            "package_root": str(packet["package_root"]),
            "book_path": str(packet["book_path"]),
            "chunks_dir": str(chunks_dir),
            "source_map_path": str(packet["source_map_path"]),
            "chunk_count": count,
            "min_chunks": min_chunks,
            "max_chunks": max_chunks,
        },
    )


def _iter_jsonl_rows(path: Path) -> Iterable[Tuple[int, Any]]:
    raw = path.read_text(encoding="utf-8")
    for i, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        yield i, json.loads(line)


def _load_source_map(packet: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    issues: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []

    source_map_path: Path = packet["source_map_path"]
    chunks_dir: Path = packet["chunks_dir"]
    package_root: Path = packet["package_root"]

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    try:
        rows = list(_iter_jsonl_rows(source_map_path))
    except Exception as exc:
        return [], [{"reason": "source_map_unreadable", "detail": str(exc), "source_map_path": str(source_map_path)}]

    if not rows:
        return [], [{"reason": "source_map_empty", "source_map_path": str(source_map_path)}]

    for line_no, row in rows:
        if not isinstance(row, dict):
            issues.append({"reason": "source_map_row_not_object", "line": line_no})
            continue

        chunk_id = row.get("chunk_id")
        chunk_path_raw = row.get("chunk_path")
        source_locator = row.get("source_locator")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            issues.append({"reason": "chunk_id_missing", "line": line_no})
            continue
        if chunk_id in seen_ids:
            issues.append({"reason": "chunk_id_duplicate", "line": line_no, "chunk_id": chunk_id})
            continue
        seen_ids.add(chunk_id)

        if not isinstance(chunk_path_raw, str) or not chunk_path_raw.strip():
            issues.append({"reason": "chunk_path_missing", "line": line_no, "chunk_id": chunk_id})
            continue

        if not isinstance(source_locator, str) or not source_locator.strip():
            issues.append({"reason": "source_locator_missing", "line": line_no, "chunk_id": chunk_id})
            continue

        chunk_path = _resolve_path(root=package_root, raw_path=chunk_path_raw)
        if not is_within(package_root, chunk_path):
            issues.append(
                {
                    "reason": "chunk_path_outside_package",
                    "line": line_no,
                    "chunk_id": chunk_id,
                    "chunk_path": str(chunk_path),
                }
            )
            continue

        if not is_within(chunks_dir, chunk_path):
            issues.append(
                {
                    "reason": "chunk_path_outside_chunks_dir",
                    "line": line_no,
                    "chunk_id": chunk_id,
                    "chunk_path": str(chunk_path),
                }
            )
            continue

        if not chunk_path.exists() or not chunk_path.is_file():
            issues.append(
                {
                    "reason": "chunk_path_unresolved",
                    "line": line_no,
                    "chunk_id": chunk_id,
                    "chunk_path": str(chunk_path),
                }
            )
            continue

        norm_path = str(chunk_path)
        if norm_path in seen_paths:
            issues.append({"reason": "chunk_path_duplicate", "line": line_no, "chunk_path": norm_path})
            continue
        seen_paths.add(norm_path)

        entries.append(
            {
                "line": line_no,
                "chunk_id": chunk_id,
                "chunk_path": chunk_path,
                "source_locator": source_locator,
            }
        )

    return entries, issues


def gate_source_map(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    packet, issues = _load_effective_packet(candidate, repo_root)
    if packet is None:
        return False, "fileset_unready", {"issues": issues}

    entries, map_issues = _load_source_map(packet)
    if map_issues:
        return False, "source_map_invalid", {"issues": map_issues}

    expect_cover = bool(packet["expectations"].get("require_source_map_chunk_cover", True))
    if expect_cover:
        chunk_files = sorted(p for p in packet["chunks_dir"].rglob("*.md") if p.is_file())
        mapped = {str(e["chunk_path"]) for e in entries}
        uncovered = [str(p) for p in chunk_files if str(p) not in mapped]
        if uncovered:
            return (
                False,
                "source_map_invalid",
                {
                    "issues": [
                        {
                            "reason": "source_map_missing_chunk_entries",
                            "uncovered": uncovered,
                        }
                    ]
                },
            )

    return (
        True,
        None,
        {
            "entry_count": len(entries),
            "covered_all_chunks": bool(expect_cover),
        },
    )


def _heading_hierarchy_issues(headings: List[Tuple[int, str]]) -> List[str]:
    issues: List[str] = []
    prev = 0
    for idx, (lvl, _txt) in enumerate(headings, start=1):
        if prev and lvl > prev + 1:
            issues.append(f"heading_jump_at:{idx}:{prev}->{lvl}")
        prev = lvl
    return issues


def gate_structure(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    packet, issues = _load_effective_packet(candidate, repo_root)
    if packet is None:
        return False, "fileset_unready", {"issues": issues}

    entries, map_issues = _load_source_map(packet)
    if map_issues:
        return False, "source_map_invalid", {"issues": map_issues}

    try:
        book_text = packet["book_path"].read_text(encoding="utf-8")
    except Exception as exc:
        return False, "structure_unfaithful", {"issues": [{"reason": "book_unreadable", "detail": str(exc)}]}

    book_headings = _extract_headings(book_text)
    if not book_headings:
        return False, "structure_unfaithful", {"issues": [{"reason": "book_missing_headings"}]}

    book_titles = [t for _lvl, t in book_headings]
    book_index = 0

    align = bool(packet["expectations"].get("require_chunk_heading_alignment", True))
    issues_out: List[Dict[str, Any]] = []

    for entry in entries:
        chunk_path: Path = entry["chunk_path"]
        try:
            chunk_text = chunk_path.read_text(encoding="utf-8")
        except Exception as exc:
            issues_out.append(
                {
                    "reason": "chunk_unreadable",
                    "chunk_id": entry["chunk_id"],
                    "chunk_path": str(chunk_path),
                    "detail": str(exc),
                }
            )
            continue

        chunk_headings = _extract_headings(chunk_text)
        if not chunk_headings:
            issues_out.append(
                {
                    "reason": "chunk_missing_headings",
                    "chunk_id": entry["chunk_id"],
                    "chunk_path": str(chunk_path),
                }
            )
            continue

        if align:
            chunk_first = chunk_headings[0][1]
            found_idx = -1
            for idx in range(book_index, len(book_titles)):
                if book_titles[idx] == chunk_first:
                    found_idx = idx
                    break
            if found_idx < 0:
                issues_out.append(
                    {
                        "reason": "chunk_heading_not_in_book",
                        "chunk_id": entry["chunk_id"],
                        "chunk_path": str(chunk_path),
                        "chunk_heading": chunk_first,
                    }
                )
            else:
                book_index = found_idx

    if issues_out:
        return False, "structure_unfaithful", {"issues": issues_out}

    return (
        True,
        None,
        {
            "book_heading_count": len(book_headings),
            "chunk_alignment_checked": align,
            "chunk_count": len(entries),
        },
    )


def gate_markdown_profile(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    packet, issues = _load_effective_packet(candidate, repo_root)
    if packet is None:
        return False, "fileset_unready", {"issues": issues}

    profile = packet["profile"]
    max_line_length = int(profile.get("max_line_length", DEFAULT_PROFILE["max_line_length"]))
    required_frontmatter_keys = list(profile.get("required_frontmatter_keys") or [])

    paths: List[Path] = [packet["book_path"]]
    paths.extend(sorted(p for p in packet["chunks_dir"].rglob("*.md") if p.is_file()))

    violations: List[Dict[str, Any]] = []

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            violations.append({"path": str(path), "reason": "file_unreadable", "detail": str(exc)})
            continue

        if CONTROL_CHAR_RE.search(text):
            violations.append({"path": str(path), "reason": "control_chars_present"})

        if bool(profile.get("disallow_raw_html", True)) and SCRIPT_TAG_RE.search(text):
            violations.append({"path": str(path), "reason": "unsafe_html_tag_detected"})

        if bool(profile.get("disallow_javascript_links", True)) and JS_LINK_RE.search(text):
            violations.append({"path": str(path), "reason": "javascript_link_detected"})

        if max_line_length > 0:
            too_long = [i for i, line in enumerate(text.splitlines(), start=1) if len(line) > max_line_length]
            if too_long:
                violations.append(
                    {
                        "path": str(path),
                        "reason": "line_length_exceeded",
                        "max_line_length": max_line_length,
                        "line_count": len(too_long),
                        "sample_lines": too_long[:5],
                    }
                )

        headings = _extract_headings(text)
        if bool(profile.get("require_heading_hierarchy", True)):
            jumps = _heading_hierarchy_issues(headings)
            if jumps:
                violations.append({"path": str(path), "reason": "heading_hierarchy_jump", "details": jumps[:8]})

        if bool(profile.get("require_frontmatter", False)):
            keys = _frontmatter_keys(text)
            if keys is None:
                violations.append({"path": str(path), "reason": "frontmatter_missing"})
            else:
                missing = [k for k in required_frontmatter_keys if not str(keys.get(k, "")).strip()]
                if missing:
                    violations.append({"path": str(path), "reason": "frontmatter_required_keys_missing", "missing": missing})

    if violations:
        return False, "markdown_profile_violation", {"issues": violations}

    return (
        True,
        None,
        {
            "files_checked": len(paths),
            "profile": {
                "markdown_flavor": profile.get("markdown_flavor"),
                "disallow_raw_html": bool(profile.get("disallow_raw_html", True)),
                "disallow_javascript_links": bool(profile.get("disallow_javascript_links", True)),
                "max_line_length": max_line_length,
            },
        },
    )


def append_decision_record(
    *,
    decision_log_path: Optional[Path],
    repo_root: Path,
    decision_row: Dict[str, Any],
) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = decision_log_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not is_within(repo_root, path):
        return {
            "enabled": True,
            "appended": False,
            "reason": "unsafe_path",
            "path": str(path),
        }

    try:
        if path.exists() and not path.is_file():
            return {
                "enabled": True,
                "appended": False,
                "reason": "path_not_file",
                "path": str(path),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(decision_row) + "\n")
        return {
            "enabled": True,
            "appended": True,
            "path": str(path),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "appended": False,
            "reason": "append_failed",
            "path": str(path),
            "error": str(exc),
        }


def evaluate_candidate(
    *,
    candidate: Any,
    candidate_path: Path,
    repo_root: Path,
    schema_path: Path,
) -> Dict[str, Any]:
    decision_at = now_iso()

    conversion_id: Optional[str] = None
    if isinstance(candidate, dict):
        raw_id = candidate.get("conversion_id")
        if isinstance(raw_id, str):
            conversion_id = raw_id

    gate_rows: List[Dict[str, Any]] = []
    blocked = False
    block_reason: Optional[str] = None
    block_gate: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(candidate, schema_path)),
        ("file_set", lambda: gate_file_set(candidate if isinstance(candidate, dict) else {}, repo_root)),
        ("source_map", lambda: gate_source_map(candidate if isinstance(candidate, dict) else {}, repo_root)),
        ("structure", lambda: gate_structure(candidate if isinstance(candidate, dict) else {}, repo_root)),
        ("markdown_profile", lambda: gate_markdown_profile(candidate if isinstance(candidate, dict) else {}, repo_root)),
    ]

    for gate_name, gate_fn in gate_specs:
        if blocked:
            gate_rows.append(
                {
                    "gate": gate_name,
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                }
            )
            continue

        try:
            ok, reason, details = gate_fn()
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            ok = False
            reason = "gate_unavailable"
            details = {"error": "gate_exception", "detail": str(exc)}

        if ok:
            gate_rows.append({"gate": gate_name, "status": "pass", "details": details})
            continue

        blocked = True
        block_reason = reason or "gate_unavailable"
        block_gate = gate_name
        gate_rows.append(
            {
                "gate": gate_name,
                "status": "fail",
                "reason": block_reason,
                "details": details,
            }
        )

    decision = "BLOCK" if blocked else "PASS"
    final_state = "BLOCKED" if blocked else "CONVERSION_QUALITY_VERIFIED"

    try:
        candidate_sha = file_sha256(candidate_path)
    except Exception:
        candidate_sha = None

    effective_profile = _merge_dict(DEFAULT_PROFILE, candidate.get("profile") if isinstance(candidate, dict) else None)
    effective_expectations = _merge_dict(DEFAULT_EXPECTATIONS, candidate.get("expectations") if isinstance(candidate, dict) else None)

    return {
        "schema": "clawd.markdown_conversion_gate.decision.v1",
        "evaluated_at": decision_at,
        "decision": decision,
        "final_state": final_state,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "conversion_id": conversion_id,
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
        },
        "policy": {
            "expectations": effective_expectations,
            "profile": effective_profile,
        },
        "gates": gate_rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic markdown conversion quality gate runner (v1)")
    ap.add_argument("--candidate", required=True, help="Path to markdown conversion candidate JSON")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root for relative path resolution")
    ap.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to markdown conversion candidate schema",
    )
    ap.add_argument(
        "--decision-log",
        default=str(DEFAULT_DECISION_LOG),
        help="Append-only decision log path (relative to repo root unless absolute)",
    )
    ap.add_argument("--no-decision-log", action="store_true", help="Disable append-only decision recording")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()

    try:
        candidate_doc = load_json_file(candidate_path)
    except Exception as exc:
        result = {
            "schema": "clawd.markdown_conversion_gate.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "conversion_id": None,
            "candidate": {
                "path": str(candidate_path),
                "sha256": None,
            },
            "policy": {
                "expectations": DEFAULT_EXPECTATIONS,
                "profile": DEFAULT_PROFILE,
            },
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {
                        "error": "candidate_json_unreadable",
                        "detail": str(exc),
                    },
                },
                {"gate": "file_set", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "source_map", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "structure", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "markdown_profile", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate_candidate(
            candidate=candidate_doc,
            candidate_path=candidate_path,
            repo_root=repo_root,
            schema_path=schema_path,
        )

    decision_log_path: Optional[Path] = None
    if not args.no_decision_log:
        decision_log_path = Path(args.decision_log).expanduser()

    record = append_decision_record(decision_log_path=decision_log_path, repo_root=repo_root, decision_row=result)
    result["decision_record"] = record

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
