#!/usr/bin/env python3
"""Prepare a deterministic coverage audit surface for repo-mining exhaustiveness.

Primary intent:
- extract canonical repo slugs from Kimi/repo-wave report surfaces,
- compare against repo-level entries in repo_mining_archive_index,
- emit a machine-readable baseline for a later chatlog cross-reference pass.

This script does NOT claim final exhaustiveness by itself; it prepares the join surface
and clearly labels what is repo-covered, collection/report-path-only, or missing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


STRICT_KIMI_SURFACE_GLOBS = [
    # direct wave/fold-in evidence of repos actually pushed through Kimi-lane outputs
    "reports/repo_wave*.md",
    "reports/repo_batch_kimi_clean_synthesis_openclaw_*.md",
    "reports/kimi_repo_findings_queue_architecture_integration_*.md",
    "reports/openbb_repo_orientation_kimi_*.md",
    "reports/claude_mem_repo_orientation_kimi_*.md",
    "reports/repo_cross_wave_full_synthesis_openclaw_*.md",
]

BROAD_QUEUE_CONTEXT_GLOBS = STRICT_KIMI_SURFACE_GLOBS + [
    # queue truth/context surfaces (broader superset; includes repos not necessarily run on Kimi)
    "reports/repo_queue_*_normalization*_*.md",
    "reports/repo_queue_*_2026-03-26.md",
    "reports/github_repo_review_queue_openclaw_patterns_2026-03-20_NORMALIZED.md",
]

SURFACE_PROFILES = {
    "strict": STRICT_KIMI_SURFACE_GLOBS,
    "broad": BROAD_QUEUE_CONTEXT_GLOBS,
}

REPORTS_DIR = "reports"
ALIAS_INDEX = "repo_mining_archive_index_latest.json"
DATED_INDEX_GLOB = "repo_mining_archive_index_*.json"

OWNER_BLOCKLIST = {
    "reports",
    "report",
    "docs",
    "doc",
    "memory",
    "state",
    "scripts",
    "tests",
    "tmp",
    "repo-reviews",
    "collections",
}

CANONICAL_ALIAS_MAP = {
    # queue-truth caveats called out in repo_wave29_exhaustion_foldin_2026-03-26.md
    "opendevin/opendevin": "all-hands-ai/openhands",
    "openhands/openhands": "all-hands-ai/openhands",
    "wanghuacan/repomaster": "quantaalpha/repomaster",
}

RE_GH_URL = re.compile(r"https?://github\.com/([^/\s`]+)/([^/\s`#?]+)", re.IGNORECASE)
RE_BACKTICK_SLUG = re.compile(r"`([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)`")
RE_HEADING_SLUG = re.compile(
    r"^#{1,6}\s+(?:\d+\)\s+)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\s*$",
    re.MULTILINE,
)
RE_BOLD_SLUG = re.compile(r"\*\*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\*\*")


def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_registry_path(root: Path, explicit: str | None) -> tuple[Path, dict[str, Any]]:
    reports = root / REPORTS_DIR
    if explicit:
        chosen = (root / explicit).resolve() if not Path(explicit).is_absolute() else Path(explicit)
        chosen = chosen.relative_to(root) if chosen.is_absolute() and str(chosen).startswith(str(root)) else chosen
        chosen_path = root / chosen if not Path(chosen).is_absolute() else Path(chosen)
        return chosen_path, {"mode": "explicit", "alias_drift": None}

    alias_path = reports / ALIAS_INDEX
    dated = sorted(
        p for p in reports.glob(DATED_INDEX_GLOB) if p.name != ALIAS_INDEX
    )

    if dated:
        chosen_path = dated[-1]
        drift = None
        if alias_path.exists():
            try:
                alias = json.loads(alias_path.read_text(encoding="utf-8"))
                newest = json.loads(chosen_path.read_text(encoding="utf-8"))
                alias_entries = len(alias.get("entries") or [])
                newest_entries = len(newest.get("entries") or [])
                alias_version = str(alias.get("version") or "")
                newest_version = str(newest.get("version") or "")
                if alias_entries != newest_entries or alias_version != newest_version:
                    drift = {
                        "detected": True,
                        "alias_path": str(alias_path.relative_to(root)),
                        "alias_version": alias_version,
                        "alias_entries": alias_entries,
                        "newest_path": str(chosen_path.relative_to(root)),
                        "newest_version": newest_version,
                        "newest_entries": newest_entries,
                    }
            except Exception:
                drift = {"detected": True, "note": "alias_unreadable_or_invalid_json"}
        return chosen_path, {"mode": "latest_dated", "alias_drift": drift}

    if alias_path.exists():
        return alias_path, {"mode": "alias_only", "alias_drift": None}

    raise FileNotFoundError("No repo_mining_archive_index JSON found in reports/")


def normalize_slug(owner: str, repo: str) -> str | None:
    owner = owner.strip().strip("`")
    repo = repo.strip().strip("`")
    repo = repo.removesuffix(".git")

    if not owner or not repo:
        return None
    if owner.lower() in OWNER_BLOCKLIST:
        return None
    if re.fullmatch(r"[A-Z]\d+", owner) or re.fullmatch(r"[A-Z]\d+", repo):
        return None
    if "/" in owner or "/" in repo:
        return None
    if not re.search(r"[A-Za-z]", owner) or not re.search(r"[A-Za-z]", repo):
        return None

    raw = f"{owner}/{repo}".lower()
    return CANONICAL_ALIAS_MAP.get(raw, raw)


def extract_slugs_from_text(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []

    for owner, repo in RE_GH_URL.findall(text):
        slug = normalize_slug(owner, repo)
        if slug:
            out.append(
                {
                    "slug": slug,
                    "raw": f"{owner}/{repo}",
                    "kind": "github_url",
                }
            )

    for token in RE_BACKTICK_SLUG.findall(text):
        owner, repo = token.split("/", 1)
        slug = normalize_slug(owner, repo)
        if slug:
            out.append(
                {
                    "slug": slug,
                    "raw": token,
                    "kind": "backtick_slug",
                }
            )

    for token in RE_HEADING_SLUG.findall(text):
        owner, repo = token.split("/", 1)
        slug = normalize_slug(owner, repo)
        if slug:
            out.append(
                {
                    "slug": slug,
                    "raw": token,
                    "kind": "heading_slug",
                }
            )

    for token in RE_BOLD_SLUG.findall(text):
        owner, repo = token.split("/", 1)
        slug = normalize_slug(owner, repo)
        if slug:
            out.append(
                {
                    "slug": slug,
                    "raw": token,
                    "kind": "bold_slug",
                }
            )

    # stable dedupe while preserving first-seen evidence kind/raw
    dedup: dict[str, dict[str, str]] = {}
    for row in out:
        dedup.setdefault(row["slug"], row)
    return [dedup[k] for k in sorted(dedup)]


def collect_surface_files(root: Path, globs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for pattern in globs:
        files.update(root.glob(pattern))
    return sorted(p for p in files if p.is_file())


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def parse_chatlog_slug_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Accept free-form lines/paragraphs by reusing extractors.
    rows = extract_slugs_from_text(text)
    if rows:
        return sorted({r["slug"] for r in rows})

    # fallback: line-based owner/repo tokens
    slugs: set[str] = set()
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith("#"):
            continue
        if "/" not in line:
            continue
        owner, repo = line.split("/", 1)
        slug = normalize_slug(owner, repo)
        if slug:
            slugs.add(slug)
    return sorted(slugs)


def build_audit_payload(
    root: Path,
    registry_path: Path,
    registry_meta: dict[str, Any],
    surface_globs: list[str],
    chatlog_slug_file: Path | None,
) -> dict[str, Any]:
    registry = read_json(registry_path)
    entries = [e for e in (registry.get("entries") or []) if isinstance(e, dict)]

    repo_entry_ids_by_slug: dict[str, list[str]] = {}
    report_paths_to_entry_ids: dict[str, set[str]] = {}

    for e in entries:
        entry_id = str(e.get("id") or "")
        repo = e.get("repo") if isinstance(e.get("repo"), dict) else None
        slug = None
        if repo and repo.get("slug"):
            slug = str(repo.get("slug")).lower()
            repo_entry_ids_by_slug.setdefault(slug, []).append(entry_id)

        report_paths = e.get("report_paths") if isinstance(e.get("report_paths"), list) else []
        for rp in report_paths:
            if not isinstance(rp, str):
                continue
            report_paths_to_entry_ids.setdefault(rp, set()).add(entry_id)

    surface_files = collect_surface_files(root, surface_globs)

    slug_evidence: dict[str, dict[str, Any]] = {}
    missing_surface_files: list[str] = []

    for f in surface_files:
        rel = str(f.relative_to(root))
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            missing_surface_files.append(rel)
            continue

        found = extract_slugs_from_text(text)
        for row in found:
            slug = row["slug"]
            item = slug_evidence.setdefault(
                slug,
                {
                    "slug": slug,
                    "source_files": set(),
                    "evidence": [],
                },
            )
            item["source_files"].add(rel)
            item["evidence"].append(
                {
                    "file": rel,
                    "kind": row["kind"],
                    "raw": row["raw"],
                }
            )

    # finalize slug rows
    slug_rows: list[dict[str, Any]] = []
    for slug in sorted(slug_evidence):
        item = slug_evidence[slug]
        source_files = sorted(item["source_files"])
        repo_entry_ids = sorted(repo_entry_ids_by_slug.get(slug, []))

        source_files_in_registry = sorted(
            f for f in source_files if f in report_paths_to_entry_ids
        )

        if repo_entry_ids:
            coverage_class = "repo_entry_present"
        elif source_files_in_registry:
            coverage_class = "report_path_only"
        else:
            coverage_class = "uncovered"

        slug_rows.append(
            {
                "slug": slug,
                "coverage_class": coverage_class,
                "registry_repo_entry_ids": repo_entry_ids,
                "source_files": source_files,
                "source_files_in_registry": source_files_in_registry,
                "evidence_count": len(item["evidence"]),
            }
        )

    coverage_counts = {
        "repo_entry_present": sum(1 for r in slug_rows if r["coverage_class"] == "repo_entry_present"),
        "report_path_only": sum(1 for r in slug_rows if r["coverage_class"] == "report_path_only"),
        "uncovered": sum(1 for r in slug_rows if r["coverage_class"] == "uncovered"),
    }

    surface_slug_set = {r["slug"] for r in slug_rows}
    registry_repo_slug_set = set(repo_entry_ids_by_slug.keys())

    chatlog_section: dict[str, Any] | None = None
    if chatlog_slug_file:
        chatlog_slugs = parse_chatlog_slug_file(chatlog_slug_file)
        surface_set = {r["slug"] for r in slug_rows}
        registry_set = set(repo_entry_ids_by_slug.keys())

        chatlog_section = {
            "chatlog_slug_file": str(chatlog_slug_file),
            "chatlog_slug_count": len(chatlog_slugs),
            "chatlog_slugs_missing_from_surface": sorted(set(chatlog_slugs) - surface_set),
            "chatlog_slugs_missing_from_registry_repo_entries": sorted(set(chatlog_slugs) - registry_set),
            "registry_repo_slugs_not_in_chatlog": sorted(registry_set - set(chatlog_slugs)),
        }

    payload: dict[str, Any] = {
        "schema": "openclaw.repo_mining_exhaustiveness_audit_prep.v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workspace_root": str(root),
        "registry": {
            "path": str(registry_path.relative_to(root)),
            "version": registry.get("version"),
            "status": registry.get("status"),
            "entry_count": len(entries),
            "repo_entry_count": sum(1 for e in entries if isinstance(e.get("repo"), dict) and e.get("repo", {}).get("slug")),
            "collection_entry_count": sum(1 for e in entries if not (isinstance(e.get("repo"), dict) and e.get("repo", {}).get("slug"))),
            "resolution": registry_meta,
        },
        "surface": {
            "globs": surface_globs,
            "file_count": len(surface_files),
            "files": [str(f.relative_to(root)) for f in surface_files],
            "read_errors": missing_surface_files,
        },
        "coverage": {
            "surface_repo_slug_count": len(slug_rows),
            "counts": coverage_counts,
            "surface_slugs_missing_registry_repo_entries": sorted(surface_slug_set - registry_repo_slug_set),
            "registry_repo_slugs_not_seen_in_surface": sorted(registry_repo_slug_set - surface_slug_set),
            "slug_rows": slug_rows,
        },
        "deterministic_crossref_instructions": {
            "chatlog_slug_file_format": "plain text with one repo slug (owner/repo) per line OR raw chatlog text with GitHub URLs; parser normalizes both",
            "command_template": "python3 scripts/repo_mining_exhaustiveness_audit.py --chatlog-slugs <path/to/chatlog_repo_slugs.txt>",
        },
    }

    if chatlog_section is not None:
        payload["chatlog_crossref"] = chatlog_section

    return payload


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def default_output_path(root: Path) -> Path:
    today = dt.date.today().isoformat()
    return root / "reports" / f"repo_mining_exhaustiveness_audit_prep_{today}.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare deterministic repo-mining exhaustiveness audit baseline")
    p.add_argument("--registry", help="Registry JSON path (default: latest dated repo_mining_archive_index_*.json)")
    p.add_argument(
        "--surface-profile",
        choices=sorted(SURFACE_PROFILES.keys()),
        default="strict",
        help="Predefined report-surface profile. strict=only direct Kimi/wave evidence, broad=adds queue-context superset",
    )
    p.add_argument(
        "--surface",
        action="append",
        default=[],
        help="Additional surface glob(s) relative to workspace root (repeatable)",
    )
    p.add_argument("--chatlog-slugs", help="Optional path to chatlog-derived repo slugs/text for cross-reference")
    p.add_argument("--out", help="Output JSON path")
    p.add_argument("--print-summary", action="store_true", help="Print summary to stdout")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = workspace_root()

    try:
        registry_path, registry_meta = resolve_registry_path(root, args.registry)
        globs = SURFACE_PROFILES[args.surface_profile] + args.surface
        chatlog_file = Path(args.chatlog_slugs).resolve() if args.chatlog_slugs else None

        payload = build_audit_payload(
            root=root,
            registry_path=registry_path,
            registry_meta=registry_meta,
            surface_globs=globs,
            chatlog_slug_file=chatlog_file,
        )
        payload.setdefault("surface", {})["profile"] = args.surface_profile

        out_path = Path(args.out).resolve() if args.out else default_output_path(root)
        write_payload(out_path, payload)

        latest_alias = root / "reports" / "repo_mining_exhaustiveness_audit_prep_latest.json"
        write_payload(latest_alias, payload)

        if args.print_summary:
            counts = payload["coverage"]["counts"]
            print(
                " ".join(
                    [
                        f"registry={payload['registry']['path']}",
                        f"surface_files={payload['surface']['file_count']}",
                        f"surface_repos={payload['coverage']['surface_repo_slug_count']}",
                        f"repo_entry_present={counts['repo_entry_present']}",
                        f"report_path_only={counts['report_path_only']}",
                        f"uncovered={counts['uncovered']}",
                        f"out={out_path}",
                    ]
                )
            )

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
