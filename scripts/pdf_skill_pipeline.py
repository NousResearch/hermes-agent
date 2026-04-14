#!/usr/bin/env python3
"""PDF -> knowledge -> skill-update pipeline.

Stages:
1) ingest   - detect PDFs + dedupe via sha256
2) extract  - extract text (reuse pre-extracted text if available)
3) index    - create run manifest + update global registry
4) distill  - produce modular playbooks with source citations
5) propose  - generate diff-ready markdown patch proposals
6) approve  - manual approval gate + optional apply + rollback bundle
7) evaluate - post-change quality checks

Workspace assumptions:
- repository root is parent of this script's directory.
- pipeline data is stored in memory/pdf_skill_pipeline
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
PIPELINE_ROOT = REPO_ROOT / "memory" / "pdf_skill_pipeline"
STATE_DIR = PIPELINE_ROOT / "state"
RUNS_DIR = PIPELINE_ROOT / "runs"
PLAYBOOK_ROOT = PIPELINE_ROOT / "playbooks"
PROPOSAL_ROOT = PIPELINE_ROOT / "proposals"
TARGET_SKILL_ROOT = REPO_ROOT / "memory" / "skills" / "twitter_digest_playbooks"

REGISTRY_PATH = STATE_DIR / "pdf_registry.json"


MODULE_SPECS: List[Dict[str, Any]] = [
    {
        "key": "cluster_event_detection",
        "title": "Cluster/Event Detection",
        "file": "playbook_cluster_event_detection.md",
        "source_keywords": ["novel and viral", "event-trigger", "trend mining"],
        "rules": [
            "Use **cluster-first detection** (noise filter → clustering → event typing) instead of direct keyword alerts.",
            "Run two operating feeds: high-recall discovery feed and high-precision operator feed; tune separately.",
            "Use unique-author / entropy signals as significance gates before raw post volume.",
            "Treat verification strictness as an explicit latency knob (stricter verification can delay alerts).",
        ],
        "thresholds": [
            "Starter telemetry target: noise filtered share ≈ 60–80% before event scoring (seed pack cites 78% in large-scale pipeline).",
            "Track cluster funnel: raw posts → clusters → candidate events → operator alerts (seed evidence: 12M/day → 16k clusters/day → 6.6k potentially newsworthy).",
            "Set separate SLOs for detection latency and verification latency; monitor p50/p90/p99.",
        ],
        "anti_patterns": [
            "Ranking trends directly from raw token/hashtag counts with no clustering.",
            "Single threshold for all audiences (operator feed gets noisy or misses events).",
            "Using volume spikes without corroboration/diversity gates (manipulation-prone).",
        ],
        "prompt": "Given clustered event candidates with metrics (velocity, author_diversity, source_quality), return top events with {event_type, why_now, confidence, citations}. Reject candidates lacking anchor evidence.",
    },
    {
        "key": "entity_anchor_grounding",
        "title": "Entity/Anchor Grounding",
        "file": "playbook_entity_anchor_grounding.md",
        "source_keywords": ["entity-first", "anchor", "dominating"],
        "rules": [
            "Perform deterministic extraction for strong anchors first: @handles, cashtags, domains/URLs, contract addresses.",
            "Canonicalize to stable IDs (chain+contract, verified domain, platform entity id) before scoring/ranking.",
            "Apply umbrella-topic demotion unless anchor evidence is present and validated.",
            "Use two-stage linking: candidate retrieval + reranking/disambiguation for alias collisions and tail entities.",
        ],
        "thresholds": [
            "Anchor coverage gate: require ≥1 validated anchor for auto-ship trend items.",
            "Demotion rule: generic topic labels with weak specificity must fail operator-feed gate.",
            "Keep alias tables versioned and refresh daily for high-churn entities.",
        ],
        "anti_patterns": [
            "Symbol-only trend detection in crypto (collisions/rebrands create false joins).",
            "Treating broad categories (e.g., 'crypto') as event outputs with no referent.",
            "Skipping canonicalization and then attempting dedupe late in pipeline.",
        ],
        "prompt": "Ground each candidate to canonical entities. Output {anchor_type, canonical_id, disambiguation_notes}. If grounding confidence is low, mark NEEDS_REVIEW and suppress umbrella labels.",
    },
    {
        "key": "llm_bounded_rerank_guardrails",
        "title": "LLM Bounded Rerank Guardrails",
        "file": "playbook_llm_bounded_rerank_guardrails.md",
        "source_keywords": ["llm-assisted ranking", "cost-efficient architectures", "bounded"],
        "rules": [
            "Treat reranking as constrained structured prediction over deterministic candidate IDs only.",
            "Enforce strict output schema + deterministic validator (subset check, dedupe, exactly-k or ABSTAIN).",
            "Use bounded windows/tournament rerank for large candidate sets to keep context and cost controlled.",
            "Fallback policy is mandatory: timeout/retry/circuit-breaker + deterministic baseline ranker when LLM unavailable.",
        ],
        "thresholds": [
            "Auto-ship only above calibrated confidence threshold; otherwise human review or ABSTAIN.",
            "Set hard token/output caps per call and pre-count tokens before execution.",
            "Citations must be machine-verifiable against known evidence IDs; reject malformed references.",
        ],
        "anti_patterns": [
            "Allowing model to invent IDs not in candidate set.",
            "Running one monolithic listwise prompt over oversized candidate sets.",
            "No fallback when provider rate-limits or model endpoint degrades.",
        ],
        "prompt": "You are a bounded reranker. Input: candidate_ids[], evidence_ids[]. Output JSON only: {ranked_ids:[...], confidence:0-1, abstain:boolean, abstain_reason, citations:[evidence_ids]}. Never emit out-of-set IDs.",
    },
    {
        "key": "novelty_scoring",
        "title": "Novelty Scoring",
        "file": "playbook_novelty_scoring.md",
        "source_keywords": ["novelty scoring", "first story", "event-window"],
        "rules": [
            "Compute novelty from short-vs-long window shift + burst/deviation + evidence quality (authors/clusters).",
            "Use dual windows (short and long) with explicit slide/tumble cadence; keep formulas and parameters versioned.",
            "Apply FDR-aware alert gating (score + significance) instead of score-only triggers.",
            "Use optional LLM entailment as bounded auxiliary feature, never as sole novelty detector.",
        ],
        "thresholds": [
            "Seed defaults from corpus: Δ=10–30s, τs=1–5m, τl=1–24h (tune on local data).",
            "Alert gate template: S > θ and p < α under calibrated FDR policy.",
            "Monitor FAR, precision/recall, and latency jointly; drift in one metric triggers recalibration.",
        ],
        "anti_patterns": [
            "Novelty judged only by LLM narrative quality with no statistical baseline.",
            "Using static thresholds indefinitely while stream dynamics drift.",
            "Ignoring author-diversity evidence (easy to game via coordinated bursts).",
        ],
        "prompt": "Score candidate cluster novelty against recent baseline. Return {novelty_score, uncertainty, burst_features, anchor_evidence, recommendation}. Abstain when evidence is contradictory.",
    },
    {
        "key": "evaluation_calibration_loop",
        "title": "Evaluation/Calibration Loop",
        "file": "playbook_evaluation_calibration_loop.md",
        "source_keywords": ["evaluation frameworks", "calibration", "quality"],
        "rules": [
            "Track quality with a fixed daily suite: Precision@k, novelty hit-rate, miss-rate, false alarms, time-to-detect.",
            "Run daily calibration: inspect misses/false alarms, retune thresholds, and log rationale.",
            "Use operator-centric telemetry (accept/override rate, triage time) alongside model metrics.",
            "Trigger blameless postmortems for silent failures and major drift events.",
        ],
        "thresholds": [
            "Define explicit SLOs for shortlist precision and time-to-detect (median + p90).",
            "Set escalation triggers (e.g., precision drop, miss-rate spike, latency regression) to force review.",
            "Keep evaluation labels and protocol versioned for reproducible week-over-week comparisons.",
        ],
        "anti_patterns": [
            "Optimizing only one metric (e.g., precision) while missing timeliness/coverage collapse.",
            "Unversioned threshold changes with no change log.",
            "Manual QA notes not linked back to run IDs and source evidence.",
        ],
        "prompt": "Evaluate daily digest output vs labeled truth. Produce metric table, top misses, top false alarms, and threshold-change suggestions with expected trade-offs.",
    },
    {
        "key": "digest_30_second_ux",
        "title": "30-Second Digest UX Format",
        "file": "playbook_30_second_digest_ux.md",
        "source_keywords": ["30-second", "digest ux", "scannability"],
        "rules": [
            "Digest lead must answer in order: what changed, why it matters, what to do next.",
            "Constrain output to scan-first micro-structure (micro-headline + short bullets + single action).",
            "Default to impact/outcome language; defer implementation detail via links or expand sections.",
            "Include one anchoring number/date only when it improves decision confidence.",
        ],
        "thresholds": [
            "Target ~120 words total for 30-second read-time envelope.",
            "Micro-headline ≤60 characters where feasible.",
            "Action section should be executable in one sentence with owner/time hint.",
        ],
        "anti_patterns": [
            "Long paragraph summaries that hide the decision/action.",
            "Multiple competing CTAs in a short digest surface.",
            "Jargon-heavy implementation dumps in primary digest card.",
        ],
        "prompt": "Write a 30-second digest from structured trend inputs. Format: [What changed] [Why it matters] [Action now] [1 proof point]. Keep under 120 words.",
    },
]

CONTRADICTION_FLAGS = [
    {
        "flag": "Precision vs timeliness",
        "detail": "Higher verification and stricter confidence gates improve precision but can delay first detection.",
        "mitigation": "Run dual lanes (fast provisional + verified operator) and report latency/quality separately.",
    },
    {
        "flag": "Cost control vs context completeness",
        "detail": "Aggressive chunking/token caps reduce spend but may hide cross-window dependencies.",
        "mitigation": "Use hierarchical summarization and periodic long-context audits on sampled runs.",
    },
    {
        "flag": "Deterministic constraints vs expressive reasoning",
        "detail": "Strict bounded rerank schema prevents invention but can suppress nuanced free-form explanations.",
        "mitigation": "Keep strict ID output channel; attach optional rationale channel with citation validator.",
    },
    {
        "flag": "Novelty sensitivity vs false-alarm risk",
        "detail": "Lower novelty thresholds catch early tails but can over-alert on noise/manipulation bursts.",
        "mitigation": "Use FDR-aware gates + diversity/anchor corroboration before notification.",
    },
]

PRIORITY_TIERS = {
    "playbook_cluster_event_detection": "P0",
    "playbook_entity_anchor_grounding": "P0",
    "playbook_llm_bounded_rerank_guardrails": "P0",
    "playbook_novelty_scoring": "P1",
    "playbook_evaluation_calibration_loop": "P1",
    "playbook_30_second_digest_ux": "P2",
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def load_registry() -> Dict[str, Any]:
    return read_json(REGISTRY_PATH, default={}) or {}


def save_registry(registry: Dict[str, Any]) -> None:
    write_json(REGISTRY_PATH, registry)


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def latest_run_id() -> Optional[str]:
    if not RUNS_DIR.exists():
        return None
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name


def resolve_run_id(args_run_id: Optional[str]) -> str:
    if args_run_id:
        return args_run_id
    rid = latest_run_id()
    if not rid:
        raise SystemExit("No previous run found. Provide --run-id.")
    return rid


def sanitize_for_search(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def discover_pdfs(inbound_dirs: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for root in inbound_dirs:
        if not root.exists():
            continue
        for pdf_path in sorted(root.rglob("*.pdf")):
            out.append(
                {
                    "pdf_path": str(pdf_path.resolve()),
                    "title_hint": pdf_path.stem,
                    "source": f"inbound:{root}",
                }
            )
    return out


def load_manifest_entries(manifest_paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for mp in manifest_paths:
        data = read_json(mp, default=[])
        if not isinstance(data, list):
            continue
        for row in data:
            if not isinstance(row, dict):
                continue
            if "pdf" not in row:
                continue
            out.append(
                {
                    "pdf_path": row.get("pdf"),
                    "title_hint": row.get("title_hint") or Path(str(row.get("pdf"))).stem,
                    "text_hint": row.get("text_path"),
                    "source": f"manifest:{mp}",
                    "manifest_row": row,
                }
            )
    return out


def infer_text_hint(pdf_path: Path) -> Optional[str]:
    txt_sibling = pdf_path.with_suffix(".txt")
    if txt_sibling.exists():
        return str(txt_sibling.resolve())
    return None


def stage_ingest(args: argparse.Namespace) -> None:
    ensure_dir(PIPELINE_ROOT)
    ensure_dir(STATE_DIR)
    ensure_dir(RUNS_DIR)

    rid = args.run_id or dt.datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
    rdir = run_dir(rid)
    idir = rdir / "ingest"
    ensure_dir(idir)

    inbound_dirs = [Path(p).expanduser().resolve() for p in (args.inbound_dir or [])]
    manifest_paths = [Path(p).expanduser().resolve() for p in (args.manifest_json or [])]

    discovered = discover_pdfs(inbound_dirs)
    discovered.extend(load_manifest_entries(manifest_paths))

    if not discovered:
        raise SystemExit("No PDFs discovered. Provide --inbound-dir and/or --manifest-json.")

    registry = load_registry()
    seen_sha: Dict[str, Dict[str, Any]] = {}
    items: List[Dict[str, Any]] = []

    for row in discovered:
        pdf_path = Path(str(row.get("pdf_path"))).expanduser().resolve()
        base_item: Dict[str, Any] = {
            "pdf_path": str(pdf_path),
            "title_hint": row.get("title_hint"),
            "source": row.get("source"),
            "discovered_at": now_utc(),
        }

        if not pdf_path.exists():
            base_item.update({"status": "missing_pdf", "error": "file_not_found"})
            items.append(base_item)
            continue

        sha = compute_sha256(pdf_path)
        if sha in seen_sha:
            base_item.update(
                {
                    "status": "duplicate_in_run",
                    "sha256": sha,
                    "duplicate_of": seen_sha[sha]["pdf_path"],
                }
            )
            items.append(base_item)
            continue

        text_hint = row.get("text_hint")
        if not text_hint:
            text_hint = infer_text_hint(pdf_path)

        status = "duplicate_known" if sha in registry else "new"
        base_item.update(
            {
                "status": status,
                "sha256": sha,
                "size_bytes": pdf_path.stat().st_size,
                "mtime": dt.datetime.fromtimestamp(pdf_path.stat().st_mtime, dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "text_hint": text_hint,
            }
        )
        seen_sha[sha] = base_item
        items.append(base_item)

    manifest = {
        "run_id": rid,
        "stage": "ingest",
        "created_at": now_utc(),
        "inputs": {
            "inbound_dirs": [str(p) for p in inbound_dirs],
            "manifest_json": [str(p) for p in manifest_paths],
        },
        "counts": {
            "total": len(items),
            "new": sum(1 for i in items if i.get("status") == "new"),
            "duplicate_known": sum(1 for i in items if i.get("status") == "duplicate_known"),
            "duplicate_in_run": sum(1 for i in items if i.get("status") == "duplicate_in_run"),
            "missing_pdf": sum(1 for i in items if i.get("status") == "missing_pdf"),
        },
        "items": items,
    }
    out_json = idir / "ingest_manifest.json"
    write_json(out_json, manifest)

    md_lines = [
        f"# Ingest Manifest — {rid}",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Total discovered: {manifest['counts']['total']}",
        f"- New: {manifest['counts']['new']}",
        f"- Duplicate known (sha256): {manifest['counts']['duplicate_known']}",
        f"- Duplicate in run (sha256): {manifest['counts']['duplicate_in_run']}",
        f"- Missing PDFs: {manifest['counts']['missing_pdf']}",
        "",
        "## Items",
        "",
    ]
    for it in items:
        md_lines.append(f"- `{it.get('status')}` `{it.get('sha256', '-')}` {it.get('pdf_path')}")
    write_text(idir / "ingest_manifest.md", "\n".join(md_lines) + "\n")


def extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pypdf unavailable. Use a python env with pypdf installed or provide pre-extracted text hints."
        ) from exc

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(parts)


def stage_extract(args: argparse.Namespace) -> None:
    rid = resolve_run_id(args.run_id)
    rdir = run_dir(rid)
    ingest_path = rdir / "ingest" / "ingest_manifest.json"
    ingest = read_json(ingest_path)
    if not ingest:
        raise SystemExit(f"Missing ingest manifest: {ingest_path}")

    edir = rdir / "extract"
    text_dir = edir / "text"
    ensure_dir(text_dir)

    include_duplicates = bool(args.include_duplicates)
    out_items: List[Dict[str, Any]] = []

    for item in ingest.get("items", []):
        status = item.get("status")
        sha = item.get("sha256")
        pdf_path = Path(item.get("pdf_path", ""))

        rec: Dict[str, Any] = {
            "pdf_path": item.get("pdf_path"),
            "sha256": sha,
            "ingest_status": status,
            "title_hint": item.get("title_hint"),
            "source": item.get("source"),
            "text_hint": item.get("text_hint"),
        }

        if status == "missing_pdf":
            rec.update({"status": "skipped_missing_pdf"})
            out_items.append(rec)
            continue

        if status in {"duplicate_known", "duplicate_in_run"} and not include_duplicates:
            rec.update({"status": "skipped_duplicate"})
            out_items.append(rec)
            continue

        text: Optional[str] = None
        extractor = None

        text_hint = item.get("text_hint")
        if text_hint and Path(text_hint).exists():
            text = Path(text_hint).read_text(encoding="utf-8", errors="ignore")
            extractor = "pre_extracted_text"

        if text is None:
            if not pdf_path.exists():
                rec.update({"status": "error", "error": "pdf_missing_at_extract"})
                out_items.append(rec)
                continue
            try:
                text = extract_with_pypdf(pdf_path)
                extractor = "pypdf"
            except Exception as exc:
                rec.update({"status": "error", "error": str(exc)})
                out_items.append(rec)
                continue

        text = normalize_whitespace(text)
        out_path = text_dir / f"{sha}.txt"
        write_text(out_path, text)

        rec.update(
            {
                "status": "extracted",
                "extractor": extractor,
                "chars": len(text),
                "words": len(text.split()),
                "text_path": str(out_path),
                "citations": [
                    {
                        "sha256": sha,
                        "pdf_path": item.get("pdf_path"),
                        "text_source": item.get("text_hint"),
                        "extractor": extractor,
                    }
                ],
            }
        )
        out_items.append(rec)

    report = {
        "run_id": rid,
        "stage": "extract",
        "created_at": now_utc(),
        "counts": {
            "total": len(out_items),
            "extracted": sum(1 for i in out_items if i.get("status") == "extracted"),
            "skipped_duplicate": sum(1 for i in out_items if i.get("status") == "skipped_duplicate"),
            "errors": sum(1 for i in out_items if i.get("status") == "error"),
        },
        "items": out_items,
    }
    write_json(edir / "extract_report.json", report)


def stage_index(args: argparse.Namespace) -> None:
    rid = resolve_run_id(args.run_id)
    rdir = run_dir(rid)
    ingest = read_json(rdir / "ingest" / "ingest_manifest.json")
    extract = read_json(rdir / "extract" / "extract_report.json")
    if not ingest:
        raise SystemExit("Run missing ingest stage output.")
    if not extract:
        raise SystemExit("Run missing extract stage output.")

    def status_rank(status: Optional[str]) -> int:
        if status == "extracted":
            return 3
        if status == "skipped_duplicate":
            return 2
        if status:
            return 1
        return 0

    # Keep best extract record per sha (prefer extracted over skipped/error rows).
    ex_by_sha: Dict[str, Dict[str, Any]] = {}
    for item in extract.get("items", []):
        sha = item.get("sha256")
        if not sha:
            continue
        prev = ex_by_sha.get(sha)
        if prev is None or status_rank(item.get("status")) > status_rank(prev.get("status")):
            ex_by_sha[sha] = item

    # Keep most informative ingest row per sha.
    ingest_priority = {"new": 3, "duplicate_known": 2, "duplicate_in_run": 1, "missing_pdf": 0}
    ingest_by_sha: Dict[str, Dict[str, Any]] = {}
    for item in ingest.get("items", []):
        sha = item.get("sha256")
        if not sha:
            continue
        prev = ingest_by_sha.get(sha)
        if prev is None or ingest_priority.get(item.get("status"), 0) > ingest_priority.get(prev.get("status"), 0):
            ingest_by_sha[sha] = item

    entries: List[Dict[str, Any]] = []
    registry = load_registry()
    include_registry_fallback = bool(getattr(args, "full_corpus", False))
    registry_fallback_count = 0

    for sha, ing in ingest_by_sha.items():
        ex = ex_by_sha.get(sha)
        if ex and ex.get("status") == "extracted":
            entries.append(
                {
                    "run_id": rid,
                    "sha256": sha,
                    "title_hint": ing.get("title_hint"),
                    "pdf_path": ing.get("pdf_path"),
                    "text_path": ex.get("text_path"),
                    "chars": ex.get("chars"),
                    "words": ex.get("words"),
                    "extractor": ex.get("extractor"),
                    "source": ing.get("source"),
                    "created_at": now_utc(),
                    "citations": ex.get("citations", []),
                }
            )
            continue

        if not include_registry_fallback:
            continue

        rec = registry.get(sha)
        if not isinstance(rec, dict):
            continue

        text_paths = [p for p in rec.get("text_paths", []) if isinstance(p, str)]
        text_path = next((p for p in reversed(text_paths) if Path(p).exists()), None)
        if not text_path:
            continue

        text_blob = Path(text_path).read_text(encoding="utf-8", errors="ignore")
        title_hints = [t for t in rec.get("title_hints", []) if isinstance(t, str) and t.strip()]
        pdf_paths = [p for p in rec.get("pdf_paths", []) if isinstance(p, str) and p.strip()]

        entries.append(
            {
                "run_id": rid,
                "sha256": sha,
                "title_hint": ing.get("title_hint") or (title_hints[0] if title_hints else None),
                "pdf_path": ing.get("pdf_path") or (pdf_paths[0] if pdf_paths else None),
                "text_path": text_path,
                "chars": len(text_blob),
                "words": len(text_blob.split()),
                "extractor": "registry_reuse",
                "source": ing.get("source") or "registry_fallback",
                "created_at": now_utc(),
                "citations": [
                    {
                        "sha256": sha,
                        "pdf_path": ing.get("pdf_path") or (pdf_paths[0] if pdf_paths else None),
                        "text_source": text_path,
                        "extractor": "registry_reuse",
                    }
                ],
            }
        )
        registry_fallback_count += 1

    # Ensure unique by sha within run. Prefer freshly extracted records over registry fallback rows.
    unique_entries: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        prev = unique_entries.get(e["sha256"])
        if prev is None:
            unique_entries[e["sha256"]] = e
            continue
        prev_is_registry = prev.get("extractor") == "registry_reuse"
        cur_is_registry = e.get("extractor") == "registry_reuse"
        if prev_is_registry and not cur_is_registry:
            unique_entries[e["sha256"]] = e
    entries = list(unique_entries.values())

    idir = rdir / "index"
    ensure_dir(idir)
    manifest = {
        "run_id": rid,
        "stage": "index",
        "created_at": now_utc(),
        "entry_count": len(entries),
        "registry_fallback_count": registry_fallback_count,
        "entries": sorted(entries, key=lambda x: x.get("title_hint", "")),
    }
    write_json(idir / "manifest.json", manifest)

    # update global registry (dedupe by sha256)
    for e in entries:
        sha = e["sha256"]
        rec = registry.get(sha, {})
        rec.setdefault("sha256", sha)
        rec.setdefault("first_seen_at", now_utc())
        rec.setdefault("first_seen_run", rid)
        rec["last_seen_at"] = now_utc()
        rec["last_seen_run"] = rid
        rec.setdefault("pdf_paths", [])
        rec.setdefault("text_paths", [])
        rec.setdefault("title_hints", [])
        rec.setdefault("runs", [])
        if e["pdf_path"] and e["pdf_path"] not in rec["pdf_paths"]:
            rec["pdf_paths"].append(e["pdf_path"])
        if e["text_path"] and e["text_path"] not in rec["text_paths"]:
            rec["text_paths"].append(e["text_path"])
        th = e.get("title_hint")
        if th and th not in rec["title_hints"]:
            rec["title_hints"].append(th)
        if rid not in rec["runs"]:
            rec["runs"].append(rid)
        registry[sha] = rec
    save_registry(registry)

    md_lines = [
        f"# Run Manifest — {rid}",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Indexed entries: {manifest['entry_count']}",
        f"- Registry fallback entries: {manifest['registry_fallback_count']}",
        "",
        "## Sources",
        "",
    ]
    for idx, e in enumerate(manifest["entries"], 1):
        md_lines.append(f"{idx}. `{e['sha256']}` — {e.get('title_hint')}")
        md_lines.append(f"   - pdf: `{e.get('pdf_path')}`")
        md_lines.append(f"   - text: `{e.get('text_path')}`")
    write_text(idir / "manifest.md", "\n".join(md_lines) + "\n")


def build_source_catalog(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(entries, 1):
        source_id = f"S{i:02d}"
        out.append(
            {
                "source_id": source_id,
                "sha256": e.get("sha256"),
                "title_hint": e.get("title_hint") or "untitled",
                "pdf_path": e.get("pdf_path"),
                "text_path": e.get("text_path"),
            }
        )
    return out


def sources_for_module(catalog: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for src in catalog:
        title = sanitize_for_search(src.get("title_hint", ""))
        if any(k in title for k in [sanitize_for_search(x) for x in keywords]):
            matched.append(src)
    if not matched:
        # fallback: cite first two docs so citation channel is never empty
        matched = catalog[:2]
    return matched


def render_playbook(
    run_id: str,
    spec: Dict[str, Any],
    module_sources: List[Dict[str, Any]],
) -> str:
    src_ids = [s["source_id"] for s in module_sources]
    src_suffix = " ".join(f"[{sid}]" for sid in src_ids)

    lines: List[str] = [
        f"# {spec['title']} — Distilled Playbook",
        "",
        f"- Run ID: `{run_id}`",
        f"- Module key: `{spec['key']}`",
        f"- Updated: {now_utc()}",
        "",
        "## Key rules",
    ]
    for rule in spec["rules"]:
        lines.append(f"- {rule} {src_suffix}")

    lines.extend(["", "## Thresholds and defaults"])
    for t in spec["thresholds"]:
        lines.append(f"- {t} {src_suffix}")

    lines.extend(["", "## Anti-patterns"])
    for ap in spec["anti_patterns"]:
        lines.append(f"- {ap} {src_suffix}")

    lines.extend(
        [
            "",
            "## Prompt starter",
            "```text",
            spec["prompt"],
            "```",
            "",
            "## Source citations",
        ]
    )
    for src in module_sources:
        lines.append(
            f"- [{src['source_id']}] `{src['sha256']}` — {src['title_hint']}"
            f"\n  - pdf: `{src['pdf_path']}`"
            f"\n  - text: `{src['text_path']}`"
        )

    return "\n".join(lines) + "\n"


def stage_distill(args: argparse.Namespace) -> None:
    rid = resolve_run_id(args.run_id)
    rdir = run_dir(rid)
    manifest = read_json(rdir / "index" / "manifest.json")
    if not manifest:
        raise SystemExit("Run missing index stage output.")

    entries = manifest.get("entries", [])
    if not entries:
        raise SystemExit("Index manifest has no entries.")

    ddir = rdir / "distill"
    run_playbooks = ddir / "playbooks"
    global_playbooks = PLAYBOOK_ROOT / rid
    ensure_dir(run_playbooks)
    ensure_dir(global_playbooks)

    catalog = build_source_catalog(entries)
    write_json(ddir / "source_catalog.json", catalog)

    index_lines = [
        f"# Distilled Playbooks — {rid}",
        "",
        f"Generated: {now_utc()}",
        "",
        "## Modules",
        "",
    ]

    for spec in MODULE_SPECS:
        m_sources = sources_for_module(catalog, spec["source_keywords"])
        content = render_playbook(rid, spec, m_sources)
        out_run = run_playbooks / spec["file"]
        out_global = global_playbooks / spec["file"]
        write_text(out_run, content)
        write_text(out_global, content)
        index_lines.append(f"- [{spec['title']}]({spec['file']})")

    contradiction_lines = [
        "# Contradiction Flag List",
        "",
        f"Run: `{rid}`",
        "",
        "Use this list during approval/evaluation to check trade-off breakage.",
        "",
    ]
    for cf in CONTRADICTION_FLAGS:
        contradiction_lines.append(f"## {cf['flag']}")
        contradiction_lines.append(f"- Tension: {cf['detail']}")
        contradiction_lines.append(f"- Mitigation: {cf['mitigation']}")
        contradiction_lines.append("")

    write_text(run_playbooks / "CONTRADICTION_FLAGS.md", "\n".join(contradiction_lines))
    write_text(global_playbooks / "CONTRADICTION_FLAGS.md", "\n".join(contradiction_lines))
    write_text(run_playbooks / "INDEX.md", "\n".join(index_lines) + "\n")
    write_text(global_playbooks / "INDEX.md", "\n".join(index_lines) + "\n")


def propose_patch(old_path: Path, new_content: str) -> str:
    old_lines = []
    if old_path.exists():
        old_lines = old_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=str(old_path),
        tofile=str(old_path),
        n=3,
    )
    return "".join(diff)


def stage_propose(args: argparse.Namespace) -> None:
    rid = resolve_run_id(args.run_id)
    rdir = run_dir(rid)
    playbook_dir = PLAYBOOK_ROOT / rid
    if not playbook_dir.exists():
        raise SystemExit(f"Missing distilled playbooks dir: {playbook_dir}")

    target_dir = Path(args.target_dir).expanduser().resolve() if args.target_dir else TARGET_SKILL_ROOT
    pdir_run = rdir / "propose"
    pdir_global = PROPOSAL_ROOT / rid
    patch_dir_run = pdir_run / "patches"
    patch_dir_global = pdir_global / "patches"
    ensure_dir(patch_dir_run)
    ensure_dir(patch_dir_global)

    analysis_refs = None
    if getattr(args, "analysis_refs_json", None):
        analysis_path = Path(args.analysis_refs_json).expanduser().resolve()
        if analysis_path.exists():
            analysis_refs = read_json(analysis_path)

    proposal_items: List[Dict[str, Any]] = []
    for playbook in sorted(playbook_dir.glob("playbook_*.md")):
        content = playbook.read_text(encoding="utf-8")
        target_path = target_dir / playbook.name
        patch = propose_patch(target_path, content)

        patch_name = playbook.stem + ".patch"
        write_text(patch_dir_run / patch_name, patch)
        write_text(patch_dir_global / patch_name, patch)

        proposed_copy = pdir_global / "proposed_files" / playbook.name
        write_text(proposed_copy, content)

        proposal_items.append(
            {
                "module": playbook.stem,
                "priority_tier": PRIORITY_TIERS.get(playbook.stem, "P2"),
                "playbook_source": str(playbook),
                "target_path": str(target_path),
                "patch_path": str((patch_dir_global / patch_name)),
                "proposed_file": str(proposed_copy),
            }
        )

    proposal_items.sort(key=lambda x: (x.get("priority_tier", "P9"), x.get("module", "")))

    proposal_manifest = {
        "run_id": rid,
        "created_at": now_utc(),
        "target_dir": str(target_dir),
        "analysis_refs_json": str(Path(args.analysis_refs_json).expanduser().resolve()) if getattr(args, "analysis_refs_json", None) else None,
        "items": proposal_items,
    }
    write_json(pdir_run / "proposal_manifest.json", proposal_manifest)
    write_json(pdir_global / "proposal_manifest.json", proposal_manifest)

    plan_lines = [
        f"# Proposed Skill Update Plan — {rid}",
        "",
        "## Scope",
        "- Convert distilled research into modular, reviewable playbook updates.",
        "- Preserve source-citation traceability from PDF/text artifacts.",
        "- Apply only after explicit manual approval gate.",
        "",
        "## Priority tiers",
        "- **P0 (core detection integrity):** cluster/event detection, entity grounding, bounded rerank guardrails.",
        "- **P1 (quality and stability):** novelty scoring and evaluation/calibration loop.",
        "- **P2 (operator presentation):** 30-second digest UX format.",
        "",
        "## Proposed files (diff-ready)",
    ]
    for it in proposal_items:
        plan_lines.append(f"- `{it['target_path']}`")
        plan_lines.append(f"  - tier: `{it.get('priority_tier', 'P2')}`")
        plan_lines.append(f"  - patch: `{it['patch_path']}`")
        plan_lines.append(f"  - proposed: `{it['proposed_file']}`")

    if isinstance(analysis_refs, dict):
        refs = analysis_refs.get("analysis_refs", [])
        plan_lines.extend([
            "",
            "## MEMORY.md analysis context",
            f"- Analysis refs included: {len(refs)}",
            f"- Source file: `{proposal_manifest.get('analysis_refs_json')}`",
        ])

    plan_lines.extend(
        [
            "",
            "## Approval gate (manual)",
            "- Reviewer: `TBD`",
            "- APPROVED: `NO` (set to `YES` to apply)",
            "- Notes: `TBD`",
            "",
            "## Safety/quality gates",
            "- [x] Dedupe by sha256 handled at ingest/index.",
            "- [x] Source citation retention present in each playbook.",
            "- [x] Contradiction flags generated in playbook bundle.",
            "- [x] Rollback notes generated before apply.",
            "",
            "## Rollback notes",
            "If apply stage is executed, backups are written under:",
            f"`{rdir / 'approve' / 'rollback'}`",
            "",
            "Rollback command pattern:",
            "```bash",
            "# restore one file",
            "cp <rollback_file> <target_file>",
            "```",
        ]
    )

    write_text(pdir_global / "skill_update_plan.md", "\n".join(plan_lines) + "\n")
    write_text(pdir_run / "skill_update_plan.md", "\n".join(plan_lines) + "\n")

    rollback_notes = [
        f"# Rollback Notes — {rid}",
        "",
        "These notes apply after `approve --apply` runs.",
        "",
        "1) Inspect apply report:",
        f"   - `{rdir / 'approve' / 'apply_report.json'}`",
        "2) Restore from backups if needed:",
        f"   - rollback dir: `{rdir / 'approve' / 'rollback'}`",
        "3) Re-run evaluation:",
        f"   - `python3 scripts/pdf_skill_pipeline.py evaluate --run-id {rid}`",
    ]
    write_text(pdir_global / "rollback_notes.md", "\n".join(rollback_notes) + "\n")
    write_text(pdir_run / "rollback_notes.md", "\n".join(rollback_notes) + "\n")


def parse_approval(decision_path: Path) -> Tuple[bool, Dict[str, str]]:
    meta = {"reviewer": "", "approved": "", "notes": ""}
    raw = decision_path.read_text(encoding="utf-8", errors="ignore")
    for line in raw.splitlines():
        if line.lower().startswith("reviewer:"):
            meta["reviewer"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("approved:"):
            meta["approved"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("notes:"):
            meta["notes"] = line.split(":", 1)[1].strip()
    approved = meta["approved"].strip().lower() in {"yes", "y", "approved", "true"}
    return approved, meta


def stage_approve(args: argparse.Namespace) -> None:
    rid = resolve_run_id(args.run_id)
    rdir = run_dir(rid)
    adir = rdir / "approve"
    ensure_dir(adir)

    proposal_manifest = read_json(PROPOSAL_ROOT / rid / "proposal_manifest.json")
    if not proposal_manifest:
        raise SystemExit("Missing proposal manifest. Run propose stage first.")

    template = adir / "APPROVAL.md"
    if not template.exists():
        write_text(
            template,
            "\n".join(
                [
                    f"# Approval Record — {rid}",
                    "",
                    "Reviewer: TBD",
                    "Approved: NO",
                    "Notes: TBD",
                    "",
                    "Set `Approved: YES` to allow apply.",
                ]
            )
            + "\n",
        )

    if not args.apply:
        return

    decision_path = Path(args.decision_file).expanduser().resolve() if args.decision_file else template
    if not decision_path.exists():
        raise SystemExit(f"Decision file not found: {decision_path}")

    approved, meta = parse_approval(decision_path)
    if not approved:
        raise SystemExit("Approval gate not satisfied. Set `Approved: YES` in decision file.")

    rollback_dir = adir / "rollback"
    ensure_dir(rollback_dir)

    applied: List[Dict[str, Any]] = []
    for item in proposal_manifest.get("items", []):
        target_path = Path(item["target_path"]).expanduser().resolve()
        proposed_file = Path(item["proposed_file"]).expanduser().resolve()
        ensure_dir(target_path.parent)

        rollback_path = None
        if target_path.exists():
            rollback_path = rollback_dir / target_path.name
            shutil.copy2(target_path, rollback_path)

        shutil.copy2(proposed_file, target_path)
        applied.append(
            {
                "target_path": str(target_path),
                "proposed_file": str(proposed_file),
                "rollback_path": str(rollback_path) if rollback_path else None,
            }
        )

    report = {
        "run_id": rid,
        "created_at": now_utc(),
        "reviewer": meta.get("reviewer"),
        "notes": meta.get("notes"),
        "decision_file": str(decision_path),
        "applied_count": len(applied),
        "applied": applied,
    }
    write_json(adir / "apply_report.json", report)

    rb_lines = [
        f"# Apply Rollback Instructions — {rid}",
        "",
        "If you need to revert applied changes:",
    ]
    for row in applied:
        if row.get("rollback_path"):
            rb_lines.append(f"- `cp {row['rollback_path']} {row['target_path']}`")
        else:
            rb_lines.append(f"- `rm -f {row['target_path']}`  # file was newly created")
    write_text(adir / "rollback_instructions.md", "\n".join(rb_lines) + "\n")


def check_playbook_quality(path: Path) -> List[str]:
    issues: List[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    if "## Source citations" not in text:
        issues.append("missing_source_citations_section")
    if text.count("- [S") < 1:
        issues.append("missing_citation_entries")
    for section in ["## Key rules", "## Thresholds and defaults", "## Anti-patterns", "## Prompt starter"]:
        if section not in text:
            issues.append(f"missing_section:{section}")
    return issues


def stage_evaluate(args: argparse.Namespace) -> None:
    rid = resolve_run_id(args.run_id)
    rdir = run_dir(rid)

    results: List[Tuple[str, str, str]] = []

    ingest = read_json(rdir / "ingest" / "ingest_manifest.json")
    extract = read_json(rdir / "extract" / "extract_report.json")
    index = read_json(rdir / "index" / "manifest.json")
    plan = (PROPOSAL_ROOT / rid / "skill_update_plan.md").exists()
    contradiction = (PLAYBOOK_ROOT / rid / "CONTRADICTION_FLAGS.md").exists()

    results.append(("ingest_manifest", "PASS" if bool(ingest) else "FAIL", "exists" if ingest else "missing"))
    results.append(("extract_report", "PASS" if bool(extract) else "FAIL", "exists" if extract else "missing"))
    results.append(("index_manifest", "PASS" if bool(index) else "FAIL", "exists" if index else "missing"))
    results.append(("skill_update_plan", "PASS" if plan else "FAIL", "found" if plan else "missing"))
    results.append(("contradiction_flags", "PASS" if contradiction else "FAIL", "found" if contradiction else "missing"))

    if index:
        entries = index.get("entries", [])
        shas = [e.get("sha256") for e in entries]
        unique_ok = len(shas) == len(set(shas))
        results.append(("sha256_dedupe", "PASS" if unique_ok else "FAIL", f"{len(set(shas))}/{len(shas)} unique"))

    playbook_dir = PLAYBOOK_ROOT / rid
    expected_files = [spec["file"] for spec in MODULE_SPECS]
    for fn in expected_files:
        fp = playbook_dir / fn
        if not fp.exists():
            results.append((f"playbook:{fn}", "FAIL", "missing"))
            continue
        issues = check_playbook_quality(fp)
        results.append((f"playbook:{fn}", "PASS" if not issues else "WARN", "ok" if not issues else ",".join(issues)))

    eval_dir = rdir / "evaluate"
    ensure_dir(eval_dir)
    report_json = {
        "run_id": rid,
        "created_at": now_utc(),
        "checks": [{"name": n, "status": s, "detail": d} for n, s, d in results],
        "summary": {
            "pass": sum(1 for _, s, _ in results if s == "PASS"),
            "warn": sum(1 for _, s, _ in results if s == "WARN"),
            "fail": sum(1 for _, s, _ in results if s == "FAIL"),
        },
    }
    write_json(eval_dir / "evaluation_report.json", report_json)

    md_lines = [f"# Evaluation Report — {rid}", "", f"Generated: {report_json['created_at']}", "", "## Checks", ""]
    for n, s, d in results:
        md_lines.append(f"- **{n}**: `{s}` — {d}")

    md_lines.extend(
        [
            "",
            "## Summary",
            f"- PASS: {report_json['summary']['pass']}",
            f"- WARN: {report_json['summary']['warn']}",
            f"- FAIL: {report_json['summary']['fail']}",
        ]
    )
    write_text(eval_dir / "evaluation_report.md", "\n".join(md_lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PDF -> knowledge -> skill pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="detect new PDFs and dedupe by sha256")
    p_ingest.add_argument("--run-id", default=None)
    p_ingest.add_argument("--inbound-dir", action="append", help="Directory to scan recursively for PDFs")
    p_ingest.add_argument("--manifest-json", action="append", help="Optional JSON manifest with pdf/text metadata")
    p_ingest.set_defaults(func=stage_ingest)

    p_extract = sub.add_parser("extract", help="extract text")
    p_extract.add_argument("--run-id", default=None)
    p_extract.add_argument("--include-duplicates", action="store_true", help="extract even if duplicate-known")
    p_extract.set_defaults(func=stage_extract)

    p_index = sub.add_parser("index", help="build run manifest and update registry")
    p_index.add_argument("--run-id", default=None)
    p_index.add_argument(
        "--full-corpus",
        action="store_true",
        help="include registry-backed entries when extract skipped duplicate-known PDFs",
    )
    p_index.set_defaults(func=stage_index)

    p_distill = sub.add_parser("distill", help="create modular distilled playbooks")
    p_distill.add_argument("--run-id", default=None)
    p_distill.set_defaults(func=stage_distill)

    p_propose = sub.add_parser("propose", help="generate diff-ready update proposal patches")
    p_propose.add_argument("--run-id", default=None)
    p_propose.add_argument("--target-dir", default=str(TARGET_SKILL_ROOT))
    p_propose.add_argument("--analysis-refs-json", default=None, help="Optional JSON listing analysis refs (e.g., from MEMORY.md)")
    p_propose.set_defaults(func=stage_propose)

    p_approve = sub.add_parser("approve", help="manual approval gate + optional apply")
    p_approve.add_argument("--run-id", default=None)
    p_approve.add_argument("--apply", action="store_true", help="apply proposed files after approval")
    p_approve.add_argument("--decision-file", default=None, help="Approval markdown file")
    p_approve.set_defaults(func=stage_approve)

    p_eval = sub.add_parser("evaluate", help="post-change quality checks")
    p_eval.add_argument("--run-id", default=None)
    p_eval.set_defaults(func=stage_evaluate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
