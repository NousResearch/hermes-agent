#!/usr/bin/env python3
"""
Research Paper Pre-processing Script — Phase 3
Offloads 4-API fetching + dedup + pre-scoring from LLM to deterministic Python.

Outputs structured JSON to stdout. The LLM cron receives only the filtered
candidate list (~10-15 papers) instead of raw API output, cutting token usage ~60%.

Usage:
  python3 research_paper_preprocess.py [--days 14] [--output scored.json]

Integration:
  Add as `script: research_paper_preprocess.py` on cron job 823708309a8e.
  Script stdout is injected as context for the LLM's scoring + cross-referencing pass.
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)

import json
import os
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import quote

# ── Config ──────────────────────────────────────────────────────────────────
ATOM_NS = "http://www.w3.org/2005/Atom"
CUTOFF_DAYS = int(os.environ.get("PAPER_CUTOFF_DAYS", "2"))
MAX_CANDIDATES = int(os.environ.get("PAPER_MAX_CANDIDATES", "30"))
OUTPUT_PATH = os.environ.get("PAPER_OUTPUT", "")
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".paper_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_SOURCE_REPO = SCRIPT_DIR.parent
REPO_ROOT = Path(os.environ.get(
    "KENSEI_REPO_ROOT",
    str(_SOURCE_REPO if (_SOURCE_REPO / "tools" / "cronjob_tools.py").is_file()
        else Path.home() / "repos" / "KenseiAgent"),
)).resolve()

_RAW_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")
_EXTERNAL_DIRECTIVE_PATTERNS = (
    re.compile(
        r"(?:^|[.!?]\s+)(?:please\s+)?(?:ignore|disregard|override)\s+"
        r"(?:(?:all|any|the|your|previous|prior|above)\s+){0,3}"
        r"(?:instructions?|rules?|guidelines?|(?:system|developer)\s+prompts?)\b",
        re.IGNORECASE,
    ),
)

MEMORY_GATE_ENABLED = os.environ.get("PAPER_MEMORY_GATE_ENABLED", "0") == "1"
MEMORY_GATE_MODE = os.environ.get("PAPER_MEMORY_GATE_MODE", "observe").lower()
MEMORY_GATE_TIMEOUT_MS = int(os.environ.get("PAPER_MEMORY_GATE_TIMEOUT_MS", "2500"))
WIKI_PATH = Path(os.environ.get("WIKI_PATH", str(Path.home() / "docs" / "wiki")))
TELEMETRY_PATH = Path(os.environ.get(
    "PAPER_MEMORY_GATE_TELEMETRY_PATH",
    str(Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        / "governance/telemetry/research-paper-memory-gate.jsonl"),
))

POLICY_VERSION = "1.0.0"
THRESHOLDS = {"relevance": .85, "coverage": .95, "freshness": .80,
              "confidence": .85, "overall": .88, "extraction": .90}
REASON_PRECEDENCE = (
    "RETRIEVAL_UNAVAILABLE", "RETRIEVAL_TIMEOUT", "REQUEST_AMBIGUOUS",
    "INTRINSIC_RESEARCH", "NO_EVIDENCE", "CONFLICTING_EVIDENCE",
    "STALE_EVIDENCE", "LOW_COVERAGE", "LOW_RELEVANCE", "LOW_CONFIDENCE",
    "BELOW_OVERALL_THRESHOLD", "SUFFICIENT",
)

# ── arXiv Category Queries ──────────────────────────────────────────────────
ARXIV_QUERIES = {
    "ai": "cat:cs.AI",
    "cl": "cat:cs.CL",
    "lg": "cat:cs.LG",
    "se": "cat:cs.SE",
    "hc": "cat:cs.HC",
    "kw1": 'all:"coding agent" OR all:"LLM agent" OR all:"MCP server" OR all:"tool calling" OR all:"context window"',
    "kw2": 'all:"agent memory" OR all:"prompt engineering" OR all:"agent orchestration" OR all:"AI workflow" OR all:"local LLM"',
}

# ── Tight Scoring Phrases (v2.1.1 — calibrated for precision) ──────────────
# Score 5: Direct stack match — exact phrases only
S5_PHRASES = [
    "mcp server", "mcp-style", "tool calling", "tool-augmented agent",
    "executable tool workflow", "tool workflow", "hyper tool",
    "context compression", "end-to-end context compression",
    "long-term agent memory", "agent memory", "graph memory",
    "selection integrity", "accumulability", "information-flow",
    "runtime enforcement", "runtime governance", "runtime memory poisoning",
    "shield synthesis", "defensibility analysis",
    "prompt injection", "red-teaming", "pi-hunter",
    "instructions-as-code", "instruction files on agentic",
    "recursive agent harness", "agent harness", "openclaw", "claw-swe",
    "delegation intelligence", "delegate intelligence",
    "multi-agent orchestration", "reward modeling for multi-agent",
    "skill self-evolution", "skill evolution", "skillcat",
    "agentic pull request", "agentic pr ", "agentic pull-request",
    "agent-native", "agent-native knowledge",
    "memory poisoning", "persistent llm agent",
    "compact agent", "inference-time evolution of executable tool",
]

# Score 4: Strong relevance, adjacent to stack
S4_PHRASES = [
    "rag ", "retrieval augmented", "fine-tuning", "prompt engineering",
    "code generation benchmark", "code review agent", "coding agent benchmark",
    "adversarial testing", "adversarial code", "adversarial",
    "llm evaluation", "llm benchmark", "agent benchmark",
    "ai-native software engineering", "ai workflow",
    "knowledge graph", "vector search", "reasoning enhanced",
    "tool-use", "tool use", "function calling",
    "agent framework", "agent platform", "agent system",
    "context management", "context retention",
    "code generation", "code synthesis",
]

# Score 3: Broader AI/ML relevance
S3_PHRASES = [
    "large language model", "transformer", "attention mechanism",
    "synthetic data", "distillation", "quantization",
    "reinforcement learning", "reasoning",
    "neural network", "deep learning",
    "natural language processing",
    "multi-modal", "multimodal",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_url(url: str, timeout: int = 15) -> str | None:
    """Fetch a URL with basic retry. Returns body text or None."""
    for attempt in range(2):
        try:
            req = Request(url, headers={"User-Agent": "KenseiResearch/1.0"})
            with urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, OSError) as e:
            if attempt == 0:
                time.sleep(2)
                continue
            print(f"  [WARN] Failed to fetch {url[:60]}: {e}", file=sys.stderr)
            return None
    return None


def parse_arxiv_id(text: str) -> str | None:
    """Extract arXiv ID from various formats."""
    text = text.strip()
    # Raw ID: "2606.00467" or "2606.00467v2"
    if text.replace(".", "").replace("v", "").isdigit() and len(text) >= 8:
        return text.split("v")[0]
    # URL: https://arxiv.org/abs/2606.00467
    if "/abs/" in text:
        return text.split("/abs/")[-1].split("v")[0]
    if "/pdf/" in text:
        return text.split("/pdf/")[-1].split(".pdf")[0].split("v")[0]
    return None


def _candidate_telemetry_id(raw_id: str, index: int) -> str:
    """Return only a validated raw arXiv ID or an opaque fallback label."""
    match = _RAW_ARXIV_ID_RE.fullmatch(raw_id.strip())
    return match.group(1) if match else f"candidate-{index}"


def _candidate_strings(candidate: dict) -> list[str]:
    """Collect nested external strings with title/summary fields kept adjacent."""
    values: list[str] = []
    preferred_keys = ("title", "summary", "abstract")

    def collect(value) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for key in preferred_keys:
                if key in value:
                    collect(value[key])
            for key in sorted(value):
                if key not in preferred_keys:
                    collect(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(candidate)
    return values


def tight_score(title: str, summary: str) -> int:
    """Two-pass scoring: exact phrases only. Returns 1-5."""
    text = (title + " " + summary).lower()

    # Pass 1: Score 5 — direct stack match
    for phrase in S5_PHRASES:
        if phrase in text:
            return 5

    # Pass 2: Score 4 — strong relevance
    for phrase in S4_PHRASES:
        if phrase in text:
            return 4

    # Pass 3: Score 3 — broader AI
    for phrase in S3_PHRASES:
        if phrase in text:
            return 3

    return 1


# ── Memory-first research gate ──────────────────────────────────────────────

def _overall(scores: dict) -> float:
    if any(scores.get(key, 0) <= 0 for key in ("relevance", "coverage", "freshness", "confidence")):
        return 0.0
    return 1 / (0.30 / scores["relevance"] + 0.35 / scores["coverage"]
                + 0.15 / scores["freshness"] + 0.20 / scores["confidence"])


def decide_memory_sufficiency(packet: dict) -> dict:
    """Apply the conservative policy to a normalised decision packet.

    This pure seam also accepts the frozen policy fixtures, keeping reason
    precedence independently testable from retrieval adapters.
    """
    retrieval = packet.get("retrieval", {})
    statuses = set(retrieval.values())
    scores = dict(packet.get("scores") or {})
    requirements = packet.get("requirements", [])
    conflicts = packet.get("conflicts", [])
    request_class = packet.get("request_class", "")
    extraction = packet.get("requirement_extraction_confidence", 1.0)

    if statuses & {"unavailable", "error", "malformed", "unauthorized"}:
        reason = "RETRIEVAL_UNAVAILABLE"
    elif "timeout" in statuses:
        reason = "RETRIEVAL_TIMEOUT"
    elif extraction < THRESHOLDS["extraction"] or "ambiguous" in request_class:
        reason = "REQUEST_AMBIGUOUS"
    elif any(token in request_class for token in ("landscape", "recommendation", "literature_review", "discovery")):
        reason = "INTRINSIC_RESEARCH"
    elif statuses and statuses <= {"valid_empty"}:
        reason = "NO_EVIDENCE"
    elif any(not conflict.get("resolved", False) for conflict in conflicts):
        reason = "CONFLICTING_EVIDENCE"
    elif scores.get("freshness", 1.0) < THRESHOLDS["freshness"]:
        reason = "STALE_EVIDENCE"
    elif (any(req.get("critical") and req.get("support", 0) < 1.0 for req in requirements)
          or scores.get("coverage", 1.0) < THRESHOLDS["coverage"]):
        reason = "LOW_COVERAGE"
    elif scores.get("relevance", 1.0) < THRESHOLDS["relevance"]:
        reason = "LOW_RELEVANCE"
    elif scores.get("confidence", 1.0) < THRESHOLDS["confidence"]:
        reason = "LOW_CONFIDENCE"
    elif scores.get("overall", _overall(scores)) < THRESHOLDS["overall"]:
        reason = "BELOW_OVERALL_THRESHOLD"
    else:
        reason = "SUFFICIENT"

    return {
        "action": "SKIP_DEEP_RESEARCH" if reason == "SUFFICIENT" else "ESCALATE",
        "reason_code": reason,
        "scores": scores or None,
    }


def _normalise_response(source: str, response) -> dict:
    if not isinstance(response, dict):
        return {"status": "malformed", "evidence": []}
    status = response.get("status", "ok")
    raw = response.get("evidence", response.get("results", response.get("memories", [])))
    if status not in {"ok", "valid_empty"} or not isinstance(raw, list):
        return {"status": status if status in {"timeout", "unavailable"} else "malformed", "evidence": []}
    evidence = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        normalised = dict(item)
        normalised.setdefault("id", f"{source}:{index}")
        normalised["source_store"] = source
        normalised.setdefault("requirement_ids", ["paper_synthesis"])
        for key in ("relevance", "support", "freshness", "confidence"):
            try:
                normalised[key] = max(0.0, min(1.0, float(normalised.get(key, 0))))
            except (TypeError, ValueError):
                normalised[key] = 0.0
        if not normalised.get("provenance"):
            normalised["confidence"] = 0.0
        evidence.append(normalised)
    return {"status": "ok" if evidence else "valid_empty", "evidence": evidence}


def lookup_mnemosyne(request: dict) -> dict:
    """Query a supported Mnemosyne adapter command using JSON over stdin/stdout."""
    command = os.environ.get("PAPER_MNEMOSYNE_COMMAND", "").strip()
    if not command:
        return {"status": "unavailable", "evidence": []}
    try:
        result = subprocess.run(
            shlex.split(command), input=json.dumps({"query": request["text"], "limit": 12}),
            text=True, capture_output=True, timeout=MEMORY_GATE_TIMEOUT_MS / 1000,
            check=False,
        )
        if result.returncode != 0:
            return {"status": "unavailable", "evidence": []}
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "evidence": []}
    except (OSError, json.JSONDecodeError):
        return {"status": "malformed", "evidence": []}


def lookup_wiki(request: dict, *, wiki_path: Path = WIKI_PATH) -> dict:
    """Bounded deterministic paper-level lookup with provenance retention."""
    if not wiki_path.is_dir():
        return {"status": "unavailable", "evidence": []}
    arxiv_id = request.get("arxiv_id", "")
    if not arxiv_id:
        return {"status": "valid_empty", "evidence": []}
    hits = []
    try:
        for path in sorted(wiki_path.rglob("*.md"))[:2000]:
            text = path.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            if arxiv_id not in text or not any(marker in lowered for marker in ("synthesi", "core insight", "what it proposes")):
                continue
            provenance = ""
            for line in text.splitlines()[:30]:
                if line.strip().startswith("sources:"):
                    provenance = line.split(":", 1)[1].strip().strip("[]").split(",")[0].strip()
                    break
            hits.append({
                "id": f"wiki:{path.relative_to(wiki_path)}", "source_store": "wiki",
                "requirement_ids": ["paper_synthesis"], "relevance": .96,
                "support": 1.0, "freshness": .95, "confidence": .90 if provenance else 0.0,
                "claim": "paper-level synthesis exists", "canonical_key": f"paper:{arxiv_id}:synthesis",
                "provenance": provenance, "path": str(path.relative_to(wiki_path)),
            })
            if len(hits) >= 8:
                break
    except OSError:
        return {"status": "unavailable", "evidence": []}
    return {"status": "ok" if hits else "valid_empty", "evidence": hits}


def memory_sufficient(request: dict, *, mnemosyne_lookup=lookup_mnemosyne,
                      wiki_lookup=lookup_wiki, now=None, deadline_ms=2500) -> dict:
    """Query both stores in parallel and fail open on every uncertain state."""
    started = time.monotonic()
    responses = {}
    latencies = {}

    def run(source, lookup):
        source_started = time.monotonic()
        try:
            response = _normalise_response(source, lookup(request))
        except Exception:  # adapters are an isolation boundary; never leak their payloads
            response = {"status": "unavailable", "evidence": []}
        latencies[source] = round((time.monotonic() - source_started) * 1000, 3)
        return response

    executor = ThreadPoolExecutor(max_workers=2)
    futures = {source: executor.submit(run, source, lookup) for source, lookup in
               (("mnemosyne", mnemosyne_lookup), ("wiki", wiki_lookup))}
    deadline = started + deadline_ms / 1000
    try:
        for source, future in futures.items():
            remaining = max(0, deadline - time.monotonic())
            try:
                responses[source] = future.result(timeout=remaining)
            except FuturesTimeout:
                responses[source] = {"status": "timeout", "evidence": []}
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    evidence = [item for source in ("mnemosyne", "wiki")
                for item in responses.get(source, {}).get("evidence", [])]
    evidence.sort(key=lambda item: (-item.get("relevance", 0), item.get("canonical_key", ""), item["id"]))
    requirements = request.get("requirements", [])
    weights = {req["id"]: req.get("weight", 0) for req in requirements}
    scores = {}
    for dimension in ("relevance", "support", "freshness", "confidence"):
        total = 0.0
        for requirement_id, weight in weights.items():
            values = [item[dimension] for item in evidence if requirement_id in item["requirement_ids"]]
            total += weight * (max(values) if values else 0.0)
        scores["coverage" if dimension == "support" else dimension] = total
    scores["overall"] = _overall(scores)
    conflicts = []
    by_key = {}
    for item in evidence:
        key = item.get("canonical_key")
        if key:
            by_key.setdefault(key, set()).add(str(item.get("claim", "")).strip().lower())
    for key, claims in by_key.items():
        if len(claims) > 1:
            conflicts.append({"requirement": key, "resolved": False})

    packet = {
        "retrieval": {source: response["status"] for source, response in responses.items()},
        "request_class": request.get("request_class", "bounded_internal_fact"),
        "requirement_extraction_confidence": request.get("requirement_extraction_confidence", 0.0),
        "requirements": [{**req, "support": max(
            [item["support"] for item in evidence if req["id"] in item["requirement_ids"]] or [0.0]
        )} for req in requirements],
        "conflicts": conflicts, "scores": scores,
    }
    decision = decide_memory_sufficiency(packet)
    decision.update({
        "requirements": packet["requirements"], "evidence_ids": [item["id"] for item in evidence],
        "evidence": evidence, "conflicts": conflicts,
        "retrieval_receipt": {
            "policy_version": POLICY_VERSION,
            "statuses": packet["retrieval"], "source_coverage": {
                source: len(response["evidence"]) for source, response in responses.items()
            }, "latency_ms_by_source": latencies,
            "latency_ms_total": round((time.monotonic() - started) * 1000, 3),
        },
    })
    return decision


def _safe_evidence(evidence: list[dict]) -> list[dict]:
    allowed = ("id", "source_store", "provenance", "path", "updated_at")
    return [{key: item[key] for key in allowed if item.get(key)} for item in evidence]


def _emit_gate_event(path: Path, event: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        pass


def apply_memory_gate(candidates: list[dict], *, enabled=MEMORY_GATE_ENABLED,
                      mode=MEMORY_GATE_MODE, mnemosyne_lookup=lookup_mnemosyne,
                      wiki_lookup=lookup_wiki, telemetry_path=TELEMETRY_PATH,
                      deadline_ms=MEMORY_GATE_TIMEOUT_MS) -> tuple[list[dict], dict]:
    """Apply the optional gate after truncation; preserve candidates on failure."""
    if not enabled:
        return candidates, {"enabled": False, "mode": "disabled", "eligible": 0,
                            "would_suppress": 0, "suppressed": 0}
    mode = mode if mode in {"observe", "enforce"} else "observe"
    run_id = str(uuid4())
    kept, supporting = [], {}
    decisions, fallbacks = Counter(), Counter()
    source_hits = Counter()
    latencies = []
    eligible = 0
    for paper in candidates:
        if paper.get("action") not in {"write_now", "ask_first"}:
            kept.append(paper)
            continue
        eligible += 1
        request = {
            "text": f"Has arXiv:{paper['arxiv_id']} already received a complete paper synthesis?",
            "arxiv_id": paper["arxiv_id"], "request_class": "bounded_source_summary",
            "requirements": [{"id": "paper_synthesis", "weight": 1.0, "critical": True}],
            "requirement_extraction_confidence": 1.0,
        }
        decision = memory_sufficient(
            request, mnemosyne_lookup=mnemosyne_lookup, wiki_lookup=wiki_lookup,
            deadline_ms=deadline_ms,
        )
        sufficient = decision["action"] == "SKIP_DEEP_RESEARCH"
        gate_decision = "suppress" if sufficient and mode == "enforce" else (
            "would_suppress" if sufficient else "fallback"
        )
        decisions[gate_decision] += 1
        if not sufficient:
            fallbacks[decision["reason_code"]] += 1
        if not (sufficient and mode == "enforce"):
            kept.append(paper)
        else:
            supporting[paper["arxiv_id"]] = _safe_evidence(decision["evidence"])
        receipt = decision["retrieval_receipt"]
        source_hits.update(receipt["source_coverage"])
        latencies.append(receipt["latency_ms_total"])
        _emit_gate_event(Path(telemetry_path), {
            "schema_version": 1, "event_type": "paper_memory_gate.candidate",
            "event_id": str(uuid4()), "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id, "cron_job_id": "463058f2566d", "candidate_id": paper["arxiv_id"],
            "candidate_action": paper["action"], "eligible": True, "gate_enabled": True,
            "gate_mode": mode, "decision": gate_decision, "decision_reason": decision["reason_code"],
            "sources_attempted": ["mnemosyne", "wiki"], "source_hits": receipt["source_coverage"],
            "latency_ms_total": receipt["latency_ms_total"],
            "latency_ms_by_source": receipt["latency_ms_by_source"],
            "fallback_reason": None if sufficient else decision["reason_code"],
            "estimated_avoided_deep_research_executions": int(sufficient and mode == "enforce"),
            "estimated_avoided_candidates": int(sufficient and mode == "enforce"),
        })
    summary = {
        "enabled": True, "mode": mode, "eligible": eligible,
        "would_suppress": decisions["would_suppress"] + decisions["suppress"],
        "suppressed": decisions["suppress"], "source_hits": dict(source_hits),
        "fallbacks": dict(fallbacks), "latency_ms": round(sum(latencies), 3),
        "supporting_evidence": supporting,
    }
    _emit_gate_event(Path(telemetry_path), {
        "schema_version": 1, "event_type": "paper_memory_gate.batch",
        "event_id": str(uuid4()), "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "cron_job_id": "463058f2566d",
        "candidate_count_before": len(candidates), "candidate_count_after": len(kept),
        "eligible_requests": eligible, "decisions": dict(decisions),
        "source_hits": dict(source_hits), "fallback_reasons": dict(fallbacks),
        "latency_ms_total": summary["latency_ms"],
        "estimated_avoided_deep_research_executions": summary["suppressed"],
        "estimated_avoided_candidates": summary["suppressed"],
    })
    return kept, summary


# ── Phase 1A: Fetch arXiv ───────────────────────────────────────────────────

def fetch_arxiv() -> list[dict]:
    """Fetch papers from arXiv API for all configured queries."""
    papers: dict[str, dict] = {}
    cutoff = datetime.now(timezone.utc) - timedelta(days=CUTOFF_DAYS)

    for label, query in ARXIV_QUERIES.items():
        url = (
            f"https://export.arxiv.org/api/query?"
            f"search_query={quote(query)}&sortBy=submittedDate&sortOrder=descending"
            f"&max_results=30"
        )
        print(f"  [arXiv] Fetching {label}...", file=sys.stderr)
        body = fetch_url(url)
        if not body:
            continue

        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            continue

        for entry in root.findall(f"{{{ATOM_NS}}}entry"):
            _id = entry.find(f"{{{ATOM_NS}}}id")
            _title = entry.find(f"{{{ATOM_NS}}}title")
            _summary = entry.find(f"{{{ATOM_NS}}}summary")
            _published = entry.find(f"{{{ATOM_NS}}}published")

            arxiv_id = parse_arxiv_id(_id.text or "") if _id is not None else None
            if not arxiv_id or arxiv_id in papers:
                continue

            title = (_title.text or "").strip() if _title is not None else ""
            summary = (_summary.text or "").strip() if _summary is not None else ""
            published = (_published.text or "").strip() if _published is not None else ""

            if not title or not summary:
                continue

            try:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except ValueError:
                pub_date = datetime.now(timezone.utc)

            if pub_date < cutoff:
                continue

            score = tight_score(title, summary)
            if score < 2:
                continue

            papers[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": title,
                "summary": summary[:500],
                "published": published,
                "score": score,
                "source": "arxiv",
                "url": f"https://arxiv.org/abs/{arxiv_id}",
            }

    return list(papers.values())


# ── Phase 1C: HuggingFace Daily Papers ──────────────────────────────────────

def fetch_hf_daily(papers: list[dict]) -> list[dict]:
    """Cross-reference HuggingFace Daily Papers against arXiv pool."""
    print("  [HF Daily] Fetching...", file=sys.stderr)
    body = fetch_url("https://huggingface.co/api/daily_papers?limit=30")
    if not body:
        return papers

    try:
        hf_papers = json.loads(body)
    except json.JSONDecodeError:
        return papers

    existing_ids = {p["arxiv_id"] for p in papers}

    for p in hf_papers:
        paper_data = p.get("paper", {})
        paper_id = paper_data.get("id", "")
        arxiv_id = parse_arxiv_id(paper_id)
        if not arxiv_id:
            continue

        title = paper_data.get("title", "")
        summary = paper_data.get("summary", "") or paper_data.get("abstract", "")

        if arxiv_id in existing_ids:
            # Mark existing paper as HF-featured
            for paper in papers:
                if paper["arxiv_id"] == arxiv_id:
                    paper["hf_featured"] = True
                    paper["hf_upvotes"] = p.get("upvotes", 0)
                    break
        else:
            # New paper from HF Daily not in arXiv results
            score = tight_score(title, summary)
            if score >= 2:
                papers.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "summary": summary[:500],
                    "published": paper_data.get("publishedAt", ""),
                    "score": score,
                    "source": "hf-daily",
                    "hf_featured": True,
                    "hf_upvotes": p.get("upvotes", 0),
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                })
                existing_ids.add(arxiv_id)

    return papers


# ── Phase 1D: Papers With Code (opportunistic) ─────────────────────────────

def fetch_pwc(papers: list[dict]) -> list[dict]:
    """Check Papers With Code for score >= 3 papers."""
    print("  [PwC] Checking top papers...", file=sys.stderr)
    for paper in papers:
        if paper["score"] < 3:
            continue
        url = f"https://paperswithcode.com/api/v1/papers/?arxiv_id={paper['arxiv_id']}"
        body = fetch_url(url, timeout=10)
        if not body:
            continue
        try:
            data = json.loads(body)
            results = data.get("results", [])
            if results:
                paper["pwc_repo"] = results[0].get("repository_url", "")
                paper["pwc_stars"] = results[0].get("stars", 0)
        except (json.JSONDecodeError, KeyError):
            pass

    return papers


# ── Scoring with Quality Weight (v2.1.1 — less harsh on new papers) ──────────

def apply_final_score(paper: dict) -> dict:
    """Apply quality weight + implementation multiplier to base relevance score.

    v2.1.1 changes from v2.1.0:
    - 0-2 citations with no HF/venue: weight raised from 0.3 to 0.5
      (brand-new papers shouldn't be penalised to Skip tier immediately)
    - 3-10 citations: weight raised from 0.5 to 0.6
    - Added recency boost: +0.1 weight for papers published in last 3 days
      (only applied if base score >= 4, prevents marginal papers from inflating)
    """
    base = paper["score"]
    qw = 0.7  # default fallback

    # Quality weight from HF featured status
    if paper.get("hf_featured"):
        qw = 0.8
        if paper.get("hf_upvotes", 0) > 50:
            qw = 0.9

    # Recency boost: +0.1 for papers <3 days old with base score >= 4
    try:
        pub = datetime.fromisoformat(paper["published"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
        if age_hours < 72 and base >= 4:
            qw = min(qw + 0.1, 1.0)
    except (ValueError, KeyError):
        pass

    # Implementation multiplier
    impl = 1.0
    if paper.get("pwc_repo"):
        stars = paper.get("pwc_stars", 0)
        if stars >= 500:
            impl = 1.25
        elif stars >= 50:
            impl = 1.2
        else:
            impl = 1.1

    final = base * qw * impl
    paper["quality_weight"] = round(qw, 2)
    paper["impl_multiplier"] = impl
    paper["final_score"] = round(final, 2)

    # Thresholds
    if final >= 5.5:
        paper["action"] = "write_now"
    elif final >= 3.0:
        paper["action"] = "ask_first"
    elif final >= 1.5:
        paper["action"] = "file"
    else:
        paper["action"] = "skip"

    return paper


def quarantine_untrusted_candidates(candidates: list[dict]) -> tuple[list[dict], dict]:
    """Remove candidates whose external fields trip Hermes' cron prompt scanner."""
    repo_text = str(REPO_ROOT)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from tools.cronjob_tools import _scan_cron_skill_assembled

    kept: list[dict] = []
    quarantined_ids: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        raw_id = str(candidate.get("arxiv_id") or "")
        candidate_id = _candidate_telemetry_id(raw_id, index)
        serialized = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        cleaned, error = _scan_cron_skill_assembled(serialized)
        candidate_strings = _candidate_strings(candidate)
        candidate_text = " ".join(candidate_strings)
        cleaned_text, text_error = _scan_cron_skill_assembled(candidate_text)
        removed_invisible = cleaned != serialized or cleaned_text != candidate_text
        external_directive = any(
            pattern.search(text)
            for text in (*candidate_strings, candidate_text)
            for pattern in _EXTERNAL_DIRECTIVE_PATTERNS
        )
        error = error or text_error
        if error or removed_invisible or external_directive:
            quarantined_ids.append(candidate_id)
            print(
                f"  [SECURITY] Quarantined candidate {candidate_id}",
                file=sys.stderr,
            )
            continue
        try:
            kept.append(json.loads(cleaned))
        except json.JSONDecodeError:
            quarantined_ids.append(candidate_id)
            print(
                f"  [SECURITY] Quarantined malformed candidate {candidate_id}",
                file=sys.stderr,
            )

    return kept, {
        "scanned": len(candidates),
        "quarantined": len(quarantined_ids),
        "candidate_ids": quarantined_ids,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[paper-preprocess] Phase 3: Fetching + scoring papers", file=sys.stderr)
    print(f"[paper-preprocess] Cutoff: {CUTOFF_DAYS} days, max candidates: {MAX_CANDIDATES}", file=sys.stderr)

    # Phase 1A: arXiv
    papers = fetch_arxiv()
    print(f"  → {len(papers)} papers from arXiv", file=sys.stderr)

    # Phase 1C: HF Daily cross-ref
    papers = fetch_hf_daily(papers)
    print(f"  → {len(papers)} after HF Daily cross-ref", file=sys.stderr)

    # Phase 1D: Papers With Code (opportunistic)
    papers = fetch_pwc(papers)
    print(f"  → {len(papers)} after PwC check", file=sys.stderr)

    # Apply final scoring
    for p in papers:
        p = apply_final_score(p)

    # Sort by final score descending
    papers.sort(key=lambda p: p["final_score"], reverse=True)

    # Filter to actionable candidates
    candidates = [p for p in papers if p["action"] != "skip"]
    candidates = candidates[:MAX_CANDIDATES]

    # External titles and abstracts are untrusted data. Quarantine only a
    # matching candidate so one hostile paper cannot block the whole cron.
    candidates, security_gate = quarantine_untrusted_candidates(candidates)

    # Optional memory-first gate: after truncation, before synthesis output.
    candidates, memory_gate = apply_memory_gate(candidates)

    # Summary
    actions = {}
    for p in candidates:
        actions.setdefault(p["action"], 0)
        actions[p["action"]] += 1

    print(f"\n[paper-preprocess] Candidates: {len(candidates)}", file=sys.stderr)
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count}", file=sys.stderr)

    # Output JSON to stdout (this is what the cron captures)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_fetched": len(papers),
        "candidates": candidates,
        "security_gate": security_gate,
        "summary": {
            "write_now": actions.get("write_now", 0),
            "ask_first": actions.get("ask_first", 0),
            "file": actions.get("file", 0),
        },
    }
    if memory_gate["enabled"]:
        output["memory_gate"] = memory_gate

    json.dump(output, sys.stdout, indent=2, ensure_ascii=False)

    # Also write to file if OUTPUT_PATH set
    if OUTPUT_PATH:
        out_path = Path(OUTPUT_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n[paper-preprocess] Written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
