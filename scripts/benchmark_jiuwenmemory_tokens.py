#!/usr/bin/env python3
"""Benchmark JiuwenMemory token reduction and recall latency.

Compares a long raw-history prompt against JiuwenMemory recall context for a
small synthetic dataset. This is an operational benchmark (not a pytest): it
uses the live local memory-server at http://127.0.0.1:8000 by default and
prints JSON + a compact table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from agent.model_metadata import estimate_messages_tokens_rough
except Exception:  # pragma: no cover - fallback for running outside repo
    def estimate_messages_tokens_rough(messages: list[dict[str, Any]]) -> int:
        return (len(str(messages)) + 3) // 4


def post(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 120) -> Any:
    data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body else {}


def get(base_url: str, path: str, timeout: float = 30) -> Any:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body) if body else {}


def extract_results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("results", "memories", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    # JiuwenMemory variants sometimes nest by memory type.
    out: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, list):
            out.extend(x for x in value if isinstance(x, dict))
    return out


def item_text(item: dict[str, Any]) -> str:
    for key in ("content", "text", "summary", "memory"):
        value = item.get(key)
        if value:
            return str(value)
    return json.dumps(item, ensure_ascii=False)


def format_recall_context(memories: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    if memories:
        sections.append("Relevant memories:\n" + "\n".join(f"- {item_text(x)[:700]}" for x in memories))
    if summaries:
        sections.append("Relevant conversation summaries:\n" + "\n".join(f"- {item_text(x)[:900]}" for x in summaries))
    if not sections:
        return ""
    return "<memory-context>\n" + "\n\n".join(sections) + "\n</memory-context>"


def build_dataset(marker: str, filler_repeat: int) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    facts = [
        {
            "topic": "Aurora backend stack",
            "fact": f"Aurora benchmark marker {marker}: Aurora uses FastAPI as backend, PostgreSQL as database, and Redis as cache.",
            "query": "Aurora benchmark marker backend stack database cache",
            "expected": ["Aurora", "FastAPI", "PostgreSQL", "Redis"],
        },
        {
            "topic": "PaymentGateway delay",
            "fact": f"Aurora benchmark marker {marker}: PaymentGateway regression failure moved the release window from Wednesday to Friday; Alice owns PaymentGateway.",
            "query": "Aurora benchmark marker release delay owner PaymentGateway",
            "expected": ["PaymentGateway", "Wednesday", "Friday", "Alice"],
        },
        {
            "topic": "Vector embedding service",
            "fact": f"Aurora benchmark marker {marker}: the embedding service is local BGE-M3 at 127.0.0.1:18080 with 1024 dimensions.",
            "query": "Aurora benchmark marker embedding service dimensions",
            "expected": ["BGE-M3", "127.0.0.1:18080", "1024"],
        },
        {
            "topic": "Operational policy",
            "fact": f"Aurora benchmark marker {marker}: production deploys require staged rollout, incident notes, and rollback owner Bob.",
            "query": "Aurora benchmark marker deploy policy rollback owner",
            "expected": ["staged rollout", "incident notes", "Bob"],
        },
    ]

    filler = (
        " This is deliberately verbose non-essential project chatter about UI colors, "
        "calendar coordination, meeting room names, status phrasing, and unrelated "
        "implementation notes."
    ) * filler_repeat

    messages: list[dict[str, str]] = []
    for idx in range(1, 33):
        fact = facts[(idx - 1) % len(facts)]
        messages.append({
            "role": "user",
            "content": f"Turn {idx}: please record project note. {fact['fact']} {filler}",
        })
        messages.append({
            "role": "assistant",
            "content": f"Recorded turn {idx} for {fact['topic']}. I will use it later when asked. {filler}",
        })
    return messages, facts


def wait_for_turbo_job(base_url: str, job_id: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_status: dict[str, Any] = {}
    while time.time() < deadline:
        # process_once is safe: it processes at most one queued/processing job.
        try:
            post(base_url, "/turbo/process_once", {}, timeout=180)
        except Exception:
            pass
        try:
            status = get(base_url, "/turbo/status", timeout=30)
            last_status = status if isinstance(status, dict) else {}
            for job in last_status.get("recent_jobs", []) or []:
                if isinstance(job, dict) and job.get("job_id") == job_id:
                    if job.get("status") in {"completed", "failed"}:
                        return job
        except Exception:
            pass
        time.sleep(2)
    return {"job_id": job_id, "status": "timeout", "last_status": last_status}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scope", default="")
    parser.add_argument("--user-id", default="bench-user")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--filler-repeat", type=int, default=10)
    parser.add_argument("--summary-limit", type=int, default=5)
    parser.add_argument("--memory-limit", type=int, default=8)
    args = parser.parse_args()

    marker = f"bench-token-{int(time.time())}"
    scope_id = args.scope or f"hermes-bench-{marker}"
    session_id = f"{scope_id}-session"

    health = get(args.base_url, "/health")
    messages, evals = build_dataset(marker, args.filler_repeat)

    baseline_prompt_tokens = estimate_messages_tokens_rough(
        messages + [{"role": "user", "content": "Answer the benchmark queries using the full conversation history."}]
    )

    t0 = time.perf_counter()
    accepted = post(args.base_url, "/turbo/add_messages_async", {
        "messages": messages,
        "scope_id": scope_id,
        "user_id": args.user_id,
        "session_id": session_id,
        "enable_summary": True,
        "enable_user_profile": True,
        "enable_semantic_memory": True,
        "enable_episodic_memory": True,
    }, timeout=60)
    enqueue_ms = (time.perf_counter() - t0) * 1000
    job_id = str(accepted.get("job_id") or "")
    job = wait_for_turbo_job(args.base_url, job_id, args.timeout) if job_id else {"status": "no_job_id"}

    rows: list[dict[str, Any]] = []
    for case in evals:
        q = case["query"]
        t_mem = time.perf_counter()
        mem_payload = post(args.base_url, "/search_memory/", {
            "query": q,
            "num": args.memory_limit,
            "scope_id": scope_id,
            "user_id": args.user_id,
            "threshold": 0.0,
        }, timeout=60)
        mem_ms = (time.perf_counter() - t_mem) * 1000
        t_sum = time.perf_counter()
        sum_payload = post(args.base_url, "/search_user_history_summary/", {
            "query": q,
            "num": args.summary_limit,
            "scope_id": scope_id,
            "user_id": args.user_id,
            "threshold": 0.0,
        }, timeout=60)
        sum_ms = (time.perf_counter() - t_sum) * 1000
        memories = extract_results(mem_payload)[: args.memory_limit]
        summaries = extract_results(sum_payload)[: args.summary_limit]
        context = format_recall_context(memories, summaries)
        recall_tokens = estimate_messages_tokens_rough([
            {"role": "user", "content": context + "\n\nQuestion: " + q}
        ])
        text = context.lower()
        expected = case["expected"]
        hits = [kw for kw in expected if kw.lower() in text]
        rows.append({
            "topic": case["topic"],
            "query": q,
            "expected": expected,
            "hits": hits,
            "hit_rate": len(hits) / max(1, len(expected)),
            "memory_results": len(memories),
            "summary_results": len(summaries),
            "search_memory_ms": round(mem_ms, 2),
            "search_summary_ms": round(sum_ms, 2),
            "recall_context_tokens": recall_tokens,
            "token_reduction_pct": round((1 - recall_tokens / baseline_prompt_tokens) * 100, 2),
            "sample_context": context[:1200],
        })

    aggregate = {
        "baseline_raw_history_tokens": baseline_prompt_tokens,
        "avg_recall_context_tokens": round(statistics.mean(r["recall_context_tokens"] for r in rows), 2),
        "avg_token_reduction_pct": round(statistics.mean(r["token_reduction_pct"] for r in rows), 2),
        "avg_hit_rate": round(statistics.mean(r["hit_rate"] for r in rows), 3),
        "avg_search_memory_ms": round(statistics.mean(r["search_memory_ms"] for r in rows), 2),
        "avg_search_summary_ms": round(statistics.mean(r["search_summary_ms"] for r in rows), 2),
        "enqueue_ms": round(enqueue_ms, 2),
    }
    result = {
        "status": "completed" if job.get("status") == "completed" else "degraded",
        "marker": marker,
        "scope_id": scope_id,
        "session_id": session_id,
        "health": health,
        "turbo_accept": accepted,
        "turbo_job": job,
        "aggregate": aggregate,
        "rows": rows,
    }

    out_dir = ROOT / "benchmark_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"jiuwenmemory_token_benchmark_{marker}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nCOMPACT")
    print("result_path", out_path)
    print("status", result["status"])
    print("baseline_raw_history_tokens", aggregate["baseline_raw_history_tokens"])
    print("avg_recall_context_tokens", aggregate["avg_recall_context_tokens"])
    print("avg_token_reduction_pct", aggregate["avg_token_reduction_pct"])
    print("avg_hit_rate", aggregate["avg_hit_rate"])
    print("avg_search_memory_ms", aggregate["avg_search_memory_ms"])
    print("avg_search_summary_ms", aggregate["avg_search_summary_ms"])
    return 0 if result["status"] == "completed" and aggregate["avg_hit_rate"] >= 0.5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
