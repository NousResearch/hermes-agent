#!/usr/bin/env python3
"""Exhaustive benchmarks for TrainingDataBridge.

Covers: throughput, scalability, dedup accuracy, format correctness,
memory profiling, edge case stress, and produces a scored report.

Usage:
    python scripts/benchmark_training_data.py
    python scripts/benchmark_training_data.py --quick      # 1K-10K only
    python scripts/benchmark_training_data.py --full       # up to 1M records
"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.training_data import TrainingDataBridge, DEFAULT_RETENTION_DAYS


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

DOMAINS = [
    "software-dev", "security", "devops", "data-science", "research",
    "writing", "creative", "general", "math", "legal",
]
SCORES = [0.1, 0.3, 0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 1.0]
VERBS = ["implement", "debug", "optimize", "refactor", "design", "analyze",
         "document", "test", "deploy", "configure", "migrate", "benchmark"]


def _make_conversation(turns: int = 2, seed: int = 0) -> List[Dict]:
    """Generate a synthetic ShareGPT conversation."""
    conv = []
    for i in range(turns):
        conv.append({"from": "human", "value": f"User message {seed}-{i}: {VERBS[(seed+i) % len(VERBS)]} the thing"})
        conv.append({"from": "gpt", "value": f"Assistant response {seed}-{i}: here is the detailed analysis for {VERBS[(seed+i) % len(VERBS)]}..."})
    return conv


def generate_records(
    n: int,
    turns: int = 2,
    duplicate_rate: float = 0.0,
    scored_rate: float = 0.8,
    domain_count: int = 5,
    seed: int = 42,
    content_offset: int = 0,
) -> List[Dict]:
    """Generate n synthetic training records with controlled distributions.

    Args:
        n: Number of records to generate
        turns: Conversation turns per record
        duplicate_rate: Fraction that are exact duplicates (0.0-1.0)
        scored_rate: Fraction with evaluation scores
        domain_count: Number of distinct domains to distribute across
        seed: Random seed for reproducibility
        content_offset: Offset added to conversation seeds (avoid overlap between batches)
    """
    import random
    rng = random.Random(seed)
    records = []
    unique_count = int(n * (1 - duplicate_rate))

    for i in range(n):
        if i < unique_count:
            conv = _make_conversation(turns=turns, seed=i + content_offset)
        else:
            # Duplicate: copy a previous record's conversation
            src_idx = rng.randint(0, unique_count - 1) if unique_count > 0 else 0
            conv = [dict(t) for t in records[src_idx]["conversations"]]

        record = {
            "conversations": conv,
            "completed": rng.random() > 0.05,
            "metadata": {
                "task_domain": DOMAINS[i % domain_count],
                "task_complexity": rng.choice(["low", "medium", "high"]),
            },
        }
        if rng.random() < scored_rate:
            record["metadata"]["score"] = rng.choice(SCORES)
        records.append(record)

    return records


def write_jsonl(records: List[Dict], path: Path) -> None:
    """Write records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_malformed_jsonl(path: Path, good: int = 90, bad: int = 10) -> int:
    """Write a JSONL with some malformed lines."""
    records = generate_records(good, turns=2)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            written += 1
        # Inject bad lines
        for i in range(bad):
            if i % 2 == 0:
                f.write("{not valid json\n")
            else:
                f.write("\n")  # empty line
    return written


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark framework
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    name: str
    passed: bool = True
    value: Any = None
    duration_ms: float = 0.0
    memory_kb: float = 0.0
    details: str = ""
    score: float = 1.0  # 0.0-1.0


@dataclass
class Suite:
    results: List[BenchResult] = field(default_factory=list)
    _start_time: float = 0.0
    _start_memory: int = 0
    _stopped: bool = False

    def start(self, name: str) -> None:
        gc.collect()
        gc.disable()
        tracemalloc.start()
        self._start_time = time.perf_counter()
        self._current_name = name
        self._stopped = False

    def stop(self, passed: bool = True, value: Any = None,
             details: str = "", score: float = 1.0) -> BenchResult:
        if self._stopped:
            return self.results[-1] if self.results else None
        self._stopped = True
        duration = (time.perf_counter() - self._start_time) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.enable()
        memory_kb = peak / 1024.0

        r = BenchResult(
            name=self._current_name,
            passed=passed,
            value=value,
            duration_ms=duration,
            memory_kb=memory_kb,
            details=details,
            score=score,
        )
        self.results.append(r)
        return r

    def summary(self) -> Dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        avg_score = sum(r.score for r in self.results) / total if total > 0 else 0
        total_time = sum(r.duration_ms for r in self.results)
        peak_mem = max((r.memory_kb for r in self.results), default=0)
        return {
            "total": total, "passed": passed, "failed": total - passed,
            "avg_score": avg_score, "total_time_ms": total_time,
            "peak_memory_kb": peak_mem,
        }


suite = Suite()


def bench(name: str):
    """Context manager for benchmarking."""
    class Ctx:
        def __enter__(self):
            suite.start(name)
            return self
        def __exit__(self, *args):
            if any(args):
                suite.stop(passed=False, details=str(args[1] or args[2] or "error"))
            else:
                suite.stop()
    return Ctx()


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 1: Throughput at scale
# ═══════════════════════════════════════════════════════════════════════════════

def bench_throughput(scales: List[int]):
    """Measure records/second for key operations across scales."""
    print("\n" + "=" * 72)
    print("BENCHMARK 1: Throughput at Scale")
    print("=" * 72)

    for n in scales:
        tmp = tempfile.mkdtemp()
        try:
            data_dir = Path(tmp) / "data"
            out_dir = Path(tmp) / "output"
            data_dir.mkdir()
            out_dir.mkdir()
            records = generate_records(n, turns=4, duplicate_rate=0.1)
            jsonl_path = data_dir / "test.jsonl"

            # ── Write ──
            with bench(f"JSONL write ({n:,} records)"):
                t0 = time.perf_counter()
                write_jsonl(records, jsonl_path)
                t1 = time.perf_counter()
                rps = n / (t1 - t0) if (t1 - t0) > 0 else float("inf")
                suite.stop(value={"records_per_sec": rps},
                           details=f"{rps:,.0f} records/sec")

            # ── _find_records ──
            bridge = TrainingDataBridge(data_dir=data_dir)
            with bench(f"_find_records ({n:,} records)"):
                t0 = time.perf_counter()
                found = bridge._find_records(days=365)
                t1 = time.perf_counter()
                rps = len(found) / (t1 - t0) if (t1 - t0) > 0 else float("inf")
                passed = len(found) == n
                suite.stop(passed=passed, value={"records_per_sec": rps},
                           details=f"{rps:,.0f} records/sec, found={len(found)}",
                           score=1.0 if passed else 0.0)

            # ── _deduplicate ──
            with bench(f"_deduplicate ({n:,} records, 10% dup)"):
                t0 = time.perf_counter()
                unique = bridge._deduplicate(found)
                t1 = time.perf_counter()
                rps = len(found) / (t1 - t0) if (t1 - t0) > 0 else float("inf")
                expected_unique = int(n * 0.9)  # 10% dup rate
                passed = abs(len(unique) - expected_unique) <= max(1, n * 0.01)
                suite.stop(passed=passed, value={"records_per_sec": rps},
                           details=f"{rps:,.0f} records/sec, {len(unique)} unique",
                           score=1.0 if passed else 0.5)

            # ── _export_sharegpt (to out_dir, not data_dir) ──
            out = out_dir / "export.jsonl"
            with bench(f"_export_sharegpt ({len(unique):,} records)"):
                t0 = time.perf_counter()
                bridge._export_sharegpt(unique, out)
                t1 = time.perf_counter()
                rps = len(unique) / (t1 - t0) if (t1 - t0) > 0 else float("inf")
                passed = out.exists() and out.stat().st_size > 0
                suite.stop(passed=passed, value={"records_per_sec": rps},
                           details=f"{rps:,.0f} records/sec, {out.stat().st_size:,} bytes",
                           score=1.0 if passed else 0.0)

            # ── _export_alpaca (to out_dir, not data_dir) ──
            out_alpaca = out_dir / "export.json"
            with bench(f"_export_alpaca ({len(unique):,} records)"):
                t0 = time.perf_counter()
                bridge._export_alpaca(unique, out_alpaca)
                t1 = time.perf_counter()
                rps = len(unique) / (t1 - t0) if (t1 - t0) > 0 else float("inf")
                passed = out_alpaca.exists()
                suite.stop(passed=passed, value={"records_per_sec": rps},
                           details=f"{rps:,.0f} records/sec",
                           score=1.0 if passed else 0.0)

            # ── get_stats (clean data_dir, no export pollution) ──
            with bench(f"get_stats ({n:,} records)"):
                t0 = time.perf_counter()
                stats = bridge.get_stats(days=365)
                t1 = time.perf_counter()
                passed = stats["total_records"] == n
                suite.stop(passed=passed,
                           details=f"{(t1-t0)*1000:.1f}ms, {stats['unique_records']} unique, "
                                    f"{stats['new_since_last_export']} new",
                           score=1.0 if passed else 0.0)

            # ── full export pipeline (to out_dir) ──
            with bench(f"export() e2e ({n:,} records)"):
                t0 = time.perf_counter()
                result = bridge.export(output_path=out_dir / "final.jsonl",
                                       fmt="sharegpt", max_records=n * 2)
                t1 = time.perf_counter()
                passed = "error" not in result and result["records_exported"] > 0
                suite.stop(passed=passed,
                           details=f"{(t1-t0)*1000:.1f}ms, exported={result.get('records_exported', 0)}",
                           score=1.0 if passed else 0.0)

        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 2: Dedup accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def bench_dedup_accuracy():
    """Verify dedup correctness under edge cases."""
    print("\n" + "=" * 72)
    print("BENCHMARK 2: Deduplication Accuracy")
    print("=" * 72)

    tmp = tempfile.mkdtemp()
    try:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        bridge = TrainingDataBridge(data_dir=data_dir)

        # Test 1: 0% duplicates
        records = generate_records(1000, duplicate_rate=0.0, seed=1)
        with bench("Dedup: 0% duplicates (all unique)"):
            unique = bridge._deduplicate(records)
            suite.stop(passed=len(unique) == 1000,
                       details=f"{len(unique)}/1000 unique")

        # Test 2: 100% duplicates
        base = _make_conversation(turns=2, seed=99)
        all_same = [{"conversations": [dict(t) for t in base]} for _ in range(1000)]
        with bench("Dedup: 100% duplicates (all same)"):
            unique = bridge._deduplicate(all_same)
            suite.stop(passed=len(unique) == 1,
                       details=f"{len(unique)}/1000 unique (expect 1)")

        # Test 3: Exact 50% duplicates
        records = generate_records(1000, duplicate_rate=0.5, seed=2)
        with bench("Dedup: 50% duplicates"):
            unique = bridge._deduplicate(records)
            expected = 500
            passed = abs(len(unique) - expected) <= 5
            suite.stop(passed=passed, details=f"{len(unique)}/1000 unique (expect ~{expected})")

        # Test 4: Near-duplicates (different metadata, same conversation)
        conv = _make_conversation(turns=3, seed=77)
        near_dupes = []
        for i in range(100):
            near_dupes.append({
                "conversations": [dict(t) for t in conv],
                "metadata": {"score": 0.1 * (i % 10), "task_domain": DOMAINS[i % len(DOMAINS)]},
            })
        with bench("Dedup: near-duplicates (same conv, diff metadata)"):
            unique = bridge._deduplicate(near_dupes)
            suite.stop(passed=len(unique) == 1,
                       details=f"{len(unique)}/100 unique (expect 1 — content hash only)")

        # Test 5: Empty record list
        with bench("Dedup: empty list"):
            unique = bridge._deduplicate([])
            suite.stop(passed=len(unique) == 0, details=f"returned {len(unique)}")

        # Test 6: Single record
        single = generate_records(1, seed=3)
        with bench("Dedup: single record"):
            unique = bridge._deduplicate(single)
            suite.stop(passed=len(unique) == 1, details=f"returned {len(unique)}")

        # Test 7: Hash stability (same content = same hash)
        r1 = {"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "hi"}]}
        r2 = {"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "hi"}]}
        with bench("Dedup: hash stability (identical content)"):
            h1 = bridge._hash_record(r1)
            h2 = bridge._hash_record(r2)
            suite.stop(passed=h1 == h2, details=f"hash1={h1[:12]}... hash2={h2[:12]}...")

        # Test 8: Hash collision resistance (different content)
        r3 = {"conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "bye"}]}
        with bench("Dedup: hash differentiation (different content)"):
            h1 = bridge._hash_record(r1)
            h3 = bridge._hash_record(r3)
            suite.stop(passed=h1 != h3, details=f"hash1={h1[:12]}... hash3={h3[:12]}...")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 3: Format correctness
# ═══════════════════════════════════════════════════════════════════════════════

def bench_format_correctness():
    """Verify export formats are valid and round-trip correctly."""
    print("\n" + "=" * 72)
    print("BENCHMARK 3: Format Correctness")
    print("=" * 72)

    tmp = tempfile.mkdtemp()
    try:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        bridge = TrainingDataBridge(data_dir=data_dir)
        records = generate_records(500, turns=4, duplicate_rate=0.0, scored_rate=1.0, seed=5)
        write_jsonl(records, data_dir / "input.jsonl")

        # ShareGPT round-trip
        out_sharegpt = data_dir / "out.jsonl"
        with bench("ShareGPT: valid JSONL, round-trip parse"):
            bridge._export_sharegpt(records, out_sharegpt)
            # Verify every line parses
            parsed = 0
            with open(out_sharegpt) as f:
                for line in f:
                    if line.strip():
                        json.loads(line)
                        parsed += 1
            passed = parsed == 500
            suite.stop(passed=passed, details=f"{parsed}/500 lines parse")

        # Alpaca format validation
        out_alpaca = data_dir / "out.json"
        with bench("Alpaca: valid JSON, has instruction/output"):
            bridge._export_alpaca(records, out_alpaca)
            content = json.loads(out_alpaca.read_text())
            valid = all("instruction" in r and "output" in r for r in content)
            suite.stop(passed=valid and len(content) == 500,
                       details=f"{len(content)} records, all have instruction+output")

        # Parquet (graceful fallback if pandas missing)
        out_pq = data_dir / "out.parquet"
        with bench("Parquet: export or graceful fallback"):
            bridge._export_parquet(records, out_pq)
            # If pandas is installed, parquet exists; otherwise jsonl fallback
            fallback = data_dir / "out.jsonl"
            parquet_ok = out_pq.exists() and out_pq.stat().st_size > 0
            fallback_ok = fallback.exists() and fallback.stat().st_size > 0
            suite.stop(passed=parquet_ok or fallback_ok,
                       details=f"parquet={'✓' if parquet_ok else '✗'} fallback={'✓' if fallback_ok else '✗'}")

        # Empty directory: export with no data
        empty_dir = Path(tmp) / "empty_data"
        empty_dir.mkdir()
        empty_bridge = TrainingDataBridge(data_dir=empty_dir)
        with bench("Format: empty directory → error message"):
            result = empty_bridge.export(fmt="sharegpt", output_path=Path(tmp) / "empty_out.jsonl")
            suite.stop(passed="error" in result,
                       details=f"message: {result.get('error', 'none')}")

        # Invalid format handling
        with bench("Format: invalid format → error"):
            result = bridge.export(fmt="invalid")
            suite.stop(passed="error" in result,
                       details=f"message: {result.get('error', 'none')}")

        # UTF-8 / Unicode handling
        unicode_records = [{
            "conversations": [
                {"from": "human", "value": "Héllo 世界 🌍 — em dash and 'quotes'"},
                {"from": "gpt", "value": "Respónse with ± ∑ ∫ √ ∞ ≈ ≠ ≤ ≥"},
            ],
            "metadata": {"task_domain": "unicode-test", "score": 0.9},
        }]
        out_unicode = data_dir / "unicode.jsonl"
        with bench("ShareGPT: unicode round-trip"):
            bridge._export_sharegpt(unicode_records, out_unicode)
            read_back = []
            with open(out_unicode, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        read_back.append(json.loads(line))
            passed = (read_back[0]["conversations"][0]["value"] == unicode_records[0]["conversations"][0]["value"])
            suite.stop(passed=passed, details=f"unicode preserved: {passed}")

        # BOM handling (UTF-8-SIG)
        out_bom = data_dir / "bom.jsonl"
        with open(out_bom, "w", encoding="utf-8-sig") as f:
            f.write(json.dumps(records[0], ensure_ascii=False) + "\n")
        with bench("Format: UTF-8-BOM handling"):
            found = bridge._find_records(days=365)
            suite.stop(passed=len(found) >= 1, details=f"BOM file parsed: {len(found)} records found")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 4: Filter accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def bench_filter_accuracy():
    """Verify min_score, domain, since_last filters work correctly."""
    print("\n" + "=" * 72)
    print("BENCHMARK 4: Filter Accuracy")
    print("=" * 72)

    tmp = tempfile.mkdtemp()
    try:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        bridge = TrainingDataBridge(data_dir=data_dir)

        records = generate_records(1000, turns=2, duplicate_rate=0.0, scored_rate=1.0, seed=7)
        # Ensure known score distribution
        for i, r in enumerate(records):
            r["metadata"]["score"] = SCORES[i % len(SCORES)]
            r["metadata"]["task_domain"] = DOMAINS[i % 5]
        write_jsonl(records, data_dir / "test.jsonl")

        # min_score filter
        with bench("Filter: min_score=0.7"):
            result = bridge.export(min_score=0.7, fmt="sharegpt")
            exported = result.get("records_exported", 0)
            # SCORES ≥ 0.7: 0.75, 0.85, 0.9, 0.95, 1.0 → 5/9 of scored records
            # All 1000 have scores → ~555 records ≥ 0.7
            expected = sum(1 for r in records if r["metadata"]["score"] >= 0.7)
            passed = exported == expected
            suite.stop(passed=passed, details=f"exported={exported}, expect={expected}")

        # min_score=1.0 (only perfect)
        with bench("Filter: min_score=1.0 (perfect only)"):
            result = bridge.export(min_score=1.0, fmt="sharegpt")
            exported = result.get("records_exported", 0)
            expected = sum(1 for r in records if r["metadata"]["score"] >= 1.0)
            passed = exported == expected
            suite.stop(passed=passed, details=f"exported={exported}, expect={expected}")

        # min_score=0.0 (all)
        with bench("Filter: min_score=0.0 (all records)"):
            result = bridge.export(min_score=0.0, fmt="sharegpt")
            exported = result.get("records_exported", 0)
            passed = exported == len(records)
            suite.stop(passed=passed, details=f"exported={exported}, expect={len(records)}")

        # Domain filter
        with bench("Filter: domain='security'"):
            result = bridge.export(domain="security", fmt="sharegpt")
            exported = result.get("records_exported", 0)
            expected = sum(1 for r in records if r["metadata"]["task_domain"] == "security")
            passed = exported == expected
            suite.stop(passed=passed, details=f"exported={exported}, expect={expected}")

        # Combined filters
        with bench("Filter: domain='software-dev' + min_score=0.8"):
            result = bridge.export(domain="software-dev", min_score=0.8, fmt="sharegpt")
            exported = result.get("records_exported", 0)
            expected = sum(1 for r in records
                          if r["metadata"]["task_domain"] == "software-dev"
                          and r["metadata"]["score"] >= 0.8)
            passed = exported == expected
            suite.stop(passed=passed, details=f"exported={exported}, expect={expected}")

        # since_last filter (fresh directory to avoid accumulated export pollution)
        sd_dir = Path(tmp) / "since_last_test"
        sd_dir.mkdir()
        sd_bridge = TrainingDataBridge(data_dir=sd_dir)
        sd_records = generate_records(500, turns=2, duplicate_rate=0.0, seed=77, content_offset=0)
        write_jsonl(sd_records, sd_dir / "orig.jsonl")
        with bench("Filter: since_last (after first export)"):
            sd_bridge.export(fmt="sharegpt", output_path=sd_dir / "first.jsonl", max_records=10000)
            new_records = generate_records(200, turns=2, seed=999, content_offset=10000)
            new_path = sd_dir / "new.jsonl"
            write_jsonl(new_records, new_path)
            future_time = time.time() + 10
            os.utime(new_path, (future_time, future_time))
            result = sd_bridge.export(since_last=True, fmt="sharegpt",
                                      output_path=sd_dir / "since_last.jsonl",
                                      max_records=10000)
            exported = result.get("records_exported", 0)
            passed = exported == 200
            suite.stop(passed=passed, details=f"new records found: {exported} (expect 200)")

        # max_records cap
        with bench("Filter: max_records=50"):
            result = bridge.export(max_records=50, fmt="sharegpt")
            exported = result.get("records_exported", 0)
            passed = exported == 50
            suite.stop(passed=passed, details=f"capped at {exported}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 5: Edge cases & stress
# ═══════════════════════════════════════════════════════════════════════════════

def bench_edge_cases():
    """Stress test edge cases and error handling."""
    print("\n" + "=" * 72)
    print("BENCHMARK 5: Edge Cases & Stress")
    print("=" * 72)

    tmp = tempfile.mkdtemp()
    try:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()

        # Empty data directory
        with bench("Edge: empty data directory"):
            bridge = TrainingDataBridge(data_dir=data_dir)
            stats = bridge.get_stats()
            suite.stop(passed=stats["total_records"] == 0,
                       details=f"total={stats['total_records']}")

        # Malformed JSONL
        mal_path = data_dir / "malformed.jsonl"
        good_count = write_malformed_jsonl(mal_path, good=90, bad=10)
        with bench("Edge: malformed JSONL (10% bad lines)"):
            bridge = TrainingDataBridge(data_dir=data_dir)
            found = bridge._find_records(days=365)
            passed = len(found) == good_count
            suite.stop(passed=passed, details=f"parsed {len(found)}/{good_count} good records, skipped 10 bad")

        # Very long conversation (200 turns)
        long_conv = [{
            "conversations": _make_conversation(turns=200, seed=42),
            "metadata": {"task_domain": "research", "score": 0.8},
        }]
        long_path = data_dir / "long.jsonl"
        write_jsonl(long_conv, long_path)
        with bench("Edge: 200-turn conversation"):
            bridge = TrainingDataBridge(data_dir=data_dir)
            found = bridge._find_records(days=365)
            has_long = any(r.get("_source_file", "").endswith("long.jsonl") for r in found)
            suite.stop(passed=has_long, details=f"long conversation parsed: {has_long}")

        # Multiple input paths
        ext1 = data_dir / "ext1"
        ext2 = data_dir / "ext2"
        ext1.mkdir()
        ext2.mkdir()
        write_jsonl(generate_records(50, seed=100), ext1 / "a.jsonl")
        write_jsonl(generate_records(75, seed=200), ext2 / "b.jsonl")
        with bench("Edge: multiple input paths"):
            bridge = TrainingDataBridge(data_dir=data_dir, input_paths=[ext1, ext2])
            stats = bridge.get_stats(days=365)
            # Initial data_dir had malformed (90) + long (1) = 91 valid records
            # Plus ext1 (50) + ext2 (75) = 125
            # Total should be >= 216
            passed = stats["total_records"] >= 216
            suite.stop(passed=passed, details=f"total across 3 paths: {stats['total_records']}")

        # Non-existent input path (should not crash)
        with bench("Edge: non-existent input path"):
            ghost = Path(tmp) / "does_not_exist"
            bridge = TrainingDataBridge(data_dir=data_dir, input_paths=[ghost])
            stats = bridge.get_stats(days=365)
            suite.stop(passed=stats["total_records"] > 0,
                       details=f"handles missing path gracefully: {stats['total_records']} records")

        # State persistence
        state_dir = Path(tmp) / "state_test"
        state_dir.mkdir()
        with bench("Edge: state persistence across instances"):
            b1 = TrainingDataBridge(data_dir=state_dir)
            write_jsonl(generate_records(10, seed=1), state_dir / "test.jsonl")
            b1.export(fmt="sharegpt")
            exports_b1 = b1._state.get("total_exports", 0)

            # New instance should load saved state
            b2 = TrainingDataBridge(data_dir=state_dir)
            passed = b2._state.get("total_exports", 0) == exports_b1
            suite.stop(passed=passed,
                       details=f"state persisted: {b2._state.get('total_exports')} exports, "
                                f"last_export_at={b2._state.get('last_export_at', 'none')[:19] if b2._state.get('last_export_at') else 'none'}")

        # Concurrent writes (no crash)
        with bench("Edge: rapid repeated exports"):
            bridge = TrainingDataBridge(data_dir=state_dir)
            for _ in range(10):
                bridge.export(fmt="sharegpt", max_records=100)
            passed = bridge._state.get("total_exports", 0) >= 10
            suite.stop(passed=passed,
                       details=f"{bridge._state.get('total_exports')} exports completed")

        # Check that _source_file and _ingested_at are populated
        with bench("Edge: metadata fields (_source_file, _ingested_at)"):
            bridge = TrainingDataBridge(data_dir=data_dir)
            found = bridge._find_records(days=365)
            has_source = all("_source_file" in r for r in found)
            has_ingested = all("_ingested_at" in r for r in found)
            suite.stop(passed=has_source and has_ingested,
                       details=f"_source_file: {has_source}, _ingested_at: {has_ingested}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark 6: Memory profiling
# ═══════════════════════════════════════════════════════════════════════════════

def bench_memory():
    """Profile memory usage under load."""
    print("\n" + "=" * 72)
    print("BENCHMARK 6: Memory Profiling")
    print("=" * 72)

    tmp = tempfile.mkdtemp()
    try:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()

        for n in [1000, 10000, 50000]:
            records = generate_records(n, turns=10, duplicate_rate=0.2, seed=8)
            write_jsonl(records, data_dir / f"mem_{n}.jsonl")

            with bench(f"Memory: {n:,} records (10-turn convos, 20% dup)"):
                bridge = TrainingDataBridge(data_dir=data_dir)
                bridge.export(fmt="sharegpt", max_records=n * 2)
                # Memory measured by tracemalloc in the bench framework
                suite.stop(details=f"peak memory recorded")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════

def print_report():
    """Print the final scored report."""
    s = suite.summary()

    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + "  TRAINING DATA BRIDGE — EXHAUSTIVE BENCHMARK REPORT".center(70) + "║")
    print("╠" + "═" * 70 + "╣")

    # Category breakdown
    categories = {}
    for r in suite.results:
        cat = r.name.split(":")[0] if ":" in r.name else r.name.split("(")[0].strip()
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "score": 0.0, "time": 0.0, "mem": 0.0}
        categories[cat]["total"] += 1
        if r.passed:
            categories[cat]["passed"] += 1
        categories[cat]["score"] += r.score
        categories[cat]["time"] += r.duration_ms
        categories[cat]["mem"] = max(categories[cat]["mem"], r.memory_kb)

    for cat, stats in sorted(categories.items()):
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg = stats["score"] / stats["total"] if stats["total"] > 0 else 0
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"║ {cat:<30s} {bar} {avg:.0%}  ({stats['passed']}/{stats['total']} pass, "
              f"{stats['time']:.0f}ms, {stats['mem']:.0f}KB peak)")

    print("╠" + "═" * 70 + "╣")

    # Individual results
    failures = [r for r in suite.results if not r.passed]
    if failures:
        print(f"║ FAILURES ({len(failures)}):")
        for r in failures:
            print(f"║   ✗ {r.name}")
            print(f"║     {r.details}")
    else:
        print("║ ✓ ALL TESTS PASSED")

    print("╠" + "═" * 70 + "╣")

    # Throughput summary
    throughput_results = [r for r in suite.results if r.value and "records_per_sec" in (r.value or {})]
    if throughput_results:
        rps_values = [r.value["records_per_sec"] for r in throughput_results]
        print(f"║ Throughput range: {min(rps_values):,.0f} – {max(rps_values):,.0f} records/sec")
        print(f"║ Median throughput: {statistics.median(rps_values):,.0f} records/sec")

    print("╠" + "═" * 70 + "╣")
    print(f"║ OVERALL: {s['passed']}/{s['total']} passed  |  "
          f"Score: {s['avg_score']:.1%}  |  "
          f"Time: {s['total_time_ms']:.0f}ms  |  "
          f"Peak mem: {s['peak_memory_kb']:.0f}KB")
    print("╚" + "═" * 70 + "╝")

    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Exhaustive TrainingDataBridge benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 1K-10K only")
    parser.add_argument("--full", action="store_true", help="Full mode: up to 1M records")
    args = parser.parse_args()

    if args.quick:
        scales = [100, 1000, 10000]
    elif args.full:
        scales = [100, 1000, 10000, 100000, 1000000]
    else:
        scales = [100, 1000, 10000, 100000]  # default: up to 100K

    print("TrainingDataBridge — Exhaustive Benchmarks")
    print(f"Scales: {[f'{s:,}' for s in scales]}")
    print(f"Python: {sys.version}")

    bench_throughput(scales)
    bench_dedup_accuracy()
    bench_format_correctness()
    bench_filter_accuracy()
    bench_edge_cases()
    bench_memory()

    summary = print_report()
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
