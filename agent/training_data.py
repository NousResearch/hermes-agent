"""Training Data Bridge — validates, deduplicates, and exports agent trajectories.

Reads Hermes batch_runner output (ShareGPT JSONL), validates format,
deduplicates by content hash, filters by score/domain/complexity, and
exports in standard formats consumable by any training framework.

Self-contained. No dependency on HAEE. Works with the existing batch_runner
pipeline today. When HAEE lands, its Atropos export uses the exact same
ShareGPT schema, so this bridge works with both without changes.

Usage:
    from agent.training_data import TrainingDataBridge
    bridge = TrainingDataBridge()
    stats = bridge.get_stats(days=30)
    bridge.export(output_path="/tmp/training.jsonl", format="sharegpt")
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "sharegpt"
SUPPORTED_FORMATS = {"sharegpt", "parquet", "alpaca"}
DEFAULT_RETENTION_DAYS = 90


class TrainingDataBridge:
    """Validates, deduplicates, and exports agent training data."""

    def __init__(self, data_dir: Optional[Path] = None, input_paths: Optional[List[Path]] = None):
        self.data_dir = data_dir or get_hermes_home() / "training_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.data_dir / ".bridge_state.json"
        self._input_paths = input_paths or []  # Additional paths to scan
        self._state = self._load_state()

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self, days: int = DEFAULT_RETENTION_DAYS) -> Dict[str, Any]:
        """Get statistics on available training data."""
        records = self._find_records(days)
        unique = self._deduplicate(records)
        scored = [r for r in unique if r.get("metadata", {}).get("score") is not None]

        by_domain: Dict[str, int] = {}
        for r in unique:
            domain = r.get("metadata", {}).get("task_domain", "general")
            by_domain[domain] = by_domain.get(domain, 0) + 1

        return {
            "total_records": len(records),
            "unique_records": len(unique),
            "scored_records": len(scored),
            "new_since_last_export": self._count_new(unique),
            "by_domain": by_domain,
            "last_export_at": self._state.get("last_export_at"),
            "period_days": days,
        }

    # ── Export ─────────────────────────────────────────────────────────

    def export(
        self,
        output_path: Optional[Path] = None,
        fmt: str = DEFAULT_FORMAT,
        min_score: float = 0.0,
        domain: Optional[str] = None,
        since_last: bool = False,
        max_records: int = 10000,
    ) -> Dict[str, Any]:
        """Export training data in the specified format.

        Args:
            output_path: Where to write. Auto-generated if None.
            fmt: "sharegpt", "parquet", or "alpaca"
            min_score: Minimum evaluation score (0.0-1.0)
            domain: Filter by task domain
            since_last: Only export records newer than last export
            max_records: Maximum records to export

        Returns:
            Dict with export stats (path, count, format, etc.)
        """
        if fmt not in SUPPORTED_FORMATS:
            return {"error": f"Unsupported format '{fmt}'. Use: {', '.join(SUPPORTED_FORMATS)}"}

        if output_path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self.data_dir / f"training_data_{ts}.{fmt if fmt != 'sharegpt' else 'jsonl'}"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Find and filter
        records = self._find_records()
        records = self._deduplicate(records)

        if min_score > 0:
            records = [
                r for r in records
                if r.get("metadata", {}).get("score", 0) >= min_score
            ]

        if domain:
            records = [
                r for r in records
                if r.get("metadata", {}).get("task_domain", "general") == domain
            ]

        if since_last and self._state.get("last_export_at"):
            records = [
                r for r in records
                if r.get("_ingested_at", "") > self._state["last_export_at"]
            ]

        records = records[:max_records]

        if not records:
            return {"error": "No records match the specified filters"}

        # Export
        if fmt == "sharegpt":
            self._export_sharegpt(records, output_path)
        elif fmt == "alpaca":
            self._export_alpaca(records, output_path)
        elif fmt == "parquet":
            self._export_parquet(records, output_path)

        # Update state
        self._state["last_export_at"] = datetime.now(timezone.utc).isoformat()
        self._state["total_exports"] = self._state.get("total_exports", 0) + 1
        self._save_state()

        return {
            "records_exported": len(records),
            "format": fmt,
            "output_path": str(output_path),
            "filters": {"min_score": min_score, "domain": domain, "since_last": since_last},
        }

    # ── Format exporters ───────────────────────────────────────────────

    def _export_sharegpt(self, records: List[Dict], path: Path) -> None:
        """Export as ShareGPT JSONL (Hermes/Atropos native format)."""
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def _export_alpaca(self, records: List[Dict], path: Path) -> None:
        """Export as Alpaca JSON (instruction/input/output)."""
        alpaca = []
        for r in records:
            conversations = r.get("conversations", [])
            instruction = ""
            output = ""
            for turn in conversations:
                if turn.get("from") == "human":
                    instruction = turn.get("value", "")
                elif turn.get("from") == "gpt":
                    output = turn.get("value", "")
            if instruction and output:
                alpaca.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "metadata": r.get("metadata", {}),
                })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(alpaca, f, ensure_ascii=False, indent=2, default=str)

    def _export_parquet(self, records: List[Dict], path: Path) -> None:
        """Export as Parquet (requires pandas/pyarrow)."""
        try:
            import pandas as pd
            df = pd.json_normalize(records, max_level=2)
            df.to_parquet(path)
        except ImportError:
            logger.warning("pandas/pyarrow not available — falling back to JSONL")
            self._export_sharegpt(records, path.with_suffix(".jsonl"))

    # ── Record management ──────────────────────────────────────────────

    def _find_records(self, days: int = DEFAULT_RETENTION_DAYS) -> List[Dict]:
        """Find all training records within the retention window."""
        records = []
        cutoff = time.time() - (days * 86400)

        # Scan data_dir + any input paths
        search_paths = [self.data_dir] + list(self._input_paths)
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for jsonl_file in (search_path.glob("*.jsonl") if search_path.is_dir() else [search_path]):
                    try:
                        if jsonl_file.stat().st_mtime < cutoff:
                            continue
                        with open(jsonl_file, encoding="utf-8-sig") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    record = json.loads(line)
                                    record["_source_file"] = str(jsonl_file)
                                    record["_ingested_at"] = datetime.fromtimestamp(
                                        jsonl_file.stat().st_mtime, tz=timezone.utc
                                    ).isoformat()
                                    records.append(record)
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        continue

        return records

    def _deduplicate(self, records: List[Dict]) -> List[Dict]:
        """Deduplicate by content hash, keeping first occurrence."""
        seen: set = set()
        unique = []
        for r in records:
            h = self._hash_record(r)
            if h not in seen:
                seen.add(h)
                unique.append(r)
        return unique

    def _count_new(self, records: List[Dict]) -> int:
        """Count records newer than the last export."""
        last = self._state.get("last_export_at", "")
        if not last:
            return len(records)
        return sum(1 for r in records if r.get("_ingested_at", "") > last)

    @staticmethod
    def _hash_record(record: Dict) -> str:
        """Stable content hash for deduplication."""
        conversations = json.dumps(record.get("conversations", []), sort_keys=True)
        return hashlib.sha256(conversations.encode()).hexdigest()

    # ── State persistence ──────────────────────────────────────────────

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {"total_exports": 0}
        try:
            with open(self._state_path) as f:
                return json.load(f)
        except Exception:
            return {"total_exports": 0}

    def _save_state(self) -> None:
        try:
            with open(self._state_path, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception:
            pass
