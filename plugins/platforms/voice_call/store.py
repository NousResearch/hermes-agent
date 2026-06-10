"""JSONL persistence for call records.

Append-only log under ``$HERMES_HOME/voice-calls/calls.jsonl`` — every
state change writes one full ``CallRecord`` line *before* control returns
to the caller, so a crash never loses more than the in-flight change. Boot
replays the file keeping the latest record per call; the runtime then
verifies each non-terminal call with the provider.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from hermes_constants import get_hermes_home

from .events import CallRecord

logger = logging.getLogger(__name__)

# Compact the log when boot replay sees this many lines.
_COMPACT_THRESHOLD = 5000


class CallStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else get_hermes_home() / "voice-calls"
        self.calls_path = self.base_dir / "calls.jsonl"

    def _ensure_dir(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: CallRecord) -> None:
        """Persist one full snapshot of ``record``."""
        self._ensure_dir()
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with open(self.calls_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_latest(self) -> Dict[str, CallRecord]:
        """Replay the log into ``{call_id: latest record}``."""
        if not self.calls_path.exists():
            return {}
        latest: Dict[str, CallRecord] = {}
        lines = 0
        try:
            with open(self.calls_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines += 1
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            raise ValueError("JSONL line is not an object")
                        record = CallRecord.from_dict(data)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        logger.warning("voice_call store: skipping corrupt JSONL line")
                        continue
                    latest[record.call_id] = record
        except OSError as e:
            logger.warning("voice_call store: failed to read %s: %s", self.calls_path, e)
            return {}
        if lines > _COMPACT_THRESHOLD:
            self._compact(latest)
        return latest

    def load_active(self) -> Dict[str, CallRecord]:
        """Latest records that are not in a terminal state."""
        return {
            call_id: record
            for call_id, record in self.load_latest().items()
            if not record.is_terminal
        }

    def history(self, limit: int = 50) -> List[CallRecord]:
        """Most recent calls (latest record per call), newest first."""
        records = sorted(
            self.load_latest().values(), key=lambda r: r.started_at, reverse=True
        )
        return records[:limit]

    def _compact(self, latest: Dict[str, CallRecord]) -> None:
        """Rewrite the log with one line per call (latest snapshot)."""
        try:
            self._ensure_dir()
            tmp_path = self.calls_path.with_suffix(".jsonl.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for record in sorted(latest.values(), key=lambda r: r.started_at):
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            tmp_path.replace(self.calls_path)
            logger.info("voice_call store: compacted calls.jsonl to %d records", len(latest))
        except OSError as e:
            logger.warning("voice_call store: compaction failed: %s", e)
