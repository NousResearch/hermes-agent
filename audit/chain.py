"""Audit hash chain for tamper protection.

Each entry contains prev_hash (SHA256 of previous entry) and hash (SHA256 of current entry).
This forms a chain that can be verified to detect any tampering with historical entries.
"""

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterator, Optional

from audit.events import AuditEvent


class AuditHashChain:
    """
    Thread-safe hash chain for audit log tamper protection.

    Each entry contains:
    - prev_hash: SHA256 of previous entry (empty string for first entry)
    - hash: SHA256 of current entry (excluding hash and prev_hash fields)

    The chain can be verified to detect any tampering with historical log entries.
    """

    def __init__(self, log_path: str, hash_algorithm: str = "sha256"):
        """
        Initialize hash chain.

        Args:
            log_path: Directory where audit logs are stored
            hash_algorithm: Hash algorithm to use (default sha256)
        """
        self._log_path = Path(log_path).expanduser().resolve()
        self._hash_algorithm = hash_algorithm
        self._lock = threading.Lock()
        self._last_hash = self._read_last_hash()

    def _read_last_hash(self) -> str:
        """Read the hash of the last entry in the chain."""
        if not self._log_path.exists():
            return ""

        # Find the most recent audit log file
        log_files = sorted(self._log_path.glob("audit-*.jsonl"), reverse=True)
        if not log_files:
            return ""

        last_file = log_files[0]
        with open(last_file, "r", encoding="utf-8") as f:
            # Read last line
            last_line = None
            for line in f:
                last_line = line.strip()
            if last_line:
                try:
                    entry = json.loads(last_line)
                    return entry.get("hash", "")
                except json.JSONDecodeError:
                    pass
        return ""

    def _compute_hash(self, entry: dict) -> str:
        """Compute hash for an entry (excludes hash and prev_hash fields)."""
        # Create canonical representation excluding hash fields
        canonical = {
            k: v for k, v in entry.items() if k not in ("hash", "prev_hash")
        }
        content = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def write_entry(self, entry: dict) -> dict:
        """
        Add hash chain fields to entry (does NOT write to file).

        Args:
            entry: Audit event dict

        Returns:
            Entry with prev_hash and hash fields added
        """
        with self._lock:
            entry["prev_hash"] = self._last_hash
            entry["hash"] = self._compute_hash(entry)
            self._last_hash = entry["hash"]
            return entry

    def _get_current_log_file(self) -> Path:
        """Get the current log file path (today's date)."""
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        return self._log_path / f"audit-{date_str}.jsonl"

    @staticmethod
    def verify(log_path: str, hash_algorithm: str = "sha256") -> tuple[bool, list]:
        """
        Verify the entire chain integrity.

        Args:
            log_path: Directory containing audit log files
            hash_algorithm: Hash algorithm used (sha256)

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        log_path = Path(log_path).expanduser().resolve()
        errors = []

        if not log_path.exists():
            return True, []  # Empty log is valid

        # Collect all log files in order
        log_files = sorted(log_path.glob("audit-*.jsonl"))

        prev_hash = ""
        entry_count = 0

        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        errors.append(
                            f"{log_file.name}:{line_num} - JSON decode error: {e}"
                        )
                        continue

                    # Verify hash field matches computed hash
                    computed = AuditHashChain._compute_hash_static(entry, hash_algorithm)
                    if entry.get("hash", "") != computed:
                        errors.append(
                            f"{log_file.name}:{line_num} - hash mismatch: "
                            f"expected {computed}, got {entry.get('hash', '')}"
                        )

                    # Verify prev_hash matches expected previous
                    if entry.get("prev_hash", "") != prev_hash:
                        errors.append(
                            f"{log_file.name}:{line_num} - prev_hash mismatch: "
                            f"expected {prev_hash}, got {entry.get('prev_hash', '')}"
                        )

                    prev_hash = entry.get("hash", "")
                    entry_count += 1

        return len(errors) == 0, errors

    @staticmethod
    def _compute_hash_static(entry: dict, algorithm: str = "sha256") -> str:
        """Static method to compute hash for an entry."""
        canonical = {
            k: v for k, v in entry.items() if k not in ("hash", "prev_hash")
        }
        content = json.dumps(canonical, sort_keys=True, ensure_ascii=True)
        if algorithm == "sha256":
            return hashlib.sha256(content.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")