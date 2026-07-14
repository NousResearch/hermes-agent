"""Auto-Export — weekly training data export for the Atropos flywheel.

Follows the curator pattern: idle-gated, runs in background, no user
interaction needed. Every 7 days (configurable), if Hermes has been idle
for 2+ hours, exports all recent evolution runs as Atropos-compatible
training records.

This closes the flywheel: HAEE usage → training data → better models.

Config:
  evolution:
    auto_export:
      enabled: true
      interval_hours: 168       # 7 days
      min_idle_minutes: 120     # 2 hours idle
      retention_days: 90        # Keep exports for 90 days
      output_dir: null          # Default: ~/.hermes/evolution/exports/
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_HOURS = 168  # 7 days
DEFAULT_MIN_IDLE_MINUTES = 120  # 2 hours
DEFAULT_RETENTION_DAYS = 90
STATE_FILE = "auto_export_state.json"


class AutoExport:
    """Background training data exporter for the Atropos flywheel."""

    def __init__(self):
        self._last_export_at: Optional[float] = None
        self._export_count: int = 0
        self._lock = threading.Lock()
        self._load_state()

    @property
    def is_due(self) -> bool:
        """Check if enough time has passed since last export."""
        if self._last_export_at is None:
            return True
        interval = self._get_interval()
        return (time.time() - self._last_export_at) >= interval

    def maybe_export(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """Export training data if due. Returns stats dict or None."""
        if not force and not self.is_due:
            return None

        # Check idle time
        if not force and not self._is_idle_enough():
            return None

        with self._lock:
            try:
                from agent.evolution.atropos_export import export_all_runs, get_export_stats
            except Exception as e:
                logger.debug("Auto-export skipped (import error): %s", e)
                return None

            try:
                stats = get_export_stats(days=self._get_retention_days())
                if stats["total_runs"] == 0 and not force:
                    self._update_last_export()
                    return None

                output_dir = self._get_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                output_path = output_dir / f"training_data_{timestamp}.jsonl"

                records = export_all_runs(
                    days=self._get_retention_days(),
                    output_path=output_path,
                )

                self._export_count += 1
                self._update_last_export()

                # Write manifest
                manifest = {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "records": len(records),
                    "period_days": self._get_retention_days(),
                    "total_exports": self._export_count,
                    "run_stats": stats,
                }
                manifest_path = output_dir / f"manifest_{timestamp}.json"
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2, default=str)

                # Clean up old exports
                self._prune_old_exports(output_dir)

                logger.info(
                    "Auto-export: %d records → %s (export #%d)",
                    len(records), output_path, self._export_count,
                )
                return manifest

            except Exception as e:
                logger.warning("Auto-export failed: %s", e)
                return None

    # ── Config helpers ─────────────────────────────────────────────────

    def _get_interval(self) -> float:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            evo = cfg.get("evolution", {}) if isinstance(cfg, dict) else {}
            auto = evo.get("auto_export", {}) if isinstance(evo, dict) else {}
            return float(auto.get("interval_hours", DEFAULT_INTERVAL_HOURS)) * 3600
        except Exception:
            return DEFAULT_INTERVAL_HOURS * 3600

    def _is_idle_enough(self) -> bool:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            evo = cfg.get("evolution", {}) if isinstance(cfg, dict) else {}
            auto = evo.get("auto_export", {}) if isinstance(evo, dict) else {}
            min_idle = float(auto.get("min_idle_minutes", DEFAULT_MIN_IDLE_MINUTES)) * 60
        except Exception:
            min_idle = DEFAULT_MIN_IDLE_MINUTES * 60

        # Check if any recent sessions are active
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            recent = db.list_sessions(limit=1)
            db.close()
            if recent and hasattr(recent[0], "ended_at") if recent else False:
                last_ended = recent[0].ended_at
                if last_ended:
                    idle_seconds = time.time() - last_ended
                    return idle_seconds >= min_idle
        except Exception:
            pass
        return True  # Default: idle enough if we can't check

    def _get_retention_days(self) -> int:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            evo = cfg.get("evolution", {}) if isinstance(cfg, dict) else {}
            auto = evo.get("auto_export", {}) if isinstance(evo, dict) else {}
            return int(auto.get("retention_days", DEFAULT_RETENTION_DAYS))
        except Exception:
            return DEFAULT_RETENTION_DAYS

    def _get_output_dir(self) -> Path:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            evo = cfg.get("evolution", {}) if isinstance(cfg, dict) else {}
            auto = evo.get("auto_export", {}) if isinstance(evo, dict) else {}
            custom = auto.get("output_dir")
            if custom:
                return Path(custom)
        except Exception:
            pass
        return get_hermes_home() / "evolution" / "exports"

    def _prune_old_exports(self, output_dir: Path, keep_days: int = 90) -> None:
        """Remove exports older than keep_days."""
        cutoff = time.time() - (keep_days * 86400)
        for f in output_dir.glob("training_data_*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass
        for f in output_dir.glob("manifest_*.json"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except OSError:
                pass

    # ── State persistence ──────────────────────────────────────────────

    def _update_last_export(self) -> None:
        self._last_export_at = time.time()
        self._save_state()

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._last_export_at = data.get("last_export_at")
            self._export_count = data.get("export_count", 0)
        except Exception:
            pass

    def _save_state(self) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "last_export_at": self._last_export_at,
                    "export_count": self._export_count,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }, f)
        except Exception:
            pass

    def _state_path(self) -> Path:
        return get_hermes_home() / "evolution" / STATE_FILE


# ── Singleton ──────────────────────────────────────────────────────────

_auto_export: Optional[AutoExport] = None


def get_auto_export() -> AutoExport:
    global _auto_export
    if _auto_export is None:
        _auto_export = AutoExport()
    return _auto_export
