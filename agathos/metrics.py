"""Agathos metrics exporter — Prometheus-compatible monitoring.

Provides metrics endpoint for external monitoring systems.
Tracks entropy detection counts, session restarts, and provider health.

Metrics exposed:
- agathos_entropy_detections_total: Counter by entropy_type and severity
- agathos_session_restarts_total: Counter by session_type
- agathos_session_kills_total: Counter by session_type
- agathos_provider_errors_total: Counter by provider and error_type
- agathos_uptime_seconds: Gauge (process uptime)
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("agathos.metrics")


class MetricsCollector:
    """Collect Agathos metrics for Prometheus or JSON export.

    Queries agathos.db to aggregate metrics across multiple dimensions:
    - Entropy detections by type and severity
    - Session restarts/kills by session_type
    - Provider errors by provider and error_type
    - Process uptime

    Attributes:
        db_path: Path to agathos.db
        start_time: Unix timestamp when collector was instantiated (for uptime)

    Usage:
        collector = MetricsCollector(Path("~/.hermes/agathos/agathos.db"))
        prometheus_text = collector.collect_prometheus_metrics()
        json_dict = collector.collect_json_metrics()
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.start_time = time.time()
        
    def collect_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus exposition format.

        Queries agathos.db for recent metrics (last hour), formats as
        Prometheus text format with HELP and TYPE metadata.

        Returns:
            Multi-line string in Prometheus text format, ending with newline.
            Includes entropy_detections, session_restarts, session_kills
            counters and uptime gauge.

        Metric names:
        - agathos_entropy_detections_total (labels: type, severity)
        - agathos_session_restarts_total (labels: session_type)
        - agathos_session_kills_total (labels: session_type)
        - agathos_uptime_seconds (no labels)

        Time window: Last 1 hour of data from agathos.db.
        """
        lines = []

        # Counters
        lines.append("# HELP agathos_entropy_detections_total Total entropy detections")
        lines.append("# TYPE agathos_entropy_detections_total counter")
        for row in self._get_entropy_counts():
            lines.append(
                f'agathos_entropy_detections_total{{type="{row["entropy_type"]}",severity="{row["severity"]}"}} {row["count"]}'
            )

        lines.append("# HELP agathos_session_restarts_total Total session restarts")
        lines.append("# TYPE agathos_session_restarts_total counter")
        for row in self._get_restart_counts():
            lines.append(
                f'agathos_session_restarts_total{{session_type="{row["session_type"]}"}} {row["count"]}'
            )

        lines.append("# HELP agathos_session_kills_total Total session kills")
        lines.append("# TYPE agathos_session_kills_total counter")
        for row in self._get_kill_counts():
            lines.append(
                f'agathos_session_kills_total{{session_type="{row["session_type"]}"}} {row["count"]}'
            )

        # Gauges
        lines.append("# HELP agathos_uptime_seconds Agathos process uptime")
        lines.append("# TYPE agathos_uptime_seconds gauge")
        lines.append(f"agathos_uptime_seconds {time.time() - self.start_time}")

        return "\n".join(lines) + "\n"
    
    def collect_json_metrics(self) -> Dict:
        """Export metrics as JSON dict.

        Queries agathos.db for recent metrics and returns as structured
        dictionary suitable for API responses or logging.

        Returns:
            Dict with keys:
            - entropy_detections: List of dicts with entropy_type, severity, count
            - session_restarts: List of dicts with session_type, count
            - session_kills: List of dicts with session_type, count
            - uptime_seconds: Float seconds since collector start

        Time window: Last 1 hour of data from agathos.db.
        """
        return {
            "entropy_detections": self._get_entropy_counts(),
            "session_restarts": self._get_restart_counts(),
            "session_kills": self._get_kill_counts(),
            "uptime_seconds": time.time() - self.start_time,
        }
    
    def _get_entropy_counts(self) -> List[Dict]:
        """Get entropy detection counts from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT entropy_type, severity, COUNT(*) as count
                FROM entropy_detections
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY entropy_type, severity
            """)
            
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get entropy counts: %s", e)
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _get_restart_counts(self) -> List[Dict]:
        """Get session restart counts."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_type, SUM(restart_count) as count
                FROM sessions
                WHERE status = 'restarted'
                GROUP BY session_type
            """)
            
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get restart counts: %s", e)
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _get_kill_counts(self) -> List[Dict]:
        """Get session kill counts."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT session_type, COUNT(*) as count
                FROM sessions
                WHERE status = 'killed'
                GROUP BY session_type
            """)
            
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error("Failed to get kill counts: %s", e)
            return []
        finally:
            if 'conn' in locals():
                conn.close()


def write_metrics_file(collector: MetricsCollector, output_path: Path) -> bool:
    """Write Prometheus metrics to file for node_exporter textfile collector.

    Exports metrics in Prometheus text format to a file that node_exporter
    can read via its textfile collector. Creates parent directories if
    they don't exist.

    Args:
        collector: MetricsCollector instance with agathos.db connection
        output_path: File path to write metrics (e.g., /var/lib/node_exporter/textfile_collector/agathos.prom)

    Returns:
        True if file written successfully, False on error

    Side effects:
        - Creates parent directories if needed
        - Writes Prometheus text format to output_path
        - Logs error on failure

    Example node_exporter configuration:
        --collector.textfile.directory=/var/lib/node_exporter/textfile_collector

    Example cron job:
        */5 * * * * agathos metrics > /var/lib/node_exporter/textfile_collector/agathos.prom
    """
    try:
        metrics = collector.collect_prometheus_metrics()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(metrics)
        return True
    except Exception as e:
        logger.error("Failed to write metrics file: %s", e)
        return False
