"""ARGUS metrics exporter — Prometheus-compatible monitoring.

Provides metrics endpoint for external monitoring systems.
Tracks entropy detection counts, session restarts, and provider health.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("argus.metrics")


class MetricsCollector:
    """Collect ARGUS metrics for export.
    
    Tracks:
    - entropy_detections_total (counter by type and severity)
    - session_restarts_total (counter by session_type)
    - session_kills_total (counter by session_type)
    - provider_errors_total (counter by provider and error_type)
    - argus_uptime_seconds (gauge)
    - argus_poll_interval_seconds (gauge)
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.start_time = time.time()
        
    def collect_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        # Counters
        lines.append("# HELP argus_entropy_detections_total Total entropy detections")
        lines.append("# TYPE argus_entropy_detections_total counter")
        for row in self._get_entropy_counts():
            lines.append(
                f'argus_entropy_detections_total{{type="{row["entropy_type"]}",severity="{row["severity"]}"}} {row["count"]}'
            )
        
        lines.append("# HELP argus_session_restarts_total Total session restarts")
        lines.append("# TYPE argus_session_restarts_total counter")
        for row in self._get_restart_counts():
            lines.append(
                f'argus_session_restarts_total{{session_type="{row["session_type"]}"}} {row["count"]}'
            )
        
        lines.append("# HELP argus_session_kills_total Total session kills")
        lines.append("# TYPE argus_session_kills_total counter")
        for row in self._get_kill_counts():
            lines.append(
                f'argus_session_kills_total{{session_type="{row["session_type"]}"}} {row["count"]}'
            )
        
        # Gauges
        lines.append("# HELP argus_uptime_seconds ARGUS process uptime")
        lines.append("# TYPE argus_uptime_seconds gauge")
        lines.append(f"argus_uptime_seconds {time.time() - self.start_time}")
        
        return "\n".join(lines) + "\n"
    
    def collect_json_metrics(self) -> Dict:
        """Export metrics as JSON dict."""
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
    
    Usage with node_exporter:
      --collector.textfile.directory=/var/lib/node_exporter/textfile_collector
    """
    try:
        metrics = collector.collect_prometheus_metrics()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(metrics)
        return True
    except Exception as e:
        logger.error("Failed to write metrics file: %s", e)
        return False
