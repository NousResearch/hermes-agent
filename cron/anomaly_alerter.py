"""Alert dispatcher for cron output anomalies.

Writes anomalies to a local JSONL log and optionally POSTs them
to a mission-control instance for dashboard visibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_ANOMALY_LOG_PATH = get_hermes_home() / "cron" / "anomalies.jsonl"


def alert(
    job_id: str,
    job_name: str,
    anomalies: List[Any],
    output_path: str = "",
) -> None:
    """Dispatch anomaly alerts through all configured channels.

    Args:
        job_id: The cron job ID.
        job_name: Human-readable job name.
        anomalies: List of Anomaly objects (from anomaly_detector.py).
        output_path: Path to the cron output file that triggered the anomaly.
    """
    if not anomalies:
        return

    record = {
        "timestamp": datetime.now().isoformat(),
        "job_id": job_id,
        "job_name": job_name,
        "output_path": output_path,
        "anomaly_count": len(anomalies),
        "anomalies": [a.to_dict() for a in anomalies],
        "max_severity": _max_severity(anomalies),
    }

    # Channel 1: Local JSONL log (always)
    _write_local_log(record)

    # Channel 2: Mission Control API (if configured)
    mc_url = os.getenv("MISSION_CONTROL_URL")
    mc_api_key = os.getenv("MC_API_KEY") or os.getenv("API_KEY")
    if mc_url:
        _post_to_mission_control(mc_url, mc_api_key, record)

    # Channel 3: Python logger (always)
    severity = record["max_severity"]
    summary = ", ".join(
        f"{a.field} (z={a.z_score:.1f})" for a in anomalies
    )
    log_fn = logger.critical if severity == "critical" else logger.warning
    log_fn(
        "Cron anomaly detected for job '%s' [%s]: %s",
        job_name, severity.upper(), summary,
    )


def _max_severity(anomalies: List[Any]) -> str:
    """Return the highest severity from a list of anomalies."""
    if any(a.severity == "critical" for a in anomalies):
        return "critical"
    return "warning"


def _write_local_log(record: Dict[str, Any]) -> None:
    """Append anomaly record to local JSONL file."""
    try:
        _ANOMALY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_ANOMALY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to write anomaly log: %s", e)


def _post_to_mission_control(
    base_url: str,
    api_key: str | None,
    record: Dict[str, Any],
) -> None:
    """POST anomaly report to mission-control alerts API."""
    import urllib.request
    import urllib.error

    url = f"{base_url.rstrip('/')}/api/cron/anomalies"
    payload = json.dumps(record).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status < 300:
                logger.debug("Anomaly reported to mission-control: %s", resp.status)
            else:
                logger.warning("Mission-control returned %s", resp.status)
    except urllib.error.URLError as e:
        logger.debug("Could not reach mission-control at %s: %s", url, e)
    except Exception as e:
        logger.debug("Failed to post anomaly to mission-control: %s", e)
