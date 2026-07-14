"""Best-effort JSON usage reports for non-interactive Hermes runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def write_usage_file(path: Optional[str], result: dict, failure: Optional[str] = None) -> None:
    """Write a pipeline usage report without masking the run outcome."""
    if not path:
        return
    try:
        report = {
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "cost_status": result.get("cost_status"),
            "cost_source": result.get("cost_source"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_read_tokens": result.get("cache_read_tokens"),
            "cache_write_tokens": result.get("cache_write_tokens"),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "total_tokens": result.get("total_tokens"),
            "api_calls": result.get("api_calls"),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "session_id": result.get("session_id"),
            "completed": result.get("completed"),
            "failed": bool(result.get("failed")) or failure is not None,
            "service_tier": result.get("service_tier"),
        }
        if failure is not None:
            report["failure"] = failure
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
