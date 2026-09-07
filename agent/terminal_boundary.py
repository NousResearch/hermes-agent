"""External Core Control application adapter for Hermes turn metadata."""
from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_INTERNAL_PLATFORMS = frozenset({"subagent", "internal", "tool"})
_CORE_CONTROL_ROOT = Path("/Users/peterbeddow/Core/core-control")


def enforce_public_terminal_response(result: dict[str, Any], *, platform: str = "") -> dict[str, Any]:
    """Ask Core Control for a disposition while preserving Hermes's result verbatim."""
    if not isinstance(result, dict) or platform in _INTERNAL_PLATFORMS:
        return result

    request = {
        "protocol_version": "hermes-core-control/v1",
        "request_id": uuid.uuid4().hex,
        "source": "hermes",
        "platform": platform,
        "turn_result": result,
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "core_control", "validate-public-response", json.dumps(result)],
            cwd=_CORE_CONTROL_ROOT,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Core Control exited with status {completed.returncode}")
        disposition = json.loads(completed.stdout)
        if not isinstance(disposition, dict):
            raise ValueError("Core Control returned a non-object disposition")
        result["core_control"] = {
            "protocol_version": disposition.get("protocol_version"),
            "request_id": disposition.get("request_id"),
            "outcome": disposition.get("disposition"),
            "terminal": disposition.get("terminal") is True,
            "verified": disposition.get("verified") is True,
            "response_preserved": disposition.get("response_preserved") is True,
        }
    except Exception as exc:
        # Core Control is advisory application metadata. Its failure must not erase
        # or replace a successful provider response.
        result["core_control"] = {
            "protocol_version": request["protocol_version"],
            "request_id": request["request_id"],
            "terminal": False,
            "available": False,
            "error": type(exc).__name__,
        }
    return result
