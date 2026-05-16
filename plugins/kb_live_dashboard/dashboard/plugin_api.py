"""KB live dashboard plugin backend.

The dashboard plugin renders the same kb-engine ``dashboard.live`` packet used
by Telegram.  The command is intentionally supplied by deployment config so the
web UI does not learn KB topology or bypass MCP boundaries.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from typing import Any

from fastapi import APIRouter, HTTPException, Query


router = APIRouter()


@router.get("/live")
async def live_dashboard(limit: int = Query(default=8, ge=1, le=50)) -> dict[str, Any]:
    command = _dashboard_command(limit=limit)
    try:
        proc = subprocess.run(
            command,
            cwd=os.environ.get("HERMES_KB_WORKSPACE") or None,
            text=True,
            capture_output=True,
            timeout=float(os.environ.get("HERMES_KB_DASHBOARD_TIMEOUT_SECONDS", "20") or 20),
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="KB dashboard command is not available in this Hermes environment.",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="KB dashboard command timed out.") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "KB dashboard command failed.").strip()
        raise HTTPException(status_code=502, detail=_safe_detail(detail))
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="KB dashboard command returned non-JSON output.") from exc
    payload = _unwrap_payload(raw)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=502, detail="KB dashboard command did not return an object packet.")
    return {
        "ok": True,
        "source": "dashboard.live",
        "payload": payload,
    }


def _dashboard_command(*, limit: int) -> list[str]:
    _refresh_hermes_env()
    configured = (
        os.environ.get("HERMES_KB_DASHBOARD_COMMAND")
        or os.environ.get("KB_DASHBOARD_COMMAND")
        or ""
    ).strip()
    args_json = json.dumps(
        {
            "limit": int(limit),
            "include_feedback": True,
            "include_publication": True,
            "include_readiness": True,
        },
        separators=(",", ":"),
    )
    if configured:
        return [part.replace("{args_json}", args_json).replace("{limit}", str(limit)) for part in shlex.split(configured)]
    ssh_target = os.environ.get("HERMES_KB_MCP_SSH_TARGET", "").strip()
    workspace = os.environ.get("HERMES_KB_WORKSPACE", "").strip()
    kb_bin = (os.environ.get("HERMES_KB_BIN") or os.environ.get("HERMES_KB_PROD_BIN") or "").strip()
    if ssh_target and workspace and kb_bin:
        remote = " ".join(
            [
                "source ~/.localrc 2>/dev/null || true;",
                "source ~/.secrets 2>/dev/null || true;",
                "export KB_ENGINE_MCP_PROFILE=journey_first_strict;",
                "cd",
                shlex.quote(workspace),
                "&&",
                shlex.quote(kb_bin),
                "mcp call dashboard.live --arguments-json",
                shlex.quote(args_json),
                "--json",
            ]
        )
        return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", ssh_target, "zsh -lc " + shlex.quote(remote)]
    return [
        "kb",
        "mcp",
        "call",
        "dashboard.live",
        "--arguments-json",
        args_json,
        "--json",
    ]


def _refresh_hermes_env() -> None:
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
    except Exception:
        return
    try:
        load_hermes_dotenv()
    except Exception:
        return


def _unwrap_payload(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        return structured.get("result") or structured.get("payload") or structured
    return raw.get("result") or raw.get("payload") or raw


def _safe_detail(value: str) -> str:
    text = value.strip().replace(os.path.expanduser("~"), "~")
    if len(text) > 400:
        text = text[:397] + "..."
    return text
