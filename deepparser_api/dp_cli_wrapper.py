from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .config import (
    DPCLI_APP_KEY,
    DPCLI_BASE_URL,
    DPCLI_BIN,
    DPCLI_FOLDER_ID,
    PARSE_TIMEOUT_SECS,
)
from .models import Citation, DPCLIAskResult, DPCLIParseResult

logger = logging.getLogger(__name__)


def _build_env() -> dict[str, str] | None:
    """Inject DPCLI_* env vars if configured, so the subprocess picks them up."""
    import os
    env = dict(os.environ)
    if DPCLI_BASE_URL:
        env["DPCLI_BASE_URL"] = DPCLI_BASE_URL
    if DPCLI_APP_KEY:
        env["DPCLI_APP_KEY"] = DPCLI_APP_KEY
    return env


async def run_dp_parse(file_path: str) -> DPCLIParseResult:
    """Upload a file to DeepParser, trigger parse, wait for READY.

    Raises RuntimeError with stderr detail on non-zero exit or timeout.
    """
    if not DPCLI_FOLDER_ID:
        raise RuntimeError(
            "DPCLI_FOLDER_ID env var is not set. "
            "Create a folder in DeepParser and set DPCLI_FOLDER_ID to its ID."
        )

    cmd = [
        DPCLI_BIN, "file", "upload",
        "--file", file_path,
        "--folder-id", DPCLI_FOLDER_ID,
        "--trigger-parse",
        "--wait-ready",
        "--wait-timeout", str(PARSE_TIMEOUT_SECS),
    ]
    logger.info("dp parse start file=%s", Path(file_path).name)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_build_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=PARSE_TIMEOUT_SECS + 10
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("dp_cli process exceeded hard timeout")

    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"dp exit {proc.returncode}: {stderr_text[:500]}")

    raw = stdout.decode(errors="replace").strip()
    return _parse_upload_result(raw)


def _parse_upload_result(raw: str) -> DPCLIParseResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dp_cli returned non-JSON: {raw[:200]}") from exc

    if not data.get("ok"):
        err = data.get("error") or str(data)
        raise RuntimeError(f"dp upload failed: {err[:400]}")

    uploads = data.get("uploads") or []
    if not uploads:
        raise RuntimeError(f"dp upload returned no uploads: {raw[:200]}")

    first = uploads[0]
    result_body = (
        first.get("result", {})
        .get("response", {})
        .get("body", {})
    )
    file_data = result_body.get("data") or {}
    file_id = str(file_data.get("id") or "")
    folder_id = str(file_data.get("folder_id") or DPCLI_FOLDER_ID)
    file_name = str(file_data.get("file_name") or "")
    extension = str(file_data.get("extension") or "")

    if not file_id:
        raise RuntimeError(f"dp upload: no file_id in response: {raw[:200]}")

    pages = None
    tables = None
    # Try to extract from ready.status if present
    ready_status = first.get("ready", {}).get("status", {})
    tasks_info = ready_status.get("tasks") or {}
    parse_task = tasks_info.get("plugin_parse") or {}
    if isinstance(parse_task, dict):
        pages = parse_task.get("pages")
        tables = parse_task.get("tables")

    return DPCLIParseResult(
        file_id=file_id,
        folder_id=folder_id,
        file_name=file_name,
        extension=extension,
        pages=pages,
        tables=tables,
    )


async def run_dp_ask(file_id: str, question: str) -> DPCLIAskResult:
    """Ask a question against a parsed file using dp chat ask."""
    cmd = [
        DPCLI_BIN, "chat", "ask",
        "--config-type", "file",
        "--config-id", file_id,
        "--query", question,
    ]
    logger.info("dp ask file_id=%s question=%.60s", file_id, question)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_build_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=35)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("dp chat ask timed out after 35s")

    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"dp ask exit {proc.returncode}: {stderr_text[:500]}")

    raw = stdout.decode(errors="replace").strip()
    return _parse_ask_result(raw)


def _parse_ask_result(raw: str) -> DPCLIAskResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dp chat ask non-JSON: {raw[:200]}") from exc

    if not data.get("ok"):
        err = data.get("error") or str(data)
        raise RuntimeError(f"dp ask failed: {err[:400]}")

    body = data.get("response", {}).get("body", {}) or {}
    answer = str(body.get("answer") or body.get("content") or "")

    raw_cites = body.get("citations") or body.get("references") or []
    citations: list[Citation] = []
    for c in raw_cites:
        if isinstance(c, dict):
            citations.append(Citation(
                filename=str(c.get("filename") or c.get("file_name") or ""),
                page=c.get("page"),
                cell=c.get("cell"),
            ))

    return DPCLIAskResult(answer=answer, citations=citations)
