"""Agent-facing MoneyPrinter tool implementations.

These call the same capability adapter layer used by Desktop Video Studio.
They never import MoneyPrinterTurbo internals.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional


def _run(coro: Any) -> Any:
    """Run an async adapter call from a sync MCP tool handler."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Nested event loop (rare for MCP stdio): use a private loop thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _json_result(payload: dict[str, Any], *, status: int = 200) -> str:
    body = dict(payload)
    body.setdefault("httpStatus", status)
    return json.dumps(body, ensure_ascii=False, indent=2)


async def _ensure_service_if_needed() -> dict[str, Any]:
    from capabilities.moneyprinter import adapter

    health_resp = await adapter.health()
    health = json.loads(health_resp.body.decode("utf-8"))
    data = health.get("data") or {}
    if data.get("serviceRunning"):
        return health
    if not data.get("installed"):
        return health
    start_resp = await adapter.start_service()
    return json.loads(start_resp.body.decode("utf-8"))


def moneyprinter_health_check(start_if_needed: bool = False) -> str:
    """Check MoneyPrinterTurbo install/config/service status.

    Args:
        start_if_needed: If true, attempt to start the sidecar when installed but not running.
    """

    async def _go() -> dict[str, Any]:
        from capabilities.moneyprinter import adapter

        if start_if_needed:
            return await _ensure_service_if_needed()
        response = await adapter.health()
        return json.loads(response.body.decode("utf-8"))

    return _json_result(_run(_go()))


def moneyprinter_start_service() -> str:
    """Start the MoneyPrinterTurbo FastAPI sidecar if it is not already running."""

    async def _go() -> dict[str, Any]:
        from capabilities.moneyprinter import adapter

        response = await adapter.start_service()
        return json.loads(response.body.decode("utf-8"))

    return _json_result(_run(_go()))


def moneyprinter_generate_video(
    video_subject: str,
    video_script: str = "",
    video_language: str = "zh-CN",
    video_aspect: str = "9:16",
    video_count: int = 1,
    video_clip_duration: int = 5,
    video_source: str = "pexels",
    voice_name: str = "zh-CN-XiaoxiaoNeural-Female",
    subtitle_enabled: bool = True,
    bgm_type: str = "random",
    auto_start: bool = True,
) -> str:
    """Create a full video generation task. Do not wait for completion.

    Returns task_id immediately. Poll with moneyprinter_get_task.

    Args:
        video_subject: Required topic / title.
        video_script: Optional full script. Empty means MoneyPrinter generates one.
        video_language: e.g. zh-CN, en-US.
        video_aspect: 9:16 (default short), 16:9, or 1:1.
        video_count: Number of output variants (1-5 recommended).
        video_clip_duration: Seconds per clip segment.
        video_source: pexels | pixabay | coverr | local.
        voice_name: Edge/Azure TTS voice id.
        subtitle_enabled: Whether to burn subtitles.
        bgm_type: random | none | custom path semantics upstream.
        auto_start: Start sidecar automatically when needed.
    """

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        body = {
            "video_subject": video_subject,
            "video_script": video_script,
            "video_language": video_language,
            "video_aspect": video_aspect,
            "video_count": video_count,
            "video_clip_duration": video_clip_duration,
            "video_source": video_source,
            "voice_name": voice_name,
            "subtitle_enabled": subtitle_enabled,
            "bgm_type": bgm_type,
        }
        status, payload = await adapter.create_video_data(body)
        if payload.get("ok") and isinstance(payload.get("data"), dict):
            task = payload["data"].get("task") or {}
            payload = {
                **payload,
                "data": {
                    **payload["data"],
                    "task_id": task.get("id"),
                    "state": task.get("state"),
                    "message": "Video generation task created. Poll moneyprinter_get_task for status.",
                },
            }
        return status, payload

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_get_task(task_id: str) -> str:
    """Get one task by id (state, progress, script, videos)."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        return await adapter.get_task_data(task_id)

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_list_tasks() -> str:
    """List known MoneyPrinter tasks (upstream + local disk recovery)."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        return await adapter.list_tasks_data()

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_list_outputs() -> str:
    """List completed video outputs with stream/download URLs."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        return await adapter.list_outputs_data()

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_delete_task(task_id: str) -> str:
    """Delete a task (upstream and/or local storage)."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        return await adapter.delete_task_data(task_id)

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_generate_script(
    video_subject: str,
    video_language: str = "zh-CN",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    auto_start: bool = True,
) -> str:
    """Generate a video script only (no render)."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        return await adapter.generate_script_data(
            {
                "video_subject": video_subject,
                "video_language": video_language,
                "paragraph_number": paragraph_number,
                "video_script_prompt": video_script_prompt,
            }
        )

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_generate_terms(
    video_subject: str,
    video_script: str = "",
    amount: int = 5,
    match_materials_to_script: bool = False,
    auto_start: bool = True,
) -> str:
    """Generate stock-footage search terms for a subject/script."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        return await adapter.generate_terms_data(
            {
                "video_subject": video_subject,
                "video_script": video_script,
                "amount": amount,
                "match_materials_to_script": match_materials_to_script,
            }
        )

    status, payload = _run(_go())
    return _json_result(payload, status=status)


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "moneyprinter_health_check",
        "description": "Check MoneyPrinterTurbo install, config, and service health.",
        "fn": moneyprinter_health_check,
    },
    {
        "name": "moneyprinter_start_service",
        "description": "Start the MoneyPrinterTurbo FastAPI sidecar process.",
        "fn": moneyprinter_start_service,
    },
    {
        "name": "moneyprinter_generate_video",
        "description": (
            "Create a short-form video generation task from a subject/script. "
            "Returns task_id immediately; poll with moneyprinter_get_task."
        ),
        "fn": moneyprinter_generate_video,
    },
    {
        "name": "moneyprinter_get_task",
        "description": "Get MoneyPrinter task status, progress, script, and output URLs.",
        "fn": moneyprinter_get_task,
    },
    {
        "name": "moneyprinter_list_tasks",
        "description": "List MoneyPrinter video generation tasks.",
        "fn": moneyprinter_list_tasks,
    },
    {
        "name": "moneyprinter_list_outputs",
        "description": "List completed video outputs with stream/download URLs.",
        "fn": moneyprinter_list_outputs,
    },
    {
        "name": "moneyprinter_delete_task",
        "description": "Delete a MoneyPrinter task and its local outputs when possible.",
        "fn": moneyprinter_delete_task,
    },
    {
        "name": "moneyprinter_generate_script",
        "description": "Generate only a video script via MoneyPrinter LLM (no render).",
        "fn": moneyprinter_generate_script,
    },
    {
        "name": "moneyprinter_generate_terms",
        "description": "Generate stock-footage search keywords for a subject/script.",
        "fn": moneyprinter_generate_terms,
    },
]
