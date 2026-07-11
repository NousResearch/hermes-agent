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


def moneyprinter_cache_local_material(source_path: str, filename: str) -> str:
    """Copy one selected materialized shot into MoneyPrinter's local whitelist."""
    from capabilities.moneyprinter import adapter

    status, payload = adapter.upload_local_material_data(
        {"filename": filename, "sourcePath": source_path}
    )
    return _json_result(payload, status=status)


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
    local_materials: Optional[list[str]] = None,
    custom_audio_file: str = "",
    match_materials_to_script: bool = False,
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
        local_materials: Cached MoneyPrinter local-material filenames, in render order.
        custom_audio_file: Cached custom-audio filename, if TTS should be skipped.
        match_materials_to_script: Preserve the supplied material order during rendering.
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
            "custom_audio_file": custom_audio_file,
            "match_materials_to_script": match_materials_to_script,
        }
        if video_source == "local":
            body["video_materials"] = [
                {"duration": 0, "provider": "local", "url": name}
                for name in (local_materials or [])
            ]
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


def moneyprinter_minimax_list_voices() -> str:
    """List MiniMax provider voices plus locally previewed clones."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        return await adapter.list_minimax_voices_data()

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_minimax_clone_voice(
    voice_id: str,
    clone_audio_source_path: str,
    prompt_audio_source_path: str = "",
    prompt_text: str = "",
    trial_text: str = "",
    model: str = "",
    activate: bool = False,
    auto_start: bool = True,
) -> str:
    """Clone a MiniMax voice through the MoneyPrinter sidecar."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        body: dict[str, Any] = {
            "activate": activate,
            "clone_audio": {"filename": clone_audio_source_path, "sourcePath": clone_audio_source_path},
            "model": model,
            "prompt_text": prompt_text,
            "trial_text": trial_text,
            "voice_id": voice_id,
        }
        if prompt_audio_source_path:
            body["prompt_audio"] = {"filename": prompt_audio_source_path, "sourcePath": prompt_audio_source_path}
        return await adapter.clone_minimax_voice_data(body)

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_minimax_generate_tts(
    text: str,
    voice_id: str,
    model: str = "",
    save_as_custom_audio: bool = True,
    auto_start: bool = True,
) -> str:
    """Generate MiniMax TTS audio through the MoneyPrinter sidecar."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        return await adapter.generate_minimax_tts_data(
            {
                "model": model,
                "save_as_custom_audio": save_as_custom_audio,
                "text": text,
                "voice_id": voice_id,
            }
        )

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_minimax_generate_lyrics(
    prompt: str,
    mode: str = "write_full_song",
    lyrics: str = "",
    title: str = "",
    auto_start: bool = True,
) -> str:
    """Generate or edit song lyrics through the MoneyPrinter MiniMax sidecar."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        return await adapter.generate_minimax_lyrics_data(
            {
                "lyrics": lyrics,
                "mode": mode,
                "prompt": prompt,
                "title": title,
            }
        )

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def moneyprinter_minimax_generate_music(
    prompt: str,
    is_instrumental: bool = False,
    lyrics: str = "",
    lyrics_optimizer: bool = True,
    model: str = "",
    save_as_bgm: bool = True,
    auto_start: bool = True,
) -> str:
    """Generate MiniMax music through MoneyPrinter and optionally save it as BGM."""

    async def _go() -> tuple[int, dict[str, Any]]:
        from capabilities.moneyprinter import adapter

        if auto_start:
            await _ensure_service_if_needed()
        return await adapter.generate_minimax_music_data(
            {
                "is_instrumental": is_instrumental,
                "lyrics": lyrics,
                "lyrics_optimizer": lyrics_optimizer,
                "model": model,
                "prompt": prompt,
                "save_as_bgm": save_as_bgm,
            }
        )

    status, payload = _run(_go())
    return _json_result(payload, status=status)


def video_library_import_asset(source_path: str, library_id: str = "") -> str:
    """Import a local video into a shot library and return its asset id."""
    from capabilities.video_library import adapter

    body = {"sourcePath": source_path}
    if library_id:
        body["libraryId"] = library_id
    status, payload = adapter.import_asset_data(body)
    return _json_result(payload, status=status)


def video_library_scan_library(library_id: str, dry_run: bool = False) -> str:
    """Scan one configured video library and index all new or changed videos."""
    from capabilities.video_library import adapter

    status, payload = adapter.scan_library_data(library_id, {"dryRun": dry_run})
    return _json_result(payload, status=status)


def video_library_analyze_asset(
    asset_id: str,
    threshold: float = 0.32,
    min_clip_seconds: float = 1.0,
    fallback_clip_seconds: float = 5.0,
    library_id: str = "",
) -> str:
    """Split one imported video into managed clips, keyframes, and technical tags."""
    from capabilities.video_library import adapter

    body = {
        "fallbackClipSeconds": fallback_clip_seconds,
        "minClipSeconds": min_clip_seconds,
        "threshold": threshold,
    }
    if library_id:
        body["libraryId"] = library_id
    status, payload = adapter.analyze_asset_data(asset_id, body)
    return _json_result(payload, status=status)


def video_library_get_status(library_id: str) -> str:
    """Return asset, clip, and failure counts for one configured named library."""
    from capabilities.video_library import adapter

    status, payload = adapter.library_status_data(library_id)
    return _json_result(payload, status=status)


def video_library_search_clips(
    asset_id: str = "",
    tag: str = "",
    library_id: str = "",
    query: str = "",
) -> str:
    """Search analyzed clips by named library, text, source asset, or exact tag."""
    from capabilities.video_library import adapter

    status, payload = adapter.list_clips_data(
        asset_id=asset_id or None,
        library_id=library_id or None,
        query=query or None,
        tag=tag or None,
    )
    return _json_result(payload, status=status)


def video_library_create_timeline(
    clip_ids: list[str],
    aspect: str = "9:16",
    library_id: str = "",
    script: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Create a timeline.json from an ordered list of managed clip ids."""
    from capabilities.video_library import adapter

    body: dict[str, Any] = {"aspect": aspect, "clipIds": clip_ids, "script": script or []}
    if library_id:
        body["libraryId"] = library_id
    status, payload = adapter.create_timeline_data(body)
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
        "name": "moneyprinter_cache_local_material",
        "description": "Copy one Agent-selected materialized shot into MoneyPrinter's local whitelist.",
        "fn": moneyprinter_cache_local_material,
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
    {
        "name": "moneyprinter_minimax_list_voices",
        "description": "List locally known MiniMax voices saved by MoneyPrinter.",
        "fn": moneyprinter_minimax_list_voices,
    },
    {
        "name": "moneyprinter_minimax_clone_voice",
        "description": "Clone a MiniMax voice and save its metadata in MoneyPrinter storage.",
        "fn": moneyprinter_minimax_clone_voice,
    },
    {
        "name": "moneyprinter_minimax_generate_tts",
        "description": "Generate TTS audio with a MiniMax voice.",
        "fn": moneyprinter_minimax_generate_tts,
    },
    {
        "name": "moneyprinter_minimax_generate_lyrics",
        "description": "Generate or edit lyrics with MiniMax.",
        "fn": moneyprinter_minimax_generate_lyrics,
    },
    {
        "name": "moneyprinter_minimax_generate_music",
        "description": "Generate MiniMax music and optionally save it as MoneyPrinter BGM.",
        "fn": moneyprinter_minimax_generate_music,
    },
    {
        "name": "video_library_import_asset",
        "description": "Import a local video into a shot library. Pass library_id for named Obsidian libraries.",
        "fn": video_library_import_asset,
    },
    {
        "name": "video_library_get_status",
        "description": "Get asset, clip, and failure counts for one configured named video library.",
        "fn": video_library_get_status,
    },
    {
        "name": "video_library_scan_library",
        "description": "Scan a configured local video library and semantically index new or changed videos.",
        "fn": video_library_scan_library,
    },
    {
        "name": "video_library_analyze_asset",
        "description": "Split an imported video into clips, keyframes, and tags. Pass library_id for named libraries.",
        "fn": video_library_analyze_asset,
    },
    {
        "name": "video_library_search_clips",
        "description": "Search configured video-library clips by free text, source asset, or exact tag.",
        "fn": video_library_search_clips,
    },
    {
        "name": "video_library_create_timeline",
        "description": "Create a renderer-neutral timeline.json from ordered clip ids in one named library.",
        "fn": video_library_create_timeline,
    },
]
