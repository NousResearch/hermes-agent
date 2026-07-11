#!/usr/bin/env python3
"""Plan and optionally render a deterministic named-library acceptance video."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from capabilities.moneyprinter import adapter as moneyprinter_adapter
from capabilities.video_library import adapter as video_library_adapter
from capabilities.video_library.config import is_generated_library_path, resolve_library_config


REQUIRED_INTENTS = ("前厅顾客吃面", "员工端碗上餐", "成品牛肉面特写")


def _require_payload(status: int, payload: dict[str, Any], operation: str) -> dict[str, Any]:
    if status >= 400 or not payload.get("ok"):
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        raise RuntimeError(str(error.get("message") or f"{operation} failed"))
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"{operation} returned no data")
    return data


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _safe_part(value: Any, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "")).strip("_")
    return normalized or fallback


def _cache_filename(library_id: str, shot: dict[str, Any], source_file: str) -> str:
    suffix = Path(source_file).suffix.lower()
    if suffix not in {".avi", ".flv", ".mkv", ".mov", ".mp4", ".webm"}:
        suffix = ".mp4"
    return "-".join(
        [
            _safe_part(library_id, "library"),
            _safe_part(shot.get("assetId"), "asset"),
            _safe_part(shot.get("clipId"), "clip"),
            _safe_part(str(shot.get("sourceSha256") or "")[:12], "nohash"),
        ]
    ) + suffix


class AcceptanceClient:
    def search(self, library_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        status, payload = video_library_adapter.list_clips_data(
            library_id=library_id,
            query=query,
            limit=limit,
        )
        data = _require_payload(status, payload, f"search {query}")
        return list(data.get("clips") or [])

    def create_timeline(
        self,
        library_id: str,
        clip_ids: list[str],
        script: list[dict[str, str]],
    ) -> dict[str, Any]:
        status, payload = video_library_adapter.create_timeline_data(
            {
                "aspect": "9:16",
                "clipIds": clip_ids,
                "libraryId": library_id,
                "script": script,
            }
        )
        return _require_payload(status, payload, "create timeline")

    def cache_material(self, source_path: str, filename: str) -> str:
        status, payload = moneyprinter_adapter.upload_local_material_data(
            {"filename": filename, "sourcePath": source_path}
        )
        data = _require_payload(status, payload, "cache local material")
        material = data.get("material") if isinstance(data.get("material"), dict) else {}
        cached = str(material.get("file") or "")
        if not cached:
            raise RuntimeError("cache local material returned no filename")
        return cached

    def cache_audio(self, source_path: Path) -> str:
        status, payload = moneyprinter_adapter.upload_custom_audio_data(
            {"filename": source_path.name, "sourcePath": str(source_path)}
        )
        data = _require_payload(status, payload, "cache custom audio")
        audio = data.get("audio") if isinstance(data.get("audio"), dict) else {}
        cached = str(audio.get("file") or "")
        if not cached:
            raise RuntimeError("cache custom audio returned no filename")
        return cached

    def create_video(self, body: dict[str, Any]) -> str:
        status, payload = _run(moneyprinter_adapter.create_video_data(body))
        data = _require_payload(status, payload, "create video")
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        task_id = str(task.get("id") or "")
        if not task_id:
            raise RuntimeError("create video returned no task id")
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any]:
        status, payload = _run(moneyprinter_adapter.get_task_data(task_id))
        return _require_payload(status, payload, "get task")


def build_acceptance_plan(client: AcceptanceClient, library_id: str) -> dict[str, Any]:
    candidates_by_query: dict[str, list[dict[str, Any]]] = {}
    for query in REQUIRED_INTENTS:
        candidates = client.search(library_id, query, limit=12)
        if not candidates:
            raise RuntimeError(f"no candidate for {query}")
        candidates_by_query[query] = sorted(
            candidates,
            key=lambda row: (
                float(row.get("score") or 0.0) * 100
                + float(row.get("quality_score") or 0.0) * 10
                + float(row.get("confidence") or 0.0)
            ),
            reverse=True,
        )

    selections: list[dict[str, Any]] = []
    used_assets: set[str] = set()
    used_clips: set[str] = set()
    round_index = 0
    while True:
        added = False
        for query in REQUIRED_INTENTS:
            remaining = [
                row
                for row in candidates_by_query[query]
                if str(row.get("id") or "") not in used_clips
            ]
            selected = next(
                (
                    row
                    for row in remaining
                    if str(row.get("asset_id") or "") not in used_assets
                ),
                remaining[0] if remaining else None,
            )
            if selected is None:
                continue
            clip_id = str(selected.get("id") or "")
            if not clip_id:
                continue
            selections.append({"query": query, "round": round_index, **selected})
            used_clips.add(clip_id)
            used_assets.add(str(selected.get("asset_id") or ""))
            added = True
        if not added:
            break
        round_index += 1
    return {"library_id": library_id, "selections": selections}


def _final_video_path(task: dict[str, Any]) -> Path:
    direct = str(task.get("final_video_path") or "").strip()
    if direct:
        return Path(direct).expanduser().resolve()
    videos = task.get("videos") if isinstance(task.get("videos"), list) else []
    if not videos:
        raise RuntimeError("completed render has no video output")
    first = videos[0]
    value = str(first.get("file") if isinstance(first, dict) else first)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = moneyprinter_adapter.TASKS_DIR / candidate
    return candidate.resolve()


def execute_render(
    client: AcceptanceClient,
    plan: dict[str, Any],
    *,
    audio_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve(strict=True)
    library_id = str(plan["library_id"])
    selections = list(plan.get("selections") or [])
    clip_ids = [str(row["id"]) for row in selections]
    script = [
        {"id": f"segment-{index + 1}", "text": str(row["query"])}
        for index, row in enumerate(selections)
    ]
    timeline_result = client.create_timeline(library_id, clip_ids, script)
    timeline = timeline_result.get("timeline")
    if not isinstance(timeline, dict):
        raise RuntimeError("timeline response is missing timeline data")
    tracks = timeline.get("tracks") if isinstance(timeline.get("tracks"), dict) else {}
    video_rows = tracks.get("video") if isinstance(tracks.get("video"), list) else []
    shot_rows = timeline.get("shotPlan") if isinstance(timeline.get("shotPlan"), list) else []
    shots_by_clip = {
        str(row.get("clipId")): row
        for row in shot_rows
        if isinstance(row, dict) and row.get("clipId")
    }
    if len(video_rows) != len(clip_ids):
        raise RuntimeError("timeline did not preserve all selected clips")

    cached_materials: list[str] = []
    selected_sources: list[str] = []
    for row in video_rows:
        if not isinstance(row, dict):
            raise RuntimeError("timeline video row is invalid")
        clip_id = str(row.get("clipId") or "")
        source_file = str(row.get("file") or "")
        shot = shots_by_clip.get(clip_id)
        if not clip_id or not source_file or shot is None:
            raise RuntimeError("timeline video row lacks provenance")
        cached_materials.append(
            client.cache_material(source_file, _cache_filename(library_id, shot, source_file))
        )
        selected_sources.append(str(shot.get("sourcePath") or ""))

    custom_audio_file = client.cache_audio(audio)
    task_id = client.create_video(
        {
            "bgm_type": "none",
            "custom_audio_file": custom_audio_file,
            "match_materials_to_script": True,
            "subtitle_enabled": False,
            "video_aspect": "9:16",
            "video_clip_duration": 5,
            "video_concat_mode": "sequential",
            "video_count": 1,
            "video_language": "zh-CN",
            "video_materials": [
                {"duration": 0, "provider": "local", "url": filename}
                for filename in cached_materials
            ],
            "video_script": "。".join(REQUIRED_INTENTS) + "。",
            "video_source": "local",
            "video_subject": "牛肉面素材库 Agent 端到端验收",
        }
    )

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while True:
        task = client.get_task(task_id)
        state = str(task.get("state") or task.get("status") or "unknown").lower()
        if state in {"complete", "completed", "success"}:
            break
        if state in {"failed", "error"}:
            raise RuntimeError(str(task.get("error") or f"render task {task_id} failed"))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"render task {task_id} did not finish within {timeout_seconds} seconds")
        time.sleep(1.0)

    final_path = _final_video_path(task)
    if not final_path.is_file() or final_path.stat().st_size <= 0:
        raise RuntimeError(f"render output is missing or empty: {final_path}")
    return {
        **plan,
        "cached_materials": cached_materials,
        "final_video_path": str(final_path),
        "selected_sources": selected_sources,
        "state": "complete",
        "task_id": task_id,
        "timeline_path": str(timeline_result.get("path") or ""),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--audio")
    parser.add_argument("--timeout", type=float, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        library = resolve_library_config(args.library)
        client = AcceptanceClient()
        plan = build_acceptance_plan(client, library.id)
        for selection in plan["selections"]:
            source = str(selection.get("source_file_path") or "")
            if not source or is_generated_library_path(library, source):
                raise RuntimeError(f"selection is not an original source: {source or '<missing>'}")
        if args.render:
            if not args.audio:
                raise ValueError("--audio is required with --render")
            result = execute_render(
                client,
                plan,
                audio_path=Path(args.audio),
                timeout_seconds=args.timeout,
            )
            default_name = "agent-e2e-result.json"
        else:
            result = plan
            default_name = "agent-e2e-plan.json"
        output = Path(args.output) if args.output else library.analysis_dir / "验收" / default_name
        written = write_json_atomic(output, result)
        print(json.dumps({"ok": True, "output": str(written), **result}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error": {"message": str(exc), "type": type(exc).__name__}, "ok": False},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
