"""Application service for shot-level video analysis and timelines."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
import uuid

from . import media
from .semantic import SemanticClipResult
from .store import VideoLibraryStore


ANALYZER_VERSION = "ffmpeg-scene-v1"


def _technical_tags(metadata: dict[str, Any], duration_seconds: float) -> list[dict[str, Any]]:
    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if width and height:
        if height > width:
            orientation = "竖屏"
        elif width > height:
            orientation = "横屏"
        else:
            orientation = "方形"
    else:
        orientation = "未知画幅"
    if duration_seconds <= 2:
        duration_tag = "短镜头"
    elif duration_seconds <= 8:
        duration_tag = "中镜头"
    else:
        duration_tag = "长镜头"
    names = [orientation, duration_tag]
    if min(width, height) >= 720:
        names.append("高清")
    return [
        {"confidence": 1.0, "name": name, "source": "technical"}
        for name in names
    ]


class VideoLibraryService:
    def __init__(
        self,
        store: VideoLibraryStore | None = None,
        *,
        semantic_analyzer: Callable[..., SemanticClipResult] | None = None,
        taxonomy: str = "beef-noodle-v1",
    ):
        self.store = store or VideoLibraryStore()
        self.semantic_analyzer = semantic_analyzer
        self.taxonomy = taxonomy

    def import_asset(self, source_path: Path | str) -> dict[str, Any]:
        return self.store.import_asset(source_path)

    def materialize_clip(self, clip_id: str) -> dict[str, Any]:
        clip = self.store.get_clip(clip_id)
        if clip is None:
            raise KeyError(f"unknown video clip: {clip_id}")
        if clip["materialized"]:
            return clip

        source = Path(clip["source_file_path"]).expanduser().resolve(strict=True)
        output_dir = self.store.clips_dir / clip["asset_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"clip-{int(clip['clip_index']):04d}.mp4"
        temporary = output.with_suffix(".tmp.mp4")
        temporary.unlink(missing_ok=True)
        try:
            media.extract_clip(
                source,
                temporary,
                start_seconds=float(clip["start_seconds"]),
                end_seconds=float(clip["end_seconds"]),
                library_root=self.store.root,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return self.store.update_clip_materialization(clip_id, output)

    def analyze_asset(
        self,
        asset_id: str,
        *,
        threshold: float = 0.32,
        min_clip_seconds: float = 1.0,
        max_clips: int = 120,
        fallback_clip_seconds: float = 5.0,
    ) -> dict[str, Any]:
        asset = self.store.get_asset(asset_id)
        if asset is None:
            raise KeyError(f"unknown video asset: {asset_id}")
        job = self.store.create_analysis_job(asset_id, analyzer_version=ANALYZER_VERSION)
        self.store.update_analysis_job(job["id"], state="running", progress=5)
        source = Path(asset["managed_path"])
        asset_clip_dir = self.store.clips_dir / asset_id
        asset_keyframe_dir = self.store.keyframes_dir / asset_id
        staging_clip_dir = self.store.clips_dir / f".{asset_id}-{job['id']}-staging"
        staging_keyframe_dir = self.store.keyframes_dir / f".{asset_id}-{job['id']}-staging"
        backup_clip_dir = self.store.clips_dir / f".{asset_id}-{job['id']}-backup"
        backup_keyframe_dir = self.store.keyframes_dir / f".{asset_id}-{job['id']}-backup"
        eager_materialize = str(asset.get("source_mode") or "managed") != "linked"
        old_clip_backed_up = False
        old_keyframe_backed_up = False
        new_clip_installed = False
        new_keyframe_installed = False
        try:
            metadata = media.probe_media(source)
            if float(metadata.get("duration_seconds") or 0) <= 0:
                raise ValueError("video duration must be greater than zero")
            self.store.update_asset_metadata(asset_id, metadata, status="analyzing")
            self.store.update_analysis_job(job["id"], state="running", progress=20)
            boundaries = media.detect_scene_boundaries(
                source,
                duration_seconds=float(metadata["duration_seconds"]),
                threshold=threshold,
                min_clip_seconds=min_clip_seconds,
                max_clips=max_clips,
                fallback_clip_seconds=fallback_clip_seconds,
            )
            if not boundaries:
                raise ValueError("scene analysis produced no clips")

            for directory in (staging_clip_dir, staging_keyframe_dir, backup_clip_dir, backup_keyframe_dir):
                if directory.exists():
                    shutil.rmtree(directory)
            staging_keyframe_dir.mkdir(parents=True, exist_ok=True)
            if eager_materialize:
                staging_clip_dir.mkdir(parents=True, exist_ok=True)

            clip_inputs: list[dict[str, Any]] = []
            for index, (start, end) in enumerate(boundaries):
                filename = f"clip-{index:04d}"
                clip_path = staging_clip_dir / f"{filename}.mp4"
                keyframe_path = staging_keyframe_dir / f"{filename}.jpg"
                if eager_materialize:
                    media.extract_clip(
                        source,
                        clip_path,
                        start_seconds=start,
                        end_seconds=end,
                        library_root=self.store.root,
                    )
                media.extract_keyframe(
                    source,
                    keyframe_path,
                    at_seconds=start + ((end - start) / 2),
                    library_root=self.store.root,
                )
                clip_inputs.append(
                    {
                        "end_seconds": end,
                        "file_path": str(asset_clip_dir / f"{filename}.mp4") if eager_materialize else "",
                        "keyframe_path": str(asset_keyframe_dir / f"{filename}.jpg"),
                        "materialized": eager_materialize,
                        "source_file_path": str(source),
                        "start_seconds": start,
                        "tags": _technical_tags(metadata, end - start),
                    }
                )
                progress = 20 + int(((index + 1) / len(boundaries)) * 70)
                self.store.update_analysis_job(job["id"], state="running", progress=progress)

            if eager_materialize and asset_clip_dir.exists():
                asset_clip_dir.replace(backup_clip_dir)
                old_clip_backed_up = True
            if asset_keyframe_dir.exists():
                asset_keyframe_dir.replace(backup_keyframe_dir)
                old_keyframe_backed_up = True
            if eager_materialize:
                staging_clip_dir.replace(asset_clip_dir)
                new_clip_installed = True
            staging_keyframe_dir.replace(asset_keyframe_dir)
            new_keyframe_installed = True
            result = self.store.commit_analysis(asset_id, job["id"], metadata, clip_inputs)
            shutil.rmtree(backup_clip_dir, ignore_errors=True)
            shutil.rmtree(backup_keyframe_dir, ignore_errors=True)
            semantic_errors: list[str] = []
            if self.semantic_analyzer is not None:
                total = len(result["clips"])
                for index, clip in enumerate(result["clips"]):
                    self.store.update_analysis_job(
                        job["id"],
                        state="running",
                        progress=90 + int(((index + 1) / total) * 9),
                        stage="semantic_analysis",
                    )
                    try:
                        semantic = self.semantic_analyzer(
                            [Path(clip["keyframe_path"])],
                            taxonomy=self.taxonomy,
                        )
                        if semantic.quality_score < 0.2:
                            status = "unusable"
                        elif semantic.confidence < 0.5:
                            status = "low_confidence"
                        else:
                            status = "ready"
                        semantic_json = dict(semantic.raw)
                        semantic_json["_index"] = {
                            "model": semantic.model,
                            "search_text": semantic.search_text,
                        }
                        self.store.update_clip_semantic(
                            clip["id"],
                            confidence=semantic.confidence,
                            description=semantic.summary,
                            quality_score=semantic.quality_score,
                            semantic_json=semantic_json,
                            status=status,
                            tags=semantic.tag_records(),
                        )
                    except Exception as exc:
                        self.store.update_clip_status(clip["id"], "semantic_failed")
                        semantic_errors.append(f"{clip['id']}: {exc}")
            result["clips"] = self.store.list_clips(asset_id=asset_id)
            result["job"] = self.store.update_analysis_job(
                job["id"],
                state="complete" if not semantic_errors else "partial",
                progress=100,
                error="\n".join(semantic_errors),
                stage="indexing",
            )
            return result
        except Exception as exc:
            if new_clip_installed:
                shutil.rmtree(asset_clip_dir, ignore_errors=True)
            if new_keyframe_installed:
                shutil.rmtree(asset_keyframe_dir, ignore_errors=True)
            if old_clip_backed_up and backup_clip_dir.exists():
                backup_clip_dir.replace(asset_clip_dir)
            if old_keyframe_backed_up and backup_keyframe_dir.exists():
                backup_keyframe_dir.replace(asset_keyframe_dir)
            self.store.update_analysis_job(
                job["id"],
                state="failed",
                progress=100,
                error=str(exc),
            )
            self.store.update_asset_metadata(asset_id, {}, status="failed")
            raise
        finally:
            for directory in (staging_clip_dir, staging_keyframe_dir, backup_clip_dir, backup_keyframe_dir):
                shutil.rmtree(directory, ignore_errors=True)

    def create_timeline(
        self,
        clip_ids: list[str],
        *,
        aspect: str = "9:16",
        script: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not clip_ids:
            raise ValueError("at least one clip is required")
        all_clips = {clip["id"]: clip for clip in self.store.list_clips()}
        selected = []
        for clip_id in clip_ids:
            clip = all_clips.get(clip_id)
            if clip is None:
                raise KeyError(f"unknown video clip: {clip_id}")
            if not clip["materialized"]:
                clip = self.materialize_clip(clip_id)
            file_path = Path(clip["file_path"]).expanduser().resolve()
            try:
                file_path.relative_to(self.store.root.resolve())
            except ValueError as exc:
                raise ValueError("timeline clips must stay inside the managed library root") from exc
            selected.append((clip, file_path))

        cursor = 0.0
        video_track = []
        for clip, file_path in selected:
            duration = float(clip["duration_seconds"])
            video_track.append(
                {
                    "clipId": clip["id"],
                    "end": round(cursor + duration, 6),
                    "file": str(file_path),
                    "sourceEnd": duration,
                    "sourceStart": 0.0,
                    "start": round(cursor, 6),
                }
            )
            cursor += duration

        timeline = {
            "aspect": aspect,
            "duration": round(cursor, 6),
            "script": list(script or []),
            "tracks": {
                "music": [],
                "transitions": [],
                "video": video_track,
                "voice": [],
            },
            "version": 1,
        }
        timeline_id = f"timeline_{uuid.uuid4().hex}"
        timeline_dir = self.store.root / "timelines"
        timeline_dir.mkdir(parents=True, exist_ok=True)
        target = timeline_dir / f"{timeline_id}.json"
        fd, temporary_name = tempfile.mkstemp(prefix=f".{timeline_id}.", suffix=".tmp", dir=timeline_dir)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(timeline, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        return {"id": timeline_id, "path": str(target), "timeline": timeline}
