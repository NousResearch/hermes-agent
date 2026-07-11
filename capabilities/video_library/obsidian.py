"""Human-readable Obsidian projections for the video library index."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any


def _safe_name(value: str) -> str:
    name = re.sub(r"[\x00-\x1f/:*?\"<>|]+", "_", str(value or "素材")).strip(" .")
    return name[:120] or "素材"


def _timecode(seconds: float) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    minutes, remainder = divmod(total_ms, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def write_markdown_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text.rstrip() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class ObsidianProjector:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.analysis_dir = self.root / "04_素材分析"

    def _display_path(self, value: str) -> str:
        path = Path(value).expanduser().resolve()
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def write_asset(self, asset: dict[str, Any], clips: list[dict[str, Any]]) -> Path:
        title = Path(str(asset.get("original_name") or "素材")).stem
        target = self.analysis_dir / "单条视频分析" / f"{_safe_name(title)}-{asset['id'][-8:]}.md"
        lines = [
            f"# {title}",
            "",
            f"- 资产 ID：`{asset['id']}`",
            f"- 来源：`{self._display_path(str(asset.get('source_path') or ''))}`",
            f"- SHA-256：`{asset.get('sha256', '')}`",
            f"- 状态：`{asset.get('status', '')}`",
            f"- 镜头数：{len(clips)}",
            "",
            "## 镜头清单",
            "",
        ]
        for clip in sorted(clips, key=lambda item: int(item.get("clip_index", 0))):
            start = _timecode(float(clip.get("start_seconds") or 0))
            end = _timecode(float(clip.get("end_seconds") or 0))
            lines.extend(
                [
                    f"### {start}-{end} · {clip.get('description') or '待补充画面描述'}",
                    "",
                    f"- 镜头 ID：`{clip['id']}`",
                    f"- 状态：`{clip.get('status', '')}`",
                    f"- 质量：{float(clip.get('quality_score') or 0):.2f}",
                    f"- 置信度：{float(clip.get('confidence') or 0):.2f}",
                    "- 标签：" + "、".join(str(tag["name"]) for tag in clip.get("tags") or []),
                ]
            )
            keyframe = str(clip.get("keyframe_path") or "").strip()
            if keyframe:
                lines.extend(["", f"![[{self._display_path(keyframe)}]]"])
            lines.append("")
        return write_markdown_atomic(target, "\n".join(lines))

    def write_stats(self, assets: list[dict[str, Any]], clips: list[dict[str, Any]]) -> Path:
        unusable = sum(1 for clip in clips if clip.get("status") == "unusable")
        low_confidence = sum(1 for clip in clips if clip.get("status") == "low_confidence")
        failed = sum(1 for clip in clips if clip.get("status") == "semantic_failed")
        text = "\n".join(
            [
                "# 牛肉面素材库统计",
                "",
                "| 项目 | 数量 |",
                "| --- | ---: |",
                f"| 素材数量 | {len(assets)} |",
                f"| 镜头数量 | {len(clips)} |",
                f"| 低置信度镜头 | {low_confidence} |",
                f"| 不可用镜头 | {unusable} |",
                f"| 语义分析失败 | {failed} |",
            ]
        )
        return write_markdown_atomic(self.analysis_dir / "素材统计.md", text)


__all__ = ["ObsidianProjector", "write_markdown_atomic"]
