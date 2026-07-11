"""Configuration and path authorization for named video libraries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


_LIBRARY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SUPPORTED_MODES = {"linked", "managed"}


@dataclass(frozen=True)
class VideoLibraryConfig:
    id: str
    name: str
    root: Path
    source_roots: tuple[Path, ...]
    mode: str
    taxonomy: str

    @property
    def database_path(self) -> Path:
        return self.root / ".hermes-assets" / "index.sqlite"

    @property
    def metadata_dir(self) -> Path:
        return self.root / ".hermes-assets"

    @property
    def selected_clips_dir(self) -> Path:
        return self.root / "02_精选镜头"

    @property
    def keyframes_dir(self) -> Path:
        return self.root / "03_关键帧"

    @property
    def analysis_dir(self) -> Path:
        return self.root / "04_素材分析"


def load_library_configs(config: dict[str, Any]) -> dict[str, VideoLibraryConfig]:
    raw_libraries = config.get("video_libraries") or []
    if not isinstance(raw_libraries, list):
        raise ValueError("video_libraries must be a list")

    result: dict[str, VideoLibraryConfig] = {}
    for raw in raw_libraries:
        if not isinstance(raw, dict):
            raise ValueError("video library entries must be mappings")
        library_id = str(raw.get("id") or "").strip().lower()
        if not _LIBRARY_ID_RE.fullmatch(library_id):
            raise ValueError("video library ids must be non-empty lowercase identifiers")
        if library_id in result:
            raise ValueError("video library ids must be unique")

        root_text = str(raw.get("root") or "").strip()
        if not root_text:
            raise ValueError(f"video library {library_id} requires root")
        root = Path(root_text).expanduser().resolve()

        raw_sources = raw.get("source_roots") or []
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"video library {library_id} requires source_roots")
        source_roots = tuple(Path(str(item)).expanduser().resolve() for item in raw_sources)

        mode = str(raw.get("mode") or "linked").strip().lower()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"unsupported video library mode: {mode}")

        result[library_id] = VideoLibraryConfig(
            id=library_id,
            name=str(raw.get("name") or library_id).strip() or library_id,
            root=root,
            source_roots=source_roots,
            mode=mode,
            taxonomy=str(raw.get("taxonomy") or "beef-noodle-v1").strip(),
        )
    return result


def resolve_library_config(
    library_id: str,
    *,
    config: dict[str, Any] | None = None,
) -> VideoLibraryConfig:
    if config is None:
        from hermes_cli.config import load_config

        config = load_config()
    normalized = str(library_id or "").strip().lower()
    libraries = load_library_configs(config)
    try:
        return libraries[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown video library: {normalized or '<empty>'}") from exc


def resolve_source_path(library: VideoLibraryConfig, path: Path | str) -> Path:
    candidate = Path(path).expanduser().resolve(strict=True)
    if not candidate.is_file():
        raise ValueError("video source must be a file")
    if not any(candidate.is_relative_to(root) for root in library.source_roots):
        raise ValueError("video source is outside configured source roots")
    return candidate


__all__ = [
    "VideoLibraryConfig",
    "load_library_configs",
    "resolve_library_config",
    "resolve_source_path",
]
