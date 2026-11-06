"""
Shared data models for watch-video skill pipeline.
All scripts use these dataclasses for type-safe JSON I/O.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import json


@dataclass
class PipelineOptions:
    """CLI-parsed options for the full pipeline."""
    url: str
    start: Optional[str] = None
    end: Optional[str] = None
    max_frames: int = 80
    resolution: int = 512
    fps: Optional[float] = None
    out_dir: Optional[str] = None
    ollama_url: str = "http://localhost:11434"
    vision_model: str = "qwen3-vl:4b"
    vision_timeout: int = 120
    vision_batch_size: int = 8
    work_dir: Optional[str] = None
    json_output: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert Path objects to strings
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PipelineOptions:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TranscriptSegment:
    """A single caption segment with timestamp."""
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TranscriptSegment:
        return cls(**d)


@dataclass
class VideoAsset:
    """Downloaded video with metadata and captions."""
    path: str
    title: str
    duration: float
    captions: list[TranscriptSegment] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["captions"] = [c.to_dict() for c in self.captions]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> VideoAsset:
        captions = [TranscriptSegment.from_dict(c) for c in d.get("captions", [])]
        return cls(
            path=d["path"],
            title=d["title"],
            duration=d["duration"],
            captions=captions,
            metadata=d.get("metadata", {}),
        )


@dataclass
class Frame:
    """Extracted video frame."""
    path: str
    timestamp: float
    index: int
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Frame:
        return cls(**d)


@dataclass
class VisionResult:
    """Vision model analysis for a single frame."""
    frame_index: int
    timestamp: float
    description: str
    ocr_text: str = ""
    ui_elements: list[str] = field(default_factory=list)
    scene_type: str = "unknown"
    speaker_change: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> VisionResult:
        return cls(**d)


@dataclass
class TimelineEntry:
    """Merged timeline entry combining transcript and vision."""
    timestamp: float
    transcript: Optional[str] = None
    visual: Optional[VisionResult] = None

    def to_dict(self) -> dict:
        d = {"timestamp": self.timestamp}
        if self.transcript is not None:
            d["transcript"] = self.transcript
        if self.visual is not None:
            d["visual"] = self.visual.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TimelineEntry:
        visual = None
        if "visual" in d and d["visual"] is not None:
            visual = VisionResult.from_dict(d["visual"])
        return cls(
            timestamp=d["timestamp"],
            transcript=d.get("transcript"),
            visual=visual,
        )


@dataclass
class NotesResult:
    """Final notes output."""
    notes_path: str
    title: str
    summary: str
    tldr: list[str]
    timeline: list[TimelineEntry]
    key_quotes: list[list]  # [timestamp, quote]
    visual_notes: list[VisionResult]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timeline"] = [t.to_dict() for t in self.timeline]
        d["visual_notes"] = [v.to_dict() for v in self.visual_notes]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> NotesResult:
        timeline = [TimelineEntry.from_dict(t) for t in d.get("timeline", [])]
        visual_notes = [VisionResult.from_dict(v) for v in d.get("visual_notes", [])]
        return cls(
            notes_path=d["notes_path"],
            title=d["title"],
            summary=d["summary"],
            tldr=d["tldr"],
            timeline=timeline,
            key_quotes=d.get("key_quotes", []),
            visual_notes=visual_notes,
        )


# JSON serialization helpers
def _convert_for_json(obj):
    """Recursively convert dataclass objects to dicts for JSON serialization."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_for_json(item) for item in obj)
    else:
        return obj


def write_json(obj, path: str) -> None:
    """Write dataclass or dict to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert_for_json(obj), f, ensure_ascii=False, indent=2)


def read_json(path: str, cls) -> object:
    """Read JSON file and deserialize to dataclass."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return cls.from_dict(data)


def print_json(obj) -> None:
    """Print dataclass or dict/list containing dataclasses as JSON to stdout."""
    def convert(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert(item) for item in o]
        elif isinstance(o, tuple):
            return tuple(convert(item) for item in o)
        else:
            return o
    
    converted = convert(obj)
    print(json.dumps(converted, ensure_ascii=False))