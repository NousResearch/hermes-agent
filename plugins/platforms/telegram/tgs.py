"""Safe, best-effort frame extraction for Telegram animated stickers."""

from __future__ import annotations

import gzip
import io
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


TGS_MAGIC = b"\x1f\x8b"
MAX_TGS_BYTES = 512 * 1024
MAX_TGS_JSON_BYTES = 4 * 1024 * 1024
MAX_RENDERED_FRAME_BYTES = 10 * 1024 * 1024
MAX_RENDER_DIMENSION = 4096


@dataclass(frozen=True)
class TGSFrameResult:
    """A rendered representative frame, or metadata for graceful fallback."""

    image_bytes: Optional[bytes]
    metadata_description: str


def _read_tgs_json(data: bytes) -> dict[str, Any]:
    """Decode a bounded gzip-compressed Lottie document."""
    if not data.startswith(TGS_MAGIC):
        raise ValueError("TGS data is not gzip-compressed")
    if len(data) > MAX_TGS_BYTES:
        raise ValueError("TGS file exceeds the safe compressed-size limit")

    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        raw = stream.read(MAX_TGS_JSON_BYTES + 1)
    if len(raw) > MAX_TGS_JSON_BYTES:
        raise ValueError("TGS JSON exceeds the safe decompressed-size limit")

    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("TGS document must be a JSON object")
    if not isinstance(document.get("layers"), list):
        raise ValueError("TGS document has no layer list")
    return document


def describe_tgs_metadata(document: dict[str, Any]) -> str:
    """Build a concise description when no Lottie renderer is available."""
    width = document.get("w")
    height = document.get("h")
    layers = document.get("layers", [])
    frame_rate = document.get("fr")
    in_point = document.get("ip")
    out_point = document.get("op")

    has_dimensions = (
        isinstance(width, (int, float))
        and not isinstance(width, bool)
        and math.isfinite(width)
        and isinstance(height, (int, float))
        and not isinstance(height, bool)
        and math.isfinite(height)
    )
    dimensions = f"{width}x{height}" if has_dimensions else "unknown dimensions"
    duration = None
    if (
        isinstance(frame_rate, (int, float))
        and not isinstance(frame_rate, bool)
        and math.isfinite(frame_rate)
        and frame_rate > 0
        and isinstance(in_point, (int, float))
        and not isinstance(in_point, bool)
        and math.isfinite(in_point)
        and isinstance(out_point, (int, float))
        and not isinstance(out_point, bool)
        and math.isfinite(out_point)
    ):
        duration = max(0.0, (out_point - in_point) / frame_rate)

    animated_layers = sum(
        1 for layer in layers if isinstance(layer, dict) and _contains_keyframes(layer)
    )
    layer_word = "layer" if len(layers) == 1 else "layers"
    details = [
        f"a {dimensions} animated sticker with {len(layers)} visual {layer_word}"
    ]
    if duration is not None:
        details.append(f"lasting about {duration:.1f} seconds")
    if animated_layers:
        animated_layer_word = "layer" if animated_layers == 1 else "layers"
        details.append(f"with motion in {animated_layers} {animated_layer_word}")
    return ", ".join(details)


def _contains_keyframes(value: Any) -> bool:
    """Return whether a Lottie subtree contains an animated-property marker."""
    if isinstance(value, dict):
        if value.get("a") == 1 and isinstance(value.get("k"), list):
            return True
        return any(_contains_keyframes(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_keyframes(child) for child in value)
    return False


def _representative_frame(document: dict[str, Any]) -> int:
    in_point = document.get("ip", 0)
    out_point = document.get("op", in_point)
    if (
        not isinstance(in_point, (int, float))
        or isinstance(in_point, bool)
        or not math.isfinite(in_point)
        or not isinstance(out_point, (int, float))
        or isinstance(out_point, bool)
        or not math.isfinite(out_point)
    ):
        return 0
    return max(0, round((in_point + out_point) / 2))


def _has_safe_dimensions(document: dict[str, Any]) -> bool:
    return all(
        isinstance(document.get(key), (int, float))
        and not isinstance(document.get(key), bool)
        and math.isfinite(document[key])
        and 0 < document[key] <= MAX_RENDER_DIMENSION
        for key in ("w", "h")
    )


def extract_tgs_frame(data: bytes) -> TGSFrameResult:
    """Render a representative PNG using an installed python-lottie CLI.

    Hermes intentionally does not add a renderer dependency for one platform.
    If ``lottie_convert.py`` is unavailable or rejects the animation, callers
    still receive bounded, parsed metadata suitable for a textual fallback.
    """
    document = _read_tgs_json(data)
    metadata_description = describe_tgs_metadata(document)
    converter = shutil.which("lottie_convert.py")
    if converter is None or not _has_safe_dimensions(document):
        return TGSFrameResult(None, metadata_description)

    with tempfile.TemporaryDirectory(prefix="hermes-tgs-") as temp_dir:
        input_path = Path(temp_dir) / "sticker.tgs"
        output_path = Path(temp_dir) / "frame.png"
        input_path.write_bytes(data)
        try:
            subprocess.run(
                [
                    converter,
                    str(input_path),
                    str(output_path),
                    "--frame",
                    str(_representative_frame(document)),
                ],
                cwd=temp_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )
            with output_path.open("rb") as stream:
                frame = stream.read(MAX_RENDERED_FRAME_BYTES + 1)
        except (OSError, subprocess.SubprocessError):
            return TGSFrameResult(None, metadata_description)

    if (
        not frame.startswith(b"\x89PNG\r\n\x1a\n")
        or len(frame) > MAX_RENDERED_FRAME_BYTES
    ):
        return TGSFrameResult(None, metadata_description)
    return TGSFrameResult(frame, metadata_description)
