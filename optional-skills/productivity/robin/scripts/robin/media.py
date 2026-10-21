from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path
from urllib.parse import urlparse

from robin.config import media_path, state_dir
from robin.parser import topic_slug

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}


def is_remote_reference(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_video_url(value: str) -> bool:
    return is_remote_reference(value)


def validate_image_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"Image file not found: {value}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {value}")
    suffix = path.suffix.lower()
    mime, _ = mimetypes.guess_type(path.name)
    if suffix not in IMAGE_EXTENSIONS and not (mime and mime.startswith("image/")):
        raise ValueError(f"Unsupported image type: {value}")
    return path


def copy_image_to_media(
    config: dict,
    explicit_state_dir: str | None,
    topic: str,
    entry_id: str,
    image_path: str,
) -> str:
    source = validate_image_path(image_path)
    destination_dir = media_path(config, explicit_state_dir) / topic_slug(topic)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{entry_id}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return str(destination.relative_to(state_dir(explicit_state_dir)))


def copy_image_to_vault(
    config: dict,
    explicit_state_dir: str | None,
    topic: str,
    entry_id: str,
    image_path: str,
) -> str:
    return copy_image_to_media(config, explicit_state_dir, topic, entry_id, image_path)
