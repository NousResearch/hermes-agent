"""Base64 image provider for outbound API responses."""

import asyncio
import base64
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

from gateway.outbound_files.config import OutboundFilesConfigError
from gateway.outbound_files.provider import OutboundFileProvider


@dataclass(frozen=True)
class Base64OutboundFileProvider(OutboundFileProvider):
    """Inline supported images using the API server's legacy data URL format."""

    max_image_size_bytes: int = 5 * 1024 * 1024
    image_mime_types: ClassVar[Mapping[str, str]] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_image_size_bytes, bool)
            or not isinstance(self.max_image_size_bytes, int)
            or self.max_image_size_bytes <= 0
        ):
            raise OutboundFilesConfigError(
                "outbound_files.provider_options.max_image_size_bytes "
                "must be a positive integer"
            )

    def _read_image(self, path: Path) -> Optional[bytes]:
        if path.suffix.lower() not in self.image_mime_types:
            return None
        try:
            with path.open("rb") as file_handle:
                data = file_handle.read(self.max_image_size_bytes + 1)
        except OSError:
            return None
        return data if len(data) <= self.max_image_size_bytes else None

    async def render(self, path: Path) -> Optional[str]:
        if path.suffix.lower() not in self.image_mime_types:
            return "[FILE OMITTED]"
        data = await asyncio.to_thread(self._read_image, path)
        if data is None:
            return None
        encoded = base64.b64encode(data).decode("ascii")
        suffix = path.suffix.lower()
        return f"![image](data:{self.image_mime_types[suffix]};base64,{encoded})"

    def requires_valid_path(self, path: Path) -> bool:
        return path.suffix.lower() in self.image_mime_types

    def system_prompt_hint(self) -> str:
        return (
            "File/media delivery: include MEDIA:/absolute/path in your response. "
            f"Images up to {self.max_image_size_bytes} bytes are inlined as base64 image "
            "data URLs. Non-image file directives are replaced with [FILE OMITTED]."
        )
