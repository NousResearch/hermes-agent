"""Disabled provider for outbound API files."""

from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from gateway.outbound_files.provider import OutboundFileProvider


@dataclass(frozen=True)
class OmittedOutboundFileProvider(OutboundFileProvider):
    """Hide local paths when outbound file delivery is disabled."""

    image_mime_types: ClassVar[Mapping[str, str]] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def requires_valid_path(self, path: Path) -> bool:
        return False

    async def render(self, path: Path) -> str:
        if path.suffix.lower() in self.image_mime_types:
            return "[IMAGE OMITTED]"
        return "[FILE OMITTED]"

    def system_prompt_hint(self) -> str:
        return (
            "File/media delivery is disabled. Do not use MEDIA:/absolute/path directives: "
            "images would be replaced with [IMAGE OMITTED] and other files with "
            "[FILE OMITTED]."
        )
