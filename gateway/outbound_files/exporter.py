"""Outbound file provider construction and path validation."""

import logging
from pathlib import Path
from typing import Any, Optional

from gateway.outbound_files.base64 import Base64OutboundFileProvider
from gateway.outbound_files.config import OutboundFilesConfig, OutboundFilesConfigError
from gateway.outbound_files.omitted import OmittedOutboundFileProvider
from gateway.outbound_files.provider import OutboundFileProvider
from gateway.platforms.base import validate_media_delivery_path

logger = logging.getLogger(__name__)


_PROVIDER_TYPES: dict[str, type[OutboundFileProvider]] = {
    "base64": Base64OutboundFileProvider,
    "none": OmittedOutboundFileProvider,
}


def create_outbound_file_provider(config: OutboundFilesConfig) -> OutboundFileProvider:
    try:
        provider_type = _PROVIDER_TYPES[config.provider]
    except KeyError:
        raise OutboundFilesConfigError(
            f"unsupported outbound_files.provider: {config.provider}"
        ) from None
    return provider_type.from_options(
        config.provider_options, provider_name=config.provider
    )


class OutboundFileExporter:
    """Validate MEDIA paths before delegating rendering to a provider."""

    def __init__(self, provider: OutboundFileProvider):
        self.provider = provider

    @classmethod
    def from_dict(cls, raw: Any) -> "OutboundFileExporter":
        config = OutboundFilesConfig.from_dict(raw)
        return cls(create_outbound_file_provider(config))

    def system_prompt_hint(self) -> str:
        return self.provider.system_prompt_hint()

    async def export_media_path(self, path: str) -> Optional[str]:
        render_path = path
        if self.provider.requires_valid_path(Path(path)):
            safe_path = validate_media_delivery_path(path)
            if not safe_path:
                return None
            render_path = safe_path
        try:
            return await self.provider.render(Path(render_path))
        except Exception as exc:
            logger.warning(
                "Outbound file provider %s failed: %s",
                type(self.provider).__name__,
                type(exc).__name__,
            )
            return None
