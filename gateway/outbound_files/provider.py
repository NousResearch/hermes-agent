"""Base interface for outbound file providers."""

from collections.abc import Mapping
from dataclasses import fields
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Self

from gateway.outbound_files.config import OutboundFilesConfigError


class OutboundFileProvider(ABC):
    """Turn a validated local path into client-visible response text."""

    @classmethod
    def from_options(
        cls, raw: Mapping[str, Any], *, provider_name: str
    ) -> Self:
        """Parse provider-specific options from this provider's dataclass fields."""
        option_names = {item.name for item in fields(cls) if item.init}
        unknown = set(raw) - option_names
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise OutboundFilesConfigError(
                f"unsupported outbound_files.{provider_name} options: {names}"
            )
        return cls(**dict(raw))

    def requires_valid_path(self, path: Path) -> bool:
        return True

    @abstractmethod
    async def render(self, path: Path) -> Optional[str]:
        """Return replacement text, or None to preserve the MEDIA directive."""

    @abstractmethod
    def system_prompt_hint(self) -> str:
        """Describe this provider's delivery contract to the agent."""
