"""Configuration for outbound file providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class OutboundFilesConfigError(ValueError):
    """Raised when outbound file delivery is configured incorrectly."""


@dataclass(frozen=True)
class OutboundFilesConfig:
    provider: str = "base64"
    provider_options: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: Any) -> "OutboundFilesConfig":
        # Preserve the API server's existing implicit base64 behavior.
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise OutboundFilesConfigError("outbound_files must be a mapping")
        unknown = set(raw) - {"provider", "provider_options"}
        if unknown:
            names = ", ".join(sorted(str(key) for key in unknown))
            raise OutboundFilesConfigError(f"unsupported outbound_files options: {names}")
        provider = raw.get("provider", "base64")
        if provider is None:
            provider = "none"
        if not isinstance(provider, str) or not provider.strip():
            raise OutboundFilesConfigError(
                "outbound_files.provider must be a non-empty string"
            )
        provider_options = raw.get("provider_options", {})
        if provider_options is None:
            provider_options = {}
        if not isinstance(provider_options, Mapping):
            raise OutboundFilesConfigError(
                "outbound_files.provider_options must be a mapping"
            )
        return cls(provider=provider.strip().lower(), provider_options=provider_options)
