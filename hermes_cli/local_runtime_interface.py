"""Local runtime interface — contract for bundled local-models plugins.

Core defines this ABC. Bundled plugins (like plugins/local_models/) implement it.
Core never imports plugins; plugins import this ABC from core.

This avoids circular imports without lazy loading.
"""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["LocalRuntime"]


class LocalRuntime(ABC):
    """Interface for local model runtimes (llama.cpp, Ollama, etc.)."""

    @abstractmethod
    def ensure_running(self, config: dict[str, Any] | None = None) -> None:
        """Ensure the local runtime server is running.

        Args:
            config: Optional config dict. If None, loads from config.yaml.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the local runtime server."""
        ...

    @abstractmethod
    def resolve_endpoint(self, config: dict[str, Any] | None = None, wait_for_boot_s: float = 0) -> dict[str, Any] | None:
        """Resolve the endpoint URL for the local runtime.

        Returns:
            Dict with 'base_url' and 'api_key' keys, or None if not available.
        """
        ...

    @abstractmethod
    def staged_model_ids(self) -> list[str]:
        """Return IDs of models staged on disk (downloaded and ready)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the local runtime is available (installed and functional)."""
        ...

    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Return a human-readable reason if the runtime is unavailable, else None."""
        ...
