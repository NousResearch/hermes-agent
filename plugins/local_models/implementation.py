"""LlamaCpp runtime implementation of LocalRuntime ABC.

This is the bundled plugin implementation for llama.cpp.
Core uses the LocalRuntime interface; this class provides the llama.cpp backend.
"""

from typing import Any

from hermes_cli.local_runtime_interface import LocalRuntime


class LlamaCppRuntime(LocalRuntime):
    """Llama.cpp implementation of the LocalRuntime interface."""

    def ensure_running(self, config: dict[str, Any] | None = None) -> None:
        """Ensure llama-server is running."""
        from hermes_cli.local_runtime.bootstrap import ensure_local_runtime
        ensure_local_runtime(config)

    def shutdown(self) -> None:
        """Shutdown llama-server."""
        from hermes_cli.local_runtime.bootstrap import shutdown_local_runtime
        shutdown_local_runtime()

    def resolve_endpoint(self, config: dict[str, Any] | None = None, wait_for_boot_s: float = 0) -> dict[str, Any] | None:
        """Resolve the llama-server endpoint."""
        from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint
        try:
            return resolve_llamacpp_endpoint(config=config, wait_for_boot_s=wait_for_boot_s)
        except Exception:
            return None

    def staged_model_ids(self) -> list[str]:
        """Return staged model IDs."""
        from hermes_cli.local_runtime.bootstrap import staged_model_ids
        return staged_model_ids()

    def is_available(self) -> bool:
        """Check if llama.cpp runtime is available."""
        from hermes_cli.local_runtime.bootstrap import local_models_available
        return local_models_available()

    def unavailable_reason(self) -> str | None:
        """Return reason if unavailable."""
        from hermes_cli.local_runtime.bootstrap import unavailable_reason
        return unavailable_reason()


# Singleton instance
_instance: LlamaCppRuntime | None = None


def get_runtime() -> LocalRuntime:
    """Get the LlamaCpp runtime instance."""
    global _instance
    if _instance is None:
        _instance = LlamaCppRuntime()
    return _instance
