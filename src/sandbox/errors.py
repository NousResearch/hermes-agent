"""Sandbox-specific errors."""


class SandboxNotSupportedError(RuntimeError):
    """Raised when the active host or provider cannot run the requested sandbox."""


class SandboxConfigError(ValueError):
    """Invalid sandbox configuration (e.g. missing image)."""
