class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""

    pass


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""


class RuntimeContractViolation(Exception):
    """The runtime assembled a provider request that violates an internal invariant.

    This is **not** a model failure and **not** a provider outage.  It means
    Hermes itself built an illegal payload (e.g. ``tool_choice`` without
    ``tools``).  Callers must fail fast **before** the HTTP request is sent.
    """

    def __init__(self, message: str, *, field: str | None = None, context: dict | None = None):
        super().__init__(message)
        self.field = field
        self.context = context or {}
