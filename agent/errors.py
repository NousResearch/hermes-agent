class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""

    pass


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""
