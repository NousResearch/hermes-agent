class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""
