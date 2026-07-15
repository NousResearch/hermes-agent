class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""

    pass
