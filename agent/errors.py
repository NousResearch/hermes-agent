class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""

    pass


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""


class WrongModelServedError(Exception):
    """The provider answered with a different model than the one requested.

    Deterministic per request (LM Studio serves whatever is loaded when the
    requested identifier isn't) — retrying reproduces it, and falling back
    would just pick ANOTHER unrequested model, so the classifier marks this
    non-retryable with no fallback.
    """
    pass
