import re


_STREAMING_NOT_SUPPORTED_RE = re.compile(
    r"(?<!non-)(?<!non )\bstream(?:ing)?\b"
    r"(?:\s+is)?(?:\s+currently)?\s+not supported\b"
    r"|\b(?:does not|doesn't)\s+support\s+stream(?:ing)?\b",
    re.IGNORECASE,
)


def is_streaming_not_supported_error(exc: BaseException) -> bool:
    """Return whether an error explicitly rejects response streaming."""
    return _STREAMING_NOT_SUPPORTED_RE.search(str(exc)) is not None


class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""
