class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""


class MoAFanoutAbortedError(RuntimeError):
    """Raised when a ``moa_fanout_verdict`` plugin hook aborts the turn fail-closed."""

    def __init__(self, message: str, *, preset_name: str | None = None, metadata: dict | None = None) -> None:
        super().__init__(message)
        self.preset_name = preset_name
        self.metadata = dict(metadata or {})


class InvalidProviderResponseError(RuntimeError):
    """An HTTP-200 answer with no usable ``choices[0].message``: the route cannot answer this request.
    The provider's own ``error`` object, when the body carried one, rides ``provider_error``."""
