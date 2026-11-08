class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""


class InvalidProviderResponseError(RuntimeError):
    """An HTTP-200 answer with no usable ``choices[0].message``: the route cannot answer this request.
    The provider's own ``error`` object, when the body carried one, rides ``provider_error``."""
