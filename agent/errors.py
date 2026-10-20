class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""


class LocalModelAdmissionError(RuntimeError):
    """Raised before a request when loading the managed local model would over-commit memory."""
