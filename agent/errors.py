class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""
    pass


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""
    pass


class ProviderStreamError(RuntimeError):
    """Raised when a provider returns an error inside a streaming SSE chunk.

    Some OpenAI-compatible providers (e.g. DeepInfra) return HTTP 200 with a
    single SSE chunk whose ``choices`` is ``None`` and whose provider-specific
    ``error_type`` / ``error_message`` fields carry a validation error (e.g.
    a context-length 400).  Without inspecting these fields, the streaming
    path treats the chunk as an empty stream and retries the identical
    oversized request forever — the real error is never surfaced to the user.

    This error is **non-transient**: the request will fail the same way every
    time, so the retry machinery should not replay it.  The real error
    message from the provider is preserved in ``provider_error`` so the user
    sees actionable diagnostics instead of the misleading "empty stream".

    See issue #65631.
    """

    def __init__(self, provider_error: str, *, error_type: str | None = None):
        self.provider_error = provider_error
        self.error_type = error_type
        super().__init__(provider_error)
