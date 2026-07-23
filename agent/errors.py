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


def should_hard_abort_wrong_model(error: BaseException) -> bool:
    """True when a wrong-model-served error should abort the session instead
    of walking the configured fallback chain.

    Default is False: a fallback the user explicitly configured, announced
    with a truthful status line, is a visible recovery — not the silent
    substitution the guard exists to stop. Opt in with
    HERMES_WRONG_MODEL_HARD_ABORT=1 for setups where model identity is part
    of the session's contract (per-model task gating, privacy posture) and
    finishing on ANY substitute model is worse than stopping.
    """
    import os

    return (
        isinstance(error, WrongModelServedError)
        and os.getenv("HERMES_WRONG_MODEL_HARD_ABORT", "") == "1"
    )
