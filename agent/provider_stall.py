"""Typed failure raised after repeated provider stream stalls."""

from __future__ import annotations

from agent.provider_health_probe import ProbeOutcome


class ProviderStalledError(TimeoutError):
    """A repeated provider stall with sanitized probe evidence."""

    provider: str
    model: str
    silent_seconds: float
    attempt: int
    probe: ProbeOutcome

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        silent_seconds: float,
        attempt: int,
        probe: ProbeOutcome,
    ) -> None:
        self.provider = provider
        self.model = model
        self.silent_seconds = float(silent_seconds)
        self.attempt = int(attempt)
        self.probe = probe
        super().__init__(
            "provider stalled with no response chunks for "
            f"{int(self.silent_seconds)}s on attempt {self.attempt}; "
            f"probe={self.probe.status}"
        )


def format_provider_stall_status(
    error: ProviderStalledError, action: str
) -> str:
    """Return the canonical sanitized user-facing provider-stall status."""
    diagnosis = {
        "reachable": "endpoint reachable but request wedged",
        "unreachable": "provider endpoint unreachable",
        "unavailable": "provider health probe unavailable",
        "disabled": "provider health probe disabled",
    }.get(error.probe.status, "provider health probe unavailable")
    action_text = {
        "reconnecting": "Reconnecting once with a fresh connection.",
        "falling_back": "Switching to configured fallback.",
        "failed": (
            "No configured fallback is available. Configure fallback_providers "
            "to continue on another provider."
        ),
    }
    if action not in action_text:
        raise ValueError(f"unsupported provider stall action: {action}")
    return (
        f"⚠️ No response chunks from {error.provider}/{error.model} for "
        f"{int(error.silent_seconds)}s; {diagnosis}. {action_text[action]}"
    )
