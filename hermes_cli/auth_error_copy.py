"""Plain-language copy for OAuth and provider-setup failures."""

from __future__ import annotations


def _details(error: BaseException) -> str:
    text = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
    return text if len(text) <= 300 else text[:299].rstrip() + "…"


def sign_in_failure_lines(
    error: BaseException, *, service_host: str, retry_command: str,
) -> list[str]:
    """Explain a failed sign-in and give its provider-aware retry command."""
    return [
        f"Could not sign in to {service_host}. Check your internet connection and run "
        f"`{retry_command}` to try again.",
        f"Details: {_details(error)}",
    ]


def provider_setup_failure_lines(error: BaseException, *, retry_command: str) -> list[str]:
    """Explain a failed initial provider connection without exposing raw text as the lead."""
    return [
        "Could not finish connecting a provider. Check your internet connection, then "
        f"run `{retry_command}` to try again. Your configuration was not changed.",
        f"Details: {_details(error)}",
    ]
