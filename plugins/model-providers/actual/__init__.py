"""Actual Computer provider profile."""

import os

from providers import register_provider
from providers.base import ProviderProfile

DEFAULT_ACTUAL_BASE_URL = "https://api.actual.inc/v1"


class ActualProfile(ProviderProfile):
    """Actual Computer: hosted at api.actual.inc; local (offline-mode client)
    inference opted into via ACTUAL_BASE_URL."""

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """ACTUAL_BASE_URL wins over the caller's base_url; bare hosts get ``/v1`` appended."""
        from hermes_cli.auth import normalize_actual_base_url

        base_url = normalize_actual_base_url(os.getenv("ACTUAL_BASE_URL", "").strip() or base_url or self.base_url)
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


actual = ActualProfile(
    name="actual", aliases=("actual-computer", "actualcomputer", "aci"), display_name="Actual Computer",
    description="Actual Computer - hosted inference via api.actual.inc, or local "
        "offline inference via ACTUAL_BASE_URL",
    signup_url="https://actual.inc", env_vars=("ACTUAL_API_KEY", "ACTUAL_BASE_URL"),
    base_url=DEFAULT_ACTUAL_BASE_URL, auth_type="api_key", api_mode="codex_responses",
)

register_provider(actual)
