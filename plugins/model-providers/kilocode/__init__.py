"""Kilo Code provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class KiloCodeProfile(ProviderProfile):
    """Kilo's catalog exposes variants per model; unknown rows stay fail-open."""

    _reasoning_efforts = {
        "deepseek/deepseek-v4.1-flash": ("none", "low", "high", "max"),
    }

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        return self._reasoning_efforts.get((model or "").lower())

kilocode = KiloCodeProfile(
    name="kilocode", aliases=("kilo-code", "kilo", "kilo-gateway"), env_vars=("KILOCODE_API_KEY",),
    base_url="https://api.kilo.ai/api/gateway", default_aux_model="google/gemini-3.6-flash",
)

register_provider(kilocode)
