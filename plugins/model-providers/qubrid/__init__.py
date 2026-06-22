"""Qubrid AI Platform provider profile."""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile


class QubridProfile(ProviderProfile):
    """Qubrid catalog + pricing use the OpenRouter export, not ``/v1/models``."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        effective = (base_url or self.base_url).rstrip("/")
        saved_models_url = self.models_url
        self.models_url = effective + "/openrouter/models"
        try:
            return super().fetch_models(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
            )
        finally:
            self.models_url = saved_models_url


qubrid = QubridProfile(
    name="qubrid",
    aliases=("qubrid-ai", "qubrid-platform"),
    display_name="Qubrid AI",
    description="Qubrid Platform — OpenAI-compatible serverless models",
    signup_url="https://platform.qubrid.com/api-keys",
    env_vars=("QUBRID_API_KEY", "QUBRID_BASE_URL"),
    base_url="https://platform.qubrid.com/v1",
    auth_type="api_key",
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    default_aux_model="mistralai/Mistral-7B-Instruct-v0.3",
    fallback_models=(
        "openai/gpt-oss-120b",
        "meta-llama/Llama-3.3-70B-Instruct",
        "deepseek-ai/DeepSeek-V3.2",
        "moonshotai/Kimi-K2.6",
        "Qwen/Qwen3-Coder-Plus",
    ),
)

register_provider(qubrid)
