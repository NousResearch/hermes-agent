"""Oracle Cloud Infrastructure (OCI) Generative AI provider profile.

Uses OCI GenAI's OpenAI-compatible endpoint with a Generative AI API key (``Authorization: Bearer``).
The key is compartment-scoped, so no ``opc-compartment-id`` header is needed. The endpoint is
regional: set ``OCI_GENAI_BASE_URL`` for regions other than us-chicago-1.

Meta Llama models are omitted from the curated list: OCI rejects tool-calling requests to them unless
the first message is a user message, and every Hermes turn opens with a system prompt plus tools.
"""

from providers import register_provider
from providers.base import ProviderProfile

oci = ProviderProfile(
    name="oci",
    aliases=("oci-genai", "oracle", "oracle-cloud", "oci-generative-ai"),
    display_name="Oracle Cloud (OCI Generative AI)",
    description="OCI Generative AI — gpt-oss, Grok, Gemini via an OCI API key",
    signup_url="https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm",
    env_vars=("OCI_GENAI_API_KEY", "OCI_GENAI_BASE_URL"),
    base_url="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1",
    # Streamed responses may otherwise arrive zstd-encoded, which httpx cannot decode by default.
    default_headers={"Accept-Encoding": "gzip, deflate"},
    supports_model_listing=False,  # GET /models is not served on the OpenAI-compatible route
    supports_health_check=False,
    fallback_models=(
        "openai.gpt-oss-120b",
        "openai.gpt-oss-20b",
        "xai.grok-4.3",
        "google.gemini-2.5-pro",
        "google.gemini-2.5-flash",
    ),
    default_aux_model="openai.gpt-oss-20b",
)

register_provider(oci)
