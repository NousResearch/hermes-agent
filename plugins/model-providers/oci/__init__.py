"""Oracle Cloud Infrastructure (OCI) Generative AI provider profile.

Uses OCI GenAI's OpenAI-compatible endpoint with a Generative AI API key (``Authorization: Bearer``).
The key is compartment-scoped, so no ``opc-compartment-id`` header is needed. The endpoint is
regional: set ``OCI_GENAI_BASE_URL`` for regions other than us-chicago-1.

Meta Llama models are omitted from the curated list: OCI rejects tool-calling requests to them unless
the first message is a user message, and every Hermes turn opens with a system prompt plus tools.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


# OCI's chat route 400s function tools on these OpenAI models unless reasoning is off ("Function
# tools with reasoning_effort are not supported ... use /v1/responses"), and omitting the field
# leaves the model's non-"none" default in charge. gpt-5.5 and older accept tools as-is, and gpt-5,
# o3 and o4-mini reject "none", so the list is explicit. gpt-6.1-sol and gpt-6-astra reject "none"
# too; they need the Responses route (see the provider docs).
_TOOLS_NEED_EFFORT_NONE = ("openai.gpt-5.6-", "openai.gpt-6-sol", "openai.gpt-6-luna")


class OCIProfile(ProviderProfile):
    """OCI Generative AI — reasoning-off for the OpenAI models that require it to call tools."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Overriding also keeps the generic ``extra_body.reasoning`` fallback off this route:
        # OpenAI models on OCI reject it with HTTP 400 "Unknown parameter: 'reasoning'".
        if (model or "").strip().lower().startswith(_TOOLS_NEED_EFFORT_NONE):
            return {}, {"reasoning_effort": "none"}
        return {}, {}


oci = OCIProfile(
    name="oci",
    aliases=("oci-genai", "oracle", "oracle-cloud", "oci-generative-ai"),
    display_name="Oracle Cloud (OCI Generative AI)",
    description="OCI Generative AI — OpenAI GPT and gpt-oss, Grok, Gemini via an OCI API key",
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
        "openai.gpt-5.5",
        "openai.gpt-6-sol",
        "xai.grok-4.7",
        "xai.grok-4.3",
        "google.gemini-2.5-pro",
        "google.gemini-2.5-flash",
    ),
    default_aux_model="openai.gpt-oss-20b",
)

register_provider(oci)
