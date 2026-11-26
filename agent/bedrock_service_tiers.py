"""Documented Bedrock Responses speed tiers, separate from Converse serviceTier.

Do not infer support from an OpenAI family name or a Converse enum. AWS documents
Sol Ultrafast on Mantle in us-east-1; runtime inference profiles still use Converse
in Hermes and must not be silently rerouted (different auth/guardrail contracts).
https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-1-sol.html#model-card-openai-gpt-6-1-sol-tiers
"""

BEDROCK_ULTRAFAST_ROUTES = {
    "openai.gpt-6.1-sol": frozenset({"https://bedrock-mantle.us-east-1.api.aws/openai/v1"}),
}


def bedrock_fast_mode_overrides(model_id: str | None, base_url: str | None, tier: str | None) -> dict | None:
    """Opt in only to a documented model/endpoint/tier triple; no Priority alias.

    The caller has already checked the Bedrock provider. Require the resolved URL,
    rather than guessing a region when called by an unresolved picker or shell.
    """
    routes = BEDROCK_ULTRAFAST_ROUTES.get(model_id or "", ())
    if tier == "ultrafast" and str(base_url or "").rstrip("/") in routes:
        return {"service_tier": "ultrafast"}
    return None
