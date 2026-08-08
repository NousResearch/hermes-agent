"""Perplexity provider profile.

Perplexity's sonar models are search-grounded: answers arrive with live web
citations baked in. OpenAI-compatible Chat Completions at
https://api.perplexity.ai (note: no /v1 path segment). Auth is a Bearer API
key from https://www.perplexity.ai/settings/api — API billing is separate
from the consumer Pro subscription.

Schema strictness: the endpoint validates tool parameter schemas and 400s
("invalid request") on any object-typed parameter that omits ``properties``
— Hermes's ``tool_call.arguments`` (a schemaless free-form object) is the
canonical offender. ``sanitize_tool_schemas=True`` routes outgoing tools
through the shared schema-repair pass
(``agent.moonshot_schema.ensure_object_properties_in_tools``), which adds
``properties: {}`` to every object-typed node.
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile

perplexity = ProviderProfile(
    name="perplexity",
    aliases=("pplx", "sonar"),
    env_vars=("PERPLEXITY_API_KEY",),
    display_name="Perplexity",
    description="Perplexity — sonar search-grounded models with citations",
    signup_url="https://www.perplexity.ai/settings/api",
    fallback_models=(
        "sonar-pro",
        "sonar",
    ),
    base_url="https://api.perplexity.ai",
    sanitize_tool_schemas=True,
    default_aux_model="sonar",
)

register_provider(perplexity)
