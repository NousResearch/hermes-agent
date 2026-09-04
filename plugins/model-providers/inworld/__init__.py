"""Inworld Router provider profile.

Inworld exposes an OpenAI-compatible LLM router at ``https://api.inworld.ai/v1``
(tool calling, streaming, prompt caching, and per-model reasoning effort).
Alongside proxied upstreams it serves first-party hosted models, all of which
support tool calling — the property that makes them usable for agent work.

The router accepts standard OpenAI Chat Completions requests; the API key is
passed through the OpenAI SDK's ``api_key`` slot (the Inworld docs demonstrate
exactly that usage). No adapter or extra_body quirks are needed, so this is a
plain declarative ``ProviderProfile``.

Model IDs on the router are ``<provider>/<model>`` slugs (e.g.
``openai/gpt-5.5``, ``anthropic/claude-sonnet-4-6``) or user-defined router
names (``inworld/<router-name>``). Because the catalog is dynamic and depends
on the user's workspace, no static fallback list is pinned here — the live
``/models`` fetch (when available) or an explicit ``--model`` id is used.
"""

from providers import register_provider
from providers.base import ProviderProfile

inworld = ProviderProfile(
    name="inworld",
    aliases=("inworld-router",),
    env_vars=("INWORLD_API_KEY",),
    display_name="Inworld",
    description="Inworld Router — OpenAI-compatible LLM router",
    signup_url="https://platform.inworld.ai/",
    base_url="https://api.inworld.ai/v1",
    auth_type="api_key",
    default_aux_model="",
    fallback_models=(),
)

register_provider(inworld)
