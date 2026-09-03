"""Hindsight's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)

# Keep in sync with ``_PROVIDER_DEFAULT_MODELS`` in ``hindsight/__init__.py``.
# Duplicated rather than imported: this module may only import from
# ``plugins.memory.config_schema`` (see that module's docstring) so the
# desktop web server never pulls the agent runtime in through a schema file.
_LLM_PROVIDERS = (
    "openai",
    "anthropic",
    "gemini",
    "groq",
    "openrouter",
    "minimax",
    "ollama",
    "lmstudio",
    "openai_compatible",
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hindsight",
    label="Hindsight",
    fields=(
        ProviderField(
            key="mode",
            label="Mode",
            kind=KIND_SELECT,
            default="cloud",
            description="How Hermes connects to Hindsight.",
            options=(
                ProviderFieldOption(
                    "cloud",
                    "Cloud",
                    "Hindsight Cloud API (lightweight, just needs an API key)",
                ),
                ProviderFieldOption(
                    "local_embedded",
                    "Local Embedded",
                    "Run Hindsight's own engine and database on this machine",
                ),
                ProviderFieldOption(
                    "local_external",
                    "Local External",
                    "Connect to an existing Hindsight instance",
                ),
            ),
            inline=True,
        ),
        ProviderField(
            key="api_key",
            label="API key",
            kind=KIND_SECRET,
            env_key="HINDSIGHT_API_KEY",
            description="Used to authenticate with the Hindsight API.",
            placeholder="Enter Hindsight API key",
            inline=True,
        ),
        ProviderField(
            key="api_url",
            label="API URL",
            kind=KIND_TEXT,
            default="https://api.hindsight.vectorize.io",
            aliases=("apiUrl",),
            env_fallbacks=("HINDSIGHT_API_URL",),
            inline=True,
        ),
        ProviderField(
            key="bank_id",
            label="Bank ID",
            kind=KIND_TEXT,
            default="hermes",
            aliases=("bankId",),
            inline=True,
        ),
        ProviderField(
            key="recall_budget",
            label="Recall budget",
            kind=KIND_SELECT,
            default="mid",
            aliases=("budget",),
            options=(
                ProviderFieldOption("low", "low"),
                ProviderFieldOption("mid", "mid"),
                ProviderFieldOption("high", "high"),
            ),
            inline=True,
        ),
        # ── local_embedded-only fields: Hindsight's own engine needs an LLM to
        # run fact extraction/consolidation. Gated so cloud/local_external
        # users — who point at someone else's already-configured instance —
        # never see an LLM sub-form that doesn't apply to them.
        ProviderField(
            key="llm_provider",
            label="LLM provider",
            kind=KIND_SELECT,
            default="openai",
            group="Embedded engine",
            description="Backend used by the local Hindsight engine for fact extraction.",
            options=tuple(ProviderFieldOption(p, p) for p in _LLM_PROVIDERS),
            when=(("mode", "local_embedded"),),
        ),
        ProviderField(
            key="llm_base_url",
            label="LLM endpoint URL",
            kind=KIND_TEXT,
            default="",
            group="Embedded engine",
            placeholder="e.g. http://127.0.0.1:8080/v1",
            description="Required for the openai_compatible provider; ignored otherwise.",
            when=(("mode", "local_embedded"), ("llm_provider", "openai_compatible")),
        ),
        ProviderField(
            key="llm_api_key",
            label="LLM API key",
            kind=KIND_SECRET,
            env_key="HINDSIGHT_LLM_API_KEY",
            group="Embedded engine",
            description="Optional for the openai_compatible provider; required otherwise.",
            placeholder="Enter LLM API key",
            when=(("mode", "local_embedded"),),
        ),
        ProviderField(
            key="llm_model",
            label="LLM model",
            kind=KIND_TEXT,
            default="gpt-4o-mini",
            group="Embedded engine",
            placeholder="e.g. gpt-4o-mini, claude-haiku-4-5",
            when=(("mode", "local_embedded"),),
        ),
        ProviderField(
            key="idle_timeout",
            label="Idle timeout (seconds)",
            kind=KIND_NUMBER,
            default="300",
            group="Embedded engine",
            description="Auto-shutdown the embedded daemon after this many idle seconds. 0 disables it.",
            when=(("mode", "local_embedded"),),
        ),
    ),
)
