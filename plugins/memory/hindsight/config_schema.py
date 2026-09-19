"""Hindsight's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_BOOL, KIND_JSON, KIND_NUMBER, KIND_SECRET, KIND_SELECT, KIND_TEXT,
    ProviderConfigSchema, ProviderField, ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="hindsight",
    label="Hindsight",
    fields=(
        ProviderField(
            key="mode", label="Mode", kind=KIND_SELECT, default="cloud",
            description="How Hermes connects to Hindsight.",
            options=(
                ProviderFieldOption("cloud", "Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
                ProviderFieldOption("local_external", "Local External", "Connect to an existing Hindsight instance"),
            ),
            inline=True,
        ),
        ProviderField(
            key="api_key", label="API key", kind=KIND_SECRET, env_key="HINDSIGHT_API_KEY",
            description="Used to authenticate with the Hindsight API.",
            placeholder="Enter Hindsight API key", inline=True,
        ),
        ProviderField(
            key="api_url", label="API URL", kind=KIND_TEXT, default="https://api.hindsight.vectorize.io",
            aliases=("apiUrl",), env_fallbacks=("HINDSIGHT_API_URL",), inline=True,
        ),
        ProviderField(key="bank_id", label="Bank ID", kind=KIND_TEXT, default="hermes", aliases=("bankId",), inline=True),
        ProviderField(
            key="recall_budget", label="Recall budget", kind=KIND_SELECT, default="mid", aliases=("budget",),
            options=tuple(ProviderFieldOption(b, b) for b in ("low", "mid", "high")),
            inline=True,
        ),
        ProviderField(key="recall_max_results", label="Auto-recall result cap", kind=KIND_NUMBER, default="0",
                      group="Optional auto-recall controls", description="0 keeps all ranked results; recall method only."),
        ProviderField(key="recall_live_status_bypass", label="Skip live-status questions", kind=KIND_BOOL, default="false",
                      group="Optional auto-recall controls", description="Opt-in English/German phrase heuristic; may miss or misclassify questions."),
        ProviderField(key="recall_simple_budget", label="Short-query budget", kind=KIND_SELECT, default="",
                      group="Optional auto-recall controls", options=(ProviderFieldOption("", "Disabled"),
                      *(ProviderFieldOption(b, b) for b in ("low", "mid", "high")))),
        ProviderField(key="recall_simple_max_words", label="Short-query word limit", kind=KIND_NUMBER, default="0",
                      group="Optional auto-recall controls", description="0 disables short-query classification."),
        ProviderField(key="recall_document_tags", label="Document base tags", kind=KIND_JSON, default="[]",
                      group="Optional document filtering", description="Filters existing memories; never imports or syncs documents."),
        ProviderField(key="recall_document_terms", label="Document trigger phrases", kind=KIND_JSON, default="[]",
                      group="Optional document filtering", description="Requires base tags and a matching phrase; recall method only."),
        ProviderField(key="recall_document_tag_routes", label="Additional document tag routes", kind=KIND_JSON, default="{}",
                      group="Optional document filtering", description="Map each additional tag to a list of trigger phrases. Matched tags must all apply."),
        ProviderField(key="recall_document_types", label="Document fact types", kind=KIND_JSON, default='["world", "observation"]',
                      group="Optional document filtering", description="Fact types used only for document-filtered auto-recall."),
    ),
)
