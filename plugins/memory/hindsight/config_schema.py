"""Hindsight's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
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
        ProviderField(
            key="auto_retain_filter_enabled",
            label="Filter automatic memory noise",
            kind=KIND_BOOL,
            default="true",
            group="Automatic filtering",
            description=(
                "Remove known Hermes lifecycle artifacts from automatic retain "
                "and background auto-recall. Disabling is a true no-op."
            ),
        ),
        ProviderField(
            key="auto_retain_filter_path",
            label="Filter YAML path",
            kind=KIND_TEXT,
            group="Automatic filtering",
            description=(
                "Optional YAML file; relative paths resolve under HERMES_HOME. "
                "Default: hindsight/auto_retain_filter.yaml."
            ),
        ),
        ProviderField(
            key="auto_retain_artifact_line_patterns",
            label="Standalone artifact-line regexes",
            kind=KIND_JSON,
            group="Automatic filtering",
            description="Additional regexes removed only when they match a complete line.",
        ),
        ProviderField(
            key="auto_retain_strip_patterns",
            label="Strip regexes",
            kind=KIND_JSON,
            group="Automatic filtering",
            description="Additional regexes removed from automatic retain and each recall result.",
        ),
        ProviderField(
            key="auto_retain_skip_patterns",
            label="Skip-turn regexes",
            kind=KIND_JSON,
            group="Automatic filtering",
            description="Skip an automatic turn after sanitization when any regex matches.",
        ),
        ProviderField(
            key="auto_retain_preserve_patterns",
            label="Preserve regexes",
            kind=KIND_JSON,
            group="Automatic filtering",
            description="Preserve a turn even when a skip-turn regex matches.",
        ),
        ProviderField(
            key="recall_skip_patterns",
            label="Auto-recall skip regexes",
            kind=KIND_JSON,
            group="Automatic filtering",
            description=(
                "Suppress matching background auto-recall queries/results. "
                "Manual hindsight_recall is unchanged."
            ),
        ),
        ProviderField(
            key="max_auto_retain_chars_per_turn",
            label="Automatic retain character cap",
            kind=KIND_NUMBER,
            default="0",
            group="Automatic filtering",
            description="Per-message cap after filtering; 0 disables truncation.",
        ),
        ProviderField(
            key="auto_retain_audit_log_path",
            label="Filter audit JSONL path",
            kind=KIND_TEXT,
            group="Automatic filtering",
            description=(
                "Optional metadata-only audit log. Records actions, reasons, and "
                "character counts, never transcript text."
            ),
        ),
    ),
)
