from plugins.memory.config_schema import (
    KIND_NUMBER,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="memvid",
    label="Memvid (.mv2 file)",
    docs_url="https://github.com/memvid/claude-brain",
    fields=(
        ProviderField(
            key="file_path",
            label="Memory file path",
            kind=KIND_TEXT,
            default="~/.hermes/memvid/mind.mv2",
            placeholder="~/.hermes/memvid/mind.mv2",
            description="Path to one local .mv2 memory file.",
            inline=True,
            group="Storage",
            info="Use a single portable Memvid .mv2 file; no database or cloud service is required.",
        ),
        ProviderField(
            key="executable",
            label="memvid executable",
            kind=KIND_TEXT,
            default="memvid",
            placeholder="memvid",
            description="Command or absolute path for the memvid CLI.",
            group="Runtime",
        ),
        ProviderField(
            key="prefetch_top_k",
            label="Prefetch result limit",
            kind=KIND_NUMBER,
            default="5",
            description="Max non-empty search-result lines injected before a turn.",
            group="Recall",
        ),
    ),
)
