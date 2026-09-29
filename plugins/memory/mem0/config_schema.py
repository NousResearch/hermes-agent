"""Mem0's declared config surface — rendered by the generic desktop panel.

Field keys are the ones the plugin itself reads from ``$HERMES_HOME/mem0.json`` (see
``plugins/memory/mem0/__init__.py::_load_config``); secrets are declared with an ``env_key`` so
the panel writes them to the env store instead of the JSON file.
"""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)


def _opts(*pairs: tuple[str, str]) -> tuple[ProviderFieldOption, ...]:
    return tuple(ProviderFieldOption(value, label) for value, label in pairs)


_HOST_INFO = (
    "A self-hosted Mem0 REST server (e.g. the Docker image on http://localhost:8000). When set, the "
    "plugin talks HTTP to it and sends the key as X-API-Key. Leave blank to use Mem0 Cloud."
)
_MODE_INFO = (
    "Platform routes to the Mem0 HTTP API — Cloud, or your own server when a URL is set. OSS runs mem0 "
    "in-process against your own LLM, embedder and vector store. OSS wins over a URL, so don't set both."
)
_USER_ID_INFO = (
    "The memory namespace. Every gateway, profile and device that uses the same id recalls and writes "
    "the same memories; a different id is a different person. Left at the default, the gateway's own "
    "runtime user id is used instead."
)
_AGENT_ID_INFO = (
    "Stamped on writes so records can be narrowed by source later. Recall ignores it — a different "
    "agent id still shares the same memories."
)
_SYNC_MAX_CHARS_INFO = (
    "Per-message character cap applied before a turn is sent for fact extraction. The 450 default fits "
    "a 512-token embedder (measured: 450 OK, 600 → HTTP 500 on bge-small-zh-v1.5:f16); raise it — e.g. "
    "6000 — for a large-window embedder such as bge-m3 or text-embedding-3-small, or long messages "
    "lose their tail before extraction."
)

# Inline fields form the curated compact panel; the rest surface only in the full-config modal.
CONFIG_SCHEMA = ProviderConfigSchema(
    name="mem0",
    label="Mem0",
    docs_url="https://docs.mem0.ai/integrations/hermes",
    fields=(
        # — Connection —
        ProviderField(
            key="api_key", label="API key", kind=KIND_SECRET,
            description="Mem0 Platform API key. Not needed for a self-hosted server that runs without "
                        "auth, or for OSS mode.",
            env_key="MEM0_API_KEY", placeholder="Enter Mem0 API key",
            inline=True, group="Connection",
        ),
        ProviderField(
            key="host", label="Server URL", kind=KIND_TEXT,
            description="Self-hosted Mem0 server URL. Leave blank for Mem0 Cloud.",
            info=_HOST_INFO, env_fallbacks=("MEM0_HOST",),
            placeholder="http://localhost:8000 (self-hosted)",
            inline=True, group="Connection",
        ),
        ProviderField(
            key="mode", label="Mode", kind=KIND_SELECT,
            description="Run against the Mem0 HTTP API, or run mem0 in-process (OSS).",
            info=_MODE_INFO, default="platform", env_fallbacks=("MEM0_MODE",),
            options=_opts(("platform", "Platform / self-hosted"), ("oss", "OSS (in-process)")),
            inline=True, group="Connection",
        ),
        # — Identity —
        ProviderField(
            key="user_id", label="User identifier", kind=KIND_TEXT,
            description="Shared memory namespace. Same id everywhere shares one memory store.",
            info=_USER_ID_INFO, default="hermes-user", env_fallbacks=("MEM0_USER_ID",),
            placeholder="hermes-user", inline=True, group="Identity",
        ),
        ProviderField(
            key="agent_id", label="Agent identifier", kind=KIND_TEXT,
            description="Source tag written with each memory.",
            info=_AGENT_ID_INFO, default="hermes", env_fallbacks=("MEM0_AGENT_ID",),
            placeholder="hermes", group="Identity",
        ),
        # — Recall —
        ProviderField(
            key="rerank", label="Rerank results", kind=KIND_BOOL,
            description="Rerank search results for relevance. Mem0 Platform only — the self-hosted "
                        "server ignores it.",
            default="false", group="Recall",
        ),
        # — Sync —
        ProviderField(
            key="sync_max_chars", label="Message cap (chars)", kind=KIND_NUMBER,
            description="Per-message cap applied before a turn is sent for fact extraction.",
            info=_SYNC_MAX_CHARS_INFO, placeholder="450", group="Sync",
        ),
    ),
)
