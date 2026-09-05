"""Honcho's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_JSON,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    STORAGE_HONCHO_HOST_BLOCK,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)


# Reasoning effort levels shared by dialectic-related selects.
_REASONING_LEVELS = (
    ProviderFieldOption("minimal", "Minimal"),
    ProviderFieldOption("low", "Low"),
    ProviderFieldOption("medium", "Medium"),
    ProviderFieldOption("high", "High"),
    ProviderFieldOption("max", "Max"),
)


CONFIG_SCHEMA = ProviderConfigSchema(
    name="honcho",
    label="Honcho",
    storage=STORAGE_HONCHO_HOST_BLOCK,
    docs_url="https://docs.honcho.dev/v3/guides/integrations/hermes",
    fields=(
        # — Connection —
        ProviderField(
            key="apiKey",
            label="API key",
            kind=KIND_SECRET,
            env_key="HONCHO_API_KEY",
            description="Authenticate with Honcho Cloud. Not needed for a self-hosted base URL.",
            placeholder="Enter Honcho API key",
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="baseUrl",
            label="Base URL",
            kind=KIND_TEXT,
            aliases=("base_url",),
            env_fallbacks=("HONCHO_BASE_URL",),
            description="Self-hosted Honcho URL. Overrides the environment when set.",
            placeholder="https://… (self-hosted)",
            inline=True,
            group="Connection",
            scope="root",
        ),
        ProviderField(
            key="environment",
            label="Environment",
            kind=KIND_SELECT,
            default="production",
            env_fallbacks=("HONCHO_ENVIRONMENT",),
            description="Honcho environment. Ignored when a base URL is set.",
            options=(
                ProviderFieldOption("production", "Cloud"),
                ProviderFieldOption("local", "Local"),
            ),
            inline=True,
            group="Connection",
        ),
        ProviderField(
            key="workspace",
            label="Workspace",
            kind=KIND_TEXT,
            description="Honcho workspace ID. Defaults to the profile host.",
            inline=True,
            group="Connection",
        ),
        # — Identity —
        ProviderField(
            key="peerName",
            label="Peer name",
            kind=KIND_TEXT,
            description="Your stable user peer. Unifies memory across platforms for single-user setups.",
            placeholder="e.g. eri",
            inline=True,
            group="Identity",
        ),
        ProviderField(
            key="aiPeer",
            label="AI peer",
            kind=KIND_TEXT,
            description="The AI-side peer name. Defaults to the profile host.",
            inline=True,
            group="Identity",
        ),
        # — Session —
        ProviderField(
            key="sessionStrategy",
            label="Session strategy",
            kind=KIND_SELECT,
            default="per-directory",
            description="How conversations map to Honcho sessions.",
            info=(
                "Per session: every conversation gets its own Honcho session. "
                "Per directory: conversations from the same working directory share one. "
                "Per repo: conversations from the same git repo share one. "
                "Global: everything shares a single session."
            ),
            options=(
                ProviderFieldOption("per-session", "Per session"),
                ProviderFieldOption("per-directory", "Per directory"),
                ProviderFieldOption("per-repo", "Per repo"),
                ProviderFieldOption("global", "Global"),
            ),
            inline=True,
            group="Session",
        ),
        # —————— Full-config-only fields below (inline=False) ——————
        # — Connection —
        ProviderField(
            key="timeout",
            label="Request timeout",
            kind=KIND_NUMBER,
            aliases=("requestTimeout",),
            env_fallbacks=("HONCHO_TIMEOUT",),
            description="Request timeout in seconds for Honcho HTTP calls. Blank uses the default.",
            placeholder="30",
            group="Connection",
            scope="root",
        ),
        # — Identity —
        ProviderField(
            key="pinUserPeer",
            label="Pin user peer",
            kind=KIND_BOOL,
            default="false",
            aliases=("pinPeerName",),
            description="Pin the user peer to the peer name, ignoring gateway runtime identity. Unifies memory for single-user setups.",
            group="Identity",
        ),
        ProviderField(
            key="runtimePeerPrefix",
            label="Runtime peer prefix",
            kind=KIND_TEXT,
            description="Prefix applied to unknown gateway runtime user IDs.",
            placeholder="e.g. telegram_",
            group="Identity",
        ),
        ProviderField(
            key="userPeerAliases",
            label="User peer aliases",
            kind=KIND_JSON,
            description="Map gateway runtime user IDs to stable Honcho peers.",
            placeholder='{"telegram_123": "eri"}',
            group="Identity",
        ),
        # — Session —
        ProviderField(
            key="sessionPeerPrefix",
            label="Session peer prefix",
            kind=KIND_BOOL,
            default="false",
            description="Prefix session peer names with the host.",
            group="Session",
        ),
        ProviderField(
            key="sessions",
            label="Session overrides",
            kind=KIND_JSON,
            description="Explicit session ID overrides keyed by resolver.",
            placeholder='{"key": "session-id"}',
            group="Session",
            scope="root",
        ),
        # — Message writing —
        ProviderField(
            key="saveMessages",
            label="Save messages",
            kind=KIND_BOOL,
            default="true",
            description="Persist conversation messages to Honcho.",
            group="Message writing",
        ),
        ProviderField(
            key="writeFrequency",
            label="Write frequency",
            kind=KIND_TEXT,
            default="async",
            description="When to flush messages: async, turn, session, or every N turns.",
            info=(
                "async: write in the background as messages arrive. "
                "turn: flush after each turn. session: flush when the session ends. "
                "A number N flushes every N turns."
            ),
            placeholder="async | turn | session | N",
            group="Message writing",
        ),
        # — Dialectic —
        ProviderField(
            key="dialecticReasoningLevel",
            label="Reasoning level",
            kind=KIND_SELECT,
            default="low",
            description="Reasoning effort for dialectic (peer.chat) calls.",
            options=_REASONING_LEVELS,
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticDynamic",
            label="Dynamic reasoning",
            kind=KIND_BOOL,
            default="true",
            description="Let the model override the reasoning level per call.",
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticMaxChars",
            label="Max result chars",
            kind=KIND_NUMBER,
            description="Max chars of dialectic result injected into the system prompt.",
            placeholder="1200",
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticDepth",
            label="Depth",
            kind=KIND_NUMBER,
            description="Dialectic passes per cycle (1–3).",
            placeholder="1",
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticDepthLevels",
            label="Per-pass levels",
            kind=KIND_JSON,
            description="Reasoning level per pass; array length matches depth.",
            placeholder='["low", "medium"]',
            group="Dialectic",
        ),
        ProviderField(
            key="dialecticMaxInputChars",
            label="Max input chars",
            kind=KIND_NUMBER,
            description="Max chars of query input sent to peer.chat().",
            placeholder="10000",
            group="Dialectic",
        ),
        # — Reasoning —
        ProviderField(
            key="reasoningHeuristic",
            label="Reasoning heuristic",
            kind=KIND_BOOL,
            default="true",
            description="Scale the reasoning level up on longer queries.",
            group="Reasoning",
        ),
        ProviderField(
            key="reasoningLevelCap",
            label="Reasoning level cap",
            kind=KIND_SELECT,
            default="high",
            description="Ceiling for the heuristic-selected reasoning level.",
            options=_REASONING_LEVELS,
            group="Reasoning",
        ),
        # — Recall —
        ProviderField(
            key="recallMode",
            label="Recall mode",
            kind=KIND_SELECT,
            default="hybrid",
            description="How memory retrieval works: hybrid, context-only, or tools-only.",
            info=(
                "Hybrid: auto-injected context plus on-demand memory tools. "
                "Context only: injection without tools. "
                "Tools only: the model queries memory explicitly, nothing is injected."
            ),
            options=(
                ProviderFieldOption("hybrid", "Hybrid"),
                ProviderFieldOption("context", "Context only"),
                ProviderFieldOption("tools", "Tools only"),
            ),
            group="Recall",
        ),
        ProviderField(
            key="contextTokens",
            label="Context token cap",
            kind=KIND_NUMBER,
            description="Cap on auto-injected context tokens. Blank leaves it uncapped.",
            placeholder="(uncapped)",
            group="Recall",
        ),
        ProviderField(
            key="initOnSessionStart",
            label="Eager init",
            kind=KIND_BOOL,
            default="false",
            description="Initialize the session eagerly in tools mode instead of on first tool call.",
            group="Recall",
        ),
        # — Limits —
        ProviderField(
            key="messageMaxChars",
            label="Message max chars",
            kind=KIND_NUMBER,
            description="Max chars per message sent to Honcho.",
            placeholder="25000",
            group="Limits",
        ),
        # — Observation —
        ProviderField(
            key="observationMode",
            label="Observation mode",
            kind=KIND_SELECT,
            default="directional",
            description="Per-peer observation preset. Directional observes all directions; unified shares one view.",
            options=(
                ProviderFieldOption("directional", "Directional"),
                ProviderFieldOption("unified", "Unified"),
            ),
            group="Observation",
        ),
    ),
)
