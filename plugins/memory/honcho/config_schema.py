"""Honcho's declared config surface — rendered by the generic desktop panel."""

from plugins.memory.config_schema import (
    KIND_BOOL, KIND_JSON, KIND_NUMBER, KIND_SECRET, KIND_SELECT, KIND_TEXT, STORAGE_HONCHO_HOST_BLOCK,
    ProviderConfigSchema, ProviderField, ProviderFieldOption,
)


def _opts(*pairs: tuple[str, str]) -> tuple[ProviderFieldOption, ...]:
    return tuple(ProviderFieldOption(value, label) for value, label in pairs)


# Reasoning effort levels shared by dialectic-related selects.
_REASONING_LEVELS = _opts(("minimal", "Minimal"), ("low", "Low"), ("medium", "Medium"), ("high", "High"), ("max", "Max"))

_SESSION_STRATEGY_INFO = ("Per session: every conversation gets its own Honcho session. "
                          "Per directory: conversations from the same working directory share one. "
                          "Per repo: conversations from the same git repo share one. Global: everything shares a single session.")
_WRITE_FREQUENCY_INFO = ("async: write in the background as messages arrive. turn: flush after each turn. "
                         "session: flush when the session ends. A number N flushes every N turns.")
_RECALL_MODE_INFO = ("Hybrid: auto-injected context plus on-demand memory tools. Context only: injection without tools. "
                     "Tools only: the model queries memory explicitly, nothing is injected.")


def _field(key, label, kind, description, *, group, **kw) -> ProviderField:
    return ProviderField(key=key, label=label, kind=kind, description=description, group=group, **kw)


# Inline fields form the curated compact panel; the rest surface only in the full-config modal.
CONFIG_SCHEMA = ProviderConfigSchema(
    name="honcho",
    label="Honcho",
    storage=STORAGE_HONCHO_HOST_BLOCK,
    docs_url="https://docs.honcho.dev/v3/guides/integrations/hermes",
    fields=(
        # — Connection (inline) —
        _field("apiKey", "API key", KIND_SECRET, "Authenticate with Honcho Cloud. Not needed for a self-hosted base URL.",
               env_key="HONCHO_API_KEY", placeholder="Enter Honcho API key", inline=True, group="Connection"),
        _field("baseUrl", "Base URL", KIND_TEXT, "Self-hosted Honcho URL. Overrides the environment when set.",
               aliases=("base_url",), env_fallbacks=("HONCHO_BASE_URL",), placeholder="https://… (self-hosted)",
               inline=True, group="Connection", scope="root"),
        _field("environment", "Environment", KIND_SELECT, "Honcho environment. Ignored when a base URL is set.",
               default="production", env_fallbacks=("HONCHO_ENVIRONMENT",),
               options=_opts(("production", "Cloud"), ("local", "Local")), inline=True, group="Connection"),
        _field("workspace", "Workspace", KIND_TEXT, "Honcho workspace ID. Defaults to the profile host.",
               inline=True, group="Connection"),
        # — Identity (inline) —
        _field("peerName", "Peer name", KIND_TEXT, "Your stable user peer. Unifies memory across platforms for single-user setups.",
               placeholder="e.g. eri", inline=True, group="Identity"),
        _field("aiPeer", "AI peer", KIND_TEXT, "The AI-side peer name. Defaults to the profile host.",
               inline=True, group="Identity"),
        # — Session (inline) —
        _field("sessionStrategy", "Session strategy", KIND_SELECT, "How conversations map to Honcho sessions.",
               default="per-directory", info=_SESSION_STRATEGY_INFO, inline=True, group="Session",
               options=_opts(("per-session", "Per session"), ("per-directory", "Per directory"), ("per-repo", "Per repo"), ("global", "Global"))),
        # — Connection —
        _field("timeout", "Request timeout", KIND_NUMBER, "Request timeout in seconds for Honcho HTTP calls. Blank uses the default.",
               aliases=("requestTimeout",), env_fallbacks=("HONCHO_TIMEOUT",), placeholder="30", group="Connection", scope="root"),
        # — Identity —
        _field("pinUserPeer", "Pin user peer", KIND_BOOL, "Pin the user peer to the peer name, ignoring gateway runtime identity. Unifies memory for single-user setups.",
               default="false", aliases=("pinPeerName",), group="Identity"),
        _field("runtimePeerPrefix", "Runtime peer prefix", KIND_TEXT, "Prefix applied to unknown gateway runtime user IDs.",
               placeholder="e.g. telegram_", group="Identity"),
        _field("userPeerAliases", "User peer aliases", KIND_JSON, "Map gateway runtime user IDs to stable Honcho peers.",
               placeholder='{"telegram_123": "eri"}', group="Identity"),
        # — Session —
        _field("sessionPeerPrefix", "Session peer prefix", KIND_BOOL, "Prefix session peer names with the host.",
               default="false", group="Session"),
        _field("sessions", "Session overrides", KIND_JSON, "Explicit session ID overrides keyed by resolver.",
               placeholder='{"key": "session-id"}', group="Session", scope="root"),
        # — Message writing —
        _field("saveMessages", "Save messages", KIND_BOOL, "Persist conversation messages to Honcho.",
               default="true", group="Message writing"),
        _field("writeFrequency", "Write frequency", KIND_TEXT, "When to flush messages: async, turn, session, or every N turns.",
               default="async", info=_WRITE_FREQUENCY_INFO, placeholder="async | turn | session | N", group="Message writing"),
        # — Dialectic —
        _field("dialecticReasoningLevel", "Reasoning level", KIND_SELECT, "Reasoning effort for dialectic (peer.chat) calls.",
               default="low", options=_REASONING_LEVELS, group="Dialectic"),
        _field("dialecticDynamic", "Dynamic reasoning", KIND_BOOL, "Let the model override the reasoning level per call.",
               default="true", group="Dialectic"),
        _field("dialecticMaxChars", "Max result chars", KIND_NUMBER, "Max chars of dialectic result injected into the system prompt.",
               placeholder="1200", group="Dialectic"),
        _field("dialecticDepth", "Depth", KIND_NUMBER, "Dialectic passes per cycle (1–3).", placeholder="1", group="Dialectic"),
        _field("dialecticDepthLevels", "Per-pass levels", KIND_JSON, "Reasoning level per pass; array length matches depth.",
               placeholder='["low", "medium"]', group="Dialectic"),
        _field("dialecticMaxInputChars", "Max input chars", KIND_NUMBER, "Max chars of query input sent to peer.chat().",
               placeholder="10000", group="Dialectic"),
        # — Reasoning —
        _field("reasoningHeuristic", "Reasoning heuristic", KIND_BOOL, "Scale the reasoning level up on longer queries.",
               default="true", group="Reasoning"),
        _field("reasoningLevelCap", "Reasoning level cap", KIND_SELECT, "Ceiling for the heuristic-selected reasoning level.",
               default="high", options=_REASONING_LEVELS, group="Reasoning"),
        # — Recall —
        _field("recallMode", "Recall mode", KIND_SELECT, "How memory retrieval works: hybrid, context-only, or tools-only.",
               default="hybrid", info=_RECALL_MODE_INFO,
               options=_opts(("hybrid", "Hybrid"), ("context", "Context only"), ("tools", "Tools only")), group="Recall"),
        _field("contextTokens", "Context token cap", KIND_NUMBER, "Cap on auto-injected context tokens. Blank leaves it uncapped.",
               placeholder="(uncapped)", group="Recall"),
        _field("initOnSessionStart", "Eager init", KIND_BOOL, "Initialize the session eagerly in tools mode instead of on first tool call.",
               default="false", group="Recall"),
        # — Limits —
        _field("messageMaxChars", "Message max chars", KIND_NUMBER, "Max chars per message sent to Honcho.",
               placeholder="25000", group="Limits"),
        # — Observation —
        _field("observationMode", "Observation mode", KIND_SELECT, "Per-peer observation preset. Directional observes all directions; unified shares one view.",
               default="directional", options=_opts(("directional", "Directional"), ("unified", "Unified")), group="Observation"),
    ),
)
