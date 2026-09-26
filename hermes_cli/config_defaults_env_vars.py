"""Optional env-var catalog for Hermes config defaults.

``OPTIONAL_ENV_VARS`` plus the entry factories it is built from (``_env``, ``_category``,
``_prov``/``_tool``/``_msg``/``_skill``/``_setting``, ``_base_url``, ``_OMIT``). Sharded out of
``hermes_cli.config_defaults`` (2K-law fracture); that module re-exports every name below for
identity, so ``hermes_cli.config_defaults.<name>`` keeps resolving to the same objects.
"""


def _env(description, prompt, **keys):
    """One OPTIONAL_ENV_VARS entry; keyword order is preserved as dict key order."""
    return {"description": description, "prompt": prompt, **keys}


_OMIT = object()


def _category(category, password, advanced):
    """Entry factory for one category with its usual password/advanced defaults.

    ``url``/``help``/``tools`` are only written when passed; ``password=None`` omits the key;
    ``advanced`` is only written when true. Key order matches the plain ``_env`` entries.
    """
    def make(description, prompt, url=_OMIT, *, help=_OMIT, tools=_OMIT, password=password,
             advanced=advanced):
        d = {"description": description, "prompt": prompt}
        d.update((k, v) for k, v in (("help", help), ("url", url), ("tools", tools)) if v is not _OMIT)
        if password is not None:
            d["password"] = password
        d["category"] = category
        if advanced:
            d["advanced"] = True
        return d
    return make


_prov = _category("provider", password=True, advanced=True)
_tool = _category("tool", password=True, advanced=False)
_msg = _category("messaging", password=False, advanced=False)
_skill = _category("skill", password=True, advanced=True)
_setting = _category("setting", password=False, advanced=False)


def _base_url(name, prompt_name=None):
    """Provider ``*_BASE_URL`` override entry (advanced, not a secret)."""
    prompt = f"{prompt_name or name} base URL (leave empty for default)"
    return _prov(f"{name} base URL override", prompt, None, password=False)


# Optional environment variables that enhance functionality. Feeds the dashboard keys page and setup
# checklists; category: provider|tool|skill|messaging|setting, advanced=True hides from checklists,
# tools=[...] lists the model tools the key unlocks.
OPTIONAL_ENV_VARS = {
    # ── Provider (handled in provider selection, not shown in checklists) ──
    "NOUS_BASE_URL": _base_url("Nous Portal"),
    "HERMES_ANON_API_SECRET": _env(
        "Shared secret for the Nous free-tier sign-up endpoints while they are in their gated "
        "integration phase (not needed once the gate is removed)",
        "Nous free-tier shared secret (leave empty unless given one)", password=True,
        category="provider", advanced=True),
    "OPENROUTER_API_KEY": _env("OpenRouter API key (for vision, web scraping helpers, and MoA)",
        "OpenRouter API key", url="https://openrouter.ai/keys", password=True, tools=["vision_analyze"],
        category="provider", advanced=True),
    "GOOGLE_API_KEY": _prov("Google AI Studio API key (also recognized as GEMINI_API_KEY)",
        "Google AI Studio API key", "https://aistudio.google.com/app/apikey"),
    "GEMINI_API_KEY": _prov("Google AI Studio API key (alias for GOOGLE_API_KEY)", "Gemini API key",
        "https://aistudio.google.com/app/apikey"),
    "GEMINI_BASE_URL": _base_url("Google AI Studio", "Gemini"),
    "VERTEX_CREDENTIALS_PATH": _prov(
        "Path to a Google Cloud service account JSON for Vertex AI (Gemini). Vertex uses "
        "OAuth2, not a static API key — this points at the credentials Hermes mints short-lived "
        "tokens from. Falls back to GOOGLE_APPLICATION_CREDENTIALS, then to ADC (gcloud auth "
        "application-default login). Set project/region under vertex: in config.yaml.",
        "Vertex service account JSON path (leave empty to use ADC / "
        "GOOGLE_APPLICATION_CREDENTIALS)", "https://cloud.google.com/iam/docs/keys-create-delete",
        password=False),
    "XAI_API_KEY": _prov("xAI API key", "xAI API key", "https://console.x.ai/"),
    "XAI_BASE_URL": _base_url("xAI"),
    "NVIDIA_API_KEY": _prov("NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)",
        "NVIDIA NIM API key", "https://build.nvidia.com/"),
    "NVIDIA_BASE_URL": _prov(
        "NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)",
        "NVIDIA NIM base URL (leave empty for default)", None, password=False),
    "LM_API_KEY": _prov("LM Studio bearer token for auth-enabled local servers",
        "LM Studio API key / bearer token", None),
    "LM_BASE_URL": _base_url("LM Studio"),
    "GLM_API_KEY": _prov("Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)",
        "Z.AI / GLM API key", "https://z.ai/"),
    "ZAI_API_KEY": _prov("Z.AI API key (alias for GLM_API_KEY)", "Z.AI API key", "https://z.ai/"),
    "Z_AI_API_KEY": _prov("Z.AI API key (alias for GLM_API_KEY)", "Z.AI API key", "https://z.ai/"),
    "GLM_BASE_URL": _base_url("Z.AI / GLM"),
    "KIMI_API_KEY": _prov("Kimi / Moonshot API key", "Kimi API key",
        "https://platform.moonshot.cn/"),
    "KIMI_BASE_URL": _base_url("Kimi / Moonshot", "Kimi"),
    "KIMI_CN_API_KEY": _prov("Kimi / Moonshot China API key", "Kimi (China) API key",
        "https://platform.moonshot.cn/"),
    "STEPFUN_API_KEY": _prov("StepFun Step Plan API key", "StepFun Step Plan API key",
        "https://platform.stepfun.com/"),
    "STEPFUN_BASE_URL": _base_url("StepFun Step Plan"),
    "ARCEEAI_API_KEY": _prov("Arcee AI API key", "Arcee AI API key", "https://chat.arcee.ai/"),
    "ARCEE_BASE_URL": _base_url("Arcee AI", "Arcee"),
    "GMI_API_KEY": _prov("GMI Cloud API key", "GMI Cloud API key", "https://www.gmicloud.ai/"),
    "GMI_BASE_URL": _base_url("GMI Cloud"),
    "ACTUAL_API_KEY": _prov("Actual Computer inference key (ac_...)",
        "Actual Computer inference key", "https://actual.inc/user/keys"),
    "FIREWORKS_API_KEY": _prov("Fireworks AI API key", "Fireworks AI API key",
        "https://app.fireworks.ai/settings/users/api-keys"),
    "MINIMAX_API_KEY": _prov("MiniMax API key (international)", "MiniMax API key",
        "https://www.minimax.io/"),
    "MINIMAX_BASE_URL": _base_url("MiniMax"),
    "MINIMAX_CN_API_KEY": _prov("MiniMax API key (China endpoint)", "MiniMax (China) API key",
        "https://www.minimaxi.com/"),
    "MINIMAX_CN_BASE_URL": _base_url("MiniMax (China)"),
    "DEEPSEEK_API_KEY": _prov("DeepSeek API key for direct DeepSeek access", "DeepSeek API Key",
        "https://platform.deepseek.com/api_keys", advanced=False),
    "DEEPSEEK_BASE_URL": _prov("Custom DeepSeek API base URL (advanced)", "DeepSeek Base URL", "",
        password=False, advanced=False),
    "DASHSCOPE_API_KEY": _prov("Alibaba Cloud DashScope API key (Qwen + multi-provider models)",
        "DashScope API Key", "https://modelstudio.console.alibabacloud.com/", advanced=False),
    "DASHSCOPE_BASE_URL": _prov(
        "Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)",
        "DashScope Base URL", "", password=False),
    "HERMES_QWEN_BASE_URL": _prov(
        "Qwen Portal base URL override (default: https://portal.qwen.ai/v1)",
        "Qwen Portal base URL (leave empty for default)", None, password=False),
    "OPENCODE_ZEN_API_KEY": _prov("OpenCode Zen API key (pay-as-you-go access to curated models)",
        "OpenCode Zen API key", "https://opencode.ai/auth"),
    "COMMANDCODE_API_KEY": _prov(
        "CommandCode API key (GOAT/Pro/Max/Provider plans — 30+ models via one key)",
        "CommandCode API key", "https://commandcode.ai/studio/"),
    "OPENCODE_ZEN_BASE_URL": _base_url("OpenCode Zen"),
    "OPENCODE_GO_API_KEY": _prov("OpenCode Go API key ($10/month subscription for open models)",
        "OpenCode Go API key", "https://opencode.ai/auth"),
    "OPENCODE_GO_BASE_URL": _base_url("OpenCode Go"),
    "HF_TOKEN": _prov(
        "Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)",
        "Hugging Face Token", "https://huggingface.co/settings/tokens", advanced=False),
    "HF_BASE_URL": _base_url("Hugging Face Inference Providers", "HF"),
    "OLLAMA_API_KEY": _prov("Ollama Cloud API key (ollama.com — cloud-hosted open models)",
        "Ollama Cloud API key", "https://ollama.com/settings"),
    "OLLAMA_BASE_URL": _prov("Ollama Cloud base URL override (default: https://ollama.com/v1)",
        "Ollama base URL (leave empty for default)", None, password=False),
    "XIAOMI_API_KEY": _prov(
        "Xiaomi MiMo API key for MiMo models (mimo-v2.5-pro, mimo-v2.5, mimo-v2-pro, "
        "mimo-v2-omni, mimo-v2-flash)", "Xiaomi MiMo API Key", "https://platform.xiaomimimo.com",
        advanced=False),
    "XIAOMI_BASE_URL": _prov(
        "Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)",
        "Xiaomi base URL (leave empty for default)", None, password=False),
    "UPSTAGE_API_KEY": _prov("Upstage API key for Solar LLM models", "Upstage API Key",
        "https://console.upstage.ai/api-keys", advanced=False),
    "UPSTAGE_BASE_URL": _prov("Upstage base URL override (default: https://api.upstage.ai/v1)",
        "Upstage base URL (leave empty for default)", None, password=False),
    "AWS_REGION": _prov("AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)",
        "AWS Region", "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html",
        password=False),
    "AWS_PROFILE": _prov("AWS named profile for Bedrock authentication (from ~/.aws/credentials)",
        "AWS Profile", None, password=False),
    "AZURE_FOUNDRY_API_KEY": _prov("Azure Foundry API key for custom Azure endpoints",
        "Azure Foundry API Key", "https://ai.azure.com/", advanced=False),
    "AZURE_FOUNDRY_BASE_URL": _prov(
        "Azure Foundry base URL (set via 'hermes model' for endpoint-specific config)",
        "Azure Foundry base URL", None, password=False),
    # ── Tool API keys ──
    "EXA_API_KEY": _tool("Exa API key for AI-native web search and contents", "Exa API key",
        "https://exa.ai/", tools=["web_search", "web_extract"]),
    "PARALLEL_API_KEY": _tool("Parallel API key for AI-native web search and extract",
        "Parallel API key", "https://parallel.ai/", tools=["web_search", "web_extract"]),
    "FIRECRAWL_API_KEY": _tool("Firecrawl API key for web search and scraping", "Firecrawl API key",
        "https://firecrawl.dev/", tools=["web_search", "web_extract"]),
    "FIRECRAWL_API_URL": _tool("Firecrawl API URL for self-hosted instances (optional)",
        "Firecrawl API URL (leave empty for cloud)", None, password=False, advanced=True),
    "FIRECRAWL_GATEWAY_URL": _tool(
        "Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)",
        "Firecrawl gateway URL (leave empty to derive from domain)", None, password=False,
        advanced=True),
    "TOOL_GATEWAY_URL": _tool(
        "Exact shared tool-gateway origin for on-origin vendors and media uploads (optional)",
        "Shared tool-gateway URL (leave empty to derive from domain)", None,
        password=False, advanced=True),
    "CONNECTOR_GATEWAY_URL": _tool(
        "Exact connector-gateway origin for the connectors API (optional)",
        "Connector-gateway URL (leave empty to derive from domain)", None,
        password=False, advanced=True),
    "TOOL_GATEWAY_DOMAIN": _tool(
        "Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor "
        "hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com",
        "Tool-gateway domain suffix", None, password=False, advanced=True),
    "TOOL_GATEWAY_SCHEME": _tool(
        "Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts "
        "(`https` by default, set `http` for local gateway testing)", "Tool-gateway URL scheme",
        None, password=False, advanced=True),
    "TOOL_GATEWAY_USER_TOKEN": _tool(
        "Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise "
        "read from the Hermes auth store)", "Tool-gateway user token", None, advanced=True),
    "TAVILY_API_KEY": _tool(
        "Tavily API key for AI-native web search and extract (optional — keyless works when "
        "Tavily is selected)", "Tavily API key", "https://app.tavily.com/home",
        tools=["web_search", "web_extract"]),
    "PERPLEXITY_API_KEY": _tool(
        "Perplexity API key for the Search API web backend (ranked results + query-relevant page "
        "snippets)", "Perplexity API key", "https://www.perplexity.ai/account/api",
        tools=["web_search", "web_extract"]),
    "KEENABLE_API_KEY": _tool(
        "Keenable API key for fast independent-index web search and page fetch (optional — "
        "keyless free tier works without it)", "Keenable API key", "https://keenable.ai",
        tools=["web_search", "web_extract"]),
    "SEARXNG_URL": _tool("URL of your SearXNG instance for free self-hosted web search",
        "SearXNG URL (e.g. http://localhost:8080)", "https://searxng.github.io/searxng/",
        tools=["web_search"], password=False),
    "BRAVE_SEARCH_API_KEY": _tool(
        "Brave Search API subscription token (free tier: 2,000 queries/mo)",
        "Brave Search subscription token", "https://brave.com/search/api/", tools=["web_search"]),
    "BROWSERBASE_API_KEY": _tool(
        "Browserbase API key for cloud browser (optional — local browser works without this)",
        "Browserbase API key", "https://browserbase.com/",
        tools=["browser_navigate", "browser_click"]),
    "BROWSERBASE_PROJECT_ID": _tool(
        "Browserbase project ID (optional — only needed for cloud browser)",
        "Browserbase project ID", "https://browserbase.com/",
        tools=["browser_navigate", "browser_click"], password=False),
    "BROWSER_USE_API_KEY": _tool(
        "Browser Use API key for cloud browser (optional — local browser works without this)",
        "Browser Use API key", "https://browser-use.com/",
        tools=["browser_navigate", "browser_click"]),
    "FIRECRAWL_BROWSER_TTL": _tool(
        "Firecrawl browser session TTL in seconds (optional, default 300)",
        "Browser session TTL (seconds)", tools=["browser_navigate", "browser_click"],
        password=False),
    "AGENT_BROWSER_ENGINE": _env(
        "Local browser engine: auto (default Chrome), lightpanda (faster, no screenshots; Browser Use mode "
        "spawns lightpanda serve), chrome", "Browser engine (auto/lightpanda/chrome)",
        url="https://lightpanda.io/docs/run-locally/installation/one-liner",
        tools=["browser_exec", "browser_navigate", "browser_snapshot", "browser_click", "browser_vision"],
        password=False, category="tool", advanced=True),
    "CAMOFOX_URL": _tool(
        "Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)",
        "Camofox server URL", "https://github.com/jo-inc/camofox-browser",
        tools=["browser_navigate", "browser_click"], password=False),
    "CAMOFOX_API_KEY": _tool(
        "Optional bearer token sent as Authorization header to a remote/authenticated Camofox "
        "server", "Camofox API key", "https://github.com/jo-inc/camofox-browser",
        tools=["browser_navigate", "browser_click"], advanced=True),
    "FAL_KEY": _tool("FAL API key for image and video generation", "FAL API key", "https://fal.ai/",
        tools=["image_generate", "video_generate"]),
    "KREA_API_KEY": _tool("Krea API key for Krea 2 image generation (Medium + Large)",
        "Krea API key", "https://www.krea.ai/settings/api-tokens", tools=["image_generate"]),
    "VOICE_TOOLS_OPENAI_KEY": _tool(
        "OpenAI API key for voice transcription (Whisper) and OpenAI TTS",
        "OpenAI API Key (for Whisper STT + TTS)", "https://platform.openai.com/api-keys",
        tools=["voice_transcription", "openai_tts"]),
    "ELEVENLABS_API_KEY": _tool(
        "ElevenLabs API key for premium text-to-speech voices and Scribe transcription",
        "ElevenLabs API key", "https://elevenlabs.io/",
        tools=["elevenlabs_tts", "voice_transcription"]),
    "MISTRAL_API_KEY": _tool("Mistral API key for Voxtral TTS and transcription (STT)",
        "Mistral API key", "https://console.mistral.ai/"),
    "PORCUPINE_ACCESS_KEY": _tool(
        "Picovoice access key for the Porcupine 'Hey Hermes' wake word engine (optional; "
        "openWakeWord is the free default)", "Picovoice access key",
        "https://console.picovoice.ai/"),
    "GITHUB_TOKEN": _tool("GitHub token for Skills Hub (higher API rate limits, skill publish)",
        "GitHub Token", "https://github.com/settings/tokens"),
    # ── Bundled skills (opt-in) ── category="skill" (not "tool") so the sandbox env blocklist in
    # tools/environments/local.py does NOT rewrite them; skills need them passed through to curl
    # via tools/env_passthrough.py.
    "NOTION_API_KEY": _skill("Notion integration token (used by the `notion` skill)",
        "Notion API key", "https://www.notion.so/my-integrations"),
    "LINEAR_API_KEY": _skill("Linear personal API key (used by the `linear` skill)",
        "Linear API key", "https://linear.app/settings/account/security"),
    "AIRTABLE_API_KEY": _skill("Airtable personal access token (used by the `airtable` skill)",
        "Airtable API key", "https://airtable.com/create/tokens"),
    "TENOR_API_KEY": _skill("Tenor API key for GIF search (used by the `gif-search` skill)",
        "Tenor API key", "https://developers.google.com/tenor/guides/quickstart"),
    # ── Honcho ──
    "HONCHO_API_KEY": _tool("Honcho API key for AI-native persistent memory", "Honcho API key",
        "https://app.honcho.dev", tools=["honcho_context"]),
    "HONCHO_BASE_URL": _tool("Base URL for self-hosted Honcho instances (no API key needed)",
        "Honcho base URL (e.g. http://localhost:8000)", password=None),
    # ── Hindsight ──
    "HINDSIGHT_API_KEY": _tool("Hindsight API key for graph-aware persistent memory",
        "Hindsight API key", "https://hindsight.vectorize.io", tools=["hindsight_recall"]),
    "HINDSIGHT_API_URL": _tool(
        "Base URL for the Hindsight API (default: https://api.hindsight.vectorize.io)",
        "Hindsight API URL", password=None, advanced=True),
    # ── Supermemory ──
    "SUPERMEMORY_API_KEY": _tool("Supermemory API key for conversation-scoped persistent memory",
        "Supermemory API key", "https://supermemory.ai", tools=["supermemory_search"]),
    # ── Mem0 ──
    "MEM0_API_KEY": _tool("Mem0 Platform API key for semantic persistent memory", "Mem0 API key",
        "https://app.mem0.ai", tools=["mem0_search"]),
    # ── RetainDB ──
    "RETAINDB_API_KEY": _tool("RetainDB API key for persistent memory", "RetainDB API key",
        "https://retaindb.com", tools=["retaindb_search"]),
    "RETAINDB_BASE_URL": _tool(
        "Base URL for self-hosted RetainDB instances (default: https://api.retaindb.com)",
        "RetainDB base URL", password=None, advanced=True),
    # ── ByteRover ──
    "BRV_API_KEY": _tool("ByteRover API key (optional, for cloud sync — local-first by default)",
        "ByteRover API key", "https://app.byterover.dev", tools=["brv_query"]),
    # ── OpenViking ──
    "OPENVIKING_API_KEY": _tool("OpenViking API key (leave blank for local dev mode)",
        "OpenViking API key", tools=["viking_search"]),
    "OPENVIKING_ENDPOINT": _tool("OpenViking server URL (default: http://127.0.0.1:1933)",
        "OpenViking endpoint", password=None, advanced=True),
    # ── Langfuse observability ──
    "HERMES_LANGFUSE_PUBLIC_KEY": _tool("Langfuse project public key (pk-lf-...)",
        "Langfuse public key", "https://cloud.langfuse.com", password=False),
    "HERMES_LANGFUSE_SECRET_KEY": _tool("Langfuse project secret key (sk-lf-...)",
        "Langfuse secret key", "https://cloud.langfuse.com"),
    "HERMES_LANGFUSE_BASE_URL": _tool("Langfuse server URL (default: https://cloud.langfuse.com)",
        "Langfuse server URL (leave empty for cloud.langfuse.com)", None, password=False,
        advanced=True),
    # ── Messaging platforms ──
    "TELEGRAM_BOT_TOKEN": _msg(
        "Complete Telegram bot token created by @BotFather (numeric bot ID followed by a colon "
        "and secret)", "Telegram bot token", "https://t.me/BotFather", password=True),
    "TELEGRAM_ALLOWED_USERS": _msg(
        "Optional comma-separated numeric Telegram user IDs allowed immediately; leave blank to "
        "approve new users through DM pairing", "Allowed Telegram user IDs (comma-separated)",
        "https://t.me/userinfobot"),
    "TELEGRAM_PROXY": _msg(
        "Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, "
        "https://, socks5://", "Telegram proxy URL (optional)"),
    "DISCORD_BOT_TOKEN": _msg("Discord bot token from Developer Portal", "Discord bot token",
        "https://discord.com/developers/applications", password=True),
    "DISCORD_ALLOWED_USERS": _msg("Comma-separated Discord user IDs allowed to use the bot",
        "Allowed Discord user IDs (comma-separated)", None),
    "DISCORD_REPLY_TO_MODE": _msg(
        "Discord reply threading mode: 'off' (no reply references), 'first' (reply on first "
        "message only, default), 'all' (reply on every chunk)",
        "Discord reply mode (off/first/all)", None),
    "SLACK_BOT_TOKEN": _msg(
        "Slack bot token (xoxb-). Get from OAuth & Permissions after installing your app. "
        "Required scopes: chat:write, app_mentions:read, channels:history, groups:history, "
        "im:history, im:read, im:write, mpim:history, mpim:read, users:read, files:read, "
        "files:write", "Slack Bot Token (xoxb-...)", "https://api.slack.com/apps",
        help=("In your Slack app, add the required bot scopes, install the app to the workspace, "
        "then copy OAuth & Permissions > Bot User OAuth Token."), password=True),
    "SLACK_APP_TOKEN": _msg(
        "Slack app-level token (xapp-) for Socket Mode. Get from Basic Information → App-Level "
        "Tokens. Also ensure Event Subscriptions include: message.im, message.channels, "
        "message.groups, message.mpim, app_mention", "Slack App Token (xapp-...)",
        "https://api.slack.com/apps",
        help=("In your Slack app, enable Socket Mode, then create Basic Information > App-Level "
        "Tokens with the connections:write scope."), password=True),
    "SLACK_ALLOWED_USERS": _msg(
        "Comma-separated Slack member IDs allowed to use Hermes, e.g. U01ABC2DEF3. Without "
        "this, Slack may connect but deny messages by default.", "Allowed Slack member IDs",
        "https://api.slack.com/apps",
        help=("In Slack, open your profile, choose More or the three-dot menu, then Copy member "
        "ID. Add multiple IDs comma-separated.")),
    "MATTERMOST_URL": _msg("Mattermost server URL (e.g. https://mm.example.com)",
        "Mattermost server URL", "https://mattermost.com/deploy/"),
    "MATTERMOST_TOKEN": _msg("Mattermost bot token or personal access token",
        "Mattermost bot token", None, password=True),
    "MATTERMOST_ALLOWED_USERS": _msg("Comma-separated Mattermost user IDs allowed to use the bot",
        "Allowed Mattermost user IDs (comma-separated)", None),
    "MATTERMOST_REQUIRE_MENTION": _msg(
        "Require @mention in Mattermost channels (default: true). Set to false to respond to "
        "all messages.", "Require @mention in channels", None),
    "MATTERMOST_FREE_RESPONSE_CHANNELS": _msg(
        "Comma-separated Mattermost channel IDs where bot responds without @mention",
        "Free-response channel IDs (comma-separated)", None),
    "MATRIX_HOMESERVER": _msg("Matrix homeserver URL (e.g. https://matrix.example.org)",
        "Matrix homeserver URL", "https://matrix.org/ecosystem/servers/"),
    "MATRIX_ACCESS_TOKEN": _msg("Matrix access token (preferred over password login)",
        "Matrix access token", None, password=True),
    "MATRIX_USER_ID": _msg("Matrix user ID (e.g. @hermes:example.org)",
        "Matrix user ID (@user:server)", None),
    "MATRIX_ALLOWED_USERS": _msg(
        "Comma-separated Matrix user IDs allowed to use the bot (@user:server format)",
        "Allowed Matrix user IDs (comma-separated)", None),
    "MATRIX_REQUIRE_MENTION": _msg(
        "Require @mention in Matrix rooms (default: true). Set to false to respond to all "
        "messages.", "Require @mention in rooms (true/false)", None, advanced=True),
    "MATRIX_FREE_RESPONSE_ROOMS": _msg(
        "Comma-separated Matrix room IDs where bot responds without @mention",
        "Free-response room IDs (comma-separated)", None, advanced=True),
    "MATRIX_AUTO_THREAD": _msg("Auto-create threads for messages in Matrix rooms (default: true)",
        "Auto-create threads in rooms (true/false)", None, advanced=True),
    "MATRIX_DM_AUTO_THREAD": _msg("Auto-create threads for DM messages in Matrix (default: false)",
        "Auto-create threads in DMs (true/false)", None, advanced=True),
    "MATRIX_DEVICE_ID": _msg(
        "Stable Matrix device ID for E2EE persistence across restarts (e.g. HERMES_BOT)",
        "Matrix device ID (stable across restarts)", None, advanced=True),
    "MATRIX_RECOVERY_KEY": _msg(
        "Matrix recovery key for cross-signing verification after device key rotation (from "
        "Element: Settings → Security → Recovery Key)", "Matrix recovery key", None, password=True,
        advanced=True),
    "BLUEBUBBLES_SERVER_URL": _msg(
        "BlueBubbles server URL for iMessage integration (e.g. http://192.168.1.10:1234)",
        "BlueBubbles server URL", "https://bluebubbles.app/"),
    "BLUEBUBBLES_PASSWORD": _msg(
        "BlueBubbles server password (from BlueBubbles Server → Settings → API)",
        "BlueBubbles server password", None, password=True),
    "BLUEBUBBLES_ALLOWED_USERS": _msg(
        "Comma-separated iMessage addresses (email or phone) allowed to use the bot",
        "Allowed iMessage addresses (comma-separated)", None),
    "BLUEBUBBLES_ALLOW_ALL_USERS": _msg("Allow all BlueBubbles users without allowlist",
        "Allow All BlueBubbles Users", password=None),
    "QQ_APP_ID": _msg("QQ Bot App ID from QQ Open Platform (q.qq.com)", "QQ App ID",
        "https://q.qq.com", password=None),
    "QQ_CLIENT_SECRET": _msg("QQ Bot Client Secret from QQ Open Platform", "QQ Client Secret",
        password=True),
    "QQ_ALLOWED_USERS": _msg("Comma-separated QQ user IDs allowed to use the bot",
        "QQ Allowed Users", password=None),
    "QQ_GROUP_ALLOWED_USERS": _msg("Comma-separated QQ group IDs allowed to interact with the bot",
        "QQ Group Allowed Users", password=None),
    "QQ_ALLOW_ALL_USERS": _msg("Allow all QQ users without an allowlist (true/false)",
        "Allow All QQ Users", password=None),
    "QQBOT_HOME_CHANNEL": _msg("Default QQ channel/group for cron delivery and notifications",
        "QQ Home Channel", password=None),
    "QQBOT_HOME_CHANNEL_NAME": _msg("Display name for the QQ home channel", "QQ Home Channel Name",
        password=None),
    "QQ_SANDBOX": _msg("Enable QQ sandbox mode for development testing (true/false)",
        "QQ Sandbox Mode", password=None),
    "IRC_SERVER": _msg("IRC server hostname (e.g. irc.libera.chat)", "IRC server", None),
    "IRC_CHANNEL": _msg("IRC channel to join (e.g. #hermes)", "IRC channel", None),
    "IRC_NICKNAME": _msg("Bot nickname on IRC (default: hermes-bot)", "IRC nickname", None),
    "IRC_SERVER_PASSWORD": _msg("IRC server password (if required)", "IRC server password", None,
        password=True, advanced=True),
    "IRC_NICKSERV_PASSWORD": _msg("NickServ password for nick identification", "NickServ password",
        None, password=True, advanced=True),
    "GATEWAY_ALLOW_ALL_USERS": _msg(
        "Allow all users to interact with messaging bots (true/false). Default: false.",
        "Allow all users (true/false)", None, advanced=True),
    "API_SERVER_ENABLED": _msg(
        "Enable the OpenAI-compatible API server (true/false). Allows frontends like Open "
        "WebUI, LobeChat, etc. to connect.", "Enable API server (true/false)", None, advanced=True),
    "API_SERVER_KEY": _msg(
        "Bearer token for API server authentication. Required whenever the API server is "
        "enabled; server refuses to start without it.", "API server auth key", None, password=True,
        advanced=True),
    "API_SERVER_PORT": _msg("Port for the API server (default: 8642).", "API server port", None,
        advanced=True),
    "API_SERVER_HOST": _msg(
        "Host/bind address for the API server (default: 127.0.0.1). API_SERVER_KEY is still "
        "required even on loopback binds.", "API server host", None, advanced=True),
    "API_SERVER_MODEL_NAME": _msg(
        "Model name advertised on /v1/models. Defaults to the profile name (or 'hermes-agent' "
        "for the default profile). Useful for multi-user setups with OpenWebUI.",
        "API server model name", None, advanced=True),
    "GATEWAY_PROXY_URL": _msg(
        "URL of a remote Hermes API server to forward messages to (proxy mode). When set, the "
        "gateway handles platform I/O only — all agent work is delegated to the remote server. "
        "Use for Docker E2EE containers that relay to a host agent. Also configurable via "
        "gateway.proxy_url in config.yaml.",
        "Remote Hermes API server URL (e.g. http://192.168.1.100:8642)", None, advanced=True),
    "GATEWAY_PROXY_KEY": _msg(
        "Bearer token for authenticating with the remote Hermes API server (proxy mode). Must "
        "match the API_SERVER_KEY on the remote host.", "Remote API server auth key", None,
        password=True, advanced=True),
    "WEBHOOK_ENABLED": _msg(
        "Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.",
        "Enable webhooks (true/false)", None),
    "WEBHOOK_PORT": _msg("Port for the webhook HTTP server (default: 8644).", "Webhook port", None),
    "WEBHOOK_SECRET": _msg(
        "Global HMAC secret for webhook signature validation (overridable per route in "
        "config.yaml).", "Webhook secret", None, password=True),
    # ── Agent settings ── (MESSAGING_CWD is gone: use terminal.cwd in config.yaml, which the
    # gateway bridges to TERMINAL_CWD.)
    "SUDO_PASSWORD": _setting(
        "Sudo password for terminal commands requiring root access; set to an explicit empty "
        "string to try empty without prompting", "Sudo password", None, password=True),
    # HERMES_TOOL_PROGRESS_MODE (deprecated; use display.tool_progress) is intentionally NOT listed:
    # this dict feeds user-facing surfaces (dashboard keys page, setup checklists), so deprecated
    # knobs stay in config._EXTRA_ENV_KEYS only. HERMES_TOOL_PROGRESS is unsupported.
    "HERMES_PREFILL_MESSAGES_FILE": _setting(
        "Path to JSON file with ephemeral prefill messages for few-shot priming",
        "Prefill messages file path", None),
    "HERMES_EPHEMERAL_SYSTEM_PROMPT": _setting(
        "Ephemeral system prompt injected at API-call time (never persisted to sessions)",
        "Ephemeral system prompt", None),
}
