"""Unified tool configuration for Hermes Agent."""

import json as _json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from hermes_cli.cli_output import print_info as _print_info
from hermes_cli.colors import Colors, color
from hermes_cli.config import cfg_get, load_config, save_config, get_env_value
from hermes_cli.nous_subscription import (
    MANAGED_FEATURE_COVERAGE_CATEGORY,
    NousSubscriptionFeatures,
    apply_nous_managed_defaults,
    get_nous_subscription_features,
)
from hermes_cli.nous_account import format_nous_portal_entitlement_message
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, fal_key_is_configured
from utils import base_url_hostname, is_truthy_value

logger = logging.getLogger(__name__)

# Platforms already warned about an all-invalid platform_toolsets list (warn once, not per resolution).
_warned_invalid_platform_toolsets: Set[str] = set()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Platform display config derived from the canonical registry (dict-of-dicts for ``PLATFORMS[key]["label"]``).
PLATFORMS = {k: {"label": info.label, "default_toolset": info.default_toolset} for k, info in _PLATFORMS_REGISTRY.items()}

# --- Toolset Registry ---
# Toolsets shown in the configurator: (toolset key in toolsets.py TOOLSETS, label, description).
CONFIGURABLE_TOOLSETS = [
    ("web",             "🔍 Web Search & Scraping",    "web_search, web_extract"),
    ("browser",         "🌐 Browser Automation",       "navigate, click, type, scroll"),
    ("terminal",        "💻 Terminal & Processes",      "terminal, process"),
    ("file",            "📁 File Operations",           "read, write, patch, search"),
    ("code_execution",  "⚡ Code Execution",            "execute_code"),
    ("vision",          "👁️  Vision / Image Analysis",  "vision_analyze"),
    ("video",           "🎬 Video Analysis",            "video_analyze (requires video-capable model)"),
    ("image_gen",       "🎨 Image Generation",          "image_generate"),
    ("video_gen",       "🎬 Video Generation",          "video_generate (text/image/reference)"),
    ("x_search",        "🐦 X (Twitter) Search",        "x_search (requires xAI OAuth or XAI_API_KEY)"),
    ("tts",             "🔊 Text-to-Speech",            "text_to_speech"),
    ("stt",             "🎙️ Speech-to-Text",           "voice transcription (gateway voice messages + voice mode)"),
    ("skills",          "📚 Skills",                    "list, view, manage"),
    ("todo",            "📋 Task Planning",             "todo_list"),
    ("kanban",          "📌 Kanban",                    "opt-in task board tools for this platform"),
    ("memory",          "💾 Memory",                    "persistent memory across sessions"),
    ("context_engine",  "🧩 Context Engine",            "runtime tools from the active context engine"),
    ("session_search",  "🔎 Session Search",            "search past conversations"),
    ("connections",     "🔌 Connections",               "remote connector tools and account authorization"),
    ("clarify",         "❓ Clarifying Questions",      "clarify"),
    ("delegation",      "👥 Task Delegation",           "delegate_task"),
    ("cronjob",         "⏰ Cron Jobs",                 "create/list/update/pause/resume/run, with optional attached skills"),
    ("homeassistant",    "🏠 Home Assistant",           "smart home device control"),
    ("spotify",          "🎵 Spotify",                  "playback, search, playlists, library"),
    ("discord",         "💬 Discord (read/participate)", "fetch messages, search members, create thread"),
    ("discord_admin",   "🛡️  Discord Server Admin",    "list channels/roles, pin, assign roles"),
    ("yuanbao",          "🤖 Yuanbao",                  "group info, member queries, DM"),
    ("computer_use",     "🖱️  Computer Use (macOS/Windows/Linux)", "background desktop control via cua-driver"),
]


def gui_toolset_label(label: str) -> str:
    """Strip the leading ``<emoji>`` from a toolset title for GUI surfaces (plugins prefix ``🔌``).
    CLI/TUI keeps the raw label — only HTTP APIs call this."""
    text = (label or "").strip()
    parts = text.split(None, 1)
    if len(parts) == 2 and not any(ch.isascii() and ch.isalnum() for ch in parts[0]):
        return parts[1].strip()
    return text


# OFF by default for new installs (still in _HERMES_CORE_TOOLS; the checklist won't pre-select them). x_search
# auto-enables when xAI creds exist (mirrors HASS_TOKEN → homeassistant); its check_fn still gates the schema.
_DEFAULT_OFF_TOOLSETS = {"homeassistant", "spotify", "discord", "discord_admin", "video", "video_gen", "x_search", "a2a", "kanban"}

# Config-only capabilities: provider setup in `hermes tools` (TOOL_CATEGORIES) but not model toolsets — zero
# schemas, own switch (``stt.enabled``), never in ``platform_toolsets`` or the per-platform checklist.
_CONFIG_ONLY_TOOLSETS = {"stt"}


def _xai_credentials_present() -> bool:
    """Cheap offline check for xAI credentials (auth store + env only); the runtime ``check_fn`` still gates
    schema registration if creds expire. Also used by ``provider_readiness_status`` for ``xai_grok`` rows."""
    try:
        from hermes_cli.auth import _read_xai_oauth_tokens
        _read_xai_oauth_tokens()
        return True
    except Exception:
        pass
    if str(get_env_value("XAI_API_KEY") or "").strip():
        return True
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # pragma: no cover — secret_scope is in-repo
        get_secret = os.environ.get
    return bool(str(get_secret("XAI_API_KEY") or "").strip())


def _homeassistant_credentials_present() -> bool:
    """Return whether the active profile has a Home Assistant token."""
    try:
        from agent.secret_scope import get_secret
        return bool((get_secret("HASS_TOKEN", "") or "").strip())
    except Exception:
        return False


def _toolset_configuration_platform(ts_key: str, default: str = "cli") -> str:
    """Platform a platform-less configuration UI should target: a toolset restricted away from ``default``
    must be configured on a supported platform, else the save helper drops it and the UI reports a no-op."""
    allowed = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return default if not allowed or default in allowed else sorted(allowed)[0]


def _get_effective_configurable_toolsets():
    """CONFIGURABLE_TOOLSETS + plugin toolsets (appended after built-ins; a plugin key already built-in is skipped)."""
    result = list(CONFIGURABLE_TOOLSETS)
    seen = {ts_key for ts_key, _, _ in result}
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_toolsets
        discover_plugins()  # idempotent — ensures plugins are loaded
        for entry in get_plugin_toolsets():
            if entry[0] not in seen:
                seen.add(entry[0])
                result.append(entry)
    except Exception:
        pass
    return result


def _get_plugin_toolset_keys() -> set:
    """Return the set of toolset keys provided by plugins."""
    try:
        # Non-blocking on CLI startup: while background discovery is still importing, serve last
        # launch's persisted key set instead of joining the discovery thread.
        from hermes_cli.plugins import get_plugin_toolset_keys_nowait
        return get_plugin_toolset_keys_nowait()
    except Exception:
        return set()


def _checklist_toolset_keys(platform: str) -> Set[str]:
    """Toolset keys the ``hermes tools`` checklist offers for ``platform`` (mirrors ``_prompt_toolset_checklist``);
    read-time-resolved toolsets (recovered composites, MCP names) are NOT here."""
    return {
        ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()
        if _toolset_allowed_for_platform(ts_key, platform) and ts_key not in _CONFIG_ONLY_TOOLSETS}


def _platform_default_toolset(platform: str) -> str:
    """Composite toolset a platform falls back to (plugin platforms derive ``hermes-<platform>``)."""
    return PLATFORMS[platform]["default_toolset"] if platform in PLATFORMS else f"hermes-{platform}"


def _cfg_section(config: dict, key: str) -> dict:
    """Return ``config[key]`` as a dict, replacing a missing or non-dict value with ``{}``."""
    section = config.setdefault(key, {})
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def _is_configurable(ts_key: str) -> bool:
    """True when the toolset has provider options or simple env-var requirements to prompt for."""
    return bool(TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key))


def _toolset_label(ts_key: str) -> str:
    """Display label for a toolset key (built-in or plugin), falling back to the key itself."""
    return next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts_key), ts_key)


# --- Tool Categories: toolset key -> provider options shown when newly enabled. Toolsets not in this map
# either need no config or use the TOOLSET_ENV_REQUIREMENTS fallback.
def _key(key: str, prompt: str, url: str = "", **extra) -> dict:
    """One ``env_vars`` entry for a provider row (key order matters for the GUI JSON)."""
    return {"key": key, "prompt": prompt, **extra, **({"url": url} if url else {})}


def _row(name: str, badge: str = "", tag: str = "", env_vars: list = (), **markers) -> dict:
    """One TOOL_CATEGORIES provider row; ``markers`` are the ``*_provider`` / ``post_setup`` / Nous keys."""
    row = {"name": name}
    if badge:
        row["badge"] = badge
    if tag:
        row["tag"] = tag
    row["env_vars"] = list(env_vars)
    row.update(markers)
    return row


_NOUS = {"requires_nous_auth": True}
_OPENAI_VOICE_KEY = _key("VOICE_TOOLS_OPENAI_KEY", "OpenAI API key", "https://platform.openai.com/api-keys")
_ELEVENLABS_KEY = _key("ELEVENLABS_API_KEY", "ElevenLabs API key", "https://elevenlabs.io/app/settings/api-keys")
_DEEPINFRA_KEY = _key("DEEPINFRA_API_KEY", "DeepInfra API key", "https://deepinfra.com/dash/api_keys")
_LANGFUSE_PUBLIC = ("HERMES_LANGFUSE_PUBLIC_KEY", "Langfuse public key (pk-lf-...)")
_LANGFUSE_SECRET = ("HERMES_LANGFUSE_SECRET_KEY", "Langfuse secret key (sk-lf-...)")

TOOL_CATEGORIES = {
    "tts": {
        "name": "Text-to-Speech", "icon": "🔊",
        "providers": [
            _row("Microsoft Edge TTS", "★ recommended · free", "Good quality, no API key needed", tts_provider="edge"),
            _row("Nous Subscription", "subscription", "Managed OpenAI TTS billed to your subscription", tts_provider="openai",
                 **_NOUS, managed_nous_feature="tts", override_env_vars=["VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"]),
            _row("OpenAI TTS", "paid", "High quality voices", [_OPENAI_VOICE_KEY], tts_provider="openai"),
            _row("xAI TTS", tag="Grok voices — uses xAI Grok OAuth or XAI_API_KEY", tts_provider="xai", post_setup="xai_grok"),
            _row("ElevenLabs", "paid", "Most natural voices", [_ELEVENLABS_KEY], tts_provider="elevenlabs"),
            # Mistral Voxtral TTS — `mistralai` SDK lazy-installs on first use.
            _row("Mistral (Voxtral TTS)", "paid", "Multilingual, native Opus",
                 [_key("MISTRAL_API_KEY", "Mistral API key", "https://console.mistral.ai/")], tts_provider="mistral"),
            _row("Google Gemini TTS", "preview", "30 prebuilt voices, controllable via prompts",
                 [_key("GEMINI_API_KEY", "Gemini API key", "https://aistudio.google.com/app/apikey")], tts_provider="gemini"),
            _row("KittenTTS", "local · free", "Lightweight local ONNX TTS (~25MB), no API key", tts_provider="kittentts",
                 post_setup="kittentts"),
            _row("Piper", "local · free", "Local neural TTS, 44 languages (voices ~20-90MB)", tts_provider="piper",
                 post_setup="piper"),
            _row("DeepInfra TTS", "paid", "Chatterbox, Qwen3-TTS, … — live catalog from api.deepinfra.com", [_DEEPINFRA_KEY],
                 tts_provider="deepinfra"),
        ],
    },
    "stt": {
        "name": "Speech-to-Text", "icon": "🎙️",
        "providers": [
            _row("Local Whisper", "★ recommended · free", "faster-whisper on-device, no API key", stt_provider="local",
                 post_setup="faster_whisper"),
            _row("Nous Subscription", "subscription", "Managed OpenAI transcription billed to your subscription",
                 stt_provider="openai", **_NOUS, managed_nous_feature="stt",
                 override_env_vars=["VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"]),
            _row("OpenAI", "paid", "whisper-1, gpt-4o-transcribe, gpt-transcribe", [_OPENAI_VOICE_KEY], stt_provider="openai"),
            _row("Groq", "free tier", "Whisper large-v3 family — very fast",
                 [_key("GROQ_API_KEY", "Groq API key", "https://console.groq.com/keys")], stt_provider="groq"),
            _row("xAI", tag="grok-stt — uses xAI Grok OAuth or XAI_API_KEY", stt_provider="xai", post_setup="xai_grok"),
            _row("ElevenLabs Scribe", "paid", "scribe_v2 — diarization + audio-event tagging", [_ELEVENLABS_KEY],
                 stt_provider="elevenlabs"),
            # Mistral Voxtral STT intentionally omitted — mistralai PyPI package quarantined (malicious 2.4.6
            # release, 2026-05-12). Restore alongside the dashboard stt.provider option.
            _row("DeepInfra", "paid", "Live STT catalog from api.deepinfra.com", [_DEEPINFRA_KEY], stt_provider="deepinfra"),
        ],
    },
    "web": {
        "name": "Web Search & Extract", "setup_title": "Select Search Provider",
        "setup_note": "A free DuckDuckGo search skill is also included — skip this if you don't need a premium provider.",
        "icon": "🔍",
        # Provider rows come from plugins.web.<vendor> via _plugin_web_search_providers(). Only the two
        # non-provider firecrawl setup-flow rows live here: managed via Nous subscription, and self-hosted.
        "providers": [
            {"name": "Nous Subscription", "badge": "subscription", "tag": "Managed Firecrawl billed to your subscription",
             "web_backend": "firecrawl", "env_vars": [], **_NOUS, "managed_nous_feature": "web",
             "override_env_vars": ["FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"]},
            {"name": "Firecrawl Self-Hosted", "badge": "free · self-hosted", "tag": "Run your own Firecrawl instance (Docker)",
             "web_backend": "firecrawl",
             "env_vars": [_key("FIRECRAWL_API_URL", "Your Firecrawl instance URL (e.g., http://localhost:3002)")]},
        ],
    },
    "image_gen": {
        "name": "Image Generation", "icon": "🎨",
        # Provider rows (FAL, OpenAI, OpenAI Codex, xAI) come from plugins.image_gen.<vendor> via
        # _plugin_image_gen_providers(). Only the managed "Nous Subscription" row lives here — fal backend, distinct UX.
        "providers": [
            _row("Nous Subscription", "subscription", "Managed FAL image generation billed to your subscription", **_NOUS,
                 managed_nous_feature="image_gen", override_env_vars=["FAL_KEY"], imagegen_backend="fal"),
        ],
    },
    "video_gen": {
        "name": "Video Generation", "icon": "🎬",
        # Mirrors image_gen: managed FAL video billed via the Nous Portal. Plugin-backed rows (FAL BYOK, xAI, …)
        # are injected at runtime by ``_plugin_video_gen_providers()`` in ``_visible_providers``. Picking this row
        # sets video_gen.provider = "fal" + use_gateway so the FAL plugin routes through the managed queue gateway.
        "providers": [
            _row("Nous Subscription", "subscription", "Managed FAL video generation billed to your subscription", **_NOUS,
                 managed_nous_feature="video_gen", override_env_vars=["FAL_KEY"], video_gen_plugin_name="fal"),
        ],
    },
    "x_search": {
        "name": "X (Twitter) Search", "setup_title": "Select xAI Credential Source",
        "setup_note": (
            "Hermes routes X searches through xAI's built-in x_search Responses tool for read-only public X "
            "discovery. Use the xurl skill for authenticated X API reads and account actions. Both credential "
            "sources hit the same https://api.x.ai/v1/responses endpoint — pick whichever you already have. "
            "SuperGrok OAuth is preferred when both are set (uses your subscription quota instead of API spend)."
        ),
        "icon": "🐦",
        "providers": [
            _row("xAI Grok OAuth (SuperGrok / Premium+)", "subscription", "Browser login at accounts.x.ai — no API key required",
                 post_setup="xai_grok"),
            _row("xAI API key", "paid", "Direct xAI API billing via XAI_API_KEY",
                 [_key("XAI_API_KEY", "xAI API key", "https://console.x.ai/")]),
        ],
    },
    "browser": {
        "name": "Browser Automation", "icon": "🌐",
        # Cloud provider rows (Browserbase, Browser Use, Firecrawl) come from plugins.browser.<vendor> via
        # _plugin_browser_providers(); only non-provider setup-flow rows live here. "Local Browser" MUST stay
        # first so a fresh install's Enter lands on the free local backend (index 0), never on the paid Nous row.
        # Lightpanda is local too (cloud_provider: local, browser.engine: lightpanda — Browser Use mode spawns
        # ``lightpanda serve``, built-in tools use ``agent-browser --engine lightpanda``; no Chromium).
        # Camofox short-circuits the cloud dispatch via _is_camofox_mode().
        "providers": [
            _row("Local Browser", "★ recommended · free", "Headless Chromium, no API key needed", browser_provider="local",
                 browser_engine="auto", post_setup="agent_browser"),
            _row("Lightpanda", "free · local · no Chromium", "Zig headless browser spawned by Hermes, text-only (no screenshots)",
                 browser_provider="local", browser_engine="lightpanda", post_setup="lightpanda"),
            # Cloud hook installs only the agent-browser CLI: Browser Use hosts its own Chromium, so the
            # local-Chromium install and readiness gate must not apply (with "agent_browser" this row read
            # "needs setup" forever on machines without a local Chromium build).
            _row("Nous Subscription (Browser Use cloud)", "subscription", "Managed Browser Use billed to your subscription",
                 browser_provider="browser-use", **_NOUS, managed_nous_feature="browser",
                 override_env_vars=["BROWSER_USE_API_KEY"], post_setup="browserbase"),
            _row("Camofox", "free · local", "Anti-detection browser (Firefox/Camoufox)",
                 [_key("CAMOFOX_URL", "Camofox server URL", "https://github.com/jo-inc/camofox-browser", default="http://localhost:9377")],
                 browser_provider="camofox", post_setup="camofox"),
            _row("Browser Use", "free · local · cloud", "New SOTA web harness (CLI 3.0)", browser_backend="browser-use",
                 post_setup="browser_use_cli"),
        ],
    },
    "homeassistant": {
        "name": "Smart Home", "icon": "🏠",
        "providers": [
            _row("Home Assistant", tag="REST API integration",
                 env_vars=[_key("HASS_TOKEN", "Home Assistant Long-Lived Access Token"),
                           _key("HASS_URL", "Home Assistant URL", default="http://homeassistant.local:8123")]),
        ],
    },
    "spotify": {
        "name": "Spotify", "icon": "🎵",
        "providers": [_row("Spotify Web API", tag="PKCE OAuth — opens the setup wizard", post_setup="spotify")],
    },
    "computer_use": {
        "name": "Computer Use (macOS/Windows/Linux)", "icon": "🖱️",
        # Runtime backends ship for macOS, Windows, Linux (X11; Wayland via XWayland). Gaps surface via `computer-use doctor`.
        "platform_gate": ["darwin", "win32", "linux"],
        # cua-driver reads HOME/TMPDIR from the process env; HERMES_CUA_DRIVER_CMD selects a specific
        # binary (e.g. a local build). There is no version-pin env var.
        "providers": [
            _row("cua-driver (background)", "★ recommended · free · local",
                 "Background computer-use via cua-driver — does NOT steal your cursor or focus. Works with any model.",
                 computer_use_backend="cua", post_setup="cua_driver"),
        ],
    },
    "langfuse": {
        "name": "Langfuse Observability", "icon": "📊",
        "providers": [
            _row("Langfuse Cloud", tag="Hosted Langfuse (cloud.langfuse.com)", post_setup="langfuse",
                 env_vars=[_key(*_LANGFUSE_PUBLIC, "https://cloud.langfuse.com"), _key(*_LANGFUSE_SECRET, "https://cloud.langfuse.com")]),
            _row("Langfuse Self-Hosted", tag="Self-hosted Langfuse instance", post_setup="langfuse",
                 env_vars=[_key(*_LANGFUSE_PUBLIC), _key(*_LANGFUSE_SECRET),
                           _key("HERMES_LANGFUSE_BASE_URL", "Langfuse server URL (e.g. http://localhost:3000)", default="http://localhost:3000")]),
        ],
    },
}

# Env-var fallback for toolsets NOT in TOOL_CATEGORIES. `vision` is only a presence marker (reconfigure menu +
# "[no API key]" suffix): setup runs `_configure_vision_backend()` and `_toolset_has_keys("vision")` uses
# `resolve_vision_provider_client()` — never forcing OpenRouter.
TOOLSET_ENV_REQUIREMENTS = {"vision": [("OPENROUTER_API_KEY", "https://openrouter.ai/keys")]}

# --- Platform / Toolset Helpers ---
_PLATFORM_ENABLE_ENV_VARS = (
    ("telegram", "TELEGRAM_BOT_TOKEN"), ("discord", "DISCORD_BOT_TOKEN"), ("slack", "SLACK_BOT_TOKEN"),
    ("whatsapp", "WHATSAPP_ENABLED"), ("qqbot", "QQ_APP_ID"))


def _get_enabled_platforms() -> List[str]:
    """Return platform keys that are configured (have tokens or are CLI)."""
    return ["cli"] + [platform for platform, env_var in _PLATFORM_ENABLE_ENV_VARS if get_env_value(env_var)]


def _platform_toolset_summary(config: dict, platforms: Optional[List[str]] = None) -> Dict[str, Set[str]]:
    """Enabled toolsets per platform (``platforms`` defaults to ``_get_enabled_platforms()``)."""
    if platforms is None:
        platforms = _get_enabled_platforms()
    return {pkey: _get_platform_tools(config, pkey) for pkey in platforms}


def _parse_enabled_flag(value, default: bool = True) -> bool:
    """Parse bool-like config values used by tool/platform settings."""
    if isinstance(value, (bool, int)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "false", "0", "no", "off"}:
            return lowered in {"true", "1", "yes", "on"}
    return default


def enabled_mcp_server_names(config: dict) -> Set[str]:
    """MCP servers globally enabled in config.yaml or by a plugin (shared by platform + cron resolvers). Enabled
    unless ``enabled`` is explicitly falsey; portable-plugin servers (in-memory) count — enabling the plugin is
    the opt-in."""
    mcp_servers = (config or {}).get("mcp_servers") or {}
    names = {
        str(name) for name, server_cfg in mcp_servers.items()
        if isinstance(server_cfg, dict) and _parse_enabled_flag(server_cfg.get("enabled", True), default=True)
    }
    try:
        from hermes_cli.plugins import get_portable_mcp_server_names_nowait
        portable = get_portable_mcp_server_names_nowait()
        names |= portable - set(mcp_servers)  # native config wins on a name collision (mirrors _load_mcp_config)
    except Exception:
        logger.debug("Failed to include portable MCP servers", exc_info=True)
    return names


#: Toolsets young enough that absence from a saved ``platform_toolsets`` list means "never offered", not
#: "declined": saving ``hermes tools`` freezes a platform's composite into an explicit list nothing adds to, so
#: a later toolset stays off forever for picker users while ``[hermes-cli]`` users inherit it.
#: MUST ship in the same release as the toolset and be emptied in the next: once a released build has put the
#: toolset on a checklist, an unchecking user's config is byte-identical to one saved before it existed and this
#: rule would turn the opt-out back on (stuck checkbox). ``check_fn``-gated toolsets cost nothing here; never
#: probe a remote service from this path — it runs on every CLI start, gateway session and cron tick.
_RECENTLY_SHIPPED_TOOLSETS: frozenset = frozenset()


#: Toolsets young enough that absence from a saved ``platform_toolsets`` list
#: means "never offered" rather than "declined".
#:
#: Saving ``hermes tools`` (or one toggle in the desktop Toolsets UI) replaces
#: a platform's composite with a frozen explicit list, and nothing ever adds to
#: that list — so a toolset shipped afterwards stays off forever for anyone who
#: has touched the picker, while everyone still on ``[hermes-cli]`` inherits it
#: on upgrade. Listing it here restores that parity.
#:
#: MUST ship in the same release as the toolset it names, and be emptied in the
#: next one. The inference only holds while no released build has put the
#: toolset on a checklist: once one has, a user who unchecks it writes a config
#: byte-identical to one saved before the toolset existed (the record below is
#: only written from that point on), and this rule turns their opt-out back on.
#: Landing late — or leaving an entry here for a second release — converts a
#: back-fill into a stuck checkbox.
#:
#: A ``check_fn``-gated toolset costs nothing here for users who cannot call
#: it: an enabled toolset still ships zero schemas when its check fails — the
#: same split Home Assistant uses. Probing a remote service from this path
#: would put a network call on every CLI start, gateway session and cron tick.
_RECENTLY_SHIPPED_TOOLSETS: frozenset = frozenset()


def _enable_recently_shipped_toolsets(
    enabled_toolsets: Set[str], config: dict, platform: str
) -> None:
    """Turn on toolsets that shipped after this platform's saved list.

    Either way of saying no outlives this: unchecking in ``hermes tools``
    records the toolset in ``known_builtin_toolsets`` so it reads as declined
    from then on, and ``agent.disabled_toolsets`` is subtracted after every
    rule in :func:`_get_platform_tools`. Mutates ``enabled_toolsets`` in place.
    """
    from toolsets import resolve_toolset

    offered = (config.get("known_builtin_toolsets") or {}).get(platform)
    declined = {str(ts) for ts in offered} if isinstance(offered, list) else set()
    default_ts = _platform_default_toolset(platform)
    composite_tools = None
    for ts_key in sorted(_RECENTLY_SHIPPED_TOOLSETS):
        if ts_key in enabled_toolsets or ts_key in declined or not _toolset_allowed_for_platform(ts_key, platform):
            continue
        # Only enable where staying on the composite would have enabled it anyway; deliberately narrow
        # composites (hermes-acp, hermes-webhook) stay narrow.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if composite_tools is None:
            composite_tools = set(resolve_toolset(default_ts))
        if not ts_tools or not ts_tools.issubset(composite_tools):
            continue
        enabled_toolsets.add(ts_key)


def _configurable_subset_of(tool_names: Set[str], platform: str) -> Set[str]:
    """Configurable toolsets whose STATIC membership is within ``tool_names`` (``include_registry=False``: a
    runtime-registered tool the composite never listed must not drop the whole toolset)."""
    from toolsets import resolve_toolset

    return {
        ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS if _toolset_allowed_for_platform(ts_key, platform)
        and (ts_tools := set(resolve_toolset(ts_key, include_registry=False))) and ts_tools <= tool_names}


def _default_off_toolsets(platform: str, explicitly_configured: bool) -> Set[str]:
    """Toolsets to strip from an implicit (composite-derived) enable set. A platform named after a default-off
    toolset (``homeassistant``) keeps it, except platform-restricted ones (``discord`` on discord stays OFF); a
    configured HASS_TOKEN is an explicit opt-in that must survive platforms resolving without a saved list.
    Platform-native default-off toolsets (``discord`` on discord) are off for unconfigured platforms as a
    security opt-in — an explicitly saved list IS that opt-in and lets them through."""
    default_off = set(_DEFAULT_OFF_TOOLSETS)
    if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
        default_off.remove(platform)
    # Home Assistant is already runtime-gated by its check_fn (requires HASS_TOKEN to register any tools).
    # When a user has configured HASS_TOKEN, they've explicitly opted in — don't also strip it via
    # _DEFAULT_OFF_TOOLSETS, which would silently drop HA from platforms (e.g. cron) that run through
    # _get_platform_tools without an explicit saved toolset list. Without this, Norbert's HA cron jobs
    # regressed after #14798 made cron honor per-platform tool config.
    if "homeassistant" in default_off and _homeassistant_credentials_present():
        default_off.remove("homeassistant")
    if explicitly_configured:
        default_off -= {ts for ts in default_off if platform in (_TOOLSET_PLATFORM_RESTRICTIONS.get(ts) or ())}
    return default_off


def _configurable_keys() -> Set[str]:
    return {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}


def _platform_default_keys() -> Set[str]:
    return {p["default_toolset"] for p in PLATFORMS.values()}


def _explicit_toolsets(
    toolset_names: List[str], explicit_known_keys: Set[str], config: dict, platform: str,
    explicitly_configured: bool) -> Set[str]:
    """Enabled set when the saved list names configurable/plugin keys directly (subset inference over
    ``hermes-cli`` would re-enable disabled toolsets). A mixed list (``[hermes-cli, spotify]``) still expands the
    composite; _DEFAULT_OFF_TOOLSETS applies to that implicit expansion only."""
    from toolsets import resolve_toolset, TOOLSETS

    enabled = {ts for ts in toolset_names if ts in explicit_known_keys and _toolset_allowed_for_platform(ts, platform)}
    composite_tools = {
        t for ts_name in toolset_names if ts_name not in explicit_known_keys and ts_name in TOOLSETS
        for t in resolve_toolset(ts_name)}
    if composite_tools:
        enabled |= _configurable_subset_of(composite_tools, platform) - _default_off_toolsets(platform, explicitly_configured)
    _enable_recently_shipped_toolsets(enabled, config, platform)
    return enabled


def _composite_toolsets(toolset_names: List[str], platform: str, explicitly_configured: bool) -> Set[str]:
    """Enabled set inferred from composite names by reverse-mapping tool names (only while no explicit list is
    saved). ``x_search`` is not in any composite, so inject it when xAI creds exist and exempt it from default-off."""
    from toolsets import resolve_toolset

    all_tool_names = {t for ts_name in toolset_names for t in resolve_toolset(ts_name)}
    enabled = _configurable_subset_of(all_tool_names, platform)
    default_off = _default_off_toolsets(platform, explicitly_configured)
    if _toolset_allowed_for_platform("x_search", platform) and _xai_credentials_present():
        enabled.add("x_search")
        default_off.discard("x_search")
    return enabled - default_off


def _enabled_plugin_toolsets(config: dict, platform: str, toolset_names: List[str], plugin_ts_keys: Set[str]) -> Set[str]:
    """Plugin toolsets: on by default unless default-off (bundled spotify) or "known" for this platform
    (``known_plugin_toolsets``, written on every save) and absent from the saved list."""
    known_for_platform = set((config.get("known_plugin_toolsets", {}) or {}).get(platform, []) or [])
    return {
        pts for pts in plugin_ts_keys
        if pts in toolset_names or (pts not in _DEFAULT_OFF_TOOLSETS and pts not in known_for_platform)
    }


def _context_engine_active(config: dict) -> bool:
    context_cfg = config.get("context") or {}
    name = str(context_cfg.get("engine") or "compressor").strip().lower() if isinstance(context_cfg, dict) else "compressor"
    return bool(name) and name != "compressor"


def _get_platform_tools(config: dict, platform: str, *, include_default_mcp_servers: bool = True) -> Set[str]:
    """Resolve which individual toolset names are enabled for a platform."""
    platform_toolsets = config.get("platform_toolsets") or {}
    toolset_names = platform_toolsets.get(platform)
    # An explicitly saved list (even a composite like ``hermes-discord``) is an opt-in to the platform's
    # native default-off toolsets — see _default_off_toolsets.
    # Track whether the user explicitly saved a toolset list for this platform (vs. falling back to the
    # platform default). See #35527.
    explicitly_configured = isinstance(toolset_names, list)
    if not explicitly_configured:
        toolset_names = [_platform_default_toolset(platform)]
    # YAML may parse bare numeric names (``12306:``) as int; normalise so sorted() never mixes types.
    toolset_names = [str(ts) for ts in toolset_names]

    configurable_keys = _configurable_keys()
    plugin_ts_keys = _get_plugin_toolset_keys()
    platform_default_keys = _platform_default_keys()
    # Plugin toolsets are first-class on a saved list: ``[hermes-cli, a2a]`` must survive filtering.
    # Plugin-provided toolsets are first-class on a platform-toolsets list — explicit config like
    # ``[hermes-cli, a2a]`` must survive filtering just like a built-in configurable toolset would. See
    # issue #81163.
    explicit_known_keys = configurable_keys | plugin_ts_keys

    if any(ts in explicit_known_keys for ts in toolset_names):
        enabled_toolsets = _explicit_toolsets(toolset_names, explicit_known_keys, config, platform, explicitly_configured)
    else:
        enabled_toolsets = _composite_toolsets(toolset_names, platform, explicitly_configured)

    _recover_platform_native_toolsets(enabled_toolsets, platform, skip=configurable_keys | plugin_ts_keys | platform_default_keys)
    if plugin_ts_keys:
        enabled_toolsets |= _enabled_plugin_toolsets(config, platform, toolset_names, plugin_ts_keys)

    # Context-engine tools are runtime-provided, not in any static composite: keep them for a non-default
    # engine even after an explicit save. An explicit EMPTY list means none, unless ``context_engine`` is added by hand.
    if _context_engine_active(config) and not (explicitly_configured and not toolset_names):
        enabled_toolsets.add("context_engine")

    # Explicit non-configurable entries (custom toolsets, MCP server names) pass through.
    explicit_passthrough = {ts for ts in toolset_names if ts not in explicit_known_keys and ts not in platform_default_keys}
    enabled_toolsets |= _merge_mcp_servers(config, toolset_names, explicit_passthrough, include_default_mcp_servers)

    # Legacy profile opt-in is a fallback only. A saved platform list (even
    # empty) is authoritative, so a later disable cannot silently re-enable it.
    if not explicitly_configured and "kanban" in (config.get("toolsets") or []):
        enabled_toolsets.add("kanban")

    # agent.disabled_toolsets is a global suppression list (#86661) and runs LAST so it overrides everything
    # above. It may arrive as a JSON-array string ("['memory']") from `hermes config set` or a JSON-mode editor.
    disabled_toolsets = (config.get("agent") or {}).get("disabled_toolsets")
    if disabled_toolsets:
        from agent.skill_utils import parse_config_string_list
        disabled_names = [name.strip() for name in parse_config_string_list(disabled_toolsets) if name.strip()]
        enabled_toolsets = _prune_toolsets_stripped_by_disabled(enabled_toolsets, disabled_names)

    if explicitly_configured and toolset_names:
        _warn_all_invalid_platform_toolsets(platform, platform_toolsets[platform])
    return enabled_toolsets


def _prune_toolsets_stripped_by_disabled(enabled_toolsets: Set[str], disabled_names: List[str]) -> Set[str]:
    """Drop disabled names AND every toolset whose tools the runtime would strip anyway.

    The agent subtracts ``agent.disabled_toolsets`` at TOOL granularity (``model_tools._select_tool_names``),
    so disabling a composite like ``debugging`` removes the terminal/web/file tools even though those names
    never appear in the list. A name-only subtraction here left inspection surfaces (``hermes tools
    --summary``, banner, ``/tools``) showing toolsets as enabled that no session could call (#97015).
    Passthrough entries (MCP server names) and toolsets with no static tools (``context_engine``) are kept.
    """
    from model_tools import _apply_toolset_selection
    from toolsets import resolve_toolset, validate_toolset

    remaining = enabled_toolsets - set(disabled_names)
    resolved = {name: set(resolve_toolset(name)) if validate_toolset(name) else set() for name in remaining}
    surviving: Set[str] = set().union(*resolved.values())
    _apply_toolset_selection(surviving, disabled_names, quiet_mode=True, disable=True)
    return {name for name, tools in resolved.items() if not tools or tools & surviving}


def _recover_platform_native_toolsets(enabled_toolsets: Set[str], platform: str, *, skip: Set[str]) -> None:
    """Add non-configurable platform toolsets (discord, feishu_*) in place: in the default composite but not in
    CONFIGURABLE_TOOLSETS, so never in a checklist or saved list. Runs for BOTH ``_get_platform_tools`` branches."""
    from toolsets import resolve_toolset, TOOLSETS

    platform_tool_universe = set(resolve_toolset(_platform_default_toolset(platform)))
    configurable_tool_universe = {t for ts_key, _, _ in CONFIGURABLE_TOOLSETS for t in resolve_toolset(ts_key)}
    claimed = {t for ts_key in enabled_toolsets for t in resolve_toolset(ts_key)}
    skip = skip | {k for k in TOOLSETS if k.startswith("hermes-")} | (set(_DEFAULT_OFF_TOOLSETS) - {platform})
    for ts_key, ts_def in TOOLSETS.items():
        # Posture toolsets (``coding``) are session-level selections made by agent/coding_context.py, not
        # per-platform capabilities to recover.
        if ts_key in skip or ts_def.get("includes") or ts_def.get("posture"):
            continue
        # Static membership: a registry-added tool absent from the platform composite must not block recovery
        # of a non-configurable toolset whose authored tools the composite lists.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if not ts_tools or not ts_tools <= platform_tool_universe or ts_tools <= configurable_tool_universe:
            continue
        if not ts_tools <= claimed:
            enabled_toolsets.add(ts_key)
            claimed.update(ts_tools)


def _merge_mcp_servers(
    config: dict, toolset_names: List[str], explicit_passthrough: Set[str], include_default_mcp_servers: bool
) -> Set[str]:
    """Explicit passthrough entries plus this platform's MCP servers: listed names form an allowlist, else every
    globally enabled server (when ``include_default_mcp_servers``); the ``no_mcp`` sentinel disables all."""
    enabled_mcp_servers = enabled_mcp_server_names(config)
    result = explicit_passthrough - enabled_mcp_servers
    if "no_mcp" in toolset_names:
        return result - {"no_mcp"}
    explicit_mcp_servers = explicit_passthrough & enabled_mcp_servers
    if include_default_mcp_servers and not explicit_mcp_servers:
        return result | enabled_mcp_servers
    return result | explicit_mcp_servers


def _warn_all_invalid_platform_toolsets(platform: str, explicit: list) -> None:
    """Warn once when an explicit platform list has only invalid names (``hermes`` for ``hermes-cli`` → no
    native tools), at session tool resolution rather than only in update/doctor."""
    from toolsets import validate_toolset

    named = [str(t) for t in explicit if isinstance(t, str) and t]
    if named and not any(validate_toolset(t) for t in named) and platform not in _warned_invalid_platform_toolsets:
        _warned_invalid_platform_toolsets.add(platform)
        logger.warning(
            "platform '%s' has no valid toolsets configured (unknown "
            "name(s): %s) - tools will be unavailable. Run `hermes tools` "
            "to reconfigure. See issue #38798.",
            platform, ", ".join(named))


def _save_platform_tools(config: dict, platform: str, enabled_toolset_keys: Set[str]):
    """Save the selected toolset keys for a platform to config."""
    config.setdefault("platform_toolsets", {})
    # Drop platform-scoped toolsets that don't apply here, so the "Configure all platforms" checklist (or a
    # hand-edited config.yaml) can't turn on `discord` for Telegram.
    enabled_toolset_keys = {ts for ts in enabled_toolset_keys if _toolset_allowed_for_platform(ts, platform)}
    plugin_keys = _get_plugin_toolset_keys()
    # Preserve only existing entries that are neither configurable nor platform defaults (i.e. MCP server
    # names): platform defaults (hermes-cli, ...) resolve to ALL tools and would silently override the user's
    # unchecked selections on the next read. Saving from the picker is consent to clear the "no_mcp" sentinel
    # (no checkbox for it; users who once set it by hand could otherwise never re-enable MCP via the UI).
    drop = _configurable_keys() | plugin_keys | _platform_default_keys() | {"no_mcp"}
    existing_toolsets = cfg_get(config, "platform_toolsets", platform, default=[])
    preserved_entries = {str(e) for e in (existing_toolsets if isinstance(existing_toolsets, list) else [])
                         if str(e) not in drop}
    config["platform_toolsets"][platform] = sorted(enabled_toolset_keys | preserved_entries)
    # Record which plugin toolsets this platform "knows" (distinguishes "new plugin, default enabled" from
    # "user disabled it"). _cfg_section normalizes a present-but-null key that setdefault alone would not replace.
    if plugin_keys:
        _cfg_section(config, "known_plugin_toolsets")[platform] = sorted(plugin_keys)
    # Same record for builtin toolsets the checklist offered; without it an unchecked toolset is
    # indistinguishable from one shipped after the save and _enable_recently_shipped_toolsets re-enables it.
    _cfg_section(config, "known_builtin_toolsets")[platform] = sorted(_configurable_keys())
    # Reconcile with agent.disabled_toolsets, which _get_platform_tools applies as a final override: a toolset
    # listed there stays OFF no matter what this writes (Blank Slate installs pre-populate ~27 entries, making
    # the desktop Toolsets UI unable to re-enable anything). Only toolsets just explicitly enabled FOR THIS
    # PLATFORM are cleared, so the list keeps working as a cross-platform suppression list for everything else.
    # See #49995.
    agent_cfg = config.get("agent")
    newly_enabled = enabled_toolset_keys - preserved_entries
    if isinstance(agent_cfg, dict) and agent_cfg.get("disabled_toolsets") and newly_enabled:
        from agent.skill_utils import parse_config_string_list
        parsed_disabled = parse_config_string_list(agent_cfg["disabled_toolsets"])
        remaining = [ts for ts in parsed_disabled if ts not in newly_enabled]
        if remaining != parsed_disabled:
            agent_cfg["disabled_toolsets"] = remaining
    save_config(config)


def _provider_env_ready(provider: dict) -> bool:
    """True when every env var a provider row declares is set (trivially true for no-key rows)."""
    return all(get_env_value(e["key"]) for e in provider.get("env_vars", []))


def _toolset_has_keys(
    ts_key: str, config: dict = None, *, force_fresh: bool = False, features: Optional[NousSubscriptionFeatures] = None,
) -> bool:
    """Check if a toolset's required API keys are configured."""
    if config is None:
        config = load_config()
    if ts_key == "vision":
        try:
            from agent.auxiliary_client import resolve_vision_provider_client
            return resolve_vision_provider_client()[1] is not None
        except Exception:
            return False
    if ts_key in {"web", "image_gen", "video_gen", "tts", "stt", "browser"}:
        if features is None:
            features = get_nous_subscription_features(config, force_fresh=force_fresh)
        feature = features.features.get(ts_key)
        if feature and (feature.available or feature.managed_by_nous):
            return True
    # Provider-aware categories first: a no-key provider (Local Browser, Edge TTS) counts as configured.
    cat = TOOL_CATEGORIES.get(ts_key)
    if cat:
        return any(_provider_env_ready(p) for p in _visible_providers(cat, config, force_fresh=force_fresh, features=features))
    return all(get_env_value(var) for var, _ in TOOLSET_ENV_REQUIREMENTS.get(ts_key, []))


def _prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu (arrow keys). Delegates to curses_radiolist."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=default)


# --- Token Estimation ---
# Profile-keyed cache so one process can serve distinct plugin tool catalogs.
_tool_token_cache: Optional[Dict[tuple[str, int], Dict[str, int]]] = None


def _estimate_tool_tokens() -> Dict[str, int]:
    """tiktoken (cl100k_base) tokens per tool name from the serialised OpenAI schema; cached per process and
    registry generation, {} if tiktoken/registry unavailable."""
    global _tool_token_cache
    from hermes_constants import hermes_home_key

    scope = hermes_home_key()
    _tool_token_cache = _tool_token_cache or {}
    try:
        import model_tools  # noqa: F401 — triggers full tool discovery
        from tools.registry import registry
        cache_key = (scope, registry._generation)
    except Exception:
        logger.debug("Tool registry unavailable; skipping token estimation")
        return _tool_token_cache.setdefault((scope, -1), {})
    if cache_key in _tool_token_cache:
        return _tool_token_cache[cache_key]
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.debug("tiktoken unavailable; skipping tool token estimation")
        return _tool_token_cache.setdefault(cache_key, {})
    # Mirror the wire shape sent to the API.
    counts = {
        name: len(enc.encode(_json.dumps({"type": "function", "function": schema})))
        for name in registry.get_all_tool_names() if (schema := registry.get_schema(name))}
    _tool_token_cache[cache_key] = counts
    return counts


def _prompt_toolset_checklist(platform_label: str, enabled: Set[str], platform: str = "cli", *, force_fresh: bool = True) -> Set[str]:
    """Multi-select checklist of toolsets. Returns set of selected toolset keys."""
    from hermes_cli.curses_ui import curses_checklist
    from toolsets import resolve_toolset

    tool_tokens = _estimate_tool_tokens()
    # Drop platform-scoped toolsets that don't apply here and config-only capabilities (stt).
    effective = [
        (k, l, d) for (k, l, d) in _get_effective_configurable_toolsets()
        if _toolset_allowed_for_platform(k, platform) and k not in _CONFIG_ONLY_TOOLSETS]
    labels = [
        f"{ts_label}  ({ts_desc})"
        + ("  [no API key]" if not _toolset_has_keys(ts_key, force_fresh=force_fresh) and _is_configurable(ts_key) else "")
        for ts_key, ts_label, ts_desc in effective]
    pre_selected = {i for i, (ts_key, _, _) in enumerate(effective) if ts_key in enabled}

    status_fn = None
    if tool_tokens:
        ts_keys = [ts_key for ts_key, _, _ in effective]

        def status_fn(chosen: set) -> str:
            """Deduplicated token cost of the selected toolsets."""
            all_tools: set = set()
            for idx in chosen:
                all_tools.update(resolve_toolset(ts_keys[idx]))
            total = sum(tool_tokens.get(name, 0) for name in all_tools)
            return f"Est. tool context: ~{total / 1000:.1f}k tokens" if total >= 1000 else f"Est. tool context: ~{total} tokens"

    chosen = curses_checklist(
        f"Tools for {platform_label}", labels, pre_selected, cancel_returns=pre_selected, status_fn=status_fn,
    )
    return {effective[i][0] for i in chosen}


# --- Provider-Aware Configuration ---
def _configure_toolset(ts_key: str, config: dict, *, force_fresh: bool = True, reconfigure: bool = False):
    """Configure a toolset: provider selection + API keys (TOOL_CATEGORIES), else simple env-var prompts."""
    cat = TOOL_CATEGORIES.get(ts_key)
    if cat:
        _configure_tool_category(ts_key, cat, config, force_fresh=force_fresh, reconfigure=reconfigure)
    else:
        _configure_simple_requirements(ts_key, reconfigure=reconfigure)


def _plugin_image_gen_providers() -> list[dict]:
    """Build picker-row dicts from plugin-registered image gen providers.

    Each returned dict looks like a regular ``TOOL_CATEGORIES`` provider
    row but carries an ``image_gen_plugin_name`` marker so downstream
    code (config writing, model picker) knows to route through the
    plugin registry. Every image-gen backend is a plugin now — there
    are no hardcoded rows left in ``TOOL_CATEGORIES["image_gen"]`` for
    this function to dedupe against (see issue #26241).
    """
    try:
        from agent.image_gen_registry import list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        providers = list_providers()
    except Exception:
        return []

    rows: list[dict] = []
    for provider in providers:
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        row = {
            "name": schema.get("name", provider.display_name),
            "badge": schema.get("badge", ""),
            "tag": schema.get("tag", ""),
            "env_vars": schema.get("env_vars", []),
            "image_gen_plugin_name": provider.name,
        }
        if schema.get("post_setup"):
            row["post_setup"] = schema["post_setup"]
        rows.append(row)
    return rows


def _plugin_video_gen_providers() -> list[dict]:
    """Build picker-row dicts from plugin-registered video gen providers.

    Mirrors ``_plugin_image_gen_providers`` exactly — every video backend
    is a plugin, so this function is the *only* source of provider rows
    for the Video Generation category. The hardcoded ``TOOL_CATEGORIES``
    entry for ``video_gen`` keeps an empty providers list.
    """
    try:
        from agent.video_gen_registry import list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        providers = list_providers()
    except Exception:
        return []

    rows: list[dict] = []
    for provider in providers:
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        row = {
            "name": schema.get("name", provider.display_name),
            "badge": schema.get("badge", ""),
            "tag": schema.get("tag", ""),
            "env_vars": schema.get("env_vars", []),
            "video_gen_plugin_name": provider.name,
        }
        if schema.get("post_setup"):
            row["post_setup"] = schema["post_setup"]
        rows.append(row)
    return rows


# Mirror of _plugin_image_gen_providers for web search backends. Surfaces
# every plugin-registered web provider so it appears in the
# "Web Search & Extract" picker. All seven providers (brave-free, ddgs,
# searxng, exa, parallel, tavily, firecrawl) live as plugins after
# PR #25182 — this helper is the sole source of truth for the category's
# provider rows. The hardcoded entries that used to drive the category
# were deleted in the same PR; only the two non-provider UX rows
# ("Nous Subscription" managed-gateway entry, "Firecrawl Self-Hosted")
# remain in TOOL_CATEGORIES because they describe alternative *setup
# flows* for the firecrawl backend rather than distinct providers.
def _plugin_web_search_providers() -> list[dict]:
    """Build picker-row dicts from plugin-registered web search providers.

    Each returned dict is a regular ``TOOL_CATEGORIES`` provider row. It
    populates both ``web_backend`` (legacy field consumed by setup +
    selection helpers) and ``web_search_plugin_name`` (informational
    marker) so the picker behaves identically whether a provider is
    hardcoded or plugin-registered.

    After PR #25182, all seven web providers (brave-free, ddgs, searxng,
    exa, parallel, tavily, firecrawl) are plugins; this helper is the sole
    source of provider rows for the Web Search & Extract category.
    """
    try:
        from agent.web_search_registry import list_providers as _list_web_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        providers = _list_web_providers()
    except Exception:
        return []

    rows: list[dict] = []
    for provider in providers:
        name = getattr(provider, "name", None)
        if not name:
            continue
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        # A schema may expose tier ``variants`` (e.g. Exa/Parallel free
        # keyless endpoint vs paid SDK) — flatten the base row plus each
        # variant into separate picker rows sharing the same backend name,
        # distinguished by ``web_tier`` (persisted to
        # ``web.provider_tier.<name>`` on selection).
        schemas = [schema] + [
            v for v in (schema.get("variants") or []) if isinstance(v, dict)
        ]
        for entry in schemas:
            row = {
                "name": entry.get("name", provider.display_name),
                "badge": entry.get("badge", ""),
                "tag": entry.get("tag", ""),
                "env_vars": entry.get("env_vars", []),
                "web_backend": name,
                "web_search_plugin_name": name,
            }
            if entry.get("web_tier"):
                row["web_tier"] = entry["web_tier"]
            # Optional pass-through fields the schema can opt into.
            if entry.get("post_setup"):
                row["post_setup"] = entry["post_setup"]
            rows.append(row)
    return rows


def web_provider_capabilities(backend: str) -> list:
    """Return the capabilities (``search`` / ``extract``) a web backend supports.

    Consults the plugin registry's provider instance (``supports_search`` /
    ``supports_extract``) so the Capabilities GUI can offer per-capability
    selection (``web.search_backend`` / ``web.extract_backend``) only where it
    makes sense — e.g. ddgs and brave-free are search-only. Falls back to both
    capabilities when the backend isn't registered (hardcoded setup-flow rows
    like the managed Firecrawl entries resolve before plugin discovery in some
    test contexts, and firecrawl itself supports both).
    """
    try:
        from agent.web_search_registry import get_provider

        provider = get_provider(backend)
        if provider is not None:
            caps = []
            if provider.supports_search():
                caps.append("search")
            if provider.supports_extract():
                caps.append("extract")
            return caps
    except Exception:
        pass
    return ["search", "extract"]


# Mirror of _plugin_web_search_providers for cloud browser backends. After
# PR #25214, Browserbase / Browser Use / Firecrawl live as plugins under
# plugins/browser/<vendor>/; this helper is the sole source of provider rows
# for those three in the "Browser Automation" picker. The hardcoded
# ``TOOL_CATEGORIES["browser"]`` entries that drove the category before
# were deleted in the same PR; only non-provider UX setup-flow rows remain
# ("Nous Subscription", "Local Browser", "Camofox") — see the comment block
# in ``TOOL_CATEGORIES["browser"]`` for why each one stays hardcoded.
def _plugin_browser_providers() -> list[dict]:
    """Build picker-row dicts from plugin-registered cloud browser providers.

    Each returned dict mirrors the legacy ``TOOL_CATEGORIES["browser"]``
    schema (``name`` / ``badge`` / ``tag`` / ``env_vars`` /
    ``browser_provider`` / ``post_setup``) so the picker behaves identically
    whether a provider was hardcoded or plugin-registered.

    Populates ``browser_provider`` (the legacy config key written to
    ``browser.cloud_provider``) and a ``browser_plugin_name`` marker so
    setup / write paths can route through the registry when they want to.
    """
    try:
        from agent.browser_registry import list_providers as _list_browser_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        providers = _list_browser_providers()
    except Exception:
        return []

    rows: list[dict] = []
    for provider in providers:
        name = getattr(provider, "name", None)
        if not name:
            continue
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        row = {
            "name": schema.get("name", provider.display_name),
            "badge": schema.get("badge", ""),
            "tag": schema.get("tag", ""),
            "env_vars": schema.get("env_vars", []),
            "browser_provider": name,
            "browser_plugin_name": name,
        }
        # Pass-through optional fields the schema can opt into.
        if schema.get("post_setup"):
            row["post_setup"] = schema["post_setup"]
        rows.append(row)
    return rows


def _plugin_tts_providers() -> list[dict]:
    """Build picker-row dicts from plugin-registered TTS providers.

    Issue #30398 — the ``register_tts_provider()`` plugin hook
    coexists alongside the 10 built-in TTS providers
    (``edge``/``openai``/``elevenlabs``/…) and the
    ``tts.providers.<name>: type: command`` registry from PR #17843.
    Built-in rows stay hardcoded in ``TOOL_CATEGORIES["tts"]``; this
    function only injects PLUGIN-registered providers.

    Defensive: plugins whose name collides with a built-in TTS provider
    are filtered out — even though the registry already rejects them
    at registration time, a future code path that registers directly
    via :func:`agent.tts_registry.register_provider` could slip
    through. Filtering here keeps the picker invariant.
    """
    try:
        from agent.tts_registry import _BUILTIN_NAMES, list_providers
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        providers = list_providers()
    except Exception:
        return []

    rows: list[dict] = []
    for provider in providers:
        name = getattr(provider, "name", None)
        if not name:
            continue
        # Defensive: reject built-in shadowing at the picker layer too.
        if name.lower().strip() in _BUILTIN_NAMES:
            continue
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        row = {
            "name": schema.get("name", provider.display_name),
            "badge": schema.get("badge", ""),
            "tag": schema.get("tag", ""),
            "env_vars": schema.get("env_vars", []),
            # Selecting this row writes ``tts.provider: <name>`` — the
            # same write-path used by hardcoded rows. The plugin
            # dispatcher picks it up automatically from there.
            "tts_provider": name,
            "tts_plugin_name": name,
        }
        if schema.get("post_setup"):
            row["post_setup"] = schema["post_setup"]
        rows.append(row)
    return rows


def _visible_providers(
    cat: dict,
    config: dict,
    *,
    force_fresh: bool = False,
    features: Optional[NousSubscriptionFeatures] = None,
) -> list[dict]:
    """Return provider entries visible for the current auth/config state.

    Nous-managed Tool Gateway rows (``managed_nous_feature``) are always
    shown — even to logged-out / unentitled users — so the picker advertises
    that the capability exists.  Selecting one drives an inline Nous Portal
    login + entitlement check (see ``_configure_provider``); the row only
    *activates* the gateway once paid access is confirmed.
    """
    if features is None:
        features = get_nous_subscription_features(config, force_fresh=force_fresh)
    acct = features.account_info
    # Pool-only users (entitled to managed tools via the free tool pool but with
    # no paid access) get image gen but NOT video gen — the pool doesn't fund
    # `fal-video`. Rather than advertise a managed video row that would be denied
    # on select, hide it for them. Logged-out users still see it (advertising)
    # and paid users are entitled to it.
    pool_only = bool(
        acct
        and acct.logged_in
        and acct.paid_service_access is not True
        and acct.tool_gateway_entitled
    )
    visible = []
    for provider in cat.get("providers", []):
        # Nous-managed Tool Gateway rows stay visible regardless of auth —
        # selecting one drives an inline Portal login. A `requires_nous_auth`
        # row that is NOT a managed gateway feature (pure pre-auth UX) is
        # still hidden until the user is logged in.
        if (
            provider.get("requires_nous_auth")
            and not provider.get("managed_nous_feature")
            and not features.nous_auth_present
        ):
            continue
        # Hide the managed video-gen row from pool-only users — their free tool
        # pool doesn't cover video, so showing it would only lead to a denial.
        if (
            pool_only
            and provider.get("managed_nous_feature") == "video_gen"
            and not (acct and acct.tool_gateway_entitled_for("fal-video"))
        ):
            continue
        visible.append(provider)

    # Inject plugin-registered image_gen backends (OpenAI today, more
    # later) so the picker lists them alongside FAL / Nous Subscription.
    if cat.get("name") == "Image Generation":
        visible.extend(_plugin_image_gen_providers())

    # Inject plugin-registered video_gen backends. Unlike image_gen,
    # video_gen has NO hardcoded providers — every backend is a plugin.
    if cat.get("name") == "Video Generation":
        visible.extend(_plugin_video_gen_providers())

    # Inject plugin-registered web search backends. After PR #25182, this
    # is the SOLE source of provider rows for the Web Search & Extract
    # category — the per-provider hardcoded entries were deleted. The two
    # remaining hardcoded rows ("Nous Subscription", "Firecrawl
    # Self-Hosted") are non-provider UX setup-flow rows for firecrawl.
    if cat.get("name") == "Web Search & Extract":
        visible.extend(_plugin_web_search_providers())

    # Inject plugin-registered cloud browser backends. After PR #25214,
    # Browserbase / Browser Use / Firecrawl are the plugin-supplied rows;
    # the hardcoded "Nous Subscription" / "Local Browser" / "Camofox" rows
    # stay because they're non-provider UX setup flows (subscription auth,
    # local fallback, and the REST-API anti-detection backend respectively).
    if cat.get("name") == "Browser Automation":
        visible.extend(_plugin_browser_providers())

    # Inject plugin-registered TTS backends (issue #30398). Plugin rows
    # render BELOW the 10 hardcoded built-in rows. Built-in shadowing
    # is filtered out by ``_plugin_tts_providers`` defensively.
    if cat.get("name") == "Text-to-Speech":
        visible.extend(_plugin_tts_providers())

    return visible


def _hidden_nous_gateway_message(
    cat: dict,
    config: dict,
    capability: str,
    *,
    force_fresh: bool = False,
) -> str:
    """Deprecated: Nous Tool Gateway rows are no longer hidden.

    Previously this returned a "log in / upgrade" banner shown above a
    category when its Nous-managed rows were filtered out for unentitled
    users. Those rows are now always listed (see ``_visible_providers``), and
    the login + entitlement guidance happens inline when the user selects one
    (``ensure_nous_portal_access``). Kept as a no-op so call sites stay simple;
    always returns an empty string.
    """
    return ""


_POST_SETUP_INSTALLED: dict = {
    # post_setup_key -> predicate(): True when the install side-effect
    # is already satisfied. Used by `_toolset_needs_configuration_prompt`
    # to force the provider-setup flow when a no-key provider still needs
    # a binary/dependency install (otherwise an already-configured user
    # who toggles the toolset on via `hermes tools` gets a silent no-op
    # because the gate sees "no env vars to ask about" and skips the
    # provider-setup flow that would have run the post_setup hook).
    #
    # Only entries here are gated; other post_setup hooks (kittentts,
    # piper, agent_browser, etc.) keep their existing behaviour. Add an
    # entry when (a) the post_setup is the ONLY install side-effect for
    # a no-key provider, and (b) an installed-state check is local, bounded,
    # and doesn't trigger a heavy import.
    "cua_driver": lambda: _cua_driver_install_ready(),
}


def _post_setup_already_installed(post_setup_key: str) -> bool:
    """Return True when the post_setup install side-effect is satisfied."""
    predicate = _POST_SETUP_INSTALLED.get(post_setup_key)
    if predicate is None:
        # No install-state check registered → assume satisfied (don't
        # change behaviour for hooks we haven't explicitly opted in).
        return True
    try:
        return bool(predicate())
    except Exception:
        return True


def _module_installed(module_name: str) -> bool:
    """Cheap importable-without-importing check (no heavy side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


# Python dependencies installed explicitly through ``hermes tools`` are not
# part of the managed runtime's locked ``all`` sync. A runtime replacement
# therefore needs a small, static allowlist that can be snapshotted before the
# old site-packages disappears and restored afterward. Keep these install
# arguments in sync with the corresponding ``_run_post_setup`` branches.
_RESTORABLE_PYTHON_TOOL_DEPENDENCIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "faster_whisper": ("faster_whisper", ("-U", "faster-whisper")),
    "kittentts": (
        "kittentts",
        (
            "-U",
            "https://github.com/KittenML/KittenTTS/releases/download/"
            "0.8.1/kittentts-0.8.1-py3-none-any.whl",
            "soundfile",
        ),
    ),
    "piper": ("piper", ("-U", "piper-tts")),
    "ddgs": ("ddgs", ("-U", "ddgs")),
    "langfuse": ("langfuse", ("langfuse",)),
}


def active_restorable_python_tool_dependencies() -> list[str]:
    """Return ``hermes tools`` Python dependencies present in this runtime."""
    return [
        name
        for name, (module_name, _install_args) in (
            _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.items()
        )
        if _module_installed(module_name)
    ]


def restorable_python_tool_dependency(
    name: str,
) -> tuple[str, tuple[str, ...]] | None:
    """Return the import probe and pip arguments for an allowlisted tool."""
    return _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.get(name)


def _agent_browser_installed() -> bool:
    """True when everything ``_run_post_setup("agent_browser")`` installs is
    present: the agent-browser CLI *and* the Chromium build it drives (or the
    Lightpanda engine, which needs no Chromium). Mirrors the hook so "Run
    setup" flips to an installed state only when re-running it would be a
    no-op."""
    import sys

    from hermes_cli.nous_subscription import _local_browser_runnable

    # The install hook runs in a spawned ``hermes tools post-setup`` process,
    # but this probe runs in the long-lived web-server/CLI process, whose
    # browser_tool module may have cached a stale "Chromium missing" result
    # from before the install. Drop the cache (when the module is loaded) so
    # the readiness pill flips to Ready right after a successful setup run.
    bt = sys.modules.get("tools.browser_tool")
    if bt is not None:
        bt._cached_chromium_installed = None

    return _local_browser_runnable()


def _camofox_installed() -> bool:
    """True when the Camofox npm package ``_run_post_setup("camofox")``
    installs is already in node_modules."""
    return (PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser").exists()


# post_setup_key -> predicate(): True when the install side-effect is already
# satisfied. Used by ``provider_readiness_status`` to decide whether a keyless
# post_setup row (KittenTTS, Piper, Local Browser, …) is honestly "ready" or
# still "needs_setup". Mirrors the installed-checks ``_run_post_setup`` itself
# performs before installing. ``xai_grok`` is intentionally absent — it is a
# credential bootstrap, not an install, and is handled as an auth check.
_POST_SETUP_READY: dict = {
    "kittentts": lambda: _module_installed("kittentts"),
    "piper": lambda: _module_installed("piper"),
    "faster_whisper": lambda: _module_installed("faster_whisper"),
    "ddgs": lambda: _module_installed("ddgs"),
    "langfuse": lambda: _module_installed("langfuse"),
    "agent_browser": lambda: _agent_browser_installed(),
    "browserbase": lambda: _cloud_agent_browser_installed(),
    "camofox": lambda: _camofox_installed(),
    "cua_driver": lambda: _cua_driver_install_ready(),
}


def _cloud_agent_browser_installed() -> bool:
    """Installed-check for the ``browserbase`` hook (cloud provider rows).

    Cloud providers host their own Chromium, so their hook only installs the
    agent-browser npm package — presence of the CLI is the whole contract."""
    from hermes_cli.nous_subscription import _has_agent_browser

    return _has_agent_browser()


def provider_readiness_status(
    provider: dict,
    config: dict,
    *,
    features=None,
    is_active: Optional[bool] = None,
) -> str:
    """Compute an honest readiness state for a provider picker row.

    Returns one of:

    - ``"ready"``       — usable as-is (keys set / entitled / installed).
    - ``"needs_keys"``  — declares env vars and at least one is unset.
    - ``"needs_auth"``  — needs a sign-in: Nous Portal login/entitlement for
      managed Tool Gateway rows, or xAI Grok OAuth / XAI_API_KEY for
      ``post_setup: "xai_grok"`` rows.
    - ``"needs_setup"`` — keyless row whose ``post_setup`` install hook has
      verifiably not run yet (see ``_POST_SETUP_READY``).

    Keyless ≠ usable: this is the server-side truth the GUI "Ready" pill
    renders from (the old client-side heuristic showed Ready for every
    zero-env-var row, including logged-out Nous Subscription rows).

    ``features`` (a ``NousSubscriptionFeatures``) can be passed to avoid
    re-fetching portal state per row. ``is_active`` is the completed-setup
    fallback signal for post_setup hooks with no registered installed-check
    (selecting a row runs its hook, so the active row has been set up).
    """
    env_vars = provider.get("env_vars", [])
    if env_vars:
        if all(get_env_value(e["key"]) for e in env_vars):
            return "ready"
        return "needs_keys"

    managed_feature = provider.get("managed_nous_feature")
    if provider.get("requires_nous_auth") or managed_feature:
        if features is None:
            features = get_nous_subscription_features(config)
        if not features.nous_auth_present:
            return "needs_auth"
        if managed_feature:
            # Same per-category entitlement gate the CLI applies at selection
            # time (free tool-pool users get image gen but not video gen).
            acct = features.account_info
            category = MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature)
            entitled = bool(
                acct
                and acct.logged_in
                and (
                    acct.tool_gateway_entitled_for(category)
                    if category
                    else acct.tool_gateway_entitled
                )
            )
            if not entitled:
                return "needs_auth"
        # Signed in and entitled — fall through: a managed row may still
        # carry a local install hook (e.g. the managed browser row needs
        # the agent-browser CLI on this machine).

    post_setup = provider.get("post_setup")
    if post_setup:
        if post_setup == "xai_grok":
            return "ready" if _xai_credentials_present() else "needs_auth"
        predicate = _POST_SETUP_READY.get(post_setup)
        if predicate is not None:
            try:
                return "ready" if predicate() else "needs_setup"
            except Exception:
                # Flaky detection must not manufacture a warning state.
                return "ready"
        # No reliable installed-check registered → treat the active-provider
        # signal as "setup completed" (selecting the row runs the hook).
        if is_active is None:
            is_active = _is_provider_active(provider, config)
        return "ready" if is_active else "needs_setup"

    return "ready"


def _toolset_needs_configuration_prompt(
    ts_key: str,
    config: dict,
    *,
    force_fresh: bool = False,
) -> bool:
    """Return True when enabling this toolset should open provider setup."""
    cat = TOOL_CATEGORIES.get(ts_key)
    if not cat:
        return not _toolset_has_keys(ts_key, config, force_fresh=force_fresh)

    # If any visible provider has a registered post_setup install-state
    # check that hasn't been satisfied (e.g. cua-driver binary not on
    # PATH yet), force the configuration flow so `_configure_provider`
    # invokes `_run_post_setup` and the install actually runs.
    for provider in _visible_providers(cat, config, force_fresh=force_fresh):
        post_setup = provider.get("post_setup")
        if post_setup and not _post_setup_already_installed(post_setup):
            return True

    if ts_key == "tts":
        tts_cfg = config.get("tts", {})
        return not isinstance(tts_cfg, dict) or "provider" not in tts_cfg
    if ts_key == "web":
        web_cfg = config.get("web", {})
        return not isinstance(web_cfg, dict) or "backend" not in web_cfg
    if ts_key == "browser":
        browser_cfg = config.get("browser", {})
        return not isinstance(browser_cfg, dict) or "cloud_provider" not in browser_cfg
    if ts_key == "image_gen":
        # Satisfied when the in-tree FAL backend is configured OR any
        # plugin-registered image gen provider is available.
        if fal_key_is_configured():
            return False
        try:
            from agent.image_gen_registry import list_providers
            from hermes_cli.plugins import _ensure_plugins_discovered

            _ensure_plugins_discovered()
            for provider in list_providers():
                try:
                    if provider.is_available():
                        return False
                except Exception:
                    continue
        except Exception:
            pass
        return True
    if ts_key == "video_gen":
        # Satisfied when any plugin-registered video gen provider reports
        # available — no in-tree fallback (every backend is a plugin).
        try:
            from agent.video_gen_registry import list_providers
            from hermes_cli.plugins import _ensure_plugins_discovered

            _ensure_plugins_discovered()
            for provider in list_providers():
                try:
                    if provider.is_available():
                        return False
                except Exception:
                    continue
        except Exception:
            pass
        return True

    return not _toolset_has_keys(ts_key, config, force_fresh=force_fresh)


def _configure_tool_category(
    ts_key: str,
    cat: dict,
    config: dict,
    *,
    force_fresh: bool = True,
):
    """Configure a tool category with provider selection."""
    icon = cat.get("icon", "")
    name = cat["name"]
    providers = _visible_providers(cat, config, force_fresh=force_fresh)
    hidden_nous_message = _hidden_nous_gateway_message(
        cat,
        config,
        f"the Nous Subscription provider for {name}",
        force_fresh=force_fresh,
    )

    # Check Python version requirement
    if cat.get("requires_python"):
        req = cat["requires_python"]
        if sys.version_info < req:
            print()
            _print_error(f"  {name} requires Python {req[0]}.{req[1]}+ (current: {sys.version_info.major}.{sys.version_info.minor})")
            _print_info("  Upgrade Python and reinstall to enable this tool.")
            return

    if len(providers) == 1:
        # Single provider - configure directly
        provider = providers[0]
        print()
        print(color(f"  --- {icon} {name} ({provider['name']}) ---", Colors.CYAN))
        if provider.get("tag"):
            _print_info(f"  {provider['tag']}")
        # For single-provider tools, show a note if available
        if cat.get("setup_note"):
            _print_info(f"  {cat['setup_note']}")
        if hidden_nous_message:
            for line in hidden_nous_message.splitlines():
                _print_warning(f"  {line}")
        _configure_provider(provider, config, force_fresh=force_fresh)
    else:
        # Multiple providers - let user choose
        print()
        # Use custom title if provided (e.g. "Select Search Provider")
        title = cat.get("setup_title", "Choose a provider")
        print(color(f"  --- {icon} {name} - {title} ---", Colors.CYAN))
        if cat.get("setup_note"):
            _print_info(f"  {cat['setup_note']}")
        if hidden_nous_message:
            for line in hidden_nous_message.splitlines():
                _print_warning(f"  {line}")
        print()

        # Plain text labels only (no ANSI codes in menu items)
        # When the user is logged into Nous, surface a marker on providers
        # whose access is included in their subscription so it's visually
        # obvious which options cost extra vs. cost nothing on top of Nous.
        try:
            _nous_logged_in = bool(
                get_nous_subscription_features(
                    config,
                    force_fresh=force_fresh,
                ).nous_auth_present
            )
        except Exception:
            _nous_logged_in = False

        provider_choices = []
        for p in providers:
            badge = f" [{p['badge']}]" if p.get("badge") else ""
            tag = f" — {p['tag']}" if p.get("tag") else ""
            configured = ""
            env_vars = p.get("env_vars", [])
            if not env_vars or all(get_env_value(v["key"]) for v in env_vars):
                if _is_provider_active(p, config, force_fresh=force_fresh):
                    configured = " [active]"
                elif not env_vars:
                    configured = ""
                else:
                    configured = " [configured]"
            # Mark Nous-managed entries. Logged-in paid subscribers get the
            # "included" star; everyone else gets a "via Nous Portal" hint so
            # it's clear selecting the row triggers a Portal login. The rows
            # are always shown now (see _visible_providers) — selecting one
            # drives an inline login + entitlement check.
            sub_marker = ""
            if p.get("managed_nous_feature"):
                if _nous_logged_in:
                    sub_marker = "  ★ Included with your Nous subscription"
                else:
                    sub_marker = "  ★ via Nous Portal (login on select)"
            provider_choices.append(f"{p['name']}{badge}{tag}{configured}{sub_marker}")

        # Add skip option
        provider_choices.append("Skip — keep defaults / configure later")

        # Detect current provider as default
        default_idx = _detect_active_provider_index(
            providers,
            config,
            force_fresh=force_fresh,
        )

        provider_idx = _prompt_choice(f"  {title}:", provider_choices, default_idx)

        # Skip selected
        if provider_idx >= len(providers):
            _print_info(f"  Skipped {name}")
            return

        _configure_provider(providers[provider_idx], config, force_fresh=force_fresh)


def _web_tier_matches(provider: dict, config: dict) -> bool:
    """Return True when a web picker row's tier matches the configured tier.

    Tiered rows (Exa/Parallel Free vs Paid) share one ``web_backend`` name
    and differ only in ``web_tier``. The configured tier lives at
    ``web.provider_tier.<backend>`` (set on selection). Matching rules:

    - row has no ``web_tier`` → tier-agnostic row, matches (legacy rows)
    - configured tier set     → must equal the row's tier
    - configured tier unset   → "auto": the effective tier is paid when the
      row's env vars are all present, free otherwise — highlight the row
      the runtime would actually use
    """
    row_tier = provider.get("web_tier")
    if not row_tier:
        return True
    web_cfg = config.get("web")
    if not isinstance(web_cfg, dict):
        web_cfg = {}
    tiers = web_cfg.get("provider_tier")
    if not isinstance(tiers, dict):
        tiers = {}
    configured = str(tiers.get(provider["web_backend"], "") or "").lower().strip()
    if configured in ("free", "paid"):
        return configured == row_tier
    # Auto: mirror plugins.web.keyless_mcp.use_keyless — key present → paid.
    try:
        from agent.web_search_provider import get_provider_env

        key_var = {"exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY"}.get(
            provider["web_backend"]
        )
        has_key = bool(get_provider_env(key_var)) if key_var else False
    except Exception:
        has_key = False
    return row_tier == ("paid" if has_key else "free")


def _is_provider_active(
    provider: dict,
    config: dict,
    *,
    force_fresh: bool = False,
) -> bool:
    """Check if a provider entry matches the currently active config."""
    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name and not provider.get("managed_nous_feature"):
        # Managed (Nous-subscription) entries fall through to the
        # managed_feature branch below, which also checks use_gateway —
        # otherwise a managed FAL pick and a direct-key FAL pick would both
        # report active for the same provider name (video already guards).
        image_cfg = config.get("image_gen", {})
        if not (isinstance(image_cfg, dict) and image_cfg.get("provider") == plugin_name):
            return False
        # A direct-key entry is only active when the managed route is OFF —
        # mirror of the managed branch's use_gateway check.
        return not is_truthy_value(image_cfg.get("use_gateway"), default=False)

    video_plugin_name = provider.get("video_gen_plugin_name")
    if video_plugin_name and not provider.get("managed_nous_feature"):
        video_cfg = config.get("video_gen", {})
        return isinstance(video_cfg, dict) and video_cfg.get("provider") == video_plugin_name

    managed_feature = provider.get("managed_nous_feature")
    if managed_feature:
        features = get_nous_subscription_features(config, force_fresh=force_fresh)
        feature = features.features.get(managed_feature)
        if feature is None:
            return False
        if managed_feature == "image_gen":
            image_cfg = config.get("image_gen", {})
            if isinstance(image_cfg, dict):
                configured_provider = image_cfg.get("provider")
                if configured_provider not in {None, "", "fal", NOUS_MANAGED_PROVIDER}:
                    return False
                if (
                    configured_provider != NOUS_MANAGED_PROVIDER
                    and image_cfg.get("use_gateway") is not None
                    and not is_truthy_value(image_cfg.get("use_gateway"), default=False)
                ):
                    return False
            return feature.managed_by_nous
        if managed_feature == "video_gen":
            video_cfg = config.get("video_gen", {})
            if isinstance(video_cfg, dict):
                configured_provider = video_cfg.get("provider")
                if configured_provider not in {None, "", "fal", NOUS_MANAGED_PROVIDER}:
                    return False
                if (
                    configured_provider != NOUS_MANAGED_PROVIDER
                    and video_cfg.get("use_gateway") is not None
                    and not is_truthy_value(video_cfg.get("use_gateway"), default=False)
                ):
                    return False
            return feature.managed_by_nous
        if provider.get("tts_provider"):
            return (
                feature.managed_by_nous
                and cfg_get(config, "tts", "provider")
                in {provider["tts_provider"], NOUS_MANAGED_PROVIDER}
            )
        if provider.get("stt_provider"):
            return (
                feature.managed_by_nous
                and cfg_get(config, "stt", "provider")
                in {provider["stt_provider"], NOUS_MANAGED_PROVIDER}
            )
        if "browser_provider" in provider:
            # Browser Use mode is a driver on top of the provider (it attaches
            # to the provider's CDP endpoint), so the provider row stays
            # active alongside the Browser Use row.
            current = cfg_get(config, "browser", "cloud_provider")
            return feature.managed_by_nous and current in {
                provider["browser_provider"],
                NOUS_MANAGED_PROVIDER,
            }
        if provider.get("web_backend"):
            current = cfg_get(config, "web", "backend")
            return (
                feature.managed_by_nous
                and current in {provider["web_backend"], NOUS_MANAGED_PROVIDER}
                and _web_tier_matches(provider, config)
            )
        return feature.managed_by_nous

    if provider.get("tts_provider"):
        return cfg_get(config, "tts", "provider") == provider["tts_provider"]
    if provider.get("stt_provider"):
        # Default stt.provider is "local" — an unset key means Local Whisper.
        current = cfg_get(config, "stt", "provider") or "local"
        return current == provider["stt_provider"]
    if "browser_provider" in provider:
        # Browser Use mode composes with the provider (driver over the
        # provider's CDP endpoint) — don't deactivate the provider row.
        current = cfg_get(config, "browser", "cloud_provider")
        return provider["browser_provider"] == current
    if provider.get("browser_backend"):
        backend = cfg_get(config, "browser", "backend")
        if backend is False:
            backend = "off"  # YAML 1.1: unquoted `off` parses as boolean False
        if backend == provider["browser_backend"]:
            return True
        if backend:
            return False  # explicit other choice ("off", …) wins
        if provider["browser_backend"] != "browser-use":
            return False
        # Backend unset: Browser Use mode is the default — the row is active
        # whenever the effective mode resolves on (legacy direct-API cloud
        # config, or CLI runnable and no Camofox).
        browser_cfg = config.get("browser") if isinstance(config, dict) else None
        try:
            from tools.browser_use_cli import (
                _find_cli,
                is_legacy_browser_use_cloud_config,
            )

            if is_legacy_browser_use_cloud_config(browser_cfg or {}):
                return True
            try:
                from tools.browser_camofox import is_camofox_mode

                if is_camofox_mode():
                    return False
            except Exception:
                pass
            return _find_cli() is not None
        except Exception:
            return False
    if provider.get("web_backend"):
        current = cfg_get(config, "web", "backend")
        if current != provider["web_backend"]:
            return False
        return _web_tier_matches(provider, config)
    if provider.get("computer_use_backend"):
        current = cfg_get(config, "computer_use", "backend")
        return current == provider["computer_use_backend"]
    if provider.get("imagegen_backend"):
        image_cfg = config.get("image_gen", {})
        if not isinstance(image_cfg, dict):
            return False
        configured_provider = image_cfg.get("provider")
        return (
            provider["imagegen_backend"] == "fal"
            and configured_provider in {None, "", "fal"}
            and not is_truthy_value(image_cfg.get("use_gateway"), default=False)
        )
    return False


def _detect_active_provider_index(
    providers: list,
    config: dict,
    *,
    force_fresh: bool = False,
) -> int:
    """Return the index of the currently active provider, or 0."""
    for i, p in enumerate(providers):
        if _is_provider_active(p, config, force_fresh=force_fresh):
            return i
        # Fallback: env vars present → likely configured
        env_vars = p.get("env_vars", [])
        if env_vars and all(get_env_value(v["key"]) for v in env_vars):
            return i
    return 0


# ─── Image Generation Model Pickers ───────────────────────────────────────────
#
# IMAGEGEN_BACKENDS is a per-backend catalog. Each entry exposes:
#   - config_key:        top-level config.yaml key for this backend's settings
#   - model_catalog_fn:  returns an OrderedDict-like {model_id: metadata}
#   - default_model:     fallback when nothing is configured
#
# This prepares for future imagegen backends (Replicate, Stability, etc.):
# each new backend registers its own entry; the FAL provider entry in
# TOOL_CATEGORIES tags itself with `imagegen_backend: "fal"` to select the
# right catalog at picker time.


def _fal_model_catalog():
    """Lazy-load the FAL model catalog from the tool module."""
    from tools.image_generation_tool import FAL_MODELS, DEFAULT_MODEL
    return FAL_MODELS, DEFAULT_MODEL


IMAGEGEN_BACKENDS = {
    "fal": {
        "display": "FAL.ai",
        "config_key": "image_gen",
        "catalog_fn": _fal_model_catalog,
    },
}


def _format_imagegen_model_row(model_id: str, meta: dict, widths: dict) -> str:
    """Format a single picker row with column-aligned speed / strengths / price."""
    return (
        f"{model_id:<{widths['model']}}  "
        f"{meta.get('speed', ''):<{widths['speed']}}  "
        f"{meta.get('strengths', ''):<{widths['strengths']}}  "
        f"{meta.get('price', '')}"
    )


def _configure_imagegen_model(backend_name: str, config: dict) -> None:
    """Prompt the user to pick a model for the given imagegen backend.

    Writes selection to ``config[backend_config_key]["model"]``. Safe to
    call even when stdin is not a TTY — curses_radiolist falls back to
    keeping the current selection.
    """
    backend = IMAGEGEN_BACKENDS.get(backend_name)
    if not backend:
        return

    catalog, default_model = backend["catalog_fn"]()
    if not catalog:
        return

    cfg_key = backend["config_key"]
    cur_cfg = config.setdefault(cfg_key, {})
    if not isinstance(cur_cfg, dict):
        cur_cfg = {}
        config[cfg_key] = cur_cfg
    current_model = cur_cfg.get("model") or default_model
    if current_model not in catalog:
        # The saved model may belong to another provider (shared config key)
        # and the catalog default itself may have drifted — never index the
        # catalog with a key it doesn't contain.
        current_model = default_model if default_model in catalog else next(iter(catalog))

    model_ids = list(catalog.keys())
    # Put current model at the top so the cursor lands on it by default.
    ordered = [current_model] + [m for m in model_ids if m != current_model]

    # Column widths
    widths = {
        "model": max(len(m) for m in model_ids),
        "speed": max((len(catalog[m].get("speed", "")) for m in model_ids), default=6),
        "strengths": max((len(catalog[m].get("strengths", "")) for m in model_ids), default=0),
    }

    print()
    header = (
        f"  {'Model':<{widths['model']}}  "
        f"{'Speed':<{widths['speed']}}  "
        f"{'Strengths':<{widths['strengths']}}  "
        f"Price"
    )
    print(color(header, Colors.CYAN))

    rows = []
    for mid in ordered:
        row = _format_imagegen_model_row(mid, catalog[mid], widths)
        if mid == current_model:
            row += "  ← currently in use"
        rows.append(row)

    idx = _prompt_choice(
        f"  Choose {backend['display']} model:",
        rows,
        default=0,
    )

    chosen = ordered[idx]
    cur_cfg["model"] = chosen
    _print_success(f"  Model set to: {chosen}")


def _plugin_image_gen_catalog(plugin_name: str):
    """Return ``(catalog_dict, default_model_id)`` for a plugin provider.

    ``catalog_dict`` is shaped like the legacy ``FAL_MODELS`` table —
    ``{model_id: {"display", "speed", "strengths", "price", ...}}`` —
    so the existing picker code paths work without change. Returns
    ``({}, None)`` if the provider isn't registered or has no models.
    """
    try:
        from agent.image_gen_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider(plugin_name)
    except Exception:
        return {}, None
    if provider is None:
        return {}, None
    try:
        models = provider.list_models() or []
        default = provider.default_model()
    except Exception:
        return {}, None
    catalog = {m["id"]: m for m in models if isinstance(m, dict) and "id" in m}
    return catalog, default


def _configure_imagegen_model_for_plugin(plugin_name: str, config: dict) -> None:
    """Prompt the user to pick a model for a plugin-registered backend.

    Writes selection to ``image_gen.model``. Mirrors
    :func:`_configure_imagegen_model` but sources its catalog from the
    plugin registry instead of :data:`IMAGEGEN_BACKENDS`.
    """
    catalog, default_model = _plugin_image_gen_catalog(plugin_name)
    if not catalog:
        return

    cur_cfg = config.setdefault("image_gen", {})
    if not isinstance(cur_cfg, dict):
        cur_cfg = {}
        config["image_gen"] = cur_cfg
    current_model = cur_cfg.get("model") or default_model
    if current_model not in catalog:
        current_model = default_model if default_model in catalog else next(iter(catalog))

    model_ids = list(catalog.keys())
    ordered = [current_model] + [m for m in model_ids if m != current_model]

    widths = {
        "model": max(len(m) for m in model_ids),
        "speed": max((len(catalog[m].get("speed", "")) for m in model_ids), default=6),
        "strengths": max((len(catalog[m].get("strengths", "")) for m in model_ids), default=0),
    }

    print()
    header = (
        f"  {'Model':<{widths['model']}}  "
        f"{'Speed':<{widths['speed']}}  "
        f"{'Strengths':<{widths['strengths']}}  "
        f"Price"
    )
    print(color(header, Colors.CYAN))

    rows = []
    for mid in ordered:
        row = _format_imagegen_model_row(mid, catalog[mid], widths)
        if mid == current_model:
            row += "  ← currently in use"
        rows.append(row)

    idx = _prompt_choice(
        f"  Choose {plugin_name} model:",
        rows,
        default=0,
    )

    chosen = ordered[idx]
    cur_cfg["model"] = chosen
    _print_success(f"  Model set to: {chosen}")


def _configure_xai_imagine_storage(section_name: str, config: dict) -> None:
    """Prompt for xAI Imagine stored public URL behavior."""
    section = config.setdefault(section_name, {})
    if not isinstance(section, dict):
        section = {}
        config[section_name] = section
    xai_cfg = section.setdefault("xai", {})
    if not isinstance(xai_cfg, dict):
        xai_cfg = {}
        section["xai"] = xai_cfg
    storage_cfg = xai_cfg.setdefault("storage", {})
    if not isinstance(storage_cfg, dict):
        storage_cfg = {}
        xai_cfg["storage"] = storage_cfg

    _print_warning(
        "  xAI Imagine can store generated media and create reusable public URLs. "
        "xAI may bill for stored files and public URL hosting."
    )
    idx = _prompt_choice(
        "  Stored public URLs:",
        [
            "Enable public URLs without automatic expiry (recommended)",
            "Disable stored public URLs",
            "Enable public URLs for 2 days",
        ],
        default=0,
    )
    if idx == 1:
        storage_cfg["enabled"] = False
        _print_success("  xAI stored public URLs disabled")
    elif idx == 2:
        storage_cfg["enabled"] = True
        storage_cfg["public_url"] = True
        storage_cfg["expires_after"] = 2 * 24 * 60 * 60
        _print_success("  xAI stored public URLs enabled for 2 days")
    else:
        storage_cfg["enabled"] = True
        storage_cfg["public_url"] = True
        storage_cfg["expires_after"] = None
        _print_success("  xAI stored public URLs enabled without automatic expiry")


def _select_plugin_image_gen_provider(plugin_name: str, config: dict, *, use_gateway: bool = False) -> None:
    """Persist a plugin-backed image generation provider selection.

    ``use_gateway=True`` marks a provider picked through the Nous-managed
    flow: the stored selection becomes ``image_gen.provider: nous`` (the
    single provider string the runtime switches on). BYOK picks store the
    plugin name. Any legacy ``use_gateway`` key is removed so old-config
    read-time shims cannot override the fresh selection.
    """
    img_cfg = config.setdefault("image_gen", {})
    if not isinstance(img_cfg, dict):
        img_cfg = {}
        config["image_gen"] = img_cfg
    img_cfg["provider"] = NOUS_MANAGED_PROVIDER if use_gateway else plugin_name
    img_cfg.pop("use_gateway", None)
    _print_success(f"  image_gen.provider set to: {img_cfg['provider']}")
    _configure_imagegen_model_for_plugin(plugin_name, config)
    if plugin_name == "xai":
        _configure_xai_imagine_storage("image_gen", config)


# ─── Video Generation Model Pickers ───────────────────────────────────────────


def _plugin_video_gen_catalog(plugin_name: str):
    """Return ``(catalog_dict, default_model_id)`` for a video gen plugin.

    Mirrors :func:`_plugin_image_gen_catalog`. Returns ``({}, None)`` when
    the plugin isn't registered or has no models.
    """
    try:
        from agent.video_gen_registry import get_provider
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        provider = get_provider(plugin_name)
    except Exception:
        return {}, None
    if provider is None:
        return {}, None
    try:
        models = provider.list_models() or []
        default = provider.default_model()
    except Exception:
        return {}, None
    catalog = {m["id"]: m for m in models if isinstance(m, dict) and "id" in m}
    return catalog, default


def _configure_videogen_model_for_plugin(plugin_name: str, config: dict) -> None:
    """Prompt for a video gen model from a plugin's catalog.

    Mirrors :func:`_configure_imagegen_model_for_plugin`. Writes the
    selection to ``video_gen.model``.
    """
    catalog, default_model = _plugin_video_gen_catalog(plugin_name)
    if not catalog:
        return

    cur_cfg = config.setdefault("video_gen", {})
    if not isinstance(cur_cfg, dict):
        cur_cfg = {}
        config["video_gen"] = cur_cfg
    current_model = cur_cfg.get("model") or default_model
    if current_model not in catalog:
        # Same guard as the image pickers: a stale cross-provider model or a
        # drifted default must not become an unindexable catalog key.
        current_model = default_model if default_model in catalog else next(iter(catalog))

    model_ids = list(catalog.keys())
    ordered = [current_model] + [m for m in model_ids if m != current_model]

    widths = {
        "model": max(len(m) for m in model_ids),
        "speed": max((len(catalog[m].get("speed", "")) for m in model_ids), default=6),
        "strengths": max((len(catalog[m].get("strengths", "")) for m in model_ids), default=0),
    }

    print()
    header = (
        f"  {'Model':<{widths['model']}}  "
        f"{'Speed':<{widths['speed']}}  "
        f"{'Strengths':<{widths['strengths']}}  "
        f"Price"
    )
    print(color(header, Colors.CYAN))

    rows = []
    for mid in ordered:
        meta = catalog[mid]
        row = (
            f"  {mid:<{widths['model']}}  "
            f"{meta.get('speed', ''):<{widths['speed']}}  "
            f"{meta.get('strengths', ''):<{widths['strengths']}}  "
            f"{meta.get('price', '')}"
        )
        if mid == current_model:
            row += "  ← currently in use"
        rows.append(row)

    idx = _prompt_choice(
        f"  Choose {plugin_name} model:",
        rows,
        default=0,
    )

    chosen = ordered[idx]
    cur_cfg["model"] = chosen
    _print_success(f"  Model set to: {chosen}")


# Per-provider STT model catalogs for the interactive picker. Keys are
# ``stt.<provider>`` config sections; the first entry is the default.
# Kept in sync with the dashboard selects (hermes_cli/web_server.py
# _CONFIG_FIELD_META) and the desktop settings enums
# (apps/desktop/src/app/settings/constants.ts).
STT_MODEL_CATALOG = {
    "local": ["base", "tiny", "small", "medium", "large-v3"],
    "groq": ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"],
    "openai": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"],
    "elevenlabs": ["scribe_v2", "scribe_v1"],
}

# ElevenLabs historically uses ``model_id`` instead of ``model``.
_STT_MODEL_CONFIG_KEY = {"elevenlabs": "model_id"}


def _configure_stt_model(stt_provider: str, config: dict) -> None:
    """Prompt for the STT model after a provider pick (when a catalog exists).

    Providers without a static catalog (xai, deepinfra) skip the prompt —
    xAI has a single model and DeepInfra resolves from its live catalog.
    """
    catalog = STT_MODEL_CATALOG.get(stt_provider)
    if not catalog:
        return
    stt_cfg = config.setdefault("stt", {})
    if not isinstance(stt_cfg, dict):
        stt_cfg = {}
        config["stt"] = stt_cfg
    prov_cfg = stt_cfg.setdefault(stt_provider, {})
    if not isinstance(prov_cfg, dict):
        prov_cfg = {}
        stt_cfg[stt_provider] = prov_cfg
    model_key = _STT_MODEL_CONFIG_KEY.get(stt_provider, "model")
    current = str(prov_cfg.get(model_key) or "").strip()
    ordered = list(catalog)
    default_idx = ordered.index(current) if current in ordered else 0
    idx = _prompt_choice("  Select STT model:", ordered, default_idx)
    chosen = ordered[idx]
    prov_cfg[model_key] = chosen
    _print_success(f"  STT model set to: {chosen}")


def _select_plugin_video_gen_provider(plugin_name: str, config: dict, *, use_gateway: bool = False) -> None:
    """Persist a plugin-backed video generation provider selection.

    Mirrors :func:`_select_plugin_image_gen_provider`: managed picks store
    ``video_gen.provider: nous``; BYOK picks store the plugin name; any
    legacy ``use_gateway`` key is removed.
    """
    vid_cfg = config.setdefault("video_gen", {})
    if not isinstance(vid_cfg, dict):
        vid_cfg = {}
        config["video_gen"] = vid_cfg
    vid_cfg["provider"] = NOUS_MANAGED_PROVIDER if use_gateway else plugin_name
    vid_cfg.pop("use_gateway", None)
    _print_success(f"  video_gen.provider set to: {vid_cfg['provider']}")
    _configure_videogen_model_for_plugin(plugin_name, config)
    if plugin_name == "xai":
        _configure_xai_imagine_storage("video_gen", config)


def _write_provider_config(provider: dict, config: dict, *, managed_feature) -> None:
    """Persist the provider/backend config keys for a selected provider.

    This is the pure, non-interactive core of :func:`_configure_provider` —
    it writes ``tts.provider`` / ``browser.cloud_provider`` / ``web.backend``
    based on the provider's markers, but does NOT prompt for env vars, run
    post-setup hooks, gate on Nous auth, or run interactive model pickers.
    Both the CLI configurator and the desktop GUI ``PUT .../provider``
    endpoint call through here so there is one code path.

    Selection model: every row writes exactly ONE provider string per
    category. Managed "Nous Subscription" rows write ``nous``; BYOK rows
    write the vendor name. ``use_gateway`` is no longer written — a fresh
    pick removes any legacy key from the touched section so the read-time
    legacy shim (use_gateway: true ⇒ nous) cannot override the new choice.
    """
    def _set_selection(section_key: str, name_key: str, vendor_value) -> None:
        section = config.setdefault(section_key, {})
        if not isinstance(section, dict):
            section = {}
            config[section_key] = section
        section[name_key] = (
            NOUS_MANAGED_PROVIDER if managed_feature else vendor_value
        )
        section.pop("use_gateway", None)

    # Set TTS provider in config if applicable
    if provider.get("tts_provider"):
        _set_selection("tts", "provider", provider["tts_provider"])

    # Set STT provider in config if applicable
    if provider.get("stt_provider"):
        _set_selection("stt", "provider", provider["stt_provider"])

    # Set browser cloud provider in config if applicable
    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        browser_cfg = config.setdefault("browser", {})
        if bp or managed_feature:
            # Browser Use mode (browser.backend) composes with the provider —
            # switching providers keeps the driver choice intact.
            _set_selection("browser", "cloud_provider", bp)
        else:
            browser_cfg.pop("use_gateway", None)

    if provider.get("browser_backend"):
        browser_cfg = config.setdefault("browser", {})
        browser_cfg["backend"] = provider["browser_backend"]

    # Set web search backend in config if applicable
    if provider.get("web_backend"):
        _set_selection("web", "backend", provider["web_backend"])
        web_cfg = config.get("web")
        if isinstance(web_cfg, dict):
            if provider.get("web_tier"):
                tiers = web_cfg.setdefault("provider_tier", {})
                if isinstance(tiers, dict):
                    tiers[provider["web_backend"]] = provider["web_tier"]
            else:
                stale_tiers = web_cfg.get("provider_tier")
                if isinstance(stale_tiers, dict):
                    stale_tiers.pop(provider["web_backend"], None)

    # Set computer_use backend in config if applicable
    if provider.get("computer_use_backend"):
        cu_cfg = config.setdefault("computer_use", {})
        cu_cfg["backend"] = provider["computer_use_backend"]

    # Managed rows for categories without a marker handled above (e.g. the
    # image_gen/video_gen "Nous Subscription" rows carry only
    # managed_nous_feature) still persist the "nous" selection.
    if managed_feature and managed_feature not in {"web", "tts", "stt", "browser"}:
        section = config.setdefault(managed_feature, {})
        if isinstance(section, dict):
            section["provider"] = NOUS_MANAGED_PROVIDER
            section.pop("use_gateway", None)
    elif not managed_feature:
        # User picked a non-gateway provider — clear any stale legacy
        # use_gateway key on the category so the read-time shim cannot
        # override the fresh selection. Resolve the category from the
        # provider's own markers first (plugin-injected rows are NOT in
        # TOOL_CATEGORIES' hardcoded provider lists and previously skipped
        # this clear), then fall back to the category-membership walk.
        marker_sections = {
            "tts_provider": "tts",
            "stt_provider": "stt",
            "browser_provider": "browser",
            "web_backend": "web",
            "image_gen_plugin_name": "image_gen",
            "imagegen_backend": "image_gen",
            "video_gen_plugin_name": "video_gen",
        }
        cleared = False
        for marker, section_key in marker_sections.items():
            if provider.get(marker) or marker in provider:
                section = config.get(section_key)
                if isinstance(section, dict):
                    section.pop("use_gateway", None)
                cleared = True
        if not cleared:
            for cat_key, cat in TOOL_CATEGORIES.items():
                if provider in cat.get("providers", []):
                    section = config.get(cat_key)
                    if isinstance(section, dict):
                        section.pop("use_gateway", None)
                    break


def apply_provider_selection(ts_key: str, provider_name: str, config: dict) -> None:
    """Non-interactively persist a provider selection for a toolset.

    Resolves ``provider_name`` within ``ts_key``'s category (matching the
    rows the GUI/CLI picker shows via :func:`_visible_providers`) and writes
    the corresponding backend/provider config keys. Unlike
    :func:`_configure_provider`, this does NOT prompt for API keys, run
    post-setup hooks, gate on Nous Portal auth, or run interactive model
    pickers — those are handled separately (env endpoints, post-setup
    endpoints, the model picker) in the desktop GUI.

    Raises ``KeyError`` if the toolset has no category or the provider name
    is not found among the visible providers.
    """
    cat = TOOL_CATEGORIES.get(ts_key)
    if cat is None:
        raise KeyError(f"Toolset has no configurable category: {ts_key}")

    providers = _visible_providers(cat, config, force_fresh=True)
    provider = next((p for p in providers if p.get("name") == provider_name), None)
    if provider is None:
        raise KeyError(f"Unknown provider {provider_name!r} for toolset {ts_key!r}")

    managed_feature = provider.get("managed_nous_feature")
    _write_provider_config(provider, config, managed_feature=managed_feature)

    # Plugin-registered image/video gen backends record the provider name in
    # their own config section. Write that here (without the interactive
    # model picker the CLI runs afterwards — model choice is a separate GUI
    # flow). Managed picks store the "nous" selection.
    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name:
        img_cfg = config.setdefault("image_gen", {})
        if not isinstance(img_cfg, dict):
            img_cfg = {}
            config["image_gen"] = img_cfg
        img_cfg["provider"] = (
            NOUS_MANAGED_PROVIDER if managed_feature else plugin_name
        )
        img_cfg.pop("use_gateway", None)

    video_plugin = provider.get("video_gen_plugin_name")
    if video_plugin:
        vid_cfg = config.setdefault("video_gen", {})
        if not isinstance(vid_cfg, dict):
            vid_cfg = {}
            config["video_gen"] = vid_cfg
        vid_cfg["provider"] = (
            NOUS_MANAGED_PROVIDER if managed_feature else video_plugin
        )
        vid_cfg.pop("use_gateway", None)

    # In-tree FAL imagegen backend (BYOK): always persist the explicit
    # ``image_gen.provider: fal`` selection — historically this row could
    # leave the provider key unset, making a deliberate BYOK pick
    # indistinguishable from a never-configured install.
    if provider.get("imagegen_backend") and not managed_feature:
        img_cfg = config.setdefault("image_gen", {})
        if not isinstance(img_cfg, dict):
            img_cfg = {}
            config["image_gen"] = img_cfg
        img_cfg["provider"] = "fal"
        img_cfg.pop("use_gateway", None)


def _configure_provider(
    provider: dict,
    config: dict,
    *,
    force_fresh: bool = True,
):
    """Configure a single provider - prompt for API keys and set config."""
    env_vars = provider.get("env_vars", [])
    managed_feature = provider.get("managed_nous_feature")

    # Nous-managed Tool Gateway backends are always listed (see
    # _visible_providers), but only *activate* once the user has paid Nous
    # Portal access. Selecting one runs an inline Portal login when needed —
    # auth + entitlement only, no inference-provider switch and no bulk
    # "enable all tools" prompt (that lives in `hermes model`).
    if managed_feature:
        from hermes_cli.nous_subscription import (
            MANAGED_FEATURE_COVERAGE_CATEGORY,
            ensure_nous_portal_access,
        )

        if not ensure_nous_portal_access(
            capability=f"{provider.get('name', 'the Nous Tool Gateway')}",
            coverage_category=MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature),
        ):
            _print_warning(
                "  Not enabled — Nous Portal access is required for this backend."
            )
            return

    # Pure pre-auth UX rows (requires_nous_auth without a managed gateway
    # feature) keep the old gate. Managed rows are handled by the inline
    # login above, so don't double-check them here.
    if provider.get("requires_nous_auth") and not managed_feature:
        features = get_nous_subscription_features(config, force_fresh=force_fresh)
        entitled = bool(
            features.account_info and features.account_info.paid_service_access is True
        )
        if not features.nous_auth_present or not entitled:
            message = format_nous_portal_entitlement_message(
                features.account_info,
                capability=f"{provider.get('name', 'Nous Subscription')}",
            )
            _print_warning(
                f"  {message or 'Nous Subscription is only available after logging into Nous Portal.'}"
            )
            return

    # Set TTS provider in config if applicable
    if provider.get("tts_provider"):
        tts_cfg = config.setdefault("tts", {})
        tts_cfg["provider"] = (
            NOUS_MANAGED_PROVIDER if managed_feature else provider["tts_provider"]
        )
        tts_cfg.pop("use_gateway", None)

    # Set STT provider in config if applicable
    if provider.get("stt_provider"):
        _print_success(f"  STT provider set to: {provider['stt_provider']}")

    # Set browser cloud provider in config if applicable
    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        if bp == "local":
            _print_success("  Browser set to local mode")
        elif bp:
            _print_success(f"  Browser cloud provider set to: {bp}")

    if provider.get("browser_backend"):
        _print_success("  Browser set to Browser Use (browser_exec via CLI 3.0)")

    # Set web search backend in config if applicable
    if provider.get("web_backend"):
        _print_success(f"  Web backend set to: {provider['web_backend']}")

    # Persist the provider/backend config keys + use_gateway flags. Shared
    # with the GUI provider-select endpoint via apply_provider_selection so
    # there is a single source of truth for these writes.
    _write_provider_config(provider, config, managed_feature=managed_feature)

    if not env_vars:
        if provider.get("post_setup"):
            _run_post_setup(provider["post_setup"])
        _print_success(f"  {provider['name']} - no configuration needed!")
        if managed_feature:
            _print_info("  Requests for this tool will be billed to your Nous subscription.")
        # Plugin-registered image_gen provider: write image_gen.provider
        # and route model selection to the plugin's own catalog.
        plugin_name = provider.get("image_gen_plugin_name")
        if plugin_name:
            _select_plugin_image_gen_provider(plugin_name, config, use_gateway=bool(managed_feature))
            return
        # Plugin-registered video_gen provider — same flow, different
        # registry.
        video_plugin = provider.get("video_gen_plugin_name")
        if video_plugin:
            _select_plugin_video_gen_provider(video_plugin, config, use_gateway=bool(managed_feature))
            return
        # Imagegen backends prompt for model selection after backend pick.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            # In-tree FAL is the only non-plugin backend today. Persist the
            # explicit selection: "nous" for a managed row, "fal" for BYOK.
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict):
                img_cfg["provider"] = (
                    NOUS_MANAGED_PROVIDER if managed_feature else "fal"
                )
                img_cfg.pop("use_gateway", None)
        # STT providers prompt for model selection after backend pick
        # (skipped for managed rows — the gateway pins the model).
        if provider.get("stt_provider") and not managed_feature:
            _configure_stt_model(provider["stt_provider"], config)
        return

    # Prompt for each required env var
    all_configured = True
    # If this BYOK provider lives in a category that ALSO has a
    # Nous-managed sibling, show a single dim hint so users know
    # they can avoid the key entirely via a Portal subscription.
    # Suppressed when the user is already authed to Nous.
    _show_portal_hint = False
    if env_vars and not managed_feature and not provider.get("requires_nous_auth"):
        try:
            _has_managed_sibling = False
            for _cat_key, _cat in TOOL_CATEGORIES.items():
                _providers = _cat.get("providers", [])
                if provider in _providers and any(
                    sib.get("managed_nous_feature") for sib in _providers
                ):
                    _has_managed_sibling = True
                    break
            if _has_managed_sibling:
                _features = get_nous_subscription_features(
                    config,
                    force_fresh=force_fresh,
                )
                _show_portal_hint = not _features.nous_auth_present
        except Exception:
            _show_portal_hint = False

    if _show_portal_hint:
        _print_info("  Available through Nous Portal subscription.")

    for var in env_vars:
        existing = get_env_value(var["key"])
        if existing:
            _print_success(f"  {var['key']}: already configured")
            # Don't ask to update - this is a new enable flow.
            # Reconfigure is handled separately.
        else:
            url = var.get("url", "")
            if url:
                _print_info(f"  Get yours at: {url}")

            default_val = var.get("default", "")
            if default_val:
                value = _prompt(f"    {var.get('prompt', var['key'])}", default_val)
            else:
                value = _prompt(f"    {var.get('prompt', var['key'])}", password=True)

            if value:
                save_env_value(var["key"], value)
                _print_success("    Saved")
            else:
                _print_warning("    Skipped")
                all_configured = False

    # Run post-setup hooks if needed
    if provider.get("post_setup") and all_configured:
        _run_post_setup(provider["post_setup"])

    if all_configured:
        _print_success(f"  {provider['name']} configured!")
        plugin_name = provider.get("image_gen_plugin_name")
        if plugin_name:
            _select_plugin_image_gen_provider(plugin_name, config, use_gateway=bool(managed_feature))
            return
        video_plugin = provider.get("video_gen_plugin_name")
        if video_plugin:
            _select_plugin_video_gen_provider(video_plugin, config, use_gateway=bool(managed_feature))
            return
        # Imagegen backends prompt for model selection after env vars are in.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict):
                img_cfg["provider"] = (
                    NOUS_MANAGED_PROVIDER if managed_feature else "fal"
                )
                img_cfg.pop("use_gateway", None)
        # STT providers prompt for model selection after env vars are in.
        if provider.get("stt_provider") and not managed_feature:
            _configure_stt_model(provider["stt_provider"], config)


def _configure_vision_backend() -> None:
    """Interactive vision-backend configuration.

    Vision is an auxiliary task whose provider/model are resolved from
    ``auxiliary.vision.{provider,model,base_url}`` in config.yaml (see
    ``agent/auxiliary_client.resolve_vision_provider_client``). Rather than
    forcing the user onto OpenRouter, let them pick any authenticated
    provider + model — the same surface as ``hermes model`` — or point at a
    custom OpenAI-compatible endpoint. "Auto" leaves the config keys empty so
    the resolver uses the main model / aggregator fallback chain.
    """
    print()
    print(color("  Vision / Image Analysis needs a multimodal model.", Colors.YELLOW))
    print(color(
        "  Pick any provider + model (like /model), or let it auto-detect.",
        Colors.DIM,
    ))

    choices = [
        "Auto — use your main model / aggregator fallback (recommended)",
        "Pick a provider and model",
        "Custom OpenAI-compatible endpoint — base URL, API key, model",
        "Skip",
    ]
    idx = _prompt_choice("  Configure vision backend", choices, 0)

    config = load_config()
    aux = config.setdefault("auxiliary", {})
    if not isinstance(aux, dict):
        aux = {}
        config["auxiliary"] = aux
    vision_cfg = aux.setdefault("vision", {})
    if not isinstance(vision_cfg, dict):
        vision_cfg = {}
        aux["vision"] = vision_cfg

    if idx == 0:
        # Auto: clear any pinned override so the resolver auto-detects.
        for key in ("provider", "model", "base_url", "api_key", "api_mode"):
            vision_cfg.pop(key, None)
        save_config(config)
        _print_success("  Vision set to auto (main model / aggregator fallback)")
        return

    if idx == 1:
        _configure_vision_provider_model(config, vision_cfg)
        return

    if idx == 2:
        base_url = _prompt("    Base URL (blank for OpenAI)").strip() or "https://api.openai.com/v1"
        is_native_openai = base_url_hostname(base_url) == "api.openai.com"
        key_label = "    OPENAI_API_KEY" if is_native_openai else "    API key"
        api_key = _prompt(key_label, password=True)
        if not (api_key and api_key.strip()):
            _print_warning("    Skipped")
            return
        default_model = "gpt-4o-mini" if is_native_openai else ""
        model = _prompt(
            f"    Vision model{f' (blank for {default_model})' if default_model else ''}"
        ).strip() or default_model
        save_env_value("OPENAI_API_KEY", api_key.strip())
        # Only base_url + model go to config.yaml; the key is the secret.
        # Pin provider="custom" so the resolver routes through this endpoint —
        # leaving it at the "auto" default would make _resolve_task_provider_model
        # ignore the base_url (it only honors base_url when paired with an
        # api_key in config or a non-auto provider).
        vision_cfg["provider"] = "custom"
        vision_cfg["base_url"] = base_url
        if model:
            vision_cfg["model"] = model
        else:
            vision_cfg.pop("model", None)
        save_config(config)
        _print_success(f"  Vision set to custom endpoint{f' ({model})' if model else ''}")
        return

    # Skip
    _print_info("  Skipped vision configuration")


def _configure_vision_provider_model(config: dict, vision_cfg: dict) -> None:
    """Provider + model picker for vision, mirroring the ``/model`` surface.

    Provider rows come from ``build_aux_picker_rows()`` — the shared aux-picker
    substrate — so this picker lists exactly what the ``hermes model`` aux-task
    picker lists, including the user's own ``providers:`` / ``custom_providers:``
    endpoints. Lets the user pick a provider and then a model from its curated
    list (or type a custom id), and persists ``auxiliary.vision.provider`` +
    ``.model``.
    """
    try:
        from hermes_cli.inventory import (
            build_aux_picker_rows,
            format_aux_picker_entries,
        )
    except Exception as exc:  # pragma: no cover - import guard
        _print_warning(f"  Could not load provider list: {exc}")
        return

    current_provider = str(vision_cfg.get("provider") or "").strip()
    current_model = str(vision_cfg.get("model") or "").strip()
    current_base_url = str(vision_cfg.get("base_url") or "").strip()

    try:
        providers = build_aux_picker_rows(
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            max_models=40,
        )
    except Exception as exc:
        _print_warning(f"  Could not detect providers: {exc}")
        providers = []

    if not providers:
        _print_warning(
            "  No authenticated providers found. Configure a provider first "
            "with `hermes model`, then re-run this."
        )
        return

    provider_labels = [
        label
        for _slug, label, _models in format_aux_picker_entries(
            providers,
            current_provider=current_provider,
            current_base_url=current_base_url,
        )
    ]
    provider_labels.append("Cancel")

    pidx = _prompt_choice("  Choose vision provider:", provider_labels, 0)
    if pidx >= len(providers):
        _print_info("  Cancelled")
        return

    chosen = providers[pidx]
    slug = chosen.get("slug")
    models = list(chosen.get("models", []))

    model_choices = list(models) + ["Type a custom model id…"]
    midx = _prompt_choice(
        f"  Choose vision model for {chosen.get('name') or slug}:",
        model_choices,
        0,
    )
    if midx < len(models):
        model = models[midx]
    else:
        model = _prompt("    Model id").strip()
        if not model:
            _print_warning("  No model entered — cancelled")
            return

    vision_cfg["provider"] = slug
    vision_cfg["model"] = model
    # A provider selection supersedes any prior custom endpoint override.
    vision_cfg.pop("base_url", None)
    vision_cfg.pop("api_key", None)
    save_config(config)
    _print_success(f"  Vision set to {slug} / {model}")


def _configure_simple_requirements(ts_key: str):
    """Simple fallback for toolsets that just need env vars (no provider selection)."""
    if ts_key == "vision":
        if _toolset_has_keys("vision"):
            return
        _configure_vision_backend()
        return

    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return

    missing = [(var, url) for var, url in requirements if not get_env_value(var)]
    if not missing:
        return

    ts_label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts_key), ts_key)
    print()
    print(color(f"  {ts_label} requires configuration:", Colors.YELLOW))

    for var, url in missing:
        if url:
            _print_info(f"  Get key at: {url}")
        value = _prompt(f"    {var}", password=True)
        if value and value.strip():
            save_env_value(var, value.strip())
            _print_success("    Saved")
        else:
            _print_warning("    Skipped")


def _reconfigure_tool(
    config: dict,
    *,
    force_fresh: bool = True,
):
    """Let user reconfigure an existing tool's provider or API key."""
    configurable = [
        (ts_key, ts_label)
        for ts_key, ts_label, _ in _get_effective_configurable_toolsets()
        if _is_configurable(ts_key) and (
            _toolset_has_keys(ts_key, config, force_fresh=force_fresh)
            or _toolset_enabled_for_reconfigure(ts_key, config))]
    if not configurable:
        _print_info("No configured tools to reconfigure.")
        return
    choices = [label for _, label in configurable] + ["Cancel"]
    idx = _prompt_choice("  Which tool would you like to reconfigure?", choices, len(choices) - 1)
    if idx >= len(configurable):
        return
    _configure_toolset(configurable[idx][0], config, force_fresh=force_fresh, reconfigure=True)
    save_config(config)


def _toolset_enabled_for_reconfigure(ts_key: str, config: dict) -> bool:
    """True if the toolset is enabled on any platform, so reconfigure covers enabled-but-unconfigured ones."""
    for platform in filter(lambda p: _toolset_allowed_for_platform(ts_key, p), PLATFORMS):
        try:
            if ts_key in _current_platform_tools(config, platform):
                return True
        except Exception:
            continue
    return False


# --- Main Entry Point ---
def _shared_metrics_state(config: dict) -> tuple[bool, bool]:
    """Return (collection_enabled, send_enabled) from a config dict."""
    telemetry = config.get("telemetry")
    shared = telemetry.get("shared_metrics") if isinstance(telemetry, dict) else None
    shared = shared if isinstance(shared, dict) else {}
    return shared.get("enabled") is True, shared.get("send") is True


def _reconfigure_provider(
    provider: dict,
    config: dict,
    *,
    force_fresh: bool = True,
):
    """Reconfigure a provider - update API keys."""
    env_vars = provider.get("env_vars", [])
    managed_feature = provider.get("managed_nous_feature")

    # Same inline Nous Portal login + entitlement gate as _configure_provider:
    # managed Tool Gateway backends only activate with paid Portal access.
    if managed_feature:
        from hermes_cli.nous_subscription import (
            MANAGED_FEATURE_COVERAGE_CATEGORY,
            ensure_nous_portal_access,
        )

        if not ensure_nous_portal_access(
            capability=f"{provider.get('name', 'the Nous Tool Gateway')}",
            coverage_category=MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature),
        ):
            _print_warning(
                "  Not enabled — Nous Portal access is required for this backend."
            )
            return

    # Pure pre-auth UX rows keep the old gate; managed rows already handled
    # by the inline login above.
    if provider.get("requires_nous_auth") and not managed_feature:
        features = get_nous_subscription_features(config, force_fresh=force_fresh)
        entitled = bool(
            features.account_info and features.account_info.paid_service_access is True
        )
        if not features.nous_auth_present or not entitled:
            message = format_nous_portal_entitlement_message(
                features.account_info,
                capability=f"{provider.get('name', 'Nous Subscription')}",
            )
            _print_warning(
                f"  {message or 'Nous Subscription is only available after logging into Nous Portal.'}"
            )
            return

    # Selection model (mirrors _write_provider_config): every row writes ONE
    # provider string per category — "nous" for managed rows, the vendor name
    # for BYOK rows — and drops any legacy use_gateway key so the read-time
    # shim (use_gateway: true ⇒ nous) cannot override the fresh pick.
    if provider.get("tts_provider"):
        tts_cfg = config.setdefault("tts", {})
        tts_cfg["provider"] = (
            NOUS_MANAGED_PROVIDER if managed_feature else provider["tts_provider"]
        )
        tts_cfg.pop("use_gateway", None)
        _print_success(f"  TTS provider set to: {provider['tts_provider']}")

    if provider.get("stt_provider"):
        stt_cfg = config.setdefault("stt", {})
        stt_cfg["provider"] = (
            NOUS_MANAGED_PROVIDER if managed_feature else provider["stt_provider"]
        )
        stt_cfg.pop("use_gateway", None)
        _print_success(f"  STT provider set to: {provider['stt_provider']}")

    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        browser_cfg = config.setdefault("browser", {})
        if managed_feature:
            browser_cfg["cloud_provider"] = NOUS_MANAGED_PROVIDER
            _print_success(f"  Browser cloud provider set to: {bp or 'nous'}")
        elif bp == "local":
            browser_cfg["cloud_provider"] = "local"
            _print_success("  Browser set to local mode")
        elif bp:
            browser_cfg["cloud_provider"] = bp
            _print_success(f"  Browser cloud provider set to: {bp}")
        # Browser Use mode (browser.backend) composes with the provider —
        # switching providers keeps the driver choice intact.
        browser_cfg.pop("use_gateway", None)

    if provider.get("browser_backend"):
        browser_cfg = config.setdefault("browser", {})
        browser_cfg["backend"] = provider["browser_backend"]
        _print_success("  Browser set to Browser Use (browser_exec via CLI 3.0)")

    # Set web search backend in config if applicable
    if provider.get("web_backend"):
        web_cfg = config.setdefault("web", {})
        web_cfg["backend"] = (
            NOUS_MANAGED_PROVIDER if managed_feature else provider["web_backend"]
        )
        web_cfg.pop("use_gateway", None)
        if provider.get("web_tier"):
            tiers = web_cfg.setdefault("provider_tier", {})
            if isinstance(tiers, dict):
                tiers[provider["web_backend"]] = provider["web_tier"]
            _print_success(
                f"  Web backend set to: {provider['web_backend']} "
                f"({provider['web_tier']} tier)"
            )
        else:
            stale_tiers = web_cfg.get("provider_tier")
            if isinstance(stale_tiers, dict):
                stale_tiers.pop(provider["web_backend"], None)
            _print_success(f"  Web backend set to: {provider['web_backend']}")

    # Set computer_use backend in config if applicable
    if provider.get("computer_use_backend"):
        cu_cfg = config.setdefault("computer_use", {})
        cu_cfg["backend"] = provider["computer_use_backend"]
        _print_success(f"  Computer Use backend set to: {provider['computer_use_backend']}")

    if managed_feature and managed_feature not in {"web", "tts", "stt", "browser"}:
        section = config.setdefault(managed_feature, {})
        if not isinstance(section, dict):
            section = {}
            config[managed_feature] = section
        section["provider"] = NOUS_MANAGED_PROVIDER
        section.pop("use_gateway", None)
    elif not managed_feature:
        for cat_key, cat in TOOL_CATEGORIES.items():
            if provider in cat.get("providers", []):
                section = config.get(cat_key)
                if isinstance(section, dict):
                    section.pop("use_gateway", None)
                break

    if not env_vars:
        if provider.get("post_setup"):
            _run_post_setup(provider["post_setup"])
        _print_success(f"  {provider['name']} - no configuration needed!")
        if managed_feature:
            _print_info("  Requests for this tool will be billed to your Nous subscription.")
        plugin_name = provider.get("image_gen_plugin_name")
        if plugin_name:
            _select_plugin_image_gen_provider(plugin_name, config, use_gateway=bool(managed_feature))
            return
        # Plugin-registered video_gen provider — same flow, different registry.
        video_plugin = provider.get("video_gen_plugin_name")
        if video_plugin:
            _select_plugin_video_gen_provider(video_plugin, config, use_gateway=bool(managed_feature))
            return
        # Imagegen backends prompt for model selection on reconfig too.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            if backend == "fal":
                img_cfg = config.setdefault("image_gen", {})
                if isinstance(img_cfg, dict):
                    # A managed (Nous Subscription) row also carries
                    # imagegen_backend="fal" — store the "nous" selection
                    # for it, "fal" for BYOK, and drop any legacy
                    # use_gateway key.
                    img_cfg["provider"] = (
                        NOUS_MANAGED_PROVIDER if managed_feature else "fal"
                    )
                    img_cfg.pop("use_gateway", None)
        # STT providers prompt for model selection on reconfig too.
        if provider.get("stt_provider") and not managed_feature:
            _configure_stt_model(provider["stt_provider"], config)
        return

    for var in env_vars:
        existing = get_env_value(var["key"])
        if existing:
            _print_info(f"  {var['key']}: configured ({existing[:8]}...)")
        url = var.get("url", "")
        if url:
            _print_info(f"  Get yours at: {url}")
        default_val = var.get("default", "")
        value = _prompt(f"    {var.get('prompt', var['key'])} (Enter to keep current)", password=not default_val)
        if value and value.strip():
            save_env_value(var["key"], value.strip())
            _print_success("    Updated")
        else:
            _print_info("    Kept current")

    if provider.get("post_setup"):
        _run_post_setup(provider["post_setup"])

    # Imagegen backends prompt for model selection on reconfig too.
    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name:
        _select_plugin_image_gen_provider(plugin_name, config, use_gateway=bool(managed_feature))
        return

    # Plugin-registered video_gen provider — same flow, different registry.
    video_plugin = provider.get("video_gen_plugin_name")
    if video_plugin:
        _select_plugin_video_gen_provider(video_plugin, config, use_gateway=bool(managed_feature))
        return

    backend = provider.get("imagegen_backend")
    if backend:
        _configure_imagegen_model(backend, config)
        if backend == "fal":
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict):
                # Same managed-row guard as the no-env-vars branch above:
                # never clobber a Nous-managed pick back onto direct keys.
                img_cfg["provider"] = (
                    NOUS_MANAGED_PROVIDER if managed_feature else "fal"
                )
                img_cfg.pop("use_gateway", None)

    # STT providers prompt for model selection on reconfig too.
    if provider.get("stt_provider") and not managed_feature:
        _configure_stt_model(provider["stt_provider"], config)


def _configure_shared_metrics_interactive(config: dict) -> None:
    """Toggle shared-metrics collection/sending via the setup wizard prompt (single home for the consent rules)."""
    from hermes_cli.setup import setup_telemetry

    before = _shared_metrics_state(config)
    setup_telemetry(config)
    if _shared_metrics_state(config) != before:
        save_config(config)


def _print_toolset_diff(added: Set[str], removed: Set[str], *, indent: str = "  ") -> None:
    """Print ``+ label`` / ``- label`` lines for a checklist change."""
    for ts in sorted(added):
        print(color(f"{indent}+ {_toolset_label(ts)}", Colors.GREEN))
    for ts in sorted(removed):
        print(color(f"{indent}- {_toolset_label(ts)}", Colors.RED))


def _toolsets_needing_setup(new_enabled: Set[str], config: dict) -> List[str]:
    """Selected toolsets still missing provider/API-key setup, sorted (opened even when the selection is unchanged)."""
    return [
        ts_key for ts_key in sorted(new_enabled)
        if _is_configurable(ts_key) and _toolset_needs_configuration_prompt(ts_key, config, force_fresh=True)
    ]


def _configure_newly_added(added: Set[str], already: Set[str], config: dict) -> None:
    """Configure newly enabled toolsets that need keys, skipping those already handled."""
    for ts_key in _toolsets_needing_setup(added - already, config):
        _configure_toolset(ts_key, config)


def _platform_menu_label(config: dict, pkey: str) -> str:
    count = len(_current_platform_tools(config, pkey))
    total = len(_get_effective_configurable_toolsets())
    return f"Configure {PLATFORMS[pkey]['label']}  ({count}/{total} enabled)"


def _print_tools_summary(config: dict, enabled_platforms: List[str]) -> None:
    """``hermes tools --summary``: enabled toolsets per platform, non-interactive."""
    total = len(_get_effective_configurable_toolsets())
    print(color("☤ Tool Summary", Colors.CYAN, Colors.BOLD))
    print()
    for pkey, enabled in _platform_toolset_summary(config, enabled_platforms).items():
        print(color(f"  {PLATFORMS[pkey]['label']}", Colors.BOLD) + color(f"  ({len(enabled)}/{total})", Colors.DIM))
        for ts_key in sorted(enabled):
            print(color(f"    ✓ {_toolset_label(ts_key)}", Colors.GREEN))
        if not enabled:
            print(color("    (none enabled)", Colors.DIM))
    print()


def _configure_list(to_configure: List[str], config: dict, *, selected: bool = True) -> None:
    """Announce then configure each toolset in ``to_configure``."""
    if not to_configure:
        return
    print()
    what = "selected tool(s)" if selected else "tool(s)"
    print(color(f"  Configuring {len(to_configure)} {what}:", Colors.YELLOW))
    for ts_key in to_configure:
        print(color(f"    • {_toolset_label(ts_key)}", Colors.DIM))
    print(color("  You can skip any tool you don't need right now.", Colors.DIM))
    print()
    for ts_key in to_configure:
        _configure_toolset(ts_key, config)


def _checklist_diff(new_enabled: Set[str], prev: Set[str], platform: str) -> tuple[Set[str], Set[str]]:
    """``(added, removed)`` scoped to the checklist universe, so read-time toolsets (MCP names) the user never
    saw a checkbox for don't print as spurious removals."""
    universe = _checklist_toolset_keys(platform)
    return (new_enabled - prev) & universe, (prev - new_enabled) & universe


def _first_install_flow(config: dict, enabled_platforms: List[str]) -> None:
    """Fresh install: one checklist per platform, no menu, keys prompted for every enabled tool."""
    for pkey in enabled_platforms:
        pinfo = PLATFORMS[pkey]
        current_enabled = _current_platform_tools(config, pkey)
        new_enabled = _prompt_toolset_checklist(pinfo["label"], current_enabled - _DEFAULT_OFF_TOOLSETS, pkey)
        _print_toolset_diff(*_checklist_diff(new_enabled, current_enabled, pkey))
        auto_configured = apply_nous_managed_defaults(config, enabled_toolsets=new_enabled, force_fresh=True)
        for ts_key in sorted(auto_configured):
            label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts_key), ts_key)
            print(color(f"  ✓ {label}: using your Nous subscription defaults", Colors.GREEN))
        # Walk through ALL selected tools with provider options or key requirements, so browser (Local vs
        # Browserbase), TTS (Edge vs OpenAI vs ElevenLabs), etc. are shown even when a free provider exists.
        _configure_list(
            [ts for ts in sorted(new_enabled) if _is_configurable(ts) and ts not in auto_configured],
            config, selected=False)
        _save_platform_tools(config, pkey, new_enabled)
        save_config(config)
        print(color(f"  ✓ Saved {pinfo['label']} tool configuration", Colors.GREEN))
        print()


def _current_platform_tools(config: dict, pkey: str) -> Set[str]:
    return _get_platform_tools(config, pkey, include_default_mcp_servers=False)


def _apply_platform_checklist(config: dict, pkey: str, new_enabled: Set[str], prev: Set[str], already: Set[str],
                              *, indent: str = "  ", header: bool = False) -> None:
    """Print the diff, configure newly added toolsets not in ``already``, and write the platform list.
    Keys for newly enabled tools not already handled by the selected-tool pass, so a tool enabled globally
    but lacking provider config doesn't drop the user back to the main menu."""
    added, removed = _checklist_diff(new_enabled, prev, pkey)
    if header and (added or removed):
        print(color(f"  {PLATFORMS[pkey]['label']}:", Colors.DIM))
    _print_toolset_diff(added, removed, indent=indent)
    _configure_newly_added(added, already, config)
    _save_platform_tools(config, pkey, new_enabled)


def _configure_platforms(config: dict, platform_keys: List[str], *, all_platforms: bool = False) -> bool:
    """Checklist + key setup + save for one platform, or for every platform at once (the 'Configure all
    platforms (global)' menu entry). Returns True when config was saved."""
    label = "All platforms" if all_platforms else PLATFORMS[platform_keys[0]]["label"]
    current = {pk: _current_platform_tools(config, pk) for pk in platform_keys}
    all_current = set().union(*current.values())
    new_enabled = _prompt_toolset_checklist(label, all_current, force_fresh=True)
    selected_to_configure = _toolsets_needing_setup(new_enabled, config)
    _configure_list(selected_to_configure, config)
    if new_enabled == all_current and not selected_to_configure:
        print(color("  No changes" if all_platforms else f"  No changes to {label}", Colors.DIM))
        return False
    for pk in platform_keys:
        # Global: re-read after each save — reconciling agent.disabled_toolsets for one platform can change
        # what the next platform resolves to. Single platform: diff against the pre-checklist snapshot.
        prev = _current_platform_tools(config, pk) if all_platforms else current[pk]
        _apply_platform_checklist(config, pk, new_enabled, prev, set(selected_to_configure),
                                  indent="    " if all_platforms else "  ", header=all_platforms)
    save_config(config)
    print(color("  ✓ Saved configuration for all platforms" if all_platforms else f"  ✓ Saved {label} configuration",
                Colors.GREEN))
    return True


def tools_command(args=None, first_install: bool = False, config: dict = None):
    """Entry point for `hermes tools` / `hermes setup tools`. ``first_install`` skips the menu (checklist + key
    prompts); a wizard-passed ``config`` receives platform_toolsets so its final save_config() keeps them."""
    if config is None:
        config = load_config()
    enabled_platforms = _get_enabled_platforms()

    print()
    if getattr(args, "summary", False):
        _print_tools_summary(config, enabled_platforms)
        return
    print(color("☤ Hermes Tool Configuration", Colors.CYAN, Colors.BOLD))
    print(color("  Enable or disable tools per platform.", Colors.DIM))
    print(color("  Tools that need API keys will be configured when enabled.", Colors.DIM))
    print(color("  Guide: https://hermes-agent.nousresearch.com/docs/user-guide/features/tools", Colors.DIM))
    print()
    if first_install:
        _first_install_flow(config, enabled_platforms)
        return

    # Returning user: platform menu loop. Per-platform rows first, then the extras in this order.
    platform_keys = list(enabled_platforms)
    platform_choices = [_platform_menu_label(config, pkey) for pkey in platform_keys]

    def _add_row(label: str, present: bool = True) -> int:
        if not present:
            return -1
        platform_choices.append(label)
        return len(platform_choices) - 1

    global_idx = _add_row("Configure all platforms (global)", len(platform_keys) > 1)
    reconfig_idx = _add_row("Reconfigure an existing tool's provider or API key")
    metrics_idx = _add_row(_shared_metrics_menu_label(config))
    mcp_idx = _add_row("Configure MCP server tools", bool(config.get("mcp_servers")))
    done_idx = _add_row("Done")

    while True:
        idx = _prompt_choice("Select an option:", platform_choices, default=0)
        if idx == done_idx:
            break
        if idx == reconfig_idx:
            _reconfigure_tool(config, force_fresh=True)
        elif idx == metrics_idx:
            _configure_shared_metrics_interactive(config)
            platform_choices[metrics_idx] = _shared_metrics_menu_label(config)
        elif idx == mcp_idx:
            _configure_mcp_tools_interactive(config)
        elif idx == global_idx:
            if _configure_platforms(config, platform_keys, all_platforms=True):
                for ci, pk in enumerate(platform_keys):
                    platform_choices[ci] = _platform_menu_label(config, pk)
        else:
            _configure_platforms(config, [platform_keys[idx]])
            platform_choices[idx] = _platform_menu_label(config, platform_keys[idx])
        print()

    print()
    from hermes_constants import display_hermes_home
    print(color(f"  Tool configuration saved to {display_hermes_home()}/config.yaml", Colors.DIM))
    print(color("  Changes take effect on next 'hermes' or gateway restart.", Colors.DIM))
    print()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import shutil  # noqa: F401,E402
import subprocess  # noqa: F401,E402
import sys  # noqa: F401,E402


def _configure_mcp_tools_interactive(config: dict):
    """Probe MCP servers for available tools and let user toggle them on/off.

    Connects to each configured MCP server, discovers tools, then shows
    a per-server curses checklist.  Writes changes back as ``tools.exclude``
    entries in config.yaml.
    """
    from hermes_cli.curses_ui import curses_checklist

    mcp_servers = config.get("mcp_servers") or {}
    if not mcp_servers:
        _print_info("No MCP servers configured.")
        return

    # Count enabled servers
    enabled_names = [
        k for k, v in mcp_servers.items()
        if v.get("enabled", True) not in {False, "false", "0", "no", "off"}
    ]
    if not enabled_names:
        _print_info("All MCP servers are disabled.")
        return

    print()
    print(color("  Discovering tools from MCP servers...", Colors.YELLOW))
    print(color(f"  Connecting to {len(enabled_names)} server(s): {', '.join(enabled_names)}", Colors.DIM))

    try:
        from tools.mcp_tool import probe_mcp_server_tools
        server_tools = probe_mcp_server_tools()
    except Exception as exc:
        _print_error(f"Failed to probe MCP servers: {exc}")
        return

    if not server_tools:
        _print_warning("Could not discover tools from any MCP server.")
        _print_info("Check that server commands/URLs are correct and dependencies are installed.")
        return

    # Report discovery results
    failed = [n for n in enabled_names if n not in server_tools]
    if failed:
        for name in failed:
            _print_warning(f"  Could not connect to '{name}'")

    total_tools = sum(len(tools) for tools in server_tools.values())
    print(color(f"  Found {total_tools} tool(s) across {len(server_tools)} server(s)", Colors.GREEN))
    print()

    any_changes = False

    for server_name, tools in server_tools.items():
        if not tools:
            _print_info(f"  {server_name}: no tools found")
            continue

        srv_cfg = mcp_servers.get(server_name, {})
        tools_cfg = srv_cfg.get("tools") or {}
        include_list = tools_cfg.get("include") or []
        exclude_list = tools_cfg.get("exclude") or []

        # Build checklist labels
        labels = []
        for tool_name, description in tools:
            desc_short = description[:70] + "..." if len(description) > 70 else description
            if desc_short:
                labels.append(f"{tool_name}  ({desc_short})")
            else:
                labels.append(tool_name)

        # Determine which tools are currently enabled. Use the SAME matching
        # semantics as runtime registration (tools/mcp_tool.py): exact names
        # or fnmatch globs — a literal `in` check renders glob excludes
        # (e.g. "*team_member*" from catalog default_excluded manifests) as
        # if nothing were excluded.
        try:
            from tools.mcp_tool import matches_name_filter as _match_filter
        except ImportError:  # pragma: no cover — defensive fallback
            def _match_filter(tool_name, patterns):
                return tool_name in patterns

        pre_selected: Set[int] = set()
        tool_names = [t[0] for t in tools]
        include_set = {str(p) for p in include_list} if include_list else None
        exclude_set = {str(p) for p in exclude_list} if exclude_list else None
        for i, tool_name in enumerate(tool_names):
            if include_set:
                # Include mode: only included tools are selected
                if _match_filter(tool_name, include_set):
                    pre_selected.add(i)
            elif exclude_set:
                # Exclude mode: everything except excluded
                if not _match_filter(tool_name, exclude_set):
                    pre_selected.add(i)
            else:
                # No filter: all enabled
                pre_selected.add(i)

        chosen = curses_checklist(
            f"MCP Server: {server_name}  ({len(tools)} tools)",
            labels,
            pre_selected,
            cancel_returns=pre_selected,
        )

        if chosen == pre_selected:
            _print_info(f"  {server_name}: no changes")
            continue

        # Update config
        srv_cfg = mcp_servers.setdefault(server_name, {})
        tools_cfg = srv_cfg.setdefault("tools", {})

        exclude_mode = bool(exclude_set) and not include_set

        if len(chosen) == len(tools) and not exclude_mode:
            # All tools enabled — clear filters (cleanest config shape; the
            # server\'s native tool set is the active set, and any tools the
            # server adds later are auto-enabled).
            tools_cfg.pop("exclude", None)
            tools_cfg.pop("include", None)
        elif exclude_mode:
            # Exclude-mode server (catalog default_excluded / hand-written
            # tools.exclude): stay in exclude mode — do NOT demote the
            # dynamic filter to a frozen include list. Unchecked tools are
            # added as literal excludes; re-checked literals are dropped;
            # glob patterns are preserved (they intentionally keep matching
            # tools the vendor ships later).
            old_exclude = sorted(exclude_set or set())
            glob_entries = [p for p in old_exclude
                            if "*" in p or "?" in p or "[" in p]
            literal_entries = {p for p in old_exclude if p not in glob_entries}
            unchecked = {tool_names[i] for i in range(len(tools))
                         if i not in chosen}
            checked = {tool_names[i] for i in chosen}
            new_literals = (literal_entries - checked) | {
                tn for tn in unchecked
                if not _match_filter(tn, set(old_exclude))
            }
            new_exclude = glob_entries + sorted(new_literals)
            glob_shadowed = sorted(
                tn for tn in checked
                if glob_entries and _match_filter(tn, set(glob_entries))
            )
            if glob_shadowed:
                _print_warning(
                    f"  {server_name}: {len(glob_shadowed)} re-enabled "
                    f"tool(s) still match glob exclude pattern(s) "
                    f"{glob_entries} and stay excluded — edit "
                    f"mcp_servers.{server_name}.tools.exclude in config.yaml "
                    "to enable them."
                )
            if not new_exclude:
                tools_cfg.pop("exclude", None)
                tools_cfg.pop("include", None)
            else:
                tools_cfg["exclude"] = new_exclude
                tools_cfg.pop("include", None)
        else:
            chosen_names = [tool_names[i] for i in sorted(chosen)]
            tools_cfg["include"] = chosen_names
            # Drop any legacy exclude block — we\'re include-mode now.
            tools_cfg.pop("exclude", None)

        enabled_count = len(chosen)
        disabled_count = len(tools) - enabled_count
        _print_success(
            f"  {server_name}: {enabled_count} enabled, {disabled_count} disabled"
        )
        any_changes = True

    if any_changes:
        save_config(config)
        print()
        print(color("  ✓ MCP tool configuration saved", Colors.GREEN))
    else:
        print(color("  No changes to MCP tools", Colors.DIM))


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
