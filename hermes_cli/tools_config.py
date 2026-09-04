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
    NousSubscriptionFeatures, apply_nous_managed_defaults, get_nous_subscription_features)
from hermes_cli.platforms import PLATFORMS as _PLATFORMS_REGISTRY
from hermes_cli.toolset_scope import (
    _TOOLSET_PLATFORM_RESTRICTIONS, toolset_allowed_for_platform as _toolset_allowed_for_platform)
# Re-exports: keep ``hermes_cli.tools_config.X`` callers and test patch targets resolving.
from hermes_cli.tools_config_cua import (  # noqa: F401
    _post_setup_no_window_flags, _cua_driver_cmd, _cua_version_summary, _resolved_cua_driver_cmd, _cua_driver_env,
    _cua_driver_contract_status, _cua_driver_install_ready, _pip_install, _cua_install_target_writable,
    install_cua_driver, _CUA_INSTALLER_TIMEOUT, _CUA_INSTALLER_DRAIN_GRACE, _CUA_LOCK_STALE_AFTER,
    _clear_stale_windows_cua_install_lock, _clear_stale_cua_install_lock, _cua_install_lock_held,
    _cua_release_endpoint_reachable, _repair_cua_driver_autostart_windows, _run_cua_driver_installer)
from hermes_cli.tools_config_post_setup import (  # noqa: F401
    _ensure_browser_use_cli, _run_post_setup, valid_post_setup_keys, run_post_setup_command, _POST_SETUP_INSTALLED,
    _post_setup_already_installed, _module_installed, active_restorable_python_tool_dependencies,
    restorable_python_tool_dependency, _POST_SETUP_READY)
from hermes_cli.tools_config_providers import (  # noqa: F401
    _plugin_image_gen_providers, _plugin_video_gen_providers, _plugin_web_search_providers, _plugin_browser_providers,
    _plugin_tts_providers, web_provider_capabilities, _visible_providers, provider_readiness_status,
    _toolset_needs_configuration_prompt, _configure_tool_category, _web_tier_matches, _is_provider_active,
    _detect_active_provider_index, IMAGEGEN_BACKENDS, _plugin_image_gen_catalog, _plugin_video_gen_catalog,
    _configure_imagegen_model, _configure_imagegen_model_for_plugin, _configure_videogen_model_for_plugin,
    _select_plugin_image_gen_provider, _select_plugin_video_gen_provider, STT_MODEL_CATALOG, _configure_stt_model,
    _write_provider_config, apply_provider_selection, _configure_provider, _reconfigure_provider,
    _configure_vision_backend, _configure_vision_provider_model, _configure_simple_requirements)
from hermes_cli.tools_config_mcp import (  # noqa: F401
    _configure_mcp_tools_interactive, _apply_toolset_change, _apply_mcp_change, tools_disable_enable_command)

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
    ("memory",          "💾 Memory",                    "persistent memory across sessions"),
    ("context_engine",  "🧩 Context Engine",            "runtime tools from the active context engine"),
    ("session_search",  "🔎 Session Search",            "search past conversations"),
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
_DEFAULT_OFF_TOOLSETS = {"homeassistant", "spotify", "discord", "discord_admin", "video", "video_gen", "x_search", "a2a"}

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
    try:
        from tools.xai_http import get_env_value as _xai_get_env_value
        if str(_xai_get_env_value("XAI_API_KEY") or "").strip():
            return True
    except Exception:
        pass
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
    read-time-resolved toolsets (``kanban``, recovered composites, MCP names) are NOT here."""
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


def _enable_recently_shipped_toolsets(enabled_toolsets: Set[str], config: dict, platform: str) -> None:
    """Turn on toolsets that shipped after this platform's saved list (mutates ``enabled_toolsets``). Both "no"s
    outlive this: unchecking records ``known_builtin_toolsets`` (declined), and ``agent.disabled_toolsets`` is
    subtracted last in :func:`_get_platform_tools`."""
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

    # agent.disabled_toolsets is a global suppression list and runs LAST so it overrides everything above. It
    # may arrive as a JSON-array string ("['memory']") from `hermes config set` or a JSON-mode editor save.
    disabled_toolsets = (config.get("agent") or {}).get("disabled_toolsets")
    # Honor agent.disabled_toolsets from config.yaml — allows users to globally suppress specific toolsets
    # (e.g. "memory") across all platforms without per-platform toolset configuration. See #86661.
    if disabled_toolsets:
        from agent.skill_utils import parse_config_string_list
        enabled_toolsets -= {name.strip() for name in parse_config_string_list(disabled_toolsets) if name.strip()}

    if explicitly_configured and toolset_names:
        _warn_all_invalid_platform_toolsets(platform, platform_toolsets[platform])
    return enabled_toolsets


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


def _reconfigure_tool(config: dict, *, force_fresh: bool = True):
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


def _shared_metrics_menu_label(config: dict) -> str:
    """Menu row for shared metrics, showing both consent states."""
    enabled, send = _shared_metrics_state(config)
    state = "off" if not enabled else ("collecting + sending to Nous" if send else "collecting locally")
    return f"Configure shared metrics  ({state})"


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
    print(color("⚕ Tool Summary", Colors.CYAN, Colors.BOLD))
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
    """``(added, removed)`` scoped to the checklist universe, so read-time toolsets (``kanban``) the user never
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
    print(color("⚕ Hermes Tool Configuration", Colors.CYAN, Colors.BOLD))
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


_PLUGIN_COMPAT_LAZY = {
    'MANAGED_FEATURE_COVERAGE_CATEGORY': ('hermes_cli.nous_subscription', 'MANAGED_FEATURE_COVERAGE_CATEGORY'),
    'NOUS_MANAGED_PROVIDER': ('tools.tool_backend_helpers', 'NOUS_MANAGED_PROVIDER'),
    'base_url_hostname': ('utils', 'base_url_hostname'),
    'fal_key_is_configured': ('tools.tool_backend_helpers', 'fal_key_is_configured'),
    'format_nous_portal_entitlement_message': ('hermes_cli.nous_account', 'format_nous_portal_entitlement_message'),
    'is_truthy_value': ('utils', 'is_truthy_value'),
    'save_env_value': ('hermes_cli.config', 'save_env_value'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
