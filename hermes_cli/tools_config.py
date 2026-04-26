"""
Unified tool configuration for Hermes Agent.

`hermes tools` and `hermes setup tools` both enter this module.
Select a platform → toggle toolsets on/off → for newly enabled tools
that need API keys, run through provider-aware configuration.

Saves per-platform tool configuration to ~/.hermes/config.yaml under
the `platform_toolsets` key.
"""

import json as _json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


from hermes_cli.config import (
    load_config, save_config, get_env_value, save_env_value,
)
from hermes_cli.colors import Colors, color
from hermes_cli.nous_subscription import (
    apply_nous_managed_defaults,
    get_nous_subscription_features,
)
from tools.tool_backend_helpers import fal_key_is_configured, managed_nous_tools_enabled
from utils import base_url_hostname

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ─── UI Helpers (shared with setup.py) ────────────────────────────────────────

from hermes_cli.cli_output import (  # noqa: E402 — late import block
    print_error as _print_error,
    print_info as _print_info,
    print_success as _print_success,
    print_warning as _print_warning,
    prompt as _prompt,
)

# ─── Toolset Registry ─────────────────────────────────────────────────────────

# Toolsets shown in the configurator, grouped for display.
# Each entry: (toolset_name, label, description)
# These map to keys in toolsets.py TOOLSETS dict.
CONFIGURABLE_TOOLSETS = [
    ("web",             "🔍 Web Search & Scraping",    "web_search, web_extract"),
    ("browser",         "🌐 Browser Automation",       "navigate, click, type, scroll"),
    ("terminal",        "💻 Terminal & Processes",      "terminal, process"),
    ("file",            "📁 File Operations",           "read, write, patch, search"),
    ("code_execution",  "⚡ Code Execution",            "execute_code"),
    ("vision",          "👁️  Vision / Image Analysis",  "vision_analyze"),
    ("image_gen",       "🎨 Image Generation",          "image_generate"),
    ("moa",             "🧠 Mixture of Agents",         "mixture_of_agents"),
    ("tts",             "🔊 Text-to-Speech",            "text_to_speech"),
    ("skills",          "📚 Skills",                    "list, view, manage"),
    ("todo",            "📋 Task Planning",             "todo"),
    ("memory",          "💾 Memory",                    "persistent memory across sessions"),
    ("session_search",  "🔎 Session Search",            "search past conversations"),
    ("clarify",         "❓ Clarifying Questions",      "clarify"),
    ("delegation",      "👥 Task Delegation",           "delegate_task"),
    ("cronjob",         "⏰ Cron Jobs",                 "create/list/update/pause/resume/run, with optional attached skills"),
    ("messaging",       "📨 Cross-Platform Messaging",  "send_message"),
    ("rl",              "🧪 RL Training",               "Tinker-Atropos training tools"),
    ("homeassistant",    "🏠 Home Assistant",           "smart home device control"),
    ("spotify",          "🎵 Spotify",                  "playback, search, playlists, library"),
    ("discord",         "💬 Discord (read/participate)", "fetch messages, search members, create thread"),
    ("discord_admin",   "🛡️  Discord Server Admin",    "list channels/roles, pin, assign roles"),
]

# Toolsets that are OFF by default for new installs.
# They're still in _HERMES_CORE_TOOLS (available at runtime if enabled),
# but the setup checklist won't pre-select them for first-time users.
_DEFAULT_OFF_TOOLSETS = {"moa", "homeassistant", "rl", "spotify", "discord", "discord_admin"}

# Platform-scoped toolsets: only appear in the `hermes tools` checklist for
# these platforms, and only resolve/save for these platforms.  A toolset
# absent from this map is available on every platform (current behaviour).
#
# Use this for tools whose APIs only make sense on one platform (Discord
# server admin, Slack workspace admin, etc.).  Keeps every other platform's
# checklist from filling up with irrelevant toggles.
_TOOLSET_PLATFORM_RESTRICTIONS: Dict[str, Set[str]] = {
    "discord": {"discord"},
    "discord_admin": {"discord"},
}


def _toolset_allowed_for_platform(ts_key: str, platform: str) -> bool:
    """Return True if ``ts_key`` is configurable on ``platform``.

    Toolsets without a restriction entry are allowed everywhere (the default).
    """
    allowed = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return allowed is None or platform in allowed


def _get_effective_configurable_toolsets():
    """Return CONFIGURABLE_TOOLSETS + any plugin-provided toolsets.

    Plugin toolsets are appended at the end so they appear after the
    built-in toolsets in the TUI checklist. A plugin whose toolset key
    already appears in ``CONFIGURABLE_TOOLSETS`` is skipped — bundled
    plugins (e.g. ``plugins/spotify``) share their toolset key with the
    built-in entry, and we want the built-in label/description to win.
    Without the dedupe, ``hermes tools`` → "reconfigure existing" would
    list the same toolset twice.
    """
    result = list(CONFIGURABLE_TOOLSETS)
    seen = {ts_key for ts_key, _, _ in result}
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_toolsets
        discover_plugins()  # idempotent — ensures plugins are loaded
        for entry in get_plugin_toolsets():
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            result.append(entry)
    except Exception:
        pass
    return result


def _get_plugin_toolset_keys() -> set:
    """Return the set of toolset keys provided by plugins."""
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_toolsets
        discover_plugins()  # idempotent — ensures plugins are loaded
        return {ts_key for ts_key, _, _ in get_plugin_toolsets()}
    except Exception:
        return set()

# Platform display config — derived from the canonical registry so every
# module shares the same data.  Kept as dict-of-dicts for backward
# compatibility with existing ``PLATFORMS[key]["label"]`` access patterns.
from hermes_cli.platforms import PLATFORMS as _PLATFORMS_REGISTRY

PLATFORMS = {
    k: {"label": info.label, "default_toolset": info.default_toolset}
    for k, info in _PLATFORMS_REGISTRY.items()
}


# ─── Tool Categories (provider-aware configuration) ──────────────────────────
# Maps toolset keys to their provider options. When a toolset is newly enabled,
# we use this to show provider selection and prompt for the right API keys.
# Toolsets not in this map either need no config or use the simple fallback.

TOOL_CATEGORIES = {
    "tts": {
        "name": "Text-to-Speech",
        "icon": "🔊",
        "providers": [
            {
                "name": "Nous Subscription",
                "badge": "subscription",
                "tag": "Managed OpenAI TTS billed to your subscription",
                "env_vars": [],
                "tts_provider": "openai",
                "requires_nous_auth": True,
                "managed_nous_feature": "tts",
                "override_env_vars": ["VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"],
            },
            {
                "name": "Microsoft Edge TTS",
                "badge": "★ recommended · free",
                "tag": "Good quality, no API key needed",
                "env_vars": [],
                "tts_provider": "edge",
            },
            {
                "name": "OpenAI TTS",
                "badge": "paid",
                "tag": "High quality voices",
                "env_vars": [
                    {"key": "VOICE_TOOLS_OPENAI_KEY", "prompt": "OpenAI API key", "url": "https://platform.openai.com/api-keys"},
                ],
                "tts_provider": "openai",
            },
            {
                "name": "xAI TTS",
                "tag": "Grok voices - requires xAI API key",
                "env_vars": [
                    {"key": "XAI_API_KEY", "prompt": "xAI API key", "url": "https://console.x.ai/"},
                ],
                "tts_provider": "xai",
            },
            {
                "name": "ElevenLabs",
                "badge": "paid",
                "tag": "Most natural voices",
                "env_vars": [
                    {"key": "ELEVENLABS_API_KEY", "prompt": "ElevenLabs API key", "url": "https://elevenlabs.io/app/settings/api-keys"},
                ],
                "tts_provider": "elevenlabs",
            },
            {
                "name": "Mistral (Voxtral TTS)",
                "badge": "paid",
                "tag": "Multilingual, native Opus",
                "env_vars": [
                    {"key": "MISTRAL_API_KEY", "prompt": "Mistral API key", "url": "https://console.mistral.ai/"},
                ],
                "tts_provider": "mistral",
            },
            {
                "name": "Google Gemini TTS",
                "badge": "preview",
                "tag": "30 prebuilt voices, controllable via prompts",
                "env_vars": [
                    {"key": "GEMINI_API_KEY", "prompt": "Gemini API key", "url": "https://aistudio.google.com/app/apikey"},
                ],
                "tts_provider": "gemini",
            },
            {
                "name": "KittenTTS",
                "badge": "local · free",
                "tag": "Lightweight local ONNX TTS (~25MB), no API key",
                "env_vars": [],
                "tts_provider": "kittentts",
                "post_setup": "kittentts",
            },
        ],
    },
    "web": {
        "name": "Web Search & Extract",
        "setup_title": "Select Search Provider",
        "setup_note": "A free DuckDuckGo search skill is also included — skip this if you don't need a premium provider.",
        "icon": "🔍",
        "providers": [
            {
                "name": "Nous Subscription",
                "badge": "subscription",
                "tag": "Managed Firecrawl billed to your subscription",
                "web_backend": "firecrawl",
                "env_vars": [],
                "requires_nous_auth": True,
                "managed_nous_feature": "web",
                "override_env_vars": ["FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"],
            },
            {
                "name": "Firecrawl Cloud",
                "badge": "★ recommended",
                "tag": "Full-featured search, extract, and crawl",
                "web_backend": "firecrawl",
                "env_vars": [
                    {"key": "FIRECRAWL_API_KEY", "prompt": "Firecrawl API key", "url": "https://firecrawl.dev"},
                ],
            },
            {
                "name": "Exa",
                "badge": "paid",
                "tag": "Neural search with semantic understanding",
                "web_backend": "exa",
                "env_vars": [
                    {"key": "EXA_API_KEY", "prompt": "Exa API key", "url": "https://exa.ai"},
                ],
            },
            {
                "name": "Parallel",
                "badge": "paid",
                "tag": "AI-powered search and extract",
                "web_backend": "parallel",
                "env_vars": [
                    {"key": "PARALLEL_API_KEY", "prompt": "Parallel API key", "url": "https://parallel.ai"},
                ],
            },
            {
                "name": "Tavily",
                "badge": "free tier",
                "tag": "Search, extract, and crawl — 1000 free searches/mo",
                "web_backend": "tavily",
                "env_vars": [
                    {"key": "TAVILY_API_KEY", "prompt": "Tavily API key", "url": "https://app.tavily.com/home"},
                ],
            },
            {
                "name": "Firecrawl Self-Hosted",
                "badge": "free · self-hosted",
                "tag": "Run your own Firecrawl instance (Docker)",
                "web_backend": "firecrawl",
                "env_vars": [
                    {"key": "FIRECRAWL_API_URL", "prompt": "Your Firecrawl instance URL (e.g., http://localhost:3002)"},
                ],
            },
        ],
    },
    "image_gen": {
        "name": "Image Generation",
        "icon": "🎨",
        "providers": [
            {
                "name": "Nous Subscription",
                "badge": "subscription",
                "tag": "Managed FAL image generation billed to your subscription",
                "env_vars": [],
                "requires_nous_auth": True,
                "managed_nous_feature": "image_gen",
                "override_env_vars": ["FAL_KEY"],
                "imagegen_backend": "fal",
            },
            {
                "name": "FAL.ai",
                "badge": "paid",
                "tag": "Pick from flux-2-klein, flux-2-pro, gpt-image, nano-banana, etc.",
                "env_vars": [
                    {"key": "FAL_KEY", "prompt": "FAL API key", "url": "https://fal.ai/dashboard/keys"},
                ],
                "imagegen_backend": "fal",
            },
        ],
    },
    "browser": {
        "name": "Browser Automation",
        "icon": "🌐",
        "providers": [
            {
                "name": "Nous Subscription (Browser Use cloud)",
                "badge": "subscription",
                "tag": "Managed Browser Use billed to your subscription",
                "env_vars": [],
                "browser_provider": "browser-use",
                "requires_nous_auth": True,
                "managed_nous_feature": "browser",
                "override_env_vars": ["BROWSER_USE_API_KEY"],
                "post_setup": "agent_browser",
            },
            {
                "name": "Local Browser",
                "badge": "★ recommended · free",
                "tag": "Headless Chromium, no API key needed",
                "env_vars": [],
                "browser_provider": "local",
                "post_setup": "agent_browser",
            },
            {
                "name": "Browserbase",
                "badge": "paid",
                "tag": "Cloud browser with stealth and proxies",
                "env_vars": [
                    {"key": "BROWSERBASE_API_KEY", "prompt": "Browserbase API key", "url": "https://browserbase.com"},
                    {"key": "BROWSERBASE_PROJECT_ID", "prompt": "Browserbase project ID"},
                ],
                "browser_provider": "browserbase",
                "post_setup": "agent_browser",
            },
            {
                "name": "Browser Use",
                "badge": "paid",
                "tag": "Cloud browser with remote execution",
                "env_vars": [
                    {"key": "BROWSER_USE_API_KEY", "prompt": "Browser Use API key", "url": "https://browser-use.com"},
                ],
                "browser_provider": "browser-use",
                "post_setup": "agent_browser",
            },
            {
                "name": "Firecrawl",
                "badge": "paid",
                "tag": "Cloud browser with remote execution",
                "env_vars": [
                    {"key": "FIRECRAWL_API_KEY", "prompt": "Firecrawl API key", "url": "https://firecrawl.dev"},
                ],
                "browser_provider": "firecrawl",
                "post_setup": "agent_browser",
            },
            {
                "name": "Camofox",
                "badge": "free · local",
                "tag": "Anti-detection browser (Firefox/Camoufox)",
                "env_vars": [
                    {"key": "CAMOFOX_URL", "prompt": "Camofox server URL", "default": "http://localhost:9377",
                     "url": "https://github.com/jo-inc/camofox-browser"},
                ],
                "browser_provider": "camofox",
                "post_setup": "camofox",
            },
        ],
    },
    "homeassistant": {
        "name": "Smart Home",
        "icon": "🏠",
        "providers": [
            {
                "name": "Home Assistant",
                "tag": "REST API integration",
                "env_vars": [
                    {"key": "HASS_TOKEN", "prompt": "Home Assistant Long-Lived Access Token"},
                    {"key": "HASS_URL", "prompt": "Home Assistant URL", "default": "http://homeassistant.local:8123"},
                ],
            },
        ],
    },
    "spotify": {
        "name": "Spotify",
        "icon": "🎵",
        "providers": [
            {
                "name": "Spotify Web API",
                "tag": "PKCE OAuth — opens the setup wizard",
                "env_vars": [],
                "post_setup": "spotify",
            },
        ],
    },
    "rl": {
        "name": "RL Training",
        "icon": "🧪",
        "requires_python": (3, 11),
        "providers": [
            {
                "name": "Tinker / Atropos",
                "tag": "RL training platform",
                "env_vars": [
                    {"key": "TINKER_API_KEY", "prompt": "Tinker API key", "url": "https://tinker-console.thinkingmachines.ai/keys"},
                    {"key": "WANDB_API_KEY", "prompt": "WandB API key", "url": "https://wandb.ai/authorize"},
                ],
                "post_setup": "rl_training",
            },
        ],
    },
}

# Simple env-var requirements for toolsets NOT in TOOL_CATEGORIES.
# Used as a fallback for tools like vision/moa that just need an API key.
TOOLSET_ENV_REQUIREMENTS = {
    "vision":     [("OPENROUTER_API_KEY",   "https://openrouter.ai/keys")],
    "moa":        [("OPENROUTER_API_KEY",   "https://openrouter.ai/keys")],
}


# ─── Post-Setup Hooks ─────────────────────────────────────────────────────────

def _run_post_setup(post_setup_key: str):
    """Run post-setup hooks for tools that need extra installation steps."""
    import shutil
    if post_setup_key in ("agent_browser", "browserbase"):
        node_modules = PROJECT_ROOT / "node_modules" / "agent-browser"
        if not node_modules.exists() and shutil.which("npm"):
            _print_info("    Installing Node.js dependencies for browser tools...")
            import subprocess
            result = subprocess.run(
                ["npm", "install", "--silent"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )
            if result.returncode == 0:
                _print_success("    Node.js dependencies installed")
            else:
                from hermes_constants import display_hermes_home
                _print_warning(f"    npm install failed - run manually: cd {display_hermes_home()}/hermes-agent && npm install")
        elif not node_modules.exists():
            _print_warning("    Node.js not found - browser tools require: npm install (in hermes-agent directory)")

    elif post_setup_key == "camofox":
        camofox_dir = PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser"
        if not camofox_dir.exists() and shutil.which("npm"):
            _print_info("    Installing Camofox browser server...")
            import subprocess
            result = subprocess.run(
                ["npm", "install", "--silent"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )
            if result.returncode == 0:
                _print_success("    Camofox installed")
            else:
                _print_warning("    npm install failed - run manually: npm install")
        if camofox_dir.exists():
            _print_info("    Start the Camofox server:")
            _print_info("      npx @askjo/camofox-browser")
            _print_info("    First run downloads the Camoufox engine (~300MB)")
            _print_info("    Or use Docker: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")
        elif not shutil.which("npm"):
            _print_warning("    Node.js not found. Install Camofox via Docker:")
            _print_info("      docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")

    elif post_setup_key == "kittentts":
        try:
            __import__("kittentts")
            _print_success("    kittentts is already installed")
            return
        except ImportError:
            pass
        import subprocess
        _print_info("    Installing kittentts (~25-80MB model, CPU-only)...")
        wheel_url = (
            "https://github.com/KittenML/KittenTTS/releases/download/"
            "0.8.1/kittentts-0.8.1-py3-none-any.whl"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-U", wheel_url, "soundfile", "--quiet"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                _print_success("    kittentts installed")
                _print_info("    Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo")
                _print_info("    Models: KittenML/kitten-tts-nano-0.8-int8 (25MB), micro (41MB), mini (80MB)")
            else:
                _print_warning("    kittentts install failed:")
                _print_info(f"      {result.stderr.strip()[:300]}")
                _print_info(f"    Run manually: python -m pip install -U '{wheel_url}' soundfile")
        except subprocess.TimeoutExpired:
            _print_warning("    kittentts install timed out (>5min)")
            _print_info(f"    Run manually: python -m pip install -U '{wheel_url}' soundfile")

    elif post_setup_key == "spotify":
        # Run the full `hermes auth spotify` flow — if the user has no
        # client_id yet, this drops them into the interactive wizard
        # (opens the Spotify dashboard, prompts for client_id, persists
        # to ~/.hermes/.env), then continues straight into PKCE. If they
        # already have an app, it skips the wizard and just does OAuth.
        from types import SimpleNamespace
        try:
            from hermes_cli.auth import login_spotify_command
        except Exception as exc:
            _print_warning(f"    Could not load Spotify auth: {exc}")
            _print_info("    Run manually: hermes auth spotify")
            return
        _print_info("    Starting Spotify login...")
        try:
            login_spotify_command(SimpleNamespace(
                client_id=None, redirect_uri=None, scope=None,
                no_browser=False, timeout=None,
            ))
            _print_success("    Spotify authenticated")
        except SystemExit as exc:
            # User aborted the wizard, or OAuth failed — don't fail the
            # toolset enable; they can retry with `hermes auth spotify`.
            _print_warning(f"    Spotify login did not complete: {exc}")
            _print_info("    Run later: hermes auth spotify")
        except Exception as exc:
            _print_warning(f"    Spotify login failed: {exc}")
            _print_info("    Run manually: hermes auth spotify")

    elif post_setup_key == "rl_training":
        try:
            __import__("tinker_atropos")
        except ImportError:
            tinker_dir = PROJECT_ROOT / "tinker-atropos"
            if tinker_dir.exists() and (tinker_dir / "pyproject.toml").exists():
                _print_info("    Installing tinker-atropos submodule...")
                import subprocess
                uv_bin = shutil.which("uv")
                if uv_bin:
                    result = subprocess.run(
                        [uv_bin, "pip", "install", "--python", sys.executable, "-e", str(tinker_dir)],
                        capture_output=True, text=True
                    )
                else:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-e", str(tinker_dir)],
                        capture_output=True, text=True
                    )
                if result.returncode == 0:
                    _print_success("    tinker-atropos installed")
                else:
                    _print_warning("    tinker-atropos install failed - run manually:")
                    _print_info('      uv pip install -e "./tinker-atropos"')
            else:
                _print_warning("    tinker-atropos submodule not found - run:")
                _print_info("      git submodule update --init --recursive")
                _print_info('      uv pip install -e "./tinker-atropos"')


# ─── Platform / Toolset Helpers ───────────────────────────────────────────────

def _get_enabled_platforms() -> List[str]:
    """Return platform keys that are configured (have tokens or are CLI)."""
    enabled = ["cli"]
    if get_env_value("TELEGRAM_BOT_TOKEN"):
        enabled.append("telegram")
    if get_env_value("DISCORD_BOT_TOKEN"):
        enabled.append("discord")
    if get_env_value("SLACK_BOT_TOKEN"):
        enabled.append("slack")
    if get_env_value("WHATSAPP_ENABLED"):
        enabled.append("whatsapp")
    if get_env_value("QQ_APP_ID"):
        enabled.append("qqbot")
    return enabled


def _platform_toolset_summary(config: dict, platforms: Optional[List[str]] = None) -> Dict[str, Set[str]]:
    """Return a summary of enabled toolsets per platform.

    When ``platforms`` is None, this uses ``_get_enabled_platforms`` to
    auto-detect platforms. Tests can pass an explicit list to avoid relying
    on environment variables.
    """
    if platforms is None:
        platforms = _get_enabled_platforms()

    summary: Dict[str, Set[str]] = {}
    for pkey in platforms:
        summary[pkey] = _get_platform_tools(config, pkey)
    return summary


def _parse_enabled_flag(value, default: bool = True) -> bool:
    """Parse bool-like config values used by tool/platform settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _get_platform_tools(
    config: dict,
    platform: str,
    *,
    include_default_mcp_servers: bool = True,
) -> Set[str]:
    """Resolve which individual toolset names are enabled for a platform."""
    from toolsets import resolve_toolset, TOOLSETS

    platform_toolsets = config.get("platform_toolsets") or {}
    toolset_names = platform_toolsets.get(platform)

    if toolset_names is None or not isinstance(toolset_names, list):
        default_ts = PLATFORMS[platform]["default_toolset"]
        toolset_names = [default_ts]

    # YAML may parse bare numeric names (e.g. ``12306:``) as int.
    # Normalise to str so downstream sorted() never mixes types.
    toolset_names = [str(ts) for ts in toolset_names]

    configurable_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    plugin_ts_keys = _get_plugin_toolset_keys()
    platform_default_keys = {p["default_toolset"] for p in PLATFORMS.values()}

    # If the saved list contains any configurable keys directly, the user
    # has explicitly configured this platform — use direct membership.
    # This avoids the subset-inference bug where composite toolsets like
    # "hermes-cli" (which include all _HERMES_CORE_TOOLS) cause disabled
    # toolsets to re-appear as enabled.
    has_explicit_config = any(ts in configurable_keys for ts in toolset_names)

    if has_explicit_config:
        enabled_toolsets = {
            ts for ts in toolset_names
            if ts in configurable_keys and _toolset_allowed_for_platform(ts, platform)
        }
    else:
        # No explicit config — fall back to resolving composite toolset names
        # (e.g. "hermes-cli") to individual tool names and reverse-mapping.
        all_tool_names = set()
        for ts_name in toolset_names:
            all_tool_names.update(resolve_toolset(ts_name))

        enabled_toolsets = set()
        for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
            if not _toolset_allowed_for_platform(ts_key, platform):
                continue
            ts_tools = set(resolve_toolset(ts_key))
            if ts_tools and ts_tools.issubset(all_tool_names):
                enabled_toolsets.add(ts_key)

        default_off = set(_DEFAULT_OFF_TOOLSETS)
        # Legacy safety: if the platform's own name matches a default-off
        # toolset (e.g. `homeassistant` platform + `homeassistant` toolset),
        # keep that toolset enabled on first install.  Skip this dodge for
        # platform-restricted toolsets — those are always opt-in even on
        # their own platform (e.g. `discord` + `discord` should stay OFF).
        if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
            default_off.remove(platform)
        # Home Assistant is already runtime-gated by its check_fn (requires
        # HASS_TOKEN to register any tools). When a user has configured
        # HASS_TOKEN, they've explicitly opted in — don't also strip it via
        # _DEFAULT_OFF_TOOLSETS, which would silently drop HA from platforms
        # (e.g. cron) that run through _get_platform_tools without an
        # explicit saved toolset list. Without this, Norbert's HA cron jobs
        # regressed after #14798 made cron honor per-platform tool config.
        if "homeassistant" in default_off and os.getenv("HASS_TOKEN"):
            default_off.remove("homeassistant")
        enabled_toolsets -= default_off

    # Recover non-configurable platform toolsets (e.g. discord, feishu_doc,
    # feishu_drive).  These are part of the platform's default composite but
    # absent from CONFIGURABLE_TOOLSETS, so they can't appear in the TUI
    # checklist or in a user-saved config.  Must run in BOTH branches —
    # otherwise saving via `hermes tools` (which flips has_explicit_config
    # to True) silently drops them.
    platform_tool_universe = set(resolve_toolset(PLATFORMS[platform]["default_toolset"]))
    configurable_tool_universe = set()
    for ck in configurable_keys:
        configurable_tool_universe.update(resolve_toolset(ck))
    claimed = set()
    for ts_key in enabled_toolsets:
        claimed.update(resolve_toolset(ts_key))
    skip = configurable_keys | plugin_ts_keys | platform_default_keys
    skip |= {k for k in TOOLSETS if k.startswith("hermes-")}
    skip |= set(_DEFAULT_OFF_TOOLSETS) - {platform}
    for ts_key, ts_def in TOOLSETS.items():
        if ts_key in skip:
            continue
        if ts_def.get("includes"):
            continue
        ts_tools = set(resolve_toolset(ts_key))
        if not ts_tools or not ts_tools.issubset(platform_tool_universe):
            continue
        if ts_tools.issubset(configurable_tool_universe):
            continue
        if not ts_tools.issubset(claimed):
            enabled_toolsets.add(ts_key)
            claimed.update(ts_tools)

    # Plugin toolsets: enabled by default unless explicitly disabled, or
    # unless the toolset is in _DEFAULT_OFF_TOOLSETS (e.g. spotify —
    # shipped as a bundled plugin but user must opt in via `hermes tools`
    # so we don't ship 7 Spotify tool schemas to users who don't use it).
    # A plugin toolset is "known" for a platform once `hermes tools`
    # has been saved for that platform (tracked via known_plugin_toolsets).
    # Unknown plugins default to enabled; known-but-absent = disabled.
    if plugin_ts_keys:
        known_map = config.get("known_plugin_toolsets", {})
        known_for_platform = set(known_map.get(platform, []))
        for pts in plugin_ts_keys:
            if pts in toolset_names:
                # Explicitly listed in config — enabled
                enabled_toolsets.add(pts)
            elif pts in _DEFAULT_OFF_TOOLSETS:
                # Opt-in plugin toolset — stay off until user picks it
                continue
            elif pts not in known_for_platform:
                # New plugin not yet seen by hermes tools — default enabled
                enabled_toolsets.add(pts)
            # else: known but not in config = user disabled it

    # Preserve any explicit non-configurable toolset entries (for example,
    # custom toolsets or MCP server names saved in platform_toolsets).
    explicit_passthrough = {
        ts
        for ts in toolset_names
        if ts not in configurable_keys
        and ts not in plugin_ts_keys
        and ts not in platform_default_keys
    }

    # MCP servers are expected to be available on all platforms by default.
    # If the platform explicitly lists one or more MCP server names, treat that
    # as an allowlist. Otherwise include every globally enabled MCP server.
    # Special sentinel: "no_mcp" in the toolset list disables all MCP servers.
    mcp_servers = config.get("mcp_servers") or {}
    enabled_mcp_servers = {
        str(name)
        for name, server_cfg in mcp_servers.items()
        if isinstance(server_cfg, dict)
        and _parse_enabled_flag(server_cfg.get("enabled", True), default=True)
    }
    # Allow "no_mcp" sentinel to opt out of all MCP servers for this platform
    if "no_mcp" in toolset_names:
        explicit_mcp_servers = set()
        enabled_toolsets.update(explicit_passthrough - enabled_mcp_servers - {"no_mcp"})
    else:
        explicit_mcp_servers = explicit_passthrough & enabled_mcp_servers
        enabled_toolsets.update(explicit_passthrough - enabled_mcp_servers)
    if include_default_mcp_servers:
        if explicit_mcp_servers or "no_mcp" in toolset_names:
            enabled_toolsets.update(explicit_mcp_servers)
        else:
            enabled_toolsets.update(enabled_mcp_servers)
    else:
        enabled_toolsets.update(explicit_mcp_servers)

    return enabled_toolsets


def _save_platform_tools(config: dict, platform: str, enabled_toolset_keys: Set[str]):
    """Save the selected toolset keys for a platform to config.

    Preserves any non-configurable toolset entries (like MCP server names)
    that were already in the config for this platform.
    """
    config.setdefault("platform_toolsets", {})

    # Drop platform-scoped toolsets that don't apply here.  Prevents the
    # "Configure all platforms" checklist (or a hand-edited config.yaml)
    # from turning on, say, the `discord` toolset for Telegram.
    enabled_toolset_keys = {
        ts for ts in enabled_toolset_keys
        if _toolset_allowed_for_platform(ts, platform)
    }

    # Get the set of all configurable toolset keys (built-in + plugin)
    configurable_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    plugin_keys = _get_plugin_toolset_keys()
    configurable_keys |= plugin_keys

    # Also exclude platform default toolsets (hermes-cli, hermes-telegram, etc.)
    # These are "super" toolsets that resolve to ALL tools, so preserving them
    # would silently override the user's unchecked selections on the next read.
    platform_default_keys = {p["default_toolset"] for p in PLATFORMS.values()}

    # Get existing toolsets for this platform
    existing_toolsets = config.get("platform_toolsets", {}).get(platform, [])
    if not isinstance(existing_toolsets, list):
        existing_toolsets = []
    existing_toolsets = [str(ts) for ts in existing_toolsets]

    # Preserve any entries that are NOT configurable toolsets and NOT platform
    # defaults (i.e. only MCP server names should be preserved)
    preserved_entries = {
        entry for entry in existing_toolsets
        if entry not in configurable_keys and entry not in platform_default_keys
    }
    # Opening `hermes tools` is the user's opt-in to reconfigure tools, so treat
    # saving from the picker as consent to clear the "no_mcp" sentinel. The
    # picker has no checkbox for no_mcp, so without this users who once set it
    # by hand could never re-enable MCP servers through the UI.
    preserved_entries.discard("no_mcp")

    # Merge preserved entries with new enabled toolsets
    config["platform_toolsets"][platform] = sorted(enabled_toolset_keys | preserved_entries)

    # Track which plugin toolsets are "known" for this platform so we can
    # distinguish "new plugin, default enabled" from "user disabled it".
    if plugin_keys:
        config.setdefault("known_plugin_toolsets", {})
        config["known_plugin_toolsets"][platform] = sorted(plugin_keys)

    save_config(config)


def _toolset_has_keys(ts_key: str, config: dict = None) -> bool:
    """Check if a toolset's required API keys are configured."""
    if config is None:
        config = load_config()

    if ts_key == "vision":
        try:
            from agent.auxiliary_client import resolve_vision_provider_client

            _provider, client, _model = resolve_vision_provider_client()
            return client is not None
        except Exception:
            return False

    if ts_key in {"web", "image_gen", "tts", "browser"}:
        features = get_nous_subscription_features(config)
        feature = features.features.get(ts_key)
        if feature and (feature.available or feature.managed_by_nous):
            return True

    # Check TOOL_CATEGORIES first (provider-aware)
    cat = TOOL_CATEGORIES.get(ts_key)
    if cat:
        for provider in _visible_providers(cat, config):
            env_vars = provider.get("env_vars", [])
            if not env_vars:
                return True  # No-key provider (e.g. Local Browser, Edge TTS)
            if all(get_env_value(e["key"]) for e in env_vars):
                return True
        return False

    # Fallback to simple requirements
    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return True
    return all(get_env_value(var) for var, _ in requirements)


# ─── Menu Helpers ─────────────────────────────────────────────────────────────

def _prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu (arrow keys). Delegates to curses_radiolist."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=default)


# ─── Token Estimation ────────────────────────────────────────────────────────

# Module-level cache so discovery + tokenization runs at most once per process.
_tool_token_cache: Optional[Dict[str, int]] = None


def _estimate_tool_tokens() -> Dict[str, int]:
    """Return estimated token counts per individual tool name.

    Uses tiktoken (cl100k_base) to count tokens in the JSON-serialised
    OpenAI-format tool schema.  Triggers tool discovery on first call,
    then caches the result for the rest of the process.

    Returns an empty dict when tiktoken or the registry is unavailable.
    """
    global _tool_token_cache
    if _tool_token_cache is not None:
        return _tool_token_cache

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.debug("tiktoken unavailable; skipping tool token estimation")
        _tool_token_cache = {}
        return _tool_token_cache

    try:
        # Trigger full tool discovery (imports all tool modules).
        import model_tools  # noqa: F401
        from tools.registry import registry
    except Exception:
        logger.debug("Tool registry unavailable; skipping token estimation")
        _tool_token_cache = {}
        return _tool_token_cache

    counts: Dict[str, int] = {}
    for name in registry.get_all_tool_names():
        schema = registry.get_schema(name)
        if schema:
            # Mirror what gets sent to the API:
            # {"type": "function", "function": <schema>}
            text = _json.dumps({"type": "function", "function": schema})
            counts[name] = len(enc.encode(text))
    _tool_token_cache = counts
    return _tool_token_cache


def _prompt_toolset_checklist(platform_label: str, enabled: Set[str], platform: str = "cli") -> Set[str]:
    """Multi-select checklist of toolsets. Returns set of selected toolset keys."""
    from hermes_cli.curses_ui import curses_checklist
    from toolsets import resolve_toolset

    # Pre-compute per-tool token counts (cached after first call).
    tool_tokens = _estimate_tool_tokens()

    effective_all = _get_effective_configurable_toolsets()
    # Drop platform-scoped toolsets that don't apply to this platform.
    effective = [
        (k, l, d) for (k, l, d) in effective_all
        if _toolset_allowed_for_platform(k, platform)
    ]

    labels = []
    for ts_key, ts_label, ts_desc in effective:
        suffix = ""
        if not _toolset_has_keys(ts_key) and (TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key)):
            suffix = "  [no API key]"
        labels.append(f"{ts_label}  ({ts_desc}){suffix}")

    pre_selected = {
        i for i, (ts_key, _, _) in enumerate(effective)
        if ts_key in enabled
    }

    # Build a live status function that shows deduplicated total token cost.
    status_fn = None
    if tool_tokens:
        ts_keys = [ts_key for ts_key, _, _ in effective]

        def status_fn(chosen: set) -> str:
            # Collect unique tool names across all selected toolsets
            all_tools: set = set()
            for idx in chosen:
                all_tools.update(resolve_toolset(ts_keys[idx]))
            total = sum(tool_tokens.get(name, 0) for name in all_tools)
            if total >= 1000:
                return f"Est. tool context: ~{total / 1000:.1f}k tokens"
            return f"Est. tool context: ~{total} tokens"

    chosen = curses_checklist(
        f"Tools for {platform_label}",
        labels,
        pre_selected,
        cancel_returns=pre_selected,
        status_fn=status_fn,
    )
    return {effective[i][0] for i in chosen}


# ─── Provider-Aware Configuration ────────────────────────────────────────────

def _configure_toolset(ts_key: str, config: dict):
    """Configure a toolset - provider selection + API keys.
    
    Uses TOOL_CATEGORIES for provider-aware config, falls back to simple
    env var prompts for toolsets not in TOOL_CATEGORIES.
    """
    cat = TOOL_CATEGORIES.get(ts_key)

    if cat:
        _configure_tool_category(ts_key, cat, config)
    else:
        # Simple fallback for vision, moa, etc.
        _configure_simple_requirements(ts_key)


def _plugin_image_gen_providers() -> list[dict]:
    """Build picker-row dicts from plugin-registered image gen providers.

    Each returned dict looks like a regular ``TOOL_CATEGORIES`` provider
    row but carries an ``image_gen_plugin_name`` marker so downstream
    code (config writing, model picker) knows to route through the
    plugin registry instead of the in-tree FAL backend.

    FAL is skipped — it's already exposed by the hardcoded
    ``TOOL_CATEGORIES["image_gen"]`` entries. When FAL gets ported to
    a plugin in a follow-up PR, the hardcoded entries go away and this
    function surfaces it alongside OpenAI automatically.
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
        if getattr(provider, "name", None) == "fal":
            # FAL has its own hardcoded rows today.
            continue
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        rows.append(
            {
                "name": schema.get("name", provider.display_name),
                "badge": schema.get("badge", ""),
                "tag": schema.get("tag", ""),
                "env_vars": schema.get("env_vars", []),
                "image_gen_plugin_name": provider.name,
            }
        )
    return rows


def _visible_providers(cat: dict, config: dict) -> list[dict]:
    """Return provider entries visible for the current auth/config state."""
    features = get_nous_subscription_features(config)
    visible = []
    for provider in cat.get("providers", []):
        if provider.get("managed_nous_feature") and not managed_nous_tools_enabled():
            continue
        if provider.get("requires_nous_auth") and not features.nous_auth_present:
            continue
        visible.append(provider)

    # Inject plugin-registered image_gen backends (OpenAI today, more
    # later) so the picker lists them alongside FAL / Nous Subscription.
    if cat.get("name") == "Image Generation":
        visible.extend(_plugin_image_gen_providers())

    return visible


def _toolset_needs_configuration_prompt(ts_key: str, config: dict) -> bool:
    """Return True when enabling this toolset should open provider setup."""
    cat = TOOL_CATEGORIES.get(ts_key)
    if not cat:
        return not _toolset_has_keys(ts_key, config)

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

    return not _toolset_has_keys(ts_key, config)


def _configure_tool_category(ts_key: str, cat: dict, config: dict):
    """Configure a tool category with provider selection."""
    icon = cat.get("icon", "")
    name = cat["name"]
    providers = _visible_providers(cat, config)

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
        _configure_provider(provider, config)
    else:
        # Multiple providers - let user choose
        print()
        # Use custom title if provided (e.g. "Select Search Provider")
        title = cat.get("setup_title", "Choose a provider")
        print(color(f"  --- {icon} {name} - {title} ---", Colors.CYAN))
        if cat.get("setup_note"):
            _print_info(f"  {cat['setup_note']}")
        print()

        # Plain text labels only (no ANSI codes in menu items)
        provider_choices = []
        for p in providers:
            badge = f" [{p['badge']}]" if p.get("badge") else ""
            tag = f" — {p['tag']}" if p.get("tag") else ""
            configured = ""
            env_vars = p.get("env_vars", [])
            if not env_vars or all(get_env_value(v["key"]) for v in env_vars):
                if _is_provider_active(p, config):
                    configured = " [active]"
                elif not env_vars:
                    configured = ""
                else:
                    configured = " [configured]"
            provider_choices.append(f"{p['name']}{badge}{tag}{configured}")

        # Add skip option
        provider_choices.append("Skip — keep defaults / configure later")

        # Detect current provider as default
        default_idx = _detect_active_provider_index(providers, config)

        provider_idx = _prompt_choice(f"  {title}:", provider_choices, default_idx)

        # Skip selected
        if provider_idx >= len(providers):
            _print_info(f"  Skipped {name}")
            return

        _configure_provider(providers[provider_idx], config)


def _is_provider_active(provider: dict, config: dict) -> bool:
    """Check if a provider entry matches the currently active config."""
    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name:
        image_cfg = config.get("image_gen", {})
        return isinstance(image_cfg, dict) and image_cfg.get("provider") == plugin_name

    managed_feature = provider.get("managed_nous_feature")
    if managed_feature:
        features = get_nous_subscription_features(config)
        feature = features.features.get(managed_feature)
        if feature is None:
            return False
        if managed_feature == "image_gen":
            image_cfg = config.get("image_gen", {})
            if isinstance(image_cfg, dict):
                configured_provider = image_cfg.get("provider")
                if configured_provider not in (None, "", "fal"):
                    return False
                if image_cfg.get("use_gateway") is False:
                    return False
            return feature.managed_by_nous
        if provider.get("tts_provider"):
            return (
                feature.managed_by_nous
                and config.get("tts", {}).get("provider") == provider["tts_provider"]
            )
        if "browser_provider" in provider:
            current = config.get("browser", {}).get("cloud_provider")
            return feature.managed_by_nous and provider["browser_provider"] == current
        if provider.get("web_backend"):
            current = config.get("web", {}).get("backend")
            return feature.managed_by_nous and current == provider["web_backend"]
        return feature.managed_by_nous

    if provider.get("tts_provider"):
        return config.get("tts", {}).get("provider") == provider["tts_provider"]
    if "browser_provider" in provider:
        current = config.get("browser", {}).get("cloud_provider")
        return provider["browser_provider"] == current
    if provider.get("web_backend"):
        current = config.get("web", {}).get("backend")
        return current == provider["web_backend"]
    if provider.get("imagegen_backend"):
        image_cfg = config.get("image_gen", {})
        if not isinstance(image_cfg, dict):
            return False
        configured_provider = image_cfg.get("provider")
        return (
            provider["imagegen_backend"] == "fal"
            and configured_provider in (None, "", "fal")
            and not image_cfg.get("use_gateway")
        )
    return False


def _detect_active_provider_index(providers: list, config: dict) -> int:
    """Return the index of the currently active provider, or 0."""
    for i, p in enumerate(providers):
        if _is_provider_active(p, config):
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
        current_model = default_model

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
        current_model = default_model

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


def _select_plugin_image_gen_provider(plugin_name: str, config: dict) -> None:
    """Persist a plugin-backed image generation provider selection."""
    img_cfg = config.setdefault("image_gen", {})
    if not isinstance(img_cfg, dict):
        img_cfg = {}
        config["image_gen"] = img_cfg
    img_cfg["provider"] = plugin_name
    img_cfg["use_gateway"] = False
    _print_success(f"  image_gen.provider set to: {plugin_name}")
    _configure_imagegen_model_for_plugin(plugin_name, config)


def _configure_provider(provider: dict, config: dict):
    """Configure a single provider - prompt for API keys and set config."""
    env_vars = provider.get("env_vars", [])
    managed_feature = provider.get("managed_nous_feature")

    if provider.get("requires_nous_auth"):
        features = get_nous_subscription_features(config)
        if not features.nous_auth_present:
            _print_warning("  Nous Subscription is only available after logging into Nous Portal.")
            return

    # Set TTS provider in config if applicable
    if provider.get("tts_provider"):
        tts_cfg = config.setdefault("tts", {})
        tts_cfg["provider"] = provider["tts_provider"]
        tts_cfg["use_gateway"] = bool(managed_feature)

    # Set browser cloud provider in config if applicable
    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        browser_cfg = config.setdefault("browser", {})
        if bp == "local":
            browser_cfg["cloud_provider"] = "local"
            _print_success("  Browser set to local mode")
        elif bp:
            browser_cfg["cloud_provider"] = bp
            _print_success(f"  Browser cloud provider set to: {bp}")
        browser_cfg["use_gateway"] = bool(managed_feature)

    # Set web search backend in config if applicable
    if provider.get("web_backend"):
        web_cfg = config.setdefault("web", {})
        web_cfg["backend"] = provider["web_backend"]
        web_cfg["use_gateway"] = bool(managed_feature)
        _print_success(f"  Web backend set to: {provider['web_backend']}")

    # For tools without a specific config key (e.g. image_gen), still
    # track use_gateway so the runtime knows the user's intent.
    if managed_feature and managed_feature not in ("web", "tts", "browser"):
        config.setdefault(managed_feature, {})["use_gateway"] = True
    elif not managed_feature:
        # User picked a non-gateway provider — find which category this
        # belongs to and clear use_gateway if it was previously set.
        for cat_key, cat in TOOL_CATEGORIES.items():
            if provider in cat.get("providers", []):
                section = config.get(cat_key)
                if isinstance(section, dict) and section.get("use_gateway"):
                    section["use_gateway"] = False
                break

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
            _select_plugin_image_gen_provider(plugin_name, config)
            return
        # Imagegen backends prompt for model selection after backend pick.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            # In-tree FAL is the only non-plugin backend today. Keep
            # image_gen.provider clear so the dispatch shim falls through
            # to the legacy FAL path.
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict) and img_cfg.get("provider") not in (None, "", "fal"):
                img_cfg["provider"] = "fal"
        return

    # Prompt for each required env var
    all_configured = True
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
            _select_plugin_image_gen_provider(plugin_name, config)
            return
        # Imagegen backends prompt for model selection after env vars are in.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict) and img_cfg.get("provider") not in (None, "", "fal"):
                img_cfg["provider"] = "fal"


def _configure_simple_requirements(ts_key: str):
    """Simple fallback for toolsets that just need env vars (no provider selection)."""
    if ts_key == "vision":
        if _toolset_has_keys("vision"):
            return
        print()
        print(color("  Vision / Image Analysis requires a multimodal backend:", Colors.YELLOW))
        choices = [
            "OpenRouter — uses Gemini",
            "OpenAI-compatible endpoint — base URL, API key, and vision model",
            "Skip",
        ]
        idx = _prompt_choice("  Configure vision backend", choices, 2)
        if idx == 0:
            _print_info("  Get key at: https://openrouter.ai/keys")
            value = _prompt("    OPENROUTER_API_KEY", password=True)
            if value and value.strip():
                save_env_value("OPENROUTER_API_KEY", value.strip())
                _print_success("    Saved")
            else:
                _print_warning("    Skipped")
        elif idx == 1:
            base_url = _prompt("    OPENAI_BASE_URL (blank for OpenAI)").strip() or "https://api.openai.com/v1"
            is_native_openai = base_url_hostname(base_url) == "api.openai.com"
            key_label = "    OPENAI_API_KEY" if is_native_openai else "    API key"
            api_key = _prompt(key_label, password=True)
            if api_key and api_key.strip():
                save_env_value("OPENAI_API_KEY", api_key.strip())
                # Save vision base URL to config (not .env — only secrets go there)
                _cfg = load_config()
                _aux = _cfg.setdefault("auxiliary", {}).setdefault("vision", {})
                _aux["base_url"] = base_url
                save_config(_cfg)
                if is_native_openai:
                    save_env_value("AUXILIARY_VISION_MODEL", "gpt-4o-mini")
                _print_success("    Saved")
            else:
                _print_warning("    Skipped")
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


def _reconfigure_tool(config: dict):
    """Let user reconfigure an existing tool's provider or API key."""
    # Build list of configurable tools that are currently set up
    configurable = []
    for ts_key, ts_label, _ in _get_effective_configurable_toolsets():
        cat = TOOL_CATEGORIES.get(ts_key)
        reqs = TOOLSET_ENV_REQUIREMENTS.get(ts_key)
        if cat or reqs:
            if _toolset_has_keys(ts_key, config):
                configurable.append((ts_key, ts_label))

    if not configurable:
        _print_info("No configured tools to reconfigure.")
        return

    choices = [label for _, label in configurable]
    choices.append("Cancel")

    idx = _prompt_choice("  Which tool would you like to reconfigure?", choices, len(choices) - 1)

    if idx >= len(configurable):
        return  # Cancel

    ts_key, ts_label = configurable[idx]
    cat = TOOL_CATEGORIES.get(ts_key)

    if cat:
        _configure_tool_category_for_reconfig(ts_key, cat, config)
    else:
        _reconfigure_simple_requirements(ts_key)

    save_config(config)


def _configure_tool_category_for_reconfig(ts_key: str, cat: dict, config: dict):
    """Reconfigure a tool category - provider selection + API key update."""
    icon = cat.get("icon", "")
    name = cat["name"]
    providers = _visible_providers(cat, config)

    if len(providers) == 1:
        provider = providers[0]
        print()
        print(color(f"  --- {icon} {name} ({provider['name']}) ---", Colors.CYAN))
        _reconfigure_provider(provider, config)
    else:
        print()
        print(color(f"  --- {icon} {name} - Choose a provider ---", Colors.CYAN))
        print()

        provider_choices = []
        for p in providers:
            badge = f" [{p['badge']}]" if p.get("badge") else ""
            tag = f" — {p['tag']}" if p.get("tag") else ""
            configured = ""
            env_vars = p.get("env_vars", [])
            if not env_vars or all(get_env_value(v["key"]) for v in env_vars):
                if _is_provider_active(p, config):
                    configured = " [active]"
                elif not env_vars:
                    configured = ""
                else:
                    configured = " [configured]"
            provider_choices.append(f"{p['name']}{badge}{tag}{configured}")

        default_idx = _detect_active_provider_index(providers, config)

        provider_idx = _prompt_choice("  Select provider:", provider_choices, default_idx)
        _reconfigure_provider(providers[provider_idx], config)


def _reconfigure_provider(provider: dict, config: dict):
    """Reconfigure a provider - update API keys."""
    env_vars = provider.get("env_vars", [])
    managed_feature = provider.get("managed_nous_feature")

    if provider.get("requires_nous_auth"):
        features = get_nous_subscription_features(config)
        if not features.nous_auth_present:
            _print_warning("  Nous Subscription is only available after logging into Nous Portal.")
            return

    if provider.get("tts_provider"):
        config.setdefault("tts", {})["provider"] = provider["tts_provider"]
        _print_success(f"  TTS provider set to: {provider['tts_provider']}")

    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        if bp == "local":
            config.setdefault("browser", {})["cloud_provider"] = "local"
            _print_success("  Browser set to local mode")
        elif bp:
            config.setdefault("browser", {})["cloud_provider"] = bp
            _print_success(f"  Browser cloud provider set to: {bp}")

    # Set web search backend in config if applicable
    if provider.get("web_backend"):
        config.setdefault("web", {})["backend"] = provider["web_backend"]
        _print_success(f"  Web backend set to: {provider['web_backend']}")

    if managed_feature and managed_feature not in ("web", "tts", "browser"):
        section = config.setdefault(managed_feature, {})
        if not isinstance(section, dict):
            section = {}
            config[managed_feature] = section
        section["use_gateway"] = True
    elif not managed_feature:
        for cat_key, cat in TOOL_CATEGORIES.items():
            if provider in cat.get("providers", []):
                section = config.get(cat_key)
                if isinstance(section, dict) and section.get("use_gateway"):
                    section["use_gateway"] = False
                break

    if not env_vars:
        if provider.get("post_setup"):
            _run_post_setup(provider["post_setup"])
        _print_success(f"  {provider['name']} - no configuration needed!")
        if managed_feature:
            _print_info("  Requests for this tool will be billed to your Nous subscription.")
        plugin_name = provider.get("image_gen_plugin_name")
        if plugin_name:
            _select_plugin_image_gen_provider(plugin_name, config)
            return
        # Imagegen backends prompt for model selection on reconfig too.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            if backend == "fal":
                img_cfg = config.setdefault("image_gen", {})
                if isinstance(img_cfg, dict):
                    img_cfg["provider"] = "fal"
                    img_cfg["use_gateway"] = False
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

    # Imagegen backends prompt for model selection on reconfig too.
    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name:
        _select_plugin_image_gen_provider(plugin_name, config)
        return

    backend = provider.get("imagegen_backend")
    if backend:
        _configure_imagegen_model(backend, config)
        if backend == "fal":
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict):
                img_cfg["provider"] = "fal"
                img_cfg["use_gateway"] = False


def _reconfigure_simple_requirements(ts_key: str):
    """Reconfigure simple env var requirements."""
    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return

    ts_label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts_key), ts_key)
    print()
    print(color(f"  {ts_label}:", Colors.CYAN))

    for var, url in requirements:
        existing = get_env_value(var)
        if existing:
            _print_info(f"  {var}: configured ({existing[:8]}...)")
        if url:
            _print_info(f"  Get key at: {url}")
        value = _prompt(f"    {var} (Enter to keep current)", password=True)
        if value and value.strip():
            save_env_value(var, value.strip())
            _print_success("    Updated")
        else:
            _print_info("    Kept current")


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def tools_command(args=None, first_install: bool = False, config: dict = None):
    """Entry point for `hermes tools` and `hermes setup tools`.

    Args:
        first_install: When True (set by the setup wizard on fresh installs),
            skip the platform menu, go straight to the CLI checklist, and
            prompt for API keys on all enabled tools that need them.
        config: Optional config dict to use.  When called from the setup
            wizard, the wizard passes its own dict so that platform_toolsets
            are written into it and survive the wizard's final save_config().
    """
    if config is None:
        config = load_config()
    enabled_platforms = _get_enabled_platforms()

    print()

    # Non-interactive summary mode for CLI usage
    if getattr(args, "summary", False):
        total = len(_get_effective_configurable_toolsets())
        print(color("⚕ Tool Summary", Colors.CYAN, Colors.BOLD))
        print()
        summary = _platform_toolset_summary(config, enabled_platforms)
        for pkey in enabled_platforms:
            pinfo = PLATFORMS[pkey]
            enabled = summary.get(pkey, set())
            count = len(enabled)
            print(color(f"  {pinfo['label']}", Colors.BOLD) + color(f"  ({count}/{total})", Colors.DIM))
            if enabled:
                for ts_key in sorted(enabled):
                    label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts_key), ts_key)
                    print(color(f"    ✓ {label}", Colors.GREEN))
            else:
                print(color("    (none enabled)", Colors.DIM))
        print()
        return
    print(color("⚕ Hermes Tool Configuration", Colors.CYAN, Colors.BOLD))
    print(color("  Enable or disable tools per platform.", Colors.DIM))
    print(color("  Tools that need API keys will be configured when enabled.", Colors.DIM))
    print(color("  Guide: https://hermes-agent.nousresearch.com/docs/user-guide/features/tools", Colors.DIM))
    print()

    # ── First-time install: linear flow, no platform menu ──
    if first_install:
        for pkey in enabled_platforms:
            pinfo = PLATFORMS[pkey]
            current_enabled = _get_platform_tools(config, pkey, include_default_mcp_servers=False)

            # Uncheck toolsets that should be off by default
            checklist_preselected = current_enabled - _DEFAULT_OFF_TOOLSETS

            # Show checklist
            new_enabled = _prompt_toolset_checklist(pinfo["label"], checklist_preselected, pkey)

            added = new_enabled - current_enabled
            removed = current_enabled - new_enabled
            if added:
                for ts in sorted(added):
                    label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts), ts)
                    print(color(f"  + {label}", Colors.GREEN))
            if removed:
                for ts in sorted(removed):
                    label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts), ts)
                    print(color(f"  - {label}", Colors.RED))

            auto_configured = apply_nous_managed_defaults(
                config,
                enabled_toolsets=new_enabled,
            )
            if managed_nous_tools_enabled():
                for ts_key in sorted(auto_configured):
                    label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts_key), ts_key)
                    print(color(f"  ✓ {label}: using your Nous subscription defaults", Colors.GREEN))

            # Walk through ALL selected tools that have provider options or
            # need API keys.  This ensures browser (Local vs Browserbase),
            # TTS (Edge vs OpenAI vs ElevenLabs), etc. are shown even when
            # a free provider exists.
            to_configure = [
                ts_key for ts_key in sorted(new_enabled)
                if (TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key))
                and ts_key not in auto_configured
            ]

            if to_configure:
                print()
                print(color(f"  Configuring {len(to_configure)} tool(s):", Colors.YELLOW))
                for ts_key in to_configure:
                    label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts_key), ts_key)
                    print(color(f"    • {label}", Colors.DIM))
                print(color("  You can skip any tool you don't need right now.", Colors.DIM))
                print()
                for ts_key in to_configure:
                    _configure_toolset(ts_key, config)

            _save_platform_tools(config, pkey, new_enabled)
            save_config(config)
            print(color(f"  ✓ Saved {pinfo['label']} tool configuration", Colors.GREEN))
            print()

        return

    # ── Returning user: platform menu loop ──
    # Build platform choices
    platform_choices = []
    platform_keys = []
    for pkey in enabled_platforms:
        pinfo = PLATFORMS[pkey]
        current = _get_platform_tools(config, pkey, include_default_mcp_servers=False)
        count = len(current)
        total = len(_get_effective_configurable_toolsets())
        platform_choices.append(f"Configure {pinfo['label']}  ({count}/{total} enabled)")
        platform_keys.append(pkey)

    if len(platform_keys) > 1:
        platform_choices.append("Configure all platforms (global)")
    platform_choices.append("Reconfigure an existing tool's provider or API key")

    # Show MCP option if any MCP servers are configured
    _has_mcp = bool(config.get("mcp_servers"))
    if _has_mcp:
        platform_choices.append("Configure MCP server tools")

    platform_choices.append("Done")

    # Index offsets for the extra options after per-platform entries
    _global_idx = len(platform_keys) if len(platform_keys) > 1 else -1
    _reconfig_idx = len(platform_keys) + (1 if len(platform_keys) > 1 else 0)
    _mcp_idx = (_reconfig_idx + 1) if _has_mcp else -1
    _done_idx = _reconfig_idx + (2 if _has_mcp else 1)

    while True:
        idx = _prompt_choice("Select an option:", platform_choices, default=0)

        # "Done" selected
        if idx == _done_idx:
            break

        # "Reconfigure" selected
        if idx == _reconfig_idx:
            _reconfigure_tool(config)
            print()
            continue

        # "Configure MCP tools" selected
        if idx == _mcp_idx:
            _configure_mcp_tools_interactive(config)
            print()
            continue

        # "Configure all platforms (global)" selected
        if idx == _global_idx:
            # Use the union of all platforms' current tools as the starting state
            all_current = set()
            for pk in platform_keys:
                all_current |= _get_platform_tools(config, pk, include_default_mcp_servers=False)
            new_enabled = _prompt_toolset_checklist("All platforms", all_current)
            if new_enabled != all_current:
                for pk in platform_keys:
                    prev = _get_platform_tools(config, pk, include_default_mcp_servers=False)
                    added = new_enabled - prev
                    removed = prev - new_enabled
                    pinfo_inner = PLATFORMS[pk]
                    if added or removed:
                        print(color(f"  {pinfo_inner['label']}:", Colors.DIM))
                        for ts in sorted(added):
                            label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts), ts)
                            print(color(f"    + {label}", Colors.GREEN))
                        for ts in sorted(removed):
                            label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts), ts)
                            print(color(f"    - {label}", Colors.RED))
                    # Configure API keys for newly enabled tools
                    for ts_key in sorted(added):
                        if (TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key)):
                            if _toolset_needs_configuration_prompt(ts_key, config):
                                _configure_toolset(ts_key, config)
                    _save_platform_tools(config, pk, new_enabled)
                save_config(config)
                print(color("  ✓ Saved configuration for all platforms", Colors.GREEN))
                # Update choice labels
                for ci, pk in enumerate(platform_keys):
                    new_count = len(_get_platform_tools(config, pk, include_default_mcp_servers=False))
                    total = len(_get_effective_configurable_toolsets())
                    platform_choices[ci] = f"Configure {PLATFORMS[pk]['label']}  ({new_count}/{total} enabled)"
            else:
                print(color("  No changes", Colors.DIM))
            print()
            continue

        pkey = platform_keys[idx]
        pinfo = PLATFORMS[pkey]

        # Get current enabled toolsets for this platform
        current_enabled = _get_platform_tools(config, pkey, include_default_mcp_servers=False)

        # Show checklist
        new_enabled = _prompt_toolset_checklist(pinfo["label"], current_enabled)

        if new_enabled != current_enabled:
            added = new_enabled - current_enabled
            removed = current_enabled - new_enabled

            if added:
                for ts in sorted(added):
                    label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts), ts)
                    print(color(f"  + {label}", Colors.GREEN))
            if removed:
                for ts in sorted(removed):
                    label = next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts), ts)
                    print(color(f"  - {label}", Colors.RED))

            # Configure newly enabled toolsets that need API keys
            for ts_key in sorted(added):
                if (TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key)):
                    if _toolset_needs_configuration_prompt(ts_key, config):
                        _configure_toolset(ts_key, config)

            _save_platform_tools(config, pkey, new_enabled)
            save_config(config)
            print(color(f"  ✓ Saved {pinfo['label']} configuration", Colors.GREEN))
        else:
            print(color(f"  No changes to {pinfo['label']}", Colors.DIM))

        print()

        # Update the choice label with new count
        new_count = len(_get_platform_tools(config, pkey, include_default_mcp_servers=False))
        total = len(_get_effective_configurable_toolsets())
        platform_choices[idx] = f"Configure {pinfo['label']}  ({new_count}/{total} enabled)"

    print()
    from hermes_constants import display_hermes_home
    print(color(f"  Tool configuration saved to {display_hermes_home()}/config.yaml", Colors.DIM))
    print(color("  Changes take effect on next 'hermes' or gateway restart.", Colors.DIM))
    print()


# ─── MCP Tools Interactive Configuration ─────────────────────────────────────


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
        if v.get("enabled", True) not in (False, "false", "0", "no", "off")
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

        # Determine which tools are currently enabled
        pre_selected: Set[int] = set()
        tool_names = [t[0] for t in tools]
        for i, tool_name in enumerate(tool_names):
            if include_list:
                # Include mode: only included tools are selected
                if tool_name in include_list:
                    pre_selected.add(i)
            elif exclude_list:
                # Exclude mode: everything except excluded
                if tool_name not in exclude_list:
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

        # Compute new exclude list based on unchecked tools
        new_exclude = [tool_names[i] for i in range(len(tool_names)) if i not in chosen]

        # Update config
        srv_cfg = mcp_servers.setdefault(server_name, {})
        tools_cfg = srv_cfg.setdefault("tools", {})

        if new_exclude:
            tools_cfg["exclude"] = new_exclude
            # Remove include if present — we're switching to exclude mode
            tools_cfg.pop("include", None)
        else:
            # All tools enabled — clear filters
            tools_cfg.pop("exclude", None)
            tools_cfg.pop("include", None)

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


# ─── Non-interactive disable/enable ──────────────────────────────────────────


def _apply_toolset_change(config: dict, platform: str, toolset_names: List[str], action: str):
    """Add or remove built-in toolsets for a platform."""
    enabled = _get_platform_tools(config, platform, include_default_mcp_servers=False)
    if action == "disable":
        updated = enabled - set(toolset_names)
    else:
        updated = enabled | set(toolset_names)
    _save_platform_tools(config, platform, updated)


def _apply_mcp_change(config: dict, targets: List[str], action: str) -> Set[str]:
    """Add or remove specific MCP tools from a server's exclude list.

    Returns the set of server names that were not found in config.
    """
    failed_servers: Set[str] = set()
    mcp_servers = config.get("mcp_servers") or {}

    for target in targets:
        server_name, tool_name = target.split(":", 1)
        if server_name not in mcp_servers:
            failed_servers.add(server_name)
            continue
        tools_cfg = mcp_servers[server_name].setdefault("tools", {})
        exclude = list(tools_cfg.get("exclude") or [])
        if action == "disable":
            if tool_name not in exclude:
                exclude.append(tool_name)
        else:
            exclude = [t for t in exclude if t != tool_name]
        tools_cfg["exclude"] = exclude

    return failed_servers


def _print_tools_list(enabled_toolsets: set, mcp_servers: dict, platform: str = "cli"):
    """Print a summary of enabled/disabled toolsets and MCP tool filters."""
    effective_all = _get_effective_configurable_toolsets()
    effective = [
        (k, l, d) for (k, l, d) in effective_all
        if _toolset_allowed_for_platform(k, platform)
    ]
    builtin_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}

    print(f"Built-in toolsets ({platform}):")
    for ts_key, label, _ in effective:
        if ts_key not in builtin_keys:
            continue
        status = (color("✓ enabled", Colors.GREEN) if ts_key in enabled_toolsets
                  else color("✗ disabled", Colors.RED))
        print(f"  {status}  {ts_key}  {color(label, Colors.DIM)}")

    # Plugin toolsets
    plugin_entries = [(k, l) for k, l, _ in effective if k not in builtin_keys]
    if plugin_entries:
        print()
        print(f"Plugin toolsets ({platform}):")
        for ts_key, label in plugin_entries:
            status = (color("✓ enabled", Colors.GREEN) if ts_key in enabled_toolsets
                      else color("✗ disabled", Colors.RED))
            print(f"  {status}  {ts_key}  {color(label, Colors.DIM)}")

    if mcp_servers:
        print()
        print("MCP servers:")
        for srv_name, srv_cfg in mcp_servers.items():
            tools_cfg = srv_cfg.get("tools") or {}
            exclude = tools_cfg.get("exclude") or []
            include = tools_cfg.get("include") or []
            if include:
                _print_info(f"{srv_name}  [include only: {', '.join(include)}]")
            elif exclude:
                _print_info(f"{srv_name}  [excluded: {color(', '.join(exclude), Colors.YELLOW)}]")
            else:
                _print_info(f"{srv_name}  {color('all tools enabled', Colors.DIM)}")


def tools_disable_enable_command(args):
    """Enable, disable, or list tools for a platform.

    Built-in toolsets use plain names (e.g. ``web``, ``memory``).
    MCP tools use ``server:tool`` notation (e.g. ``github:create_issue``).
    """
    action = args.tools_action
    platform = getattr(args, "platform", "cli")
    config = load_config()

    if platform not in PLATFORMS:
        _print_error(f"Unknown platform '{platform}'. Valid: {', '.join(PLATFORMS)}")
        return

    if action == "list":
        _print_tools_list(_get_platform_tools(config, platform, include_default_mcp_servers=False),
                          config.get("mcp_servers") or {}, platform)
        return

    targets: List[str] = args.names
    toolset_targets = [t for t in targets if ":" not in t]
    mcp_targets = [t for t in targets if ":" in t]

    valid_toolsets = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS} | _get_plugin_toolset_keys()
    unknown_toolsets = [t for t in toolset_targets if t not in valid_toolsets]
    if unknown_toolsets:
        for name in unknown_toolsets:
            _print_error(f"Unknown toolset '{name}'")
        toolset_targets = [t for t in toolset_targets if t in valid_toolsets]

    # Reject platform-scoped toolsets on platforms that don't allow them.
    restricted_targets = [
        t for t in toolset_targets
        if not _toolset_allowed_for_platform(t, platform)
    ]
    if restricted_targets:
        for name in restricted_targets:
            allowed = sorted(_TOOLSET_PLATFORM_RESTRICTIONS.get(name) or set())
            _print_error(
                f"Toolset '{name}' is not available on platform '{platform}' "
                f"(only: {', '.join(allowed)})"
            )
        toolset_targets = [t for t in toolset_targets if t not in restricted_targets]

    if toolset_targets:
        _apply_toolset_change(config, platform, toolset_targets, action)

    failed_servers: Set[str] = set()
    if mcp_targets:
        failed_servers = _apply_mcp_change(config, mcp_targets, action)
        for srv in failed_servers:
            _print_error(f"MCP server '{srv}' not found in config")

    save_config(config)

    successful = [
        t for t in targets
        if t not in unknown_toolsets and (":" not in t or t.split(":")[0] not in failed_servers)
    ]
    if successful:
        verb = "Disabled" if action == "disable" else "Enabled"
        _print_success(f"{verb}: {', '.join(successful)}")
