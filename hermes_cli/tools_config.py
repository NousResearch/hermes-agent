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

# ─── Post-Setup Hooks ─────────────────────────────────────────────────────────


def _cua_driver_cmd() -> str:
    """Return the configured cua-driver override, or the bare default name."""
    return os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip() or "cua-driver"


def _cua_version_summary(raw: str, *, limit: int = 120) -> str:
    """Reduce a driver's ``--version`` output to one short status line.

    A binary selected by ``HERMES_CUA_DRIVER_CMD`` is not obliged to answer
    ``--version`` the way cua-driver does. Pointing the override at, say,
    ``cmd.exe`` yields a multi-line banner plus a prompt, which used to be
    interpolated verbatim into ``cua-driver: installed at ... (<version>)``
    and shattered the one-line summary. Keep the first non-empty line and
    bound its length.
    """
    for line in (raw or "").splitlines():
        text = line.strip()
        if text:
            return text[:limit]
    return ""


def _resolved_cua_driver_cmd() -> Optional[str]:
    """Resolve cua-driver exactly as the runtime and Desktop status do."""
    from tools.computer_use.cua_backend import resolve_cua_driver_cmd

    return resolve_cua_driver_cmd()


def _cua_driver_env() -> dict:
    """cua-driver child env with the Hermes telemetry policy applied.

    Delegates to ``cua_backend.cua_driver_child_env`` (telemetry disabled by
    default; user opt-in via ``computer_use.cua_telemetry``). Falls back to the
    current environment if the helper can't be imported, so install/status
    never break on a telemetry-helper error.
    """
    try:
        from tools.computer_use.cua_backend import cua_driver_child_env

        return cua_driver_child_env()
    except Exception:
        return dict(os.environ)


_CUA_DRIVER_CONTRACT_CACHE: dict = {}


def _cua_driver_contract_status(binary: Optional[str] = None) -> dict:
    """Inspect whether an installed driver supports Hermes' runtime contract."""
    import time

    from tools.computer_use.cua_backend import cua_driver_runtime_contract_status

    resolved = binary or _resolved_cua_driver_cmd()
    if not resolved:
        return cua_driver_runtime_contract_status(None)
    try:
        stat = os.stat(resolved)
        fingerprint = (resolved, stat.st_mtime_ns, stat.st_size)
    except OSError:
        return cua_driver_runtime_contract_status(resolved)

    now = time.monotonic()
    if (
        _CUA_DRIVER_CONTRACT_CACHE.get("fingerprint") == fingerprint
        and now - _CUA_DRIVER_CONTRACT_CACHE.get("checked_at", 0.0) < 30.0
    ):
        return dict(_CUA_DRIVER_CONTRACT_CACHE["state"])

    state = cua_driver_runtime_contract_status(resolved)
    _CUA_DRIVER_CONTRACT_CACHE.update(
        fingerprint=fingerprint,
        checked_at=now,
        state=dict(state),
    )
    return state


def _cua_driver_install_ready() -> bool:
    """Return whether an existing driver needs no install-time repair."""
    if not _cua_driver_contract_status().get("ready"):
        return False
    if sys.platform == "win32":
        return _cua_driver_autostart_registered_windows()
    return True


def _pip_install(
    args: List[str],
    *,
    timeout: int = 300,
    capture_output: bool = True,
):
    """Install Python packages from a post-setup hook.

    Strategy (in order):
    1. ``uv pip install`` if uv is on PATH — fast, doesn't need pip in the venv.
    2. ``python -m pip install`` — works on stdlib venvs.
    3. ``python -m ensurepip --upgrade`` then retry pip — covers ``uv venv``
       which creates a venv WITHOUT pip.

    Why this exists: the Windows installer creates the venv via ``uv venv``,
    which doesn't seed pip. Post-setup hooks that shelled out to
    ``[sys.executable, '-m', 'pip', 'install', ...]`` failed with
    ``No module named pip`` on every fresh install. uv-first sidesteps that.

    Returns the ``subprocess.CompletedProcess`` from whichever tier succeeded
    (or the last failure for the caller to inspect).
    """
    venv_root = Path(sys.executable).parent.parent
    uv_env = {**os.environ, "VIRTUAL_ENV": str(venv_root)}

    # Managed uv first: $HERMES_HOME/bin is never on PATH, so a bare which()
    # misses the uv Hermes installed and prefers a system one when both exist.
    # ensure_uv() rather than a pure lookup because this runs during setup,
    # where installing uv is in scope — and tier 2 is a pip that the Windows
    # installer's `uv venv` does not seed, so failing to find uv here is the
    # difference between a working post-setup hook and "No module named pip".
    from hermes_cli.managed_uv import ensure_uv

    uv_bin = ensure_uv()
    if uv_bin:
        try:
            result = subprocess.run(
                [uv_bin, "pip", "install", *args],
                capture_output=capture_output, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                env=uv_env,
                creationflags=_post_setup_no_window_flags(
                    streams_to_console=not capture_output
                ),
            )
            if result.returncode == 0:
                return result
            # Fall through to pip — uv may have failed for an unrelated reason
            # (resolution conflict, network), and pip might handle it.
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    pip_cmd = [sys.executable, "-m", "pip"]
    try:
        # Probe for pip; bootstrap via ensurepip if missing (uv venv lacks it).
        probe = subprocess.run(
            pip_cmd + ["--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            creationflags=_post_setup_no_window_flags(),
        )
        if probe.returncode != 0:
            raise FileNotFoundError("pip not in venv")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, check=True,
                creationflags=_post_setup_no_window_flags(),
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Synthesize a result so callers see a clean failure path.
            return subprocess.CompletedProcess(
                pip_cmd, returncode=1, stdout="",
                stderr=f"pip not available and ensurepip failed: {e}",
            )

    return subprocess.run(
        pip_cmd + ["install", *args],
        capture_output=capture_output, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=_post_setup_no_window_flags(
            streams_to_console=not capture_output
        ),
    )



# The asset-probe that lived here used to hit `/releases/latest` on
# trycua/cua and inspect the release's asset list before piping the
# installer to bash. It was broken in two places:
#
#   1. cua-driver-rs releases are marked **prerelease** on every cut,
#      and GitHub's `/releases/latest` endpoint explicitly skips
#      prereleases. On the live trycua/cua repo today, `/releases/latest`
#      returns the Python `cua-agent v0.8.3` package (zero binary
#      assets) instead of `cua-driver-rs-v0.6.0` (19 binary assets).
#      The probe then reported "no asset for this arch" and skipped the
#      install on every non-arm64 host — Linux x86_64, Windows, macOS
#      Intel, Linux arm64 — even when the upstream installer would have
#      succeeded.
#   2. Even with the right endpoint, we'd be duplicating tag-resolution
#      logic the upstream installer already does correctly via
#      `CUA_DRIVER_RS_BAKED_VERSION` (auto-baked by CD on every release,
#      with an API fallback). Drift between our probe and theirs is a
#      maintenance hazard.
#
# Resolution: trust the upstream installer. For fresh installs, run
# install.sh directly — it errors clean if the target arch has no
# asset. For the upgrade path, `cua_driver_update_check()` (which calls
# `cua-driver check-update --json`) gives us the canonical update
# answer from the binary itself — same tag-resolution as the installer,
# no Python-side duplication.


def _cua_install_target_writable() -> bool:
    """Return whether the upstream installer can write its app bundle target."""
    if sys.platform != "darwin":
        return True
    applications_dir = "/Applications"
    try:
        if not os.path.isdir(applications_dir):
            return True
        return os.access(applications_dir, os.W_OK)
    except Exception:
        return True


def install_cua_driver(
    upgrade: bool = False,
    require_confirmed_update: bool = False,
    show_installer_progress: bool = True,
) -> bool:
    """Install or refresh the cua-driver binary used by Computer Use.

    The upstream installer always pulls the latest release tag, so re-running
    it is the canonical way to upgrade. We expose two modes:

    * ``upgrade=False`` — keep a compatible Cua Driver 0.20 installation,
      repair an old or incomplete installation, and install when missing.
      Used by the toolset enable flow.
    * ``upgrade=True`` — always re-run the installer (or call ``cua-driver
      update`` if the binary supports it). Used by ``hermes update`` and
      by ``hermes computer-use install --upgrade``.

    ``require_confirmed_update`` (only meaningful with ``upgrade=True`` and
    an installed binary): when the driver's native ``check-update`` verb
    can't positively confirm that a newer release exists — the driver is
    too old for the verb, the GitHub check failed, we're offline, or the
    probe timed out — keep the installed version and return instead of
    falling through to the full upstream installer. ``hermes update`` sets
    this so a broken update check costs seconds, not a multi-minute silent
    reinstall on every update (the upstream installer runs up to
    ``_CUA_INSTALLER_TIMEOUT`` and install.ps1's concurrency lock can add
    a further ~600s wait on Windows). ``hermes computer-use install
    --upgrade`` leaves it False — an explicit upgrade request should still
    reinstall when the check is indeterminate. On Windows this flag also
    defers contract REPAIRS and fresh INSTALLS to the explicit command
    (those paths can legitimately need a human: first-time autostart
    elevation, SmartScreen). Routine confirmed upgrades DO run, in
    unattended-safe mode: stdin closed, version pinned, lock/network
    preflights, and the shorter background ceiling below.

    ``show_installer_progress`` controls the installer's own progress line.
    ``hermes update`` already prints a contextual line before its update
    check, so it disables this to avoid printing the refresh twice.

    The confirmed-update path is also bounded by
    ``_CUA_BACKGROUND_UPDATE_TIMEOUT``. It runs as an optional, quiet part of
    ``hermes update`` and must not inherit the explicit install command's
    11-minute ceiling when an upstream prompt or UAC dialog is unattended.

    Returns True iff cua-driver is installed (or successfully refreshed)
    when the function returns. Supported on macOS, Windows, and Linux
    (Linux is alpha). Silently returns False on unsupported platforms.
    """
    import platform as _plat
    import shutil
    import subprocess

    system = _plat.system()
    if system not in ("Darwin", "Windows", "Linux"):
        if upgrade:
            # Silent on unsupported platforms — `hermes update` calls this
            # for every user; only macOS/Windows/Linux users care.
            return False
        _print_warning("    Computer Use (cua-driver) is unsupported on this platform; skipping.")
        return False

    is_windows = system == "Windows"
    is_linux = system == "Linux"

    # The Windows installer (install.ps1) is fetched via PowerShell's `irm`,
    # so it needs PowerShell rather than curl. macOS/Linux use curl | bash.
    fetch_tool = "powershell" if is_windows else "curl"

    driver_cmd = _cua_driver_cmd()
    binary = _resolved_cua_driver_cmd()

    # An explicit override is authoritative even when it is currently broken.
    # Do not install or replace the standard system driver: that cannot repair
    # the configured path and would mutate an unrelated installation.
    override = os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip()
    if override and not binary:
        _print_warning(
            "    HERMES_CUA_DRIVER_CMD does not resolve to an executable: "
            f"{override}"
        )
        _print_info(
            "    Fix or unset the override before running computer-use install."
        )
        return False

    # Not installed → fresh install path (only when caller asked for it).
    if not binary and not upgrade:
        if not _cua_install_target_writable():
            _print_info(
                "    /Applications is not writable; skipping cua-driver install."
            )
            _print_info(
                "    Run from an admin account or install cua-driver manually."
            )
            return False
        if not shutil.which(fetch_tool):
            _print_warning(f"    {fetch_tool} not found — install manually:")
            _print_info("      https://github.com/trycua/cua/blob/main/libs/cua-driver/README.md")
            return False
        # Pre-install asset probe deleted — see comment near the top of
        # tools_config.py for why. install.sh has CUA_DRIVER_RS_BAKED_VERSION
        # baked in by CD and errors cleanly on missing-arch assets.
        return _run_cua_driver_installer(label="Installing")

    # An installed driver that fails Hermes' runtime contract (version floor,
    # missing manifest verbs) is repaired regardless of the caller's mode.
    # Hermes' own minimum requirement IS the confirmation that an upgrade is
    # needed, so the ``upgrade=True`` path must not defer to the driver's
    # ``check-update`` verb here — a cached/indeterminate "no update" answer
    # would otherwise pin users on an unusable driver forever (observed:
    # 0.19.3 installs hard-failing every computer_use call after the 0.20
    # contract landed, with `hermes update` declining to refresh).
    contract = _cua_driver_contract_status(binary) if binary else None
    repair_existing = bool(binary and contract and not contract.get("ready"))

    # A compatible existing installation needs no download. Finish the small
    # host-specific setup that the upstream installer normally owns.
    if binary and not upgrade and not repair_existing:
        try:
            version = subprocess.run(
                [binary, "--version"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, env=_cua_driver_env(),
                creationflags=_post_setup_no_window_flags(),
            ).stdout.strip()
            _print_success(f"    {driver_cmd} already installed: {version or 'unknown version'}")
        except Exception:
            _print_success(f"    {driver_cmd} already installed.")
        if is_windows:
            if not _repair_cua_driver_autostart_windows(binary, verbose=False):
                _print_warning(
                    "    cua-driver is compatible, but Windows autostart repair failed."
                )
                return False
            _print_info("    cua-driver may spawn a UIAccess worker (cua-driver-uia.exe);")
            _print_info("    Windows/SmartScreen may prompt the first time it runs.")
        elif is_linux:
            _print_warning("    Linux support is alpha.")
        else:
            _print_info("    Grant macOS permissions if not done yet:")
            _print_info("      System Settings > Privacy & Security > Accessibility")
            _print_info("      System Settings > Privacy & Security > Screen Recording")
        return True

    if repair_existing:
        version = contract.get("version") or "unknown version"
        reason = contract.get("reason") or "required runtime features are missing"
        _print_warning(
            f"    Found cua-driver {version}, but Hermes cannot use its current "
            f"runtime contract: {reason}."
        )
        if os.environ.get("HERMES_CUA_DRIVER_CMD", "").strip():
            _print_info(
                "    Update the binary selected by HERMES_CUA_DRIVER_CMD, or unset "
                "the override and run: hermes computer-use install --upgrade"
            )
            return False
        if is_windows and require_confirmed_update:
            _print_info(
                "    Automatic Windows updates cannot safely run cua-driver's "
                "interactive repair installer."
            )
            _print_info(
                "    Repair it from an interactive terminal with: "
                "hermes computer-use install --upgrade"
            )
            return False
        _print_info("    Repairing it with the current upstream installer.")

    # upgrade=True path — refresh to the latest upstream release.
    if not _cua_install_target_writable():
        _print_info(
            "    /Applications is not writable; skipping cua-driver refresh."
        )
        _print_info(
            "    Run `hermes computer-use install --upgrade` from an admin account to update it."
        )
        return bool(binary)

    if not shutil.which(fetch_tool):
        _print_warning(f"    {fetch_tool} not found — cannot refresh cua-driver.")
        return bool(binary)

    # Pre-install asset probe deleted (see top-of-file comment). The
    # `cua_driver_update_check()` call further down asks the installed
    # cua-driver binary itself whether an update exists — same
    # tag-resolution as the installer, no duplication.

    # Skip the (network) re-install when the driver itself reports it's already
    # on the latest release. Best-effort: an older driver (no check-update
    # verb) or an offline check returns None. What happens then depends on the
    # caller: `hermes update` (require_confirmed_update=True) keeps the
    # installed version — an indeterminate check must never cost the user a
    # multi-minute silent reinstall on every update. An explicit
    # `hermes computer-use install --upgrade` falls through and re-runs the
    # installer as before.
    confirmed_version = None
    if binary and not repair_existing:
        _state = None
        try:
            from tools.computer_use.cua_backend import cua_driver_update_check
            _state = cua_driver_update_check()
        except Exception:
            _state = None
        if _state is not None and not _state.get("update_available"):
            _print_success(
                f"    {driver_cmd} is already on the latest release "
                f"({_state.get('current_version') or 'unknown'})."
            )
            return True
        if _state is None and require_confirmed_update:
            _print_info(
                f"    Could not confirm a newer {driver_cmd} release "
                "(offline, rate-limited, or driver too old to check); "
                "keeping the installed version."
            )
            _print_info(
                "    Force a refresh with: hermes computer-use install --upgrade"
            )
            return True
        if _state is not None and _state.get("update_available"):
            # Windows routine upgrades run UNATTENDED-SAFE rather than
            # deferring: stdin is closed (a consent Read-Host can't block),
            # the version is pinned, the ceiling is
            # _CUA_BACKGROUND_UPDATE_TIMEOUT, and _run_cua_driver_installer's
            # preflights skip in seconds when the install lock is held or
            # GitHub is unreachable. Only contract repairs and fresh installs
            # stay interactive-only (guards above/below) — those are the
            # paths where upstream legitimately needs a human (first-time
            # autostart elevation, SmartScreen).
            # Pin the installer to the release check-update just confirmed.
            # `latest_version` comes from the GitHub Releases API, so its
            # assets are published — unlike the installer script's baked
            # version on `main`, which Release Please bumps in the release
            # PR *before* the release assets exist. Installing unpinned in
            # that window 404s (observed: baked 0.14.0 vs latest published
            # 0.13.1). Malformed values are ignored → unpinned fallback.
            import re as _re

            _latest = str(_state.get("latest_version") or "").strip().lstrip("vV")
            if _re.fullmatch(r"\d+(\.\d+)*", _latest):
                confirmed_version = _latest

    if is_windows and require_confirmed_update and not binary:
        # Missing-binary path (driver enabled in config but never installed,
        # or wiped by a failed install). Same rule as the repair and
        # confirmed-update branches above: an automatic Windows update must
        # never launch install.ps1, which can demand console/UAC consent the
        # hidden updater cannot provide (#87703).
        _print_info(
            "    cua-driver is not installed; automatic Windows updates "
            "cannot safely run its interactive installer."
        )
        _print_info(
            "    Install it from an interactive terminal with: "
            "hermes computer-use install --upgrade"
        )
        return False

    if binary:
        # Show before/after version when we have a baseline. Best-effort.
        try:
            before = subprocess.run(
                [binary, "--version"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, env=_cua_driver_env(),
                creationflags=_post_setup_no_window_flags(),
            ).stdout.strip()
        except Exception:
            before = ""
    else:
        before = ""

    ok = _run_cua_driver_installer(
        label="Repairing" if repair_existing else "Refreshing",
        verbose=False,
        pin_version=confirmed_version,
        show_progress=show_installer_progress,
        installer_timeout=(
            _CUA_BACKGROUND_UPDATE_TIMEOUT
            if require_confirmed_update
            else None
        ),
    )
    if ok and repair_existing:
        repaired = _cua_driver_contract_status()
        if not repaired.get("ready"):
            _print_warning(
                "    cua-driver was reinstalled, but its runtime contract is still "
                f"unusable: {repaired.get('reason') or 'unknown error'}."
            )
            _print_info("    Run: hermes computer-use doctor")
            return False
    if ok and before:
        try:
            after = subprocess.run(
                [binary, "--version"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, env=_cua_driver_env(),
                creationflags=_post_setup_no_window_flags(),
            ).stdout.strip()
            if after and after != before:
                _print_success(f"    {driver_cmd} upgraded: {before} → {after}")
            elif after:
                _print_info(f"    {driver_cmd} up to date: {after}")
        except Exception:
            pass
    return ok


# Ceiling for one upstream-installer run. Must exceed the installer's own
# stale-lock recovery window: _install-rust.sh serializes concurrent installs
# with a lock dir at ~/.cua-driver/packages/.install.lock.d and only
# force-releases a dead holder's lock after LOCK_STALE_AFTER_SECONDS=600 of
# waiting. With a shorter Python-side timeout, a stale lock means every run
# gets killed before the installer's recovery can fire — a permanent
# "always times out" wedge (issue #58762). 660s = 600s lock window + 60s
# headroom for the actual download/swap.
_CUA_INSTALLER_TIMEOUT = 660

# Grace period for draining the installer's pipes after a timeout kill. The
# kill is best-effort (see _reap_after_timeout), so this drain has to be
# bounded: a descendant that survived the kill still holds the inherited
# stdout handle, and an unbounded read waits on an EOF that never comes,
# which turns the ceiling above into no ceiling at all (issue #87703). A
# successful kill closes the pipe immediately, so this costs nothing in the
# normal case; it only caps how long a failed one can stall the update.
_CUA_INSTALLER_DRAIN_GRACE = 15

# Optional refreshes launched by ``hermes update`` are quiet and unattended.
# Keep their interruption bounded even when upstream waits on Read-Host or a
# consent prompt. Explicit ``computer-use install --upgrade`` runs retain the
# full installer ceiling above. (The lock/network preflights below make a
# legitimate long wait impossible on this path, so a short ceiling is safe.)
_CUA_BACKGROUND_UPDATE_TIMEOUT = 120

# Upstream installer's stale-lock threshold (LOCK_STALE_AFTER_SECONDS in
# _install-rust.sh). Used by the pre-clear below to avoid yanking a lock
# that a live-but-slow install still holds.
_CUA_LOCK_STALE_AFTER = 600


def _cua_install_home() -> "Path":
    """Package home shared by the upstream POSIX and Windows installers."""
    return Path(
        os.environ.get("CUA_DRIVER_RS_HOME")
        or str(Path.home() / ".cua-driver")
    )


def _cua_install_lock_dir() -> "Path":
    """Path of the upstream installer's concurrent-install lock dir."""
    return _cua_install_home() / "packages" / ".install.lock.d"


def _cua_windows_install_lock_file() -> "Path":
    """Path of install.ps1's FileShare::None lock file."""
    return _cua_install_home() / "install.lock"


def _clear_stale_windows_cua_install_lock() -> None:
    """Delete install.ps1's lock file only when no process still holds it.

    ``install.ps1`` serializes installs with a ``FileStream`` opened using
    ``FileShare::None``. Mirror that primitive with a zero-share
    ``CreateFileW`` probe. ``FILE_FLAG_DELETE_ON_CLOSE`` removes an unlocked
    leftover atomically when the probe handle closes, avoiding a gap where a
    new installer could acquire the file between our probe and deletion.
    """
    lock_file = _cua_windows_install_lock_file()
    try:
        if not lock_file.is_file():
            return

        import ctypes as _ctypes
        from ctypes import wintypes as _wintypes

        # Win32 constants used by install.ps1's FileShare::None equivalent.
        delete_access = 0x00010000
        generic_read = 0x80000000
        generic_write = 0x40000000
        open_existing = 3
        file_attribute_normal = 0x00000080
        file_flag_delete_on_close = 0x04000000

        kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            _wintypes.LPCWSTR,
            _wintypes.DWORD,
            _wintypes.DWORD,
            _wintypes.LPVOID,
            _wintypes.DWORD,
            _wintypes.DWORD,
            _wintypes.HANDLE,
        ]
        create_file.restype = _wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [_wintypes.HANDLE]
        close_handle.restype = _wintypes.BOOL

        handle = create_file(
            str(lock_file),
            generic_read | generic_write | delete_access,
            0,  # FileShare::None
            None,
            open_existing,
            file_attribute_normal | file_flag_delete_on_close,
            None,
        )
        invalid_handle = _wintypes.HANDLE(-1).value
        if handle == invalid_handle:
            logger.debug(
                "Windows cua install lock at %s is still held or cannot be "
                "removed (winerror %s)",
                lock_file,
                _ctypes.get_last_error(),
            )
            return

        if not close_handle(handle):
            logger.debug(
                "could not close Windows cua install lock probe at %s "
                "(winerror %s)",
                lock_file,
                _ctypes.get_last_error(),
            )
            return
        if lock_file.exists():
            logger.debug(
                "Windows cua install lock probe succeeded but %s remains",
                lock_file,
            )
            return

        logger.info("Cleared stale Windows cua-driver install lock at %s", lock_file)
        _print_info(f"    Cleared stale cua-driver install lock ({lock_file}).")
    except Exception as e:
        logger.debug("stale Windows cua install lock check failed: %s", e)


def _clear_stale_cua_install_lock() -> None:
    """Best-effort: remove a stale installer lock left by a dead holder.

    The POSIX installer stamps its holder pid into
    ``~/.cua-driver/packages/.install.lock.d/info``. The Windows installer
    instead holds ``~/.cua-driver/install.lock`` open with
    ``FileShare::None``. Clear either artifact up front only when its
    platform-specific liveness check proves that no install still holds it.
    """
    if sys.platform == "win32":
        _clear_stale_windows_cua_install_lock()
        return
    lock_dir = _cua_install_lock_dir()
    try:
        if not lock_dir.is_dir():
            return
        holder_pid = None
        info = lock_dir / "info"
        try:
            for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("pid="):
                    holder_pid = int(line.split("=", 1)[1].strip())
                    break
        except (OSError, ValueError):
            holder_pid = None

        if holder_pid is not None:
            try:
                os.kill(holder_pid, 0)  # windows-footgun: ok — function early-returns on win32
                # Holder alive → a concurrent install is running; don't touch.
                return
            except ProcessLookupError:
                pass  # dead holder → stale, clear below
            except PermissionError:
                # Alive but owned by someone else — treat as live.
                return
        else:
            # No readable pid. Only clear if the lock is old enough that the
            # upstream installer itself would consider it reclaimable.
            import time as _time
            try:
                age = _time.time() - lock_dir.stat().st_mtime
            except OSError:
                return
            if age < _CUA_LOCK_STALE_AFTER:
                return

        import shutil as _shutil
        _shutil.rmtree(lock_dir, ignore_errors=True)
        logger.info("Cleared stale cua-driver install lock at %s", lock_dir)
        _print_info(f"    Cleared stale cua-driver install lock ({lock_dir}).")
    except Exception as e:
        logger.debug("stale cua install lock check failed: %s", e)


def _cua_install_lock_held() -> bool:
    """True when the upstream installer's lock is held by a LIVE process.

    Called after ``_clear_stale_cua_install_lock()``: anything provably
    stale is already gone, so a surviving lock artifact means a concurrent
    (or orphaned-but-alive) install owns it. Upstream waits up to
    ``LOCK_STALE_AFTER_SECONDS=600`` on a held lock before probing —
    unattended refreshes must not eat that wait (the 11-minute hang class,
    #87703): they skip instead. Best-effort: unreadable state reports
    not-held so a probe failure can never block an install.
    """
    try:
        if sys.platform == "win32":
            lock_file = _cua_windows_install_lock_file()
            if not lock_file.is_file():
                return False
            # install.ps1 holds the file open with FileShare::None — any
            # open attempt fails with a sharing violation while it's held.
            # _clear_stale_windows_cua_install_lock() already deleted it if
            # it was unheld, so surviving = held; confirm with an open probe.
            try:
                with open(lock_file, "r+b"):
                    return False  # opened fine → not held (racy leftover)
            except PermissionError:
                return True
            except OSError:
                return True
        lock_dir = _cua_install_lock_dir()
        return lock_dir.is_dir()
    except Exception as e:
        logger.debug("cua install lock probe failed: %s", e)
        return False


def _cua_release_endpoint_reachable(timeout: float = 5.0) -> bool:
    """Fast probe: can we reach GitHub's release download host at all?

    The upstream installers (install.ps1 / _install-rust.sh) download from
    ``github.com/<repo>/releases/download/...``. When that host is
    unreachable (outage, DNS, firewall), the installer dies slowly inside
    its own retries while the unattended refresh eats the whole ceiling.
    A 5s HEAD tells us in seconds. Only a *connection-level* failure counts
    as unreachable — any HTTP response (including 4xx/5xx) proves the path
    works and lets the installer make its own decisions.
    """
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://github.com/trycua/cua/releases", method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True  # server answered → reachable
    except Exception as e:
        logger.debug("cua release endpoint probe failed: %s", e)
        return False


def _ps_single_quote(value: str) -> str:
    """Return a PowerShell single-quoted string literal."""
    return "'" + value.replace("'", "''") + "'"


def _cua_driver_autostart_registered_windows() -> bool:
    """Return whether the Windows cua-driver scheduled task is registered."""
    if sys.platform != "win32":
        return False
    import subprocess

    try:
        result = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", "cua-driver-serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0


def _repair_cua_driver_autostart_windows(driver_cmd: str, *, verbose: bool) -> bool:
    """Best-effort repair for Windows installer autostart quoting failures.

    Older install.ps1 builds invoked
    ``& C:\\Users\\Name With Spaces\\...\\cua-driver`` from an elevated
    PowerShell command string, which PowerShell split at the first space. If
    the installer left the scheduled task missing, retry by
    launching the resolved binary through Start-Process's structured
    ``-FilePath`` / ``-ArgumentList`` parameters instead of interpolating a
    path into a command string.
    """
    if sys.platform != "win32":
        return True
    if _cua_driver_autostart_registered_windows():
        return True

    import subprocess

    binary = shutil.which(driver_cmd)
    if not binary:
        return False

    ps = shutil.which("powershell") or shutil.which("powershell.exe") or "powershell"
    ps_cmd = (
        f"$exe = {_ps_single_quote(binary)}; "
        "$proc = Start-Process -FilePath $exe "
        "-ArgumentList @('autostart','enable') "
        "-Verb RunAs -Wait -PassThru -ErrorAction Stop; "
        "exit $proc.ExitCode"
    )

    if verbose:
        _print_info("    Registering cua-driver auto-start...")
    else:
        _print_info("    Repairing cua-driver auto-start registration...")

    try:
        result = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=_cua_driver_env(),
        )
    except subprocess.TimeoutExpired:
        _print_warning("    cua-driver autostart registration timed out.")
        return False
    except Exception as exc:
        _print_warning(f"    cua-driver autostart registration failed: {exc}")
        return False

    if result.returncode == 0:
        return True

    tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
    _print_warning("    cua-driver autostart registration failed.")
    for line in tail:
        _print_info(f"      {line[:200]}")
    _print_info("    From an elevated shell, run: cua-driver autostart enable")
    return False


def _run_cua_driver_installer(
    label: str = "Installing",
    verbose: bool = True,
    pin_version: Optional[str] = None,
    show_progress: bool = True,
    installer_timeout: Optional[float] = None,
) -> bool:
    """Run the upstream cua-driver installer for this platform.

    The scripts are idempotent: they always download the latest release, so
    re-running on an already-installed system performs an upgrade.

    * macOS / Linux → ``curl -fsSL …/install.sh | /bin/bash``.
    * Windows       → ``powershell -NoProfile -ExecutionPolicy Bypass -Command
      "irm …/install.ps1 | iex"``.

    ``pin_version`` (e.g. ``"0.13.1"``) is exported as
    ``CUA_DRIVER_RS_VERSION`` so the installer downloads that exact release
    instead of its baked-in default. The baked version on upstream ``main``
    is bumped by Release Please *before* the release assets are published,
    so an unpinned run inside that window fails with a 404; pinning to the
    version ``check-update`` confirmed sidesteps the race entirely.

    ``installer_timeout`` lets quiet callers use a shorter ceiling without
    weakening the explicit install path's stale-lock recovery window.
    """
    import platform as _plat
    import shutil
    import subprocess

    system = _plat.system()
    is_windows = system == "Windows"
    is_linux = system == "Linux"

    if is_windows:
        # Mirror the one-liner printed by cua_driver_install_hint().
        ps_oneliner = (
            "irm https://raw.githubusercontent.com/trycua/cua/main/"
            "libs/cua-driver/scripts/install.ps1 | iex"
        )
        install_cmd = [
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-Command", ps_oneliner,
        ]
        manual_hint = (
            'powershell -NoProfile -ExecutionPolicy Bypass -Command '
            f'"{ps_oneliner}"'
        )
        script_path = None
    else:
        # Download-then-exec instead of `bash -c "$(curl …)"`: no shell=True,
        # no command substitution, and the script lands in a mkstemp file
        # (unpredictable name, 0600) rather than a fixed /tmp path — avoiding
        # both the shell-injection surface and a symlink/TOCTOU race on
        # multi-user machines. The manual hint stays the upstream one-liner
        # since that's what the docs/README teach.
        import tempfile as _tempfile

        install_url = (
            "https://raw.githubusercontent.com/trycua/cua/main/"
            "libs/cua-driver/scripts/install.sh"
        )
        manual_hint = f'/bin/bash -c "$(curl -fsSL {install_url})"'
        fd, script_path = _tempfile.mkstemp(prefix="cua-driver-install-", suffix=".sh")
        os.close(fd)
        try:
            dl = subprocess.run(
                ["curl", "-fsSL", "-o", script_path, install_url],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            _print_warning(f"    cua-driver installer download failed: {e}")
            try:
                os.remove(script_path)
            except OSError:
                pass
            return False
        if dl.returncode != 0:
            _print_warning(
                "    cua-driver installer download failed: "
                f"{(dl.stderr or '').strip()[:200]}"
            )
            try:
                os.remove(script_path)
            except OSError:
                pass
            return False
        install_cmd = ["/bin/bash", script_path]
    use_shell = False

    if show_progress:
        if verbose:
            _print_info(f"    {label} cua-driver (background computer-use)...")
        else:
            _print_info(f"→ {label} cua-driver (Computer Use)...")
    driver_cmd = _cua_driver_cmd()
    timeout = (
        _CUA_INSTALLER_TIMEOUT
        if installer_timeout is None
        else installer_timeout
    )

    installer_env = _cua_driver_env()
    if pin_version:
        # Both upstream installers (install.sh and install.ps1) honour
        # CUA_DRIVER_RS_VERSION over their baked default.
        installer_env["CUA_DRIVER_RS_VERSION"] = pin_version

    # A previous timed-out install can leave the upstream installer's
    # concurrent-install lock behind; clear it when provably stale so the
    # refresh doesn't wedge waiting on a dead holder (issue #58762).
    _clear_stale_cua_install_lock()

    # Unattended refreshes (installer_timeout set by `hermes update`) fail
    # FAST on the two conditions that otherwise consume the whole ceiling:
    #
    # 1. Install lock held by a live process — upstream would poll it for up
    #    to LOCK_STALE_AFTER_SECONDS=600 before probing the holder. That is
    #    the 11-minute silent hang class (#87703; observed live 2026-08-25:
    #    "cua-driver refreshing timed out after 660s"). Skip in ~0s instead.
    # 2. Release host unreachable (outage/DNS/firewall) — the installer
    #    would die slowly inside its own retries. A 5s HEAD answers now.
    #
    # Explicit `computer-use install --upgrade` runs keep upstream's full
    # lock-recovery semantics — a human is watching and can wait or Ctrl-C.
    if installer_timeout is not None:
        if _cua_install_lock_held():
            _print_info(
                "    Another cua-driver install is in progress (upstream "
                "install lock is held) — skipping this refresh."
            )
            _print_info(
                "    If no install is really running, retry with: "
                "hermes computer-use install --upgrade"
            )
            return False
        if not _cua_release_endpoint_reachable():
            _print_info(
                "    github.com is unreachable — skipping cua-driver "
                "refresh (will retry on the next update)."
            )
            return False
        if is_windows:
            # -NoAutoStart skips Register-CuaDriverAutostart entirely — the
            # ONLY branch of install.ps1 that self-elevates (UAC). Cost: an
            # existing cua-driver-serve task keeps pointing at the previous
            # binary until the next interactive upgrade re-registers it.
            # scriptblock invocation (instead of `| iex`) is what lets us
            # pass the parameter to a piped script.
            install_cmd = [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-Command",
                "$sc = irm https://raw.githubusercontent.com/trycua/cua/"
                "main/libs/cua-driver/scripts/install.ps1; "
                "& ([scriptblock]::Create($sc)) -NoAutoStart",
            ]

    # POSIX: run the installer in its own process group so a timeout kill
    # takes out the whole `curl | bash` pipeline (and the exec'd
    # _install-rust.sh), not just the outer shell. Otherwise the surviving
    # grandchildren keep holding the install lock, wedging every later run.
    popen_kwargs = {}
    if not is_windows:
        popen_kwargs["start_new_session"] = True

    def _kill_installer_tree(proc):
        import signal as _signal
        try:
            if not is_windows:
                os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)  # windows-footgun: ok — POSIX branch only
            else:
                # PowerShell may leave download/install helpers alive after its
                # direct process is killed. Those descendants inherit stdout
                # and can keep both communicate() and install.lock wedged, so
                # collect the tree first and kill it leaf-up.
                import psutil as _psutil

                try:
                    parent = _psutil.Process(proc.pid)
                    descendants = parent.children(recursive=True)
                except _psutil.NoSuchProcess:
                    return
                except _psutil.Error as e:
                    logger.debug(
                        "could not enumerate cua-driver installer tree for pid %s: %s",
                        proc.pid,
                        e,
                    )
                    proc.kill()
                    return

                for child in reversed(descendants):
                    try:
                        child.kill()
                    except _psutil.NoSuchProcess:
                        pass
                    except _psutil.Error as e:
                        logger.debug(
                            "could not kill cua-driver installer child pid %s: %s",
                            child.pid,
                            e,
                        )
                try:
                    parent.kill()
                except _psutil.NoSuchProcess:
                    pass
                except _psutil.Error as e:
                    logger.debug(
                        "could not kill cua-driver installer parent pid %s: %s",
                        proc.pid,
                        e,
                    )
                    proc.kill()
        except (OSError, ProcessLookupError):
            proc.kill()

    def _reap_after_timeout(proc):
        """Kill the installer tree, then drain its pipes under a deadline.

        ``_kill_installer_tree`` is best-effort by construction: every
        ``psutil.Error`` it can raise is logged at debug level and stepped
        over, on the reasoning that a partly-killed tree beats none. The case
        that matters is an ``install.ps1`` which self-elevated through
        ``Start-Process -Verb RunAs``: that descendant runs at High integrity,
        a medium-integrity kill gets ``AccessDenied``, and the survivor is
        still holding the ``stdout=PIPE`` write handle it inherited.

        Draining with no deadline then blocks on an EOF that only arrives when
        someone kills that process by hand, so the ``_CUA_INSTALLER_TIMEOUT``
        ceiling stops bounding anything and ``hermes update`` hangs past its
        own timeout warning (#87703). Bound the drain instead: a kill that
        landed closes the pipe at once, and one that did not costs
        ``_CUA_INSTALLER_DRAIN_GRACE`` rather than forever. The caller
        re-raises the original ``TimeoutExpired`` either way, so the manual
        re-run hint still prints and the update unwinds. Losing the tail of a
        timed-out installer's log is the cheaper half of that trade.
        """
        _kill_installer_tree(proc)
        try:
            drained_out, _ = proc.communicate(timeout=_CUA_INSTALLER_DRAIN_GRACE)
            # Diagnosability (#87703 post-mortem): the partial output names
            # WHERE the installer was stuck (lock wait, consent prompt,
            # download) — before this, the answer died with the process and
            # the timeout line was unactionable.
            if drained_out:
                logger.warning(
                    "cua-driver installer timed out; last output before "
                    "kill:\n%s",
                    drained_out[-2000:],
                )
        except subprocess.TimeoutExpired:
            # Deliberately not closing proc.stdout here. communicate()'s
            # reader threads are still blocked on that handle and closing it
            # underneath them races; they are daemon threads, so abandoning
            # them does not keep the interpreter alive.
            logger.debug(
                "cua-driver installer pipes still open %ss after the kill — "
                "abandoning the drain, a surviving descendant holds the "
                "inherited handle",
                _CUA_INSTALLER_DRAIN_GRACE,
            )
        except (OSError, ValueError) as e:
            logger.debug("cua-driver installer drain failed: %s", e)

    try:
        # When not verbose (e.g. `hermes update`'s refresh), capture the
        # installer's chatty "Next steps" wall instead of dumping it to the
        # terminal. The combined output is logged so a failure stays
        # debuggable. Verbose installs (interactive `computer-use install`)
        # keep streaming live.
        if verbose:
            proc = subprocess.Popen(
                install_cmd, shell=use_shell, env=installer_env,
                creationflags=_post_setup_no_window_flags(streams_to_console=True),
                **popen_kwargs
            )
            try:
                proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _reap_after_timeout(proc)
                raise
            result = subprocess.CompletedProcess(
                install_cmd, proc.returncode, stdout=None, stderr=None
            )
        else:
            proc = subprocess.Popen(
                install_cmd, shell=use_shell, env=installer_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=_post_setup_no_window_flags(),
                **popen_kwargs
            )
            try:
                out, _ = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _reap_after_timeout(proc)
                raise
            result = subprocess.CompletedProcess(
                install_cmd, proc.returncode, stdout=out, stderr=None
            )
            # Preserve the full installer output. During `hermes update`,
            # sys.stdout is the mirroring _UpdateOutputStream whose `_log`
            # handle is ~/.hermes/logs/update.log — write straight to it so
            # the captured "Next steps" wall is kept in full (success AND
            # failure), without echoing it to the terminal.
            if result.stdout:
                _update_log = getattr(sys.stdout, "_log", None)
                if _update_log is not None:
                    try:
                        _update_log.write(
                            "\n--- cua-driver installer output ---\n"
                            + result.stdout
                            + "\n"
                        )
                        _update_log.flush()
                    except Exception:
                        pass
                if result.returncode != 0:
                    logger.debug("cua-driver installer output:\n%s", result.stdout)
        installed_binary = _resolved_cua_driver_cmd()
        if result.returncode == 0 and installed_binary:
            if is_windows and not _repair_cua_driver_autostart_windows(
                installed_binary, verbose=verbose
            ):
                _print_warning(
                    "    cua-driver installed, but auto-start was not registered."
                )
            if verbose:
                _print_success(f"    {driver_cmd} installed.")
                if is_windows:
                    _print_info("    cua-driver may spawn a UIAccess worker (cua-driver-uia.exe);")
                    _print_info("    Windows/SmartScreen may prompt the first time it runs.")
                elif is_linux:
                    _print_warning("    Linux support is alpha.")
                else:
                    _print_info("    IMPORTANT — grant macOS permissions now:")
                    _print_info("      System Settings > Privacy & Security > Accessibility")
                    _print_info("      System Settings > Privacy & Security > Screen Recording")
                    _print_info("    Both must allow the terminal / Hermes process.")
            return True
        _print_warning(f"    cua-driver {label.lower()} did not complete. Re-run manually:")
        _print_info(f"      {manual_hint}")
        return False
    except subprocess.TimeoutExpired:
        _print_warning(
            f"    cua-driver {label.lower()} timed out after "
            f"{timeout}s."
        )
        if not is_windows:
            _print_info(
                "    If this repeats, a stale installer lock may be present — "
                f"check {_cua_install_lock_dir()}"
            )
        _print_info(f"    Re-run manually:  {manual_hint}")
        return False
    except Exception as e:
        _print_warning(f"    cua-driver {label.lower()} failed: {e}")
        return False
    finally:
        if script_path:
            try:
                os.remove(script_path)
            except OSError:
                pass


def _ensure_browser_use_cli(*, verbose_hints: bool = False) -> None:
    """Install the Browser Use CLI if it isn't already runnable.

    The Browser Use CLI 3.0 is the primary driver engine for EVERY browser
    backend except Camofox (which is Firefox-based with no CDP surface, so
    the CDP-only browser-use harness cannot drive it). Local, Browserbase,
    Firecrawl, and the Nous-managed cloud rows all execute through
    ``browser_exec`` when the CLI is runnable — so every one of those
    picker selections must attempt this install, not just the explicit
    "Browser Use" row. Failure is non-fatal: ``browser_exec`` can still run
    zero-install via ``uvx browser-use``, and the built-in browser tools
    remain the final fallback.

    MANAGED-FIRST: a browser-use on the user's PATH does NOT satisfy this
    check — only the Hermes-managed ``$HERMES_HOME/bin`` copy does.
    ``install_cli()`` short-circuits on the managed copy and otherwise
    provisions it, so resolution always lands on a binary Hermes installs
    and updates rather than a user-level side install.
    """
    _print_info("    Ensuring browser-use CLI (managed install)...")
    try:
        from tools.browser_use_cli import install_cli

        ok, message = install_cli()
    except Exception as exc:  # pragma: no cover — defensive
        ok, message = False, f"install failed: {exc}"
    if ok:
        _print_success(f"    {message}")
    else:
        for line in str(message).splitlines():
            _print_warning(f"    {line[:200]}")
        if shutil.which("uvx"):
            _print_info("    Falling back to zero-install runs via `uvx browser-use`")
        else:
            _print_info("    Install manually: uv tool install browser-use  (https://docs.astral.sh/uv/)")
    if verbose_hints:
        _print_info("    Local Chrome needs remote debugging: chrome://inspect/#remote-debugging")
        _print_info("    Cloud browsers: browser-use auth login  (or set BROWSER_USE_API_KEY)")


def _run_post_setup(post_setup_key: str):
    """Run post-setup hooks for tools that need extra installation steps."""
    from hermes_constants import find_node_executable

    if post_setup_key in {"agent_browser", "browserbase"}:
        # Every non-Camofox browser backend drives through the Browser Use
        # CLI when it's runnable — install it here too, not only on the
        # explicit "Browser Use" picker row.
        _ensure_browser_use_cli()
        # agent-browser is no longer a root package.json dependency (#43564)
        # — it resolves lazily via npx (or a global/Hermes-managed install)
        # instead of a local `npm install`, so there's no node_modules/
        # population step here anymore.
        try:
            # Import lazily so the tools_config UI doesn't pull in the full
            # browser_tool module at import time.
            from tools.browser_tool import (
                _chromium_installed,
                _running_in_docker,
                _find_agent_browser,
                _resolve_npx_bin,
                _is_npx_agent_browser_sentinel,
                AGENT_BROWSER_NPX_SPEC,
            )
        except Exception as exc:  # pragma: no cover — defensive
            _print_warning(f"    Could not check Chromium status: {exc}")
            return

        # Reuse the same resolution cascade browser tools use at runtime
        # (PATH -> Homebrew/Hermes-managed node -> npx) rather than a bare
        # shutil.which — Hermes-managed-Node-only setups resolve agent-browser
        # / npx only through the extended fallback path, which a bare
        # shutil.which("npx") lookup misses.
        try:
            browser_cmd = _find_agent_browser(validate=False)
        except FileNotFoundError:
            _print_warning(
                "    npx not found - browser tools require Node.js: https://nodejs.org"
            )
            return

        # Step 1: only the local browser provider actually needs Chromium on
        # disk. Cloud providers (Browserbase, Browser Use, Firecrawl) host
        # their own Chromium and don't need the local install.
        if post_setup_key != "agent_browser":
            return

        # Step 2: ensure the Chromium / headless-shell build agent-browser
        # drives is actually installed. Without it the CLI hangs on first
        # use until the command timeout fires. Skip inside Docker — the
        # image bakes Chromium in at build time, and runtime users usually
        # can't write to PLAYWRIGHT_BROWSERS_PATH anyway.
        if _chromium_installed():
            _print_success("    Chromium browser already installed, nothing to do")
            return

        if _running_in_docker():
            _print_warning(
                "    Chromium is missing but you're running in Docker."
            )
            _print_info(
                "    Pull the latest image to get the bundled Chromium:"
            )
            _print_info(
                "      docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
            return

        # browser_cmd was already resolved above (same PATH -> Homebrew ->
        # Hermes-managed-node -> npx cascade _find_agent_browser uses at
        # runtime), so this can't diverge from what actually gets invoked.
        if _is_npx_agent_browser_sentinel(browser_cmd):
            # Re-resolve via the same PATH + extended-PATH cascade
            # _find_agent_browser used, rather than a bare shutil.which("npx")
            # — Hermes-managed-Node-only setups resolve npx only through the
            # extended fallback path, and a bare lookup here would silently
            # diverge and hand subprocess.run a None argument.
            npx_bin = _resolve_npx_bin()
            if not npx_bin:
                _print_warning(
                    "    npx not found - install Chromium manually: npx agent-browser install --with-deps"
                )
                return
            install_cmd = [npx_bin, "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install", "--with-deps"]
        else:
            install_cmd = [browser_cmd, "install", "--with-deps"]

        _print_info("    Installing Chromium (~170MB one-time download)...")
        import subprocess
        try:
            result = subprocess.run(
                install_cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT), timeout=600,
                creationflags=_post_setup_no_window_flags(),
            )
            if result.returncode == 0:
                _print_success("    Chromium installed")
                # Invalidate the cached "missing" result so subsequent
                # check_browser_requirements() calls see the new install.
                import tools.browser_tool as _bt
                _bt._cached_chromium_installed = None
            else:
                _print_warning("    Chromium install failed:")
                tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
                for line in tail:
                    _print_info(f"      {line[:200]}")
                _print_info("    Run manually: npx agent-browser install --with-deps")
        except subprocess.TimeoutExpired:
            _print_warning("    Chromium install timed out (>10min)")
            _print_info("    Run manually: npx agent-browser install --with-deps")
        except Exception as exc:
            _print_warning(f"    Chromium install failed: {exc}")
            _print_info("    Run manually: npx agent-browser install --with-deps")

    elif post_setup_key == "browser_use_cli":
        _ensure_browser_use_cli(verbose_hints=True)

    elif post_setup_key == "camofox":
        camofox_dir = PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser"
        _npm_bin = find_node_executable("npm")
        if camofox_dir.exists():
            _print_success("    Camofox already installed, nothing to do")
        elif _npm_bin:
            _print_info("    Installing Camofox browser server...")
            import subprocess
            # Absolute npm path so .cmd shim executes on Windows.
            result = subprocess.run(
                # --workspaces=false avoids resolving apps/desktop. See #38772.
                [_npm_bin, "install", "--silent", "--workspaces=false"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT),
                creationflags=_post_setup_no_window_flags(),
            )
            if result.returncode == 0:
                _print_success("    Camofox installed")
            else:
                _print_warning("    npm install failed - run manually: npm install --workspaces=false")
        if camofox_dir.exists():
            _print_info("    Start the Camofox server:")
            _print_info("      npx @askjo/camofox-browser")
            _print_info("    First run downloads the Camoufox engine (~300MB)")
            _print_info("    Or use Docker: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")
        elif not _npm_bin:
            _print_warning("    Node.js not found. Install Camofox via Docker:")
            _print_info("      docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")

    elif post_setup_key == "cua_driver":
        install_cua_driver(upgrade=False)

    elif post_setup_key == "faster_whisper":
        import subprocess
        try:
            __import__("faster_whisper")
            _print_success("    faster-whisper is already installed")
            return
        except ImportError:
            pass
        _print_info("    Installing faster-whisper (model ~150MB downloads on first use)...")
        try:
            result = _pip_install(["-U", "faster-whisper", "--quiet"], timeout=300)
            if result.returncode == 0:
                _print_success("    faster-whisper installed")
                _print_info("    Model sizes: tiny, base (default), small, medium, large-v3")
                _print_info("    Change via stt.local.model in ~/.hermes/config.yaml")
            else:
                _print_warning("    faster-whisper install failed:")
                _print_info(f"      {(result.stderr or '').strip()[:300]}")
                _print_info("    Run manually: uv pip install -U faster-whisper")
        except subprocess.TimeoutExpired:
            _print_warning("    faster-whisper install timed out (>5min)")
            _print_info("    Run manually: uv pip install -U faster-whisper")

    elif post_setup_key == "kittentts":
        try:
            __import__("kittentts")
            _print_success("    kittentts is already installed")
            return
        except ImportError:
            pass
        _print_info("    Installing kittentts (~25-80MB model, CPU-only)...")
        wheel_url = (
            "https://github.com/KittenML/KittenTTS/releases/download/"
            "0.8.1/kittentts-0.8.1-py3-none-any.whl"
        )
        try:
            result = _pip_install(["-U", wheel_url, "soundfile", "--quiet"], timeout=300)
            if result.returncode == 0:
                _print_success("    kittentts installed")
                _print_info("    Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo")
                _print_info("    Models: KittenML/kitten-tts-nano-0.8-int8 (25MB), micro (41MB), mini (80MB)")
            else:
                _print_warning("    kittentts install failed:")
                _print_info(f"      {(result.stderr or '').strip()[:300]}")
                _print_info(f"    Run manually: uv pip install -U '{wheel_url}' soundfile")
        except subprocess.TimeoutExpired:
            _print_warning("    kittentts install timed out (>5min)")
            _print_info(f"    Run manually: uv pip install -U '{wheel_url}' soundfile")

    elif post_setup_key == "piper":
        try:
            __import__("piper")
            _print_success("    piper-tts is already installed")
        except ImportError:
            _print_info("    Installing piper-tts (~14MB wheel, voices downloaded on first use)...")
            try:
                result = _pip_install(["-U", "piper-tts", "--quiet"], timeout=300)
                if result.returncode == 0:
                    _print_success("    piper-tts installed")
                else:
                    _print_warning("    piper-tts install failed:")
                    _print_info(f"      {(result.stderr or '').strip()[:300]}")
                    _print_info("    Run manually: uv pip install -U piper-tts")
                    return
            except subprocess.TimeoutExpired:
                _print_warning("    piper-tts install timed out (>5min)")
                _print_info("    Run manually: uv pip install -U piper-tts")
                return
        _print_info("    Default voice: en_US-lessac-medium (downloaded on first TTS call)")
        _print_info("    Full voice list: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md")
        _print_info("    Switch voices by setting tts.piper.voice in ~/.hermes/config.yaml")

    elif post_setup_key == "ddgs":
        try:
            __import__("ddgs")
            _print_success("    ddgs is already installed")
        except ImportError:
            _print_info("    Installing ddgs (DuckDuckGo search package)...")
            try:
                result = _pip_install(["-U", "ddgs", "--quiet"], timeout=300)
                if result.returncode == 0:
                    _print_success("    ddgs installed")
                else:
                    _print_warning("    ddgs install failed:")
                    _print_info(f"      {(result.stderr or '').strip()[:300]}")
                    _print_info("    Run manually: uv pip install -U ddgs")
                    return
            except subprocess.TimeoutExpired:
                _print_warning("    ddgs install timed out (>5min)")
                _print_info("    Run manually: uv pip install -U ddgs")
                return
        _print_info("    No API key required. DuckDuckGo enforces server-side rate limits.")
        _print_info("    Pair with an extract provider if you also need web_extract.")

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

    elif post_setup_key == "langfuse":
        # Install the langfuse SDK.
        try:
            __import__("langfuse")
            _print_success("    langfuse SDK already installed")
        except ImportError:
            _print_info("    Installing langfuse SDK...")
            result = _pip_install(["langfuse", "--quiet"], timeout=120)
            if result.returncode == 0:
                _print_success("    langfuse SDK installed")
            else:
                _print_warning("    langfuse SDK install failed — run manually: uv pip install langfuse")
        # Opt the bundled observability/langfuse plugin into plugins.enabled.
        # The plugin ships in the repo but doesn't load until the user enables
        # it (standalone plugins are opt-in).
        try:
            from hermes_cli.plugins_cmd import _get_enabled_set, _save_enabled_set
            enabled = _get_enabled_set()
            if "observability/langfuse" in enabled or "langfuse" in enabled:
                _print_success("    Plugin observability/langfuse already enabled")
            else:
                enabled.add("observability/langfuse")
                _save_enabled_set(enabled)
                _print_success("    Plugin observability/langfuse enabled")
        except Exception as exc:
            _print_warning(f"    Could not enable plugin automatically: {exc}")
            _print_info("    Run manually: hermes plugins enable observability/langfuse")
        _print_info("    Restart Hermes for tracing to take effect.")
        _print_info("    Verify: hermes plugins list")

    elif post_setup_key == "xai_grok":
        # Shared credential bootstrap for any picker entry that talks to xAI
        # (TTS, Video Gen, future Image Gen, etc.). Accepts either a
        # SuperGrok-tier OAuth bearer token (preferred — billed against the
        # user's existing subscription) or a raw XAI_API_KEY from
        # console.x.ai. The picker entries declare empty env_vars so we
        # drive the full auth UX here.
        try:
            from hermes_cli.auth import get_xai_oauth_auth_status
            oauth_logged_in = bool(get_xai_oauth_auth_status().get("logged_in"))
        except Exception:
            oauth_logged_in = False
        existing_api_key = get_env_value("XAI_API_KEY")

        if oauth_logged_in:
            _print_success(
                "    xAI will use your xAI Grok OAuth (SuperGrok / Premium+) credentials"
            )
            return
        if existing_api_key:
            _print_success("    xAI will use your existing XAI_API_KEY")
            return

        _print_info("    xAI needs credentials. Choose one:")
        try:
            from hermes_cli.setup import (
                _run_xai_oauth_login_from_setup,
                prompt_choice,
                prompt as _setup_prompt,
            )
            from hermes_cli.config import save_env_value
        except Exception as exc:
            _print_warning(f"    Could not load setup helpers: {exc}")
            _print_info("    Run later: hermes auth add xai-oauth   (or set XAI_API_KEY)")
            return

        idx = prompt_choice(
            "    How do you want xAI to authenticate?",
            choices=[
                "Sign in with xAI Grok OAuth (SuperGrok / Premium+) — browser login",
                "Paste an xAI API key (console.x.ai)",
                "Skip — configure later via `hermes auth add xai-oauth`",
            ],
            default=0,
        )
        if idx == 0:
            if _run_xai_oauth_login_from_setup():
                _print_success(
                    "    Logged in — xAI will use these OAuth credentials"
                )
            else:
                _print_warning(
                    "    xAI Grok OAuth login did not complete. "
                    "Run later: hermes auth add xai-oauth"
                )
        elif idx == 1:
            api_key = _setup_prompt("    xAI API key", password=True)
            if api_key:
                save_env_value("XAI_API_KEY", api_key)
                _print_success("    XAI_API_KEY saved")
            else:
                _print_warning(
                    "    No API key provided. Run later: hermes auth add xai-oauth"
                )
        else:
            _print_info("    xAI will remain inactive until credentials are configured.")


def valid_post_setup_keys() -> Set[str]:
    """Return the set of post-setup keys declared by any visible provider.

    Collected from ``TOOL_CATEGORIES`` plus the plugin-registered web /
    image-gen / video-gen / browser providers (which can also carry a
    ``post_setup``). This is the allowlist the ``hermes tools post-setup``
    command and the dashboard post-setup endpoint validate against, so a
    caller can't drive ``_run_post_setup`` with an arbitrary key.
    """
    keys: Set[str] = set()
    for cat in TOOL_CATEGORIES.values():
        for prov in cat.get("providers", []):
            ps = prov.get("post_setup")
            if ps:
                keys.add(ps)
    # Plugin-registered providers can declare their own post_setup hooks.
    for builder in (
        _plugin_web_search_providers,
        _plugin_image_gen_providers,
        _plugin_video_gen_providers,
        _plugin_browser_providers,
    ):
        try:
            for prov in builder():
                ps = prov.get("post_setup")
                if ps:
                    keys.add(ps)
        except Exception:  # pragma: no cover — defensive; plugins optional
            continue
    return keys


def run_post_setup_command(args) -> int:
    """``hermes tools post-setup <key>`` — non-interactive post-setup runner.

    Runs the install/bootstrap hook a provider declares (npm install for
    browser/Camofox, pip install for kittentts/piper/ddgs, cua-driver fetch,
    etc.). This is the stable, scriptable target the dashboard spawns so the
    GUI can drive backend setup without re-implementing the install logic.
    Returns a process exit code (0 ok, 2 unknown key).
    """
    key = getattr(args, "post_setup_key", None)
    if not key:
        _print_error("Usage: hermes tools post-setup <key>")
        return 2
    valid = valid_post_setup_keys()
    if key not in valid:
        _print_error(
            f"Unknown post-setup key: {key!r}. "
            f"Valid keys: {', '.join(sorted(valid)) or '(none)'}"
        )
        return 2
    _print_info(f"Running post-setup hook: {key}")
    try:
        _run_post_setup(key)
    except Exception as exc:  # pragma: no cover — defensive
        _print_error(f"Post-setup failed: {exc}")
        return 1
    _print_success(f"Post-setup '{key}' complete")
    return 0


# ─── Platform / Toolset Helpers ───────────────────────────────────────────────

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
# "Web Search & Extract" picker. All bundled providers (brave-free, ddgs,
# searxng, exa, parallel, tavily, firecrawl, keenable) live as plugins after
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

    After PR #25182, all bundled web providers (brave-free, ddgs, searxng,
    exa, parallel, tavily, firecrawl, keenable) are plugins; this helper is the sole
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


def _shared_metrics_state(config: dict) -> tuple[bool, bool]:
    """Return (collection_enabled, send_enabled) from a config dict."""
    telemetry = config.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    shared = telemetry.get("shared_metrics")
    shared = shared if isinstance(shared, dict) else {}
    return shared.get("enabled") is True, shared.get("send") is True


def _shared_metrics_menu_label(config: dict) -> str:
    """Menu row for shared metrics, showing both consent states."""
    enabled, send = _shared_metrics_state(config)
    if not enabled:
        state = "off"
    elif send:
        state = "collecting + sending to Nous"
    else:
        state = "collecting locally"
    return f"Configure shared metrics  ({state})"


def _configure_shared_metrics_interactive(config: dict) -> None:
    """Toggle shared-metrics collection and sending from `hermes tools`.

    Delegates to the setup wizard's prompt so the consent rules live in one
    place: sending requires collection, and turning collection off also turns
    sending off.
    """
    from hermes_cli.setup import setup_telemetry

    before = _shared_metrics_state(config)
    setup_telemetry(config)
    after = _shared_metrics_state(config)
    if before != after:
        save_config(config)


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

    if len(platform_keys) > 1:
        platform_choices.append("Configure all platforms (global)")
    platform_choices.append("Reconfigure an existing tool's provider or API key")
    platform_choices.append(_shared_metrics_menu_label(config))

    # Show MCP option if any MCP servers are configured
    _has_mcp = bool(config.get("mcp_servers"))
    if _has_mcp:
        platform_choices.append("Configure MCP server tools")

    platform_choices.append("Done")

    # Index offsets for the extra options after per-platform entries
    _global_idx = len(platform_keys) if len(platform_keys) > 1 else -1
    _reconfig_idx = len(platform_keys) + (1 if len(platform_keys) > 1 else 0)
    _metrics_idx = _reconfig_idx + 1
    _mcp_idx = (_metrics_idx + 1) if _has_mcp else -1
    _done_idx = _metrics_idx + (2 if _has_mcp else 1)

    while True:
        idx = _prompt_choice("Select an option:", platform_choices, default=0)
        if idx == done_idx:
            break
        if idx == reconfig_idx:
            _reconfigure_tool(config, force_fresh=True)
            print()
            continue

        # "Shared metrics" selected
        if idx == _metrics_idx:
            _configure_shared_metrics_interactive(config)
            platform_choices[_metrics_idx] = _shared_metrics_menu_label(config)
            print()
            continue

        # "Configure MCP tools" selected
        if idx == _mcp_idx:
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
