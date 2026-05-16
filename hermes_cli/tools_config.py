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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


from hermes_cli.config import (
    cfg_get,
    load_config, save_config, get_env_value, save_env_value,
)
from hermes_cli.colors import Colors, color
from hermes_cli.nous_subscription import (
    apply_nous_managed_defaults,
    get_nous_subscription_features,
)
from tools.tool_backend_helpers import fal_key_is_configured, managed_nous_tools_enabled
from utils import base_url_hostname, is_truthy_value

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
    ("video",           "🎬 Video Analysis",            "video_analyze (requires video-capable model)"),
    ("image_gen",       "🎨 Image Generation",          "image_generate"),
    ("video_gen",       "🎬 Video Generation",          "video_generate (text-to-video + image-to-video)"),
    ("x_search",        "🐦 X (Twitter) Search",        "x_search (requires xAI OAuth or XAI_API_KEY)"),
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
    ("homeassistant",    "🏠 Home Assistant",           "smart home device control"),
    ("spotify",          "🎵 Spotify",                  "playback, search, playlists, library"),
    ("discord",         "💬 Discord (read/participate)", "fetch messages, search members, create thread"),
    ("discord_admin",   "🛡️  Discord Server Admin",    "list channels/roles, pin, assign roles"),
    ("yuanbao",          "🤖 Yuanbao",                  "group info, member queries, DM"),
    ("computer_use",     "🖱️  Computer Use (macOS)",     "background desktop control via cua-driver"),
]

# Toolsets that are OFF by default for new installs.
# They're still in _HERMES_CORE_TOOLS (available at runtime if enabled),
# but the setup checklist won't pre-select them for first-time users.
#
# Video gen is off by default — it's a niche, paid, slow feature. Users
# who want it opt in via `hermes tools` → Video Generation, which walks
# them through provider + model selection.
#
# X search is off by default — gated on xAI credentials (SuperGrok OAuth
# or XAI_API_KEY). Users opt in via `hermes tools` → X (Twitter) Search,
# which walks them through credential setup. The tool's check_fn means
# the schema won't appear to the model even if enabled without credentials.
_DEFAULT_OFF_TOOLSETS = {"moa", "homeassistant", "spotify", "discord", "discord_admin", "video", "video_gen", "x_search"}

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
                "tag": "Grok voices — uses xAI Grok OAuth or XAI_API_KEY",
                "env_vars": [],
                "tts_provider": "xai",
                "post_setup": "xai_grok",
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
            # Mistral (Voxtral TTS) temporarily hidden — `mistralai` PyPI
            # package is currently quarantined (malicious 2.4.6 release on
            # 2026-05-12). Restore this entry once PyPI un-quarantines.
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
            {
                "name": "Piper",
                "badge": "local · free",
                "tag": "Local neural TTS, 44 languages (voices ~20-90MB)",
                "env_vars": [],
                "tts_provider": "piper",
                "post_setup": "piper",
            },
        ],
    },
    "web": {
        "name": "Web Search & Extract",
        "setup_title": "Select Search Provider",
        "setup_note": "A free DuckDuckGo search skill is also included — skip this if you don't need a premium provider.",
        "icon": "🔍",
        # Per-provider rows are injected at runtime from
        # plugins.web.<vendor>.provider via _plugin_web_search_providers()
        # in _visible_providers(). Only non-provider UX setup-flow rows
        # for the firecrawl backend are listed here:
        #   - "Nous Subscription" — managed Firecrawl billed via Nous
        #     subscription (requires_nous_auth + override_env_vars).
        #   - "Firecrawl Self-Hosted" — points firecrawl at a private
        #     Docker instance via FIRECRAWL_API_URL only.
        # See PR #25182 for the migration rationale.
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
    "video_gen": {
        "name": "Video Generation",
        "icon": "🎬",
        # Providers list is intentionally empty — every video gen backend
        # is a plugin, surfaced by ``_plugin_video_gen_providers()`` and
        # injected by ``_visible_providers``. Mirrors the design we'll
        # converge image_gen toward.
        "providers": [],
    },
    "x_search": {
        "name": "X (Twitter) Search",
        "setup_title": "Select xAI Credential Source",
        "setup_note": (
            "Hermes routes X searches through xAI's built-in x_search "
            "Responses tool. Both credential sources hit the same "
            "https://api.x.ai/v1/responses endpoint — pick whichever you "
            "already have. SuperGrok OAuth is preferred when both are set "
            "(uses your subscription quota instead of API spend)."
        ),
        "icon": "🐦",
        "providers": [
            {
                "name": "xAI Grok OAuth (SuperGrok Subscription)",
                "badge": "subscription",
                "tag": "Browser login at accounts.x.ai — no API key required",
                "env_vars": [],
                "post_setup": "xai_grok",
            },
            {
                "name": "xAI API key",
                "badge": "paid",
                "tag": "Direct xAI API billing via XAI_API_KEY",
                "env_vars": [
                    {
                        "key": "XAI_API_KEY",
                        "prompt": "xAI API key",
                        "url": "https://console.x.ai/",
                    },
                ],
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
    "computer_use": {
        "name": "Computer Use (macOS)",
        "icon": "🖱️",
        "platform_gate": "darwin",
        "providers": [
            {
                "name": "cua-driver (background)",
                "badge": "★ recommended · free · local",
                "tag": (
                    "macOS background computer-use via SkyLight SPIs — does "
                    "NOT steal your cursor or focus. Works with any model."
                ),
                "env_vars": [
                    # cua-driver reads HOME/TMPDIR from the process env, no
                    # extra keys required. HERMES_CUA_DRIVER_VERSION is an
                    # optional pin for reproducibility across macOS updates.
                ],
                "post_setup": "cua_driver",
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

    uv_bin = shutil.which("uv")
    if uv_bin:
        try:
            result = subprocess.run(
                [uv_bin, "pip", "install", *args],
                capture_output=capture_output, text=True, timeout=timeout,
                env=uv_env,
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
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode != 0:
            raise FileNotFoundError("pip not in venv")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                capture_output=True, text=True, timeout=120, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Synthesize a result so callers see a clean failure path.
            return subprocess.CompletedProcess(
                pip_cmd, returncode=1, stdout="",
                stderr=f"pip not available and ensurepip failed: {e}",
            )

    return subprocess.run(
        pip_cmd + ["install", *args],
        capture_output=capture_output, text=True, timeout=timeout,
    )


def install_cua_driver(upgrade: bool = False) -> bool:
    """Install or refresh the cua-driver binary used by Computer Use.

    The upstream installer always pulls the latest release tag, so re-running
    it is the canonical way to upgrade. We expose two modes:

    * ``upgrade=False`` — original post-setup behaviour: skip if already
      installed, install otherwise. Used by the toolset enable flow where
      we don't want to surprise the user with a network fetch.
    * ``upgrade=True`` — always re-run the installer (or call ``cua-driver
      update`` if the binary supports it). Used by ``hermes update`` and
      by ``hermes computer-use install --upgrade``.

    Returns True iff cua-driver is installed (or successfully refreshed)
    when the function returns. macOS-only — silently returns False on
    other platforms.
    """
    import platform as _plat
    import shutil
    import subprocess

    if _plat.system() != "Darwin":
        if upgrade:
            # Silent on non-macOS — `hermes update` calls this for every
            # user; only macOS users with cua-driver care.
            return False
        _print_warning("    Computer Use (cua-driver) is macOS-only; skipping.")
        return False

    binary = shutil.which("cua-driver")

    # Not installed → fresh install path (only when caller asked for it).
    if not binary and not upgrade:
        if not shutil.which("curl"):
            _print_warning("    curl not found — install manually:")
            _print_info("      https://github.com/trycua/cua/blob/main/libs/cua-driver/README.md")
            return False
        return _run_cua_driver_installer(label="Installing")

    # Already installed and caller didn't ask to upgrade → just confirm.
    if binary and not upgrade:
        try:
            version = subprocess.run(
                ["cua-driver", "--version"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            _print_success(f"    cua-driver already installed: {version or 'unknown version'}")
        except Exception:
            _print_success("    cua-driver already installed.")
        _print_info("    Grant macOS permissions if not done yet:")
        _print_info("      System Settings > Privacy & Security > Accessibility")
        _print_info("      System Settings > Privacy & Security > Screen Recording")
        return True

    # upgrade=True path — refresh to the latest upstream release.
    if not shutil.which("curl"):
        _print_warning("    curl not found — cannot refresh cua-driver.")
        return bool(binary)

    if binary:
        # Show before/after version when we have a baseline. Best-effort.
        try:
            before = subprocess.run(
                ["cua-driver", "--version"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            before = ""
    else:
        before = ""

    ok = _run_cua_driver_installer(label="Refreshing", verbose=False)
    if ok and before:
        try:
            after = subprocess.run(
                ["cua-driver", "--version"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if after and after != before:
                _print_success(f"    cua-driver upgraded: {before} → {after}")
            elif after:
                _print_info(f"    cua-driver up to date: {after}")
        except Exception:
            pass
    return ok


def _run_cua_driver_installer(label: str = "Installing", verbose: bool = True) -> bool:
    """Run the upstream cua-driver install.sh. Returns True on success.

    The script is idempotent: it always downloads the latest release, so
    re-running it on an already-installed system performs an upgrade.
    """
    import shutil
    import subprocess

    install_cmd = (
        "/bin/bash -c \"$(curl -fsSL "
        "https://raw.githubusercontent.com/trycua/cua/main/"
        "libs/cua-driver/scripts/install.sh)\""
    )
    if verbose:
        _print_info(f"    {label} cua-driver (macOS background computer-use)...")
    else:
        _print_info(f"    {label} cua-driver...")
    try:
        result = subprocess.run(install_cmd, shell=True, timeout=300)
        if result.returncode == 0 and shutil.which("cua-driver"):
            if verbose:
                _print_success("    cua-driver installed.")
                _print_info("    IMPORTANT — grant macOS permissions now:")
                _print_info("      System Settings > Privacy & Security > Accessibility")
                _print_info("      System Settings > Privacy & Security > Screen Recording")
                _print_info("    Both must allow the terminal / Hermes process.")
            return True
        _print_warning(f"    cua-driver {label.lower()} did not complete. Re-run manually:")
        _print_info(f"      {install_cmd}")
        return False
    except subprocess.TimeoutExpired:
        _print_warning(f"    cua-driver {label.lower()} timed out. Re-run manually.")
        return False
    except Exception as e:
        _print_warning(f"    cua-driver {label.lower()} failed: {e}")
        return False


def _run_post_setup(post_setup_key: str):
    """Run post-setup hooks for tools that need extra installation steps."""
    import shutil
    if post_setup_key in {"agent_browser", "browserbase"}:
        node_modules = PROJECT_ROOT / "node_modules" / "agent-browser"
        npm_bin = shutil.which("npm")
        npx_bin = shutil.which("npx")
        # Step 1: install the agent-browser npm package into node_modules/
        if not node_modules.exists() and npm_bin:
            _print_info("    Installing Node.js dependencies for browser tools...")
            import subprocess
            # Use the resolved npm_bin absolute path so subprocess.Popen can
            # execute npm.cmd on Windows (CreateProcessW otherwise rejects
            # batch shims).  On POSIX npm_bin is the plain path — same
            # behaviour as before.
            result = subprocess.run(
                [npm_bin, "install", "--silent"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )
            if result.returncode == 0:
                _print_success("    Node.js dependencies installed")
            else:
                from hermes_constants import display_hermes_home
                _print_warning(f"    npm install failed - run manually: cd {display_hermes_home()}/hermes-agent && npm install")
                if result.stderr:
                    _print_info(f"      {result.stderr.strip()[:200]}")
        elif not node_modules.exists():
            _print_warning("    Node.js not found - browser tools require: npm install (in hermes-agent directory)")
            return

        # Step 2: only the local browser provider actually needs Chromium on
        # disk. Cloud providers (Browserbase, Browser Use, Firecrawl) host
        # their own Chromium and don't need the local install.
        if post_setup_key != "agent_browser":
            return

        # Step 3: ensure the Chromium / headless-shell build agent-browser
        # drives is actually installed. Without it the CLI hangs on first
        # use until the command timeout fires. Skip inside Docker — the
        # image bakes Chromium in at build time, and runtime users usually
        # can't write to PLAYWRIGHT_BROWSERS_PATH anyway.
        try:
            # Import lazily so the tools_config UI doesn't pull in the full
            # browser_tool module at import time.
            from tools.browser_tool import (
                _chromium_installed,
                _running_in_docker,
            )
        except Exception as exc:  # pragma: no cover — defensive
            _print_warning(f"    Could not check Chromium status: {exc}")
            return

        if _chromium_installed():
            _print_success("    Chromium browser already installed")
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

        if not npx_bin:
            _print_warning(
                "    npx not found - install Chromium manually: npx agent-browser install --with-deps"
            )
            return

        _print_info("    Installing Chromium (~170MB one-time download)...")
        import subprocess
        # Prefer the bundled agent-browser install subcommand so the
        # version of Chromium matches the CLI. Fall back to npx shim on
        # setups where the local bin stub isn't present.
        local_ab = PROJECT_ROOT / "node_modules" / ".bin" / "agent-browser"
        if sys.platform == "win32":
            local_ab_win = local_ab.with_suffix(".cmd")
            if local_ab_win.exists():
                local_ab = local_ab_win
        install_cmd = (
            [str(local_ab), "install", "--with-deps"]
            if local_ab.exists()
            else [npx_bin, "-y", "agent-browser", "install", "--with-deps"]
        )
        try:
            result = subprocess.run(
                install_cmd,
                capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600,
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

    elif post_setup_key == "camofox":
        camofox_dir = PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser"
        _npm_bin = shutil.which("npm")
        if not camofox_dir.exists() and _npm_bin:
            _print_info("    Installing Camofox browser package...")
            _print_info("    First run downloads the Camoufox engine (~300MB) — this can take several minutes.")
            import subprocess
            # Install @askjo/camofox-browser on-demand. It is NOT in
            # package.json so that `hermes update` does not silently pull
            # the ~300MB Camoufox Firefox-fork binary for every user.
            # Stream output (no capture, no --silent) so the long-running
            # postinstall download is visible instead of looking frozen.
            try:
                result = subprocess.run(
                    [_npm_bin, "install", "@askjo/camofox-browser@^1.5.2",
                     "--no-fund", "--no-audit", "--progress=false"],
                    cwd=str(PROJECT_ROOT),
                )
                if result.returncode == 0:
                    _print_success("    Camofox installed")
                else:
                    _print_warning(
                        "    npm install failed — run manually: "
                        "npm install @askjo/camofox-browser"
                    )
            except Exception as exc:
                _print_warning(f"    Camofox install failed: {exc}")
                _print_info(
                    "    Run manually: npm install @askjo/camofox-browser"
                )
        if camofox_dir.exists():
            _print_info("    Start the Camofox server:")
            _print_info("      npx @askjo/camofox-browser")
            _print_info("    Or use Docker: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")
        elif not shutil.which("npm"):
            _print_warning("    Node.js not found. Install Camofox via Docker:")
            _print_info("      docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")

    elif post_setup_key == "cua_driver":
        install_cua_driver(upgrade=False)

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
                "    xAI will use your xAI Grok OAuth (SuperGrok Subscription) credentials"
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
                "Sign in with xAI Grok OAuth (SuperGrok Subscription) — browser login",
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
        plat_info = PLATFORMS.get(platform)
        if plat_info:
            default_ts = plat_info["default_toolset"]
        else:
            # Plugin platform — derive toolset name from platform key
            default_ts = f"hermes-{platform}"
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
        # Mixed config: composite toolset alongside configurables (e.g.
        # ``[hermes-cli, spotify]`` after enabling Spotify via ``hermes
        # tools``). Without expansion the composite name is silently dropped,
        # leaving sessions with only the configurable opt-ins and no native
        # tools. Mirror the else-branch's subset inference, but apply
        # _DEFAULT_OFF_TOOLSETS only to the implicit expansion — anything the
        # user explicitly listed (e.g. ``spotify``) must survive.
        composite_tools = set()
        for ts_name in toolset_names:
            if ts_name in configurable_keys or ts_name in plugin_ts_keys:
                continue
            if ts_name not in TOOLSETS:
                continue
            composite_tools.update(resolve_toolset(ts_name))

        if composite_tools:
            expanded = set()
            for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
                if not _toolset_allowed_for_platform(ts_key, platform):
                    continue
                ts_tools = set(resolve_toolset(ts_key))
                if ts_tools and ts_tools.issubset(composite_tools):
                    expanded.add(ts_key)

            default_off = set(_DEFAULT_OFF_TOOLSETS)
            if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
                default_off.remove(platform)
            if "homeassistant" in default_off and os.getenv("HASS_TOKEN"):
                default_off.remove("homeassistant")
            expanded -= default_off

            enabled_toolsets |= expanded
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
    _plat_info = PLATFORMS.get(platform)
    _default_ts = _plat_info["default_toolset"] if _plat_info else f"hermes-{platform}"
    platform_tool_universe = set(resolve_toolset(_default_ts))
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

    # Honor agent.disabled_toolsets from config.yaml — allows users to
    # globally suppress specific toolsets (e.g. "memory") across all
    # platforms without per-platform toolset configuration.  This runs
    # last so it overrides everything above.
    agent_cfg = config.get("agent") or {}
    disabled_toolsets = agent_cfg.get("disabled_toolsets") or []
    if disabled_toolsets:
        disabled_set = {str(ts) for ts in disabled_toolsets}
        enabled_toolsets -= disabled_set

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
    existing_toolsets = cfg_get(config, "platform_toolsets", platform, default=[])
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
        row = {
            "name": schema.get("name", provider.display_name),
            "badge": schema.get("badge", ""),
            "tag": schema.get("tag", ""),
            "env_vars": schema.get("env_vars", []),
            "web_backend": name,
            "web_search_plugin_name": name,
        }
        # Optional pass-through fields the schema can opt into.
        if schema.get("post_setup"):
            row["post_setup"] = schema["post_setup"]
        rows.append(row)
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

    return visible


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
    # a no-key provider, and (b) an installed-state check is cheap and
    # doesn't trigger a heavy import.
    "cua_driver": lambda: bool(shutil.which("cua-driver")),
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


def _toolset_needs_configuration_prompt(ts_key: str, config: dict) -> bool:
    """Return True when enabling this toolset should open provider setup."""
    cat = TOOL_CATEGORIES.get(ts_key)
    if not cat:
        return not _toolset_has_keys(ts_key, config)

    # If any visible provider has a registered post_setup install-state
    # check that hasn't been satisfied (e.g. cua-driver binary not on
    # PATH yet), force the configuration flow so `_configure_provider`
    # invokes `_run_post_setup` and the install actually runs.
    for provider in _visible_providers(cat, config):
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

    video_plugin_name = provider.get("video_gen_plugin_name")
    if video_plugin_name:
        video_cfg = config.get("video_gen", {})
        return isinstance(video_cfg, dict) and video_cfg.get("provider") == video_plugin_name

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
                if configured_provider not in {None, "", "fal"}:
                    return False
                if image_cfg.get("use_gateway") is not None and not is_truthy_value(image_cfg.get("use_gateway"), default=False):
                    return False
            return feature.managed_by_nous
        if provider.get("tts_provider"):
            return (
                feature.managed_by_nous
                and cfg_get(config, "tts", "provider") == provider["tts_provider"]
            )
        if "browser_provider" in provider:
            current = cfg_get(config, "browser", "cloud_provider")
            return feature.managed_by_nous and provider["browser_provider"] == current
        if provider.get("web_backend"):
            current = cfg_get(config, "web", "backend")
            return feature.managed_by_nous and current == provider["web_backend"]
        return feature.managed_by_nous

    if provider.get("tts_provider"):
        return cfg_get(config, "tts", "provider") == provider["tts_provider"]
    if "browser_provider" in provider:
        current = cfg_get(config, "browser", "cloud_provider")
        return provider["browser_provider"] == current
    if provider.get("web_backend"):
        current = cfg_get(config, "web", "backend")
        return current == provider["web_backend"]
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


def _select_plugin_video_gen_provider(plugin_name: str, config: dict) -> None:
    """Persist a plugin-backed video generation provider selection."""
    vid_cfg = config.setdefault("video_gen", {})
    if not isinstance(vid_cfg, dict):
        vid_cfg = {}
        config["video_gen"] = vid_cfg
    vid_cfg["provider"] = plugin_name
    vid_cfg["use_gateway"] = False
    _print_success(f"  video_gen.provider set to: {plugin_name}")
    _configure_videogen_model_for_plugin(plugin_name, config)


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
    if managed_feature and managed_feature not in {"web", "tts", "browser"}:
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
        # Plugin-registered video_gen provider — same flow, different
        # registry.
        video_plugin = provider.get("video_gen_plugin_name")
        if video_plugin:
            _select_plugin_video_gen_provider(video_plugin, config)
            return
        # Imagegen backends prompt for model selection after backend pick.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            # In-tree FAL is the only non-plugin backend today. Keep
            # image_gen.provider clear so the dispatch shim falls through
            # to the legacy FAL path.
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict) and img_cfg.get("provider") not in {None, "", "fal"}:
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
        video_plugin = provider.get("video_gen_plugin_name")
        if video_plugin:
            _select_plugin_video_gen_provider(video_plugin, config)
            return
        # Imagegen backends prompt for model selection after env vars are in.
        backend = provider.get("imagegen_backend")
        if backend:
            _configure_imagegen_model(backend, config)
            img_cfg = config.setdefault("image_gen", {})
            if isinstance(img_cfg, dict) and img_cfg.get("provider") not in {None, "", "fal"}:
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
            if _toolset_has_keys(ts_key, config) or _toolset_enabled_for_reconfigure(ts_key, config):
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


def _toolset_enabled_for_reconfigure(ts_key: str, config: dict) -> bool:
    """Return True if a configurable toolset is enabled anywhere.

    Reconfigure must include enabled-but-unconfigured categories so users can
    finish provider/API-key setup without disabling and re-enabling the toolset.
    """
    for platform in PLATFORMS:
        if not _toolset_allowed_for_platform(ts_key, platform):
            continue
        try:
            enabled = _get_platform_tools(
                config,
                platform,
                include_default_mcp_servers=False,
            )
        except Exception:
            continue
        if ts_key in enabled:
            return True
    return False


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
        tts_cfg = config.setdefault("tts", {})
        tts_cfg["provider"] = provider["tts_provider"]
        tts_cfg["use_gateway"] = bool(managed_feature)
        _print_success(f"  TTS provider set to: {provider['tts_provider']}")

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

    if managed_feature and managed_feature not in {"web", "tts", "browser"}:
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
        # Plugin-registered video_gen provider — same flow, different registry.
        video_plugin = provider.get("video_gen_plugin_name")
        if video_plugin:
            _select_plugin_video_gen_provider(video_plugin, config)
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

    # Plugin-registered video_gen provider — same flow, different registry.
    video_plugin = provider.get("video_gen_plugin_name")
    if video_plugin:
        _select_plugin_video_gen_provider(video_plugin, config)
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
