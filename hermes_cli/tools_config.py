"""
Unified tool configuration for Hermes Agent.

`hermes tools` and `hermes setup tools` both enter this module.
Select a platform → toggle toolsets on/off → for newly enabled tools
that need API keys, run through provider-aware configuration.

Saves per-platform tool configuration to ~/.hermes/config.yaml under
the `platform_toolsets` key.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

import os

from hermes_cli.config import (
    load_config, save_config, get_env_value, save_env_value,
    get_hermes_home,
)
from hermes_cli.colors import Colors, color

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ─── UI Helpers (shared with setup.py) ────────────────────────────────────────

def _print_info(text: str):
    print(color(f"  {text}", Colors.DIM))

def _print_success(text: str):
    print(color(f"✓ {text}", Colors.GREEN))

def _print_warning(text: str):
    print(color(f"⚠ {text}", Colors.YELLOW))

def _print_error(text: str):
    print(color(f"✗ {text}", Colors.RED))

def _prompt(question: str, default: str = None, password: bool = False) -> str:
    if default:
        display = f"{question} [{default}]: "
    else:
        display = f"{question}: "
    try:
        if password:
            import getpass
            value = getpass.getpass(color(display, Colors.YELLOW))
        else:
            value = input(color(display, Colors.YELLOW))
        return value.strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        return default or ""

def _prompt_yes_no(question: str, default: bool = True) -> bool:
    default_str = "Y/n" if default else "y/N"
    while True:
        try:
            value = input(color(f"{question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            return default
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False


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
    ("cronjob",         "⏰ Cron Jobs",                 "schedule, list, remove"),
    ("rl",              "🧪 RL Training",               "Tinker-Atropos training tools"),
    ("homeassistant",    "🏠 Home Assistant",           "smart home device control"),
]

# Platform display config
PLATFORMS = {
    "cli":      {"label": "🖥️  CLI",       "default_toolset": "hermes-cli"},
    "telegram": {"label": "📱 Telegram",   "default_toolset": "hermes-telegram"},
    "discord":  {"label": "💬 Discord",    "default_toolset": "hermes-discord"},
    "slack":    {"label": "💼 Slack",      "default_toolset": "hermes-slack"},
    "whatsapp": {"label": "📱 WhatsApp",   "default_toolset": "hermes-whatsapp"},
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
                "name": "Microsoft Edge TTS",
                "tag": "Free - no API key needed",
                "env_vars": [],
                "tts_provider": "edge",
            },
            {
                "name": "OpenAI TTS",
                "tag": "Premium - high quality voices",
                "env_vars": [
                    {"key": "VOICE_TOOLS_OPENAI_KEY", "prompt": "OpenAI API key", "url": "https://platform.openai.com/api-keys"},
                ],
                "tts_provider": "openai",
            },
            {
                "name": "ElevenLabs",
                "tag": "Premium - most natural voices",
                "env_vars": [
                    {"key": "ELEVENLABS_API_KEY", "prompt": "ElevenLabs API key", "url": "https://elevenlabs.io/app/settings/api-keys"},
                ],
                "tts_provider": "elevenlabs",
            },
        ],
    },
    "web": {
        "name": "Web Search & Extract",
        "icon": "🔍",
        "providers": [
            {
                "name": "Firecrawl Cloud",
                "tag": "Recommended - hosted service",
                "env_vars": [
                    {"key": "FIRECRAWL_API_KEY", "prompt": "Firecrawl API key", "url": "https://firecrawl.dev"},
                ],
            },
            {
                "name": "Firecrawl Self-Hosted",
                "tag": "Free - run your own instance",
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
                "name": "FAL.ai",
                "tag": "FLUX 2 Pro with auto-upscaling",
                "env_vars": [
                    {"key": "FAL_KEY", "prompt": "FAL API key", "url": "https://fal.ai/dashboard/keys"},
                ],
            },
        ],
    },
    "browser": {
        "name": "Browser Automation",
        "icon": "🌐",
        "providers": [
            {
                "name": "Local Browser",
                "tag": "Free headless Chromium (no API key needed)",
                "env_vars": [],
                "post_setup": "browserbase",  # Same npm install for agent-browser
            },
            {
                "name": "Browserbase",
                "tag": "Cloud browser with stealth & proxies",
                "env_vars": [
                    {"key": "BROWSERBASE_API_KEY", "prompt": "Browserbase API key", "url": "https://browserbase.com"},
                    {"key": "BROWSERBASE_PROJECT_ID", "prompt": "Browserbase project ID"},
                ],
                "post_setup": "browserbase",
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
    if post_setup_key == "browserbase":
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
                _print_warning("    npm install failed - run manually: cd ~/.hermes/hermes-agent && npm install")
        elif not node_modules.exists():
            _print_warning("    Node.js not found - browser tools require: npm install (in hermes-agent directory)")

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
    return enabled


def _get_platform_tools(config: dict, platform: str) -> Set[str]:
    """Resolve which individual toolset names are enabled for a platform."""
    from toolsets import resolve_toolset, TOOLSETS

    platform_toolsets = config.get("platform_toolsets", {})
    toolset_names = platform_toolsets.get(platform)

    if toolset_names is None or not isinstance(toolset_names, list):
        default_ts = PLATFORMS[platform]["default_toolset"]
        toolset_names = [default_ts]

    # Resolve to individual tool names, then map back to which
    # configurable toolsets are covered
    all_tool_names = set()
    for ts_name in toolset_names:
        all_tool_names.update(resolve_toolset(ts_name))

    # Map individual tool names back to configurable toolset keys
    enabled_toolsets = set()
    for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
        ts_tools = set(resolve_toolset(ts_key))
        if ts_tools and ts_tools.issubset(all_tool_names):
            enabled_toolsets.add(ts_key)

    return enabled_toolsets


def _save_platform_tools(config: dict, platform: str, enabled_toolset_keys: Set[str]):
    """Save the selected toolset keys for a platform to config."""
    config.setdefault("platform_toolsets", {})
    config["platform_toolsets"][platform] = sorted(enabled_toolset_keys)
    save_config(config)


def _toolset_has_keys(ts_key: str) -> bool:
    """Check if a toolset's required API keys are configured."""
    # Check TOOL_CATEGORIES first (provider-aware)
    cat = TOOL_CATEGORIES.get(ts_key)
    if cat:
        for provider in cat["providers"]:
            env_vars = provider.get("env_vars", [])
            if not env_vars:
                return True  # Free provider (e.g., Edge TTS)
            if all(get_env_value(v["key"]) for v in env_vars):
                return True
        return False

    # Fallback to simple requirements
    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return True
    return all(get_env_value(var) for var, _ in requirements)


# ─── Menu Helpers ─────────────────────────────────────────────────────────────

def _prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu (arrow keys). Uses curses to avoid simple_term_menu
    rendering bugs in tmux, iTerm, and other non-standard terminals."""

    # Curses-based single-select — works in tmux, iTerm, and standard terminals
    try:
        import curses
        result_holder = [default]

        def _curses_menu(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
            cursor = default

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                try:
                    stdscr.addnstr(0, 0, question, max_x - 1,
                                   curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0))
                except curses.error:
                    pass

                for i, c in enumerate(choices):
                    y = i + 2
                    if y >= max_y - 1:
                        break
                    arrow = "→" if i == cursor else " "
                    line = f" {arrow}  {c}"
                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(1)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ord('k')):
                    cursor = (cursor - 1) % len(choices)
                elif key in (curses.KEY_DOWN, ord('j')):
                    cursor = (cursor + 1) % len(choices)
                elif key in (curses.KEY_ENTER, 10, 13):
                    result_holder[0] = cursor
                    return
                elif key in (27, ord('q')):
                    return

        curses.wrapper(_curses_menu)
        return result_holder[0]

    except Exception:
        pass

    # Fallback: numbered input (Windows without curses, etc.)
    print(color(question, Colors.YELLOW))
    for i, c in enumerate(choices):
        marker = "●" if i == default else "○"
        style = Colors.GREEN if i == default else ""
        print(color(f"  {marker} {i+1}. {c}", style) if style else f"  {marker} {i+1}. {c}")
    while True:
        try:
            val = input(color(f"  Select [1-{len(choices)}] ({default + 1}): ", Colors.DIM))
            if not val:
                return default
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return idx
        except (ValueError, KeyboardInterrupt, EOFError):
            print()
            return default


def _prompt_toolset_checklist(platform_label: str, enabled: Set[str]) -> Set[str]:
    """Multi-select checklist of toolsets. Returns set of selected toolset keys."""

    labels = []
    for ts_key, ts_label, ts_desc in CONFIGURABLE_TOOLSETS:
        suffix = ""
        if not _toolset_has_keys(ts_key) and (TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key)):
            suffix = "  [no API key]"
        labels.append(f"{ts_label}  ({ts_desc}){suffix}")

    pre_selected_indices = [
        i for i, (ts_key, _, _) in enumerate(CONFIGURABLE_TOOLSETS)
        if ts_key in enabled
    ]

    # Curses-based multi-select — arrow keys + space to toggle + enter to confirm.
    # simple_term_menu has rendering bugs in tmux, iTerm, and other terminals.
    try:
        import curses
        selected = set(pre_selected_indices)
        result_holder = [None]

        def _curses_checklist(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_YELLOW, -1)
                curses.init_pair(3, 8, -1)  # dim gray
            cursor = 0
            scroll_offset = 0

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                header = f"Tools for {platform_label}  —  ↑↓ navigate, SPACE toggle, ENTER confirm"
                try:
                    stdscr.addnstr(0, 0, header, max_x - 1, curses.A_BOLD | curses.color_pair(2) if curses.has_colors() else curses.A_BOLD)
                except curses.error:
                    pass

                visible_rows = max_y - 3
                if cursor < scroll_offset:
                    scroll_offset = cursor
                elif cursor >= scroll_offset + visible_rows:
                    scroll_offset = cursor - visible_rows + 1

                for draw_i, i in enumerate(range(scroll_offset, min(len(labels), scroll_offset + visible_rows))):
                    y = draw_i + 2
                    if y >= max_y - 1:
                        break
                    check = "✓" if i in selected else " "
                    arrow = "→" if i == cursor else " "
                    line = f" {arrow} [{check}] {labels[i]}"

                    attr = curses.A_NORMAL
                    if i == cursor:
                        attr = curses.A_BOLD
                        if curses.has_colors():
                            attr |= curses.color_pair(1)
                    try:
                        stdscr.addnstr(y, 0, line, max_x - 1, attr)
                    except curses.error:
                        pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in (curses.KEY_UP, ord('k')):
                    cursor = (cursor - 1) % len(labels)
                elif key in (curses.KEY_DOWN, ord('j')):
                    cursor = (cursor + 1) % len(labels)
                elif key == ord(' '):
                    if cursor in selected:
                        selected.discard(cursor)
                    else:
                        selected.add(cursor)
                elif key in (curses.KEY_ENTER, 10, 13):
                    result_holder[0] = {CONFIGURABLE_TOOLSETS[i][0] for i in selected}
                    return
                elif key in (27, ord('q')):  # ESC or q
                    result_holder[0] = enabled
                    return

        curses.wrapper(_curses_checklist)
        return result_holder[0] if result_holder[0] is not None else enabled

    except Exception:
        pass  # fall through to numbered toggle

    # Final fallback: numbered toggle (Windows without curses, etc.)
    selected = set(pre_selected_indices)
    print(color(f"\n  Tools for {platform_label}", Colors.YELLOW))
    print(color("  Toggle by number, Enter to confirm.\n", Colors.DIM))

    while True:
        for i, label in enumerate(labels):
            marker = color("[✓]", Colors.GREEN) if i in selected else "[ ]"
            print(f"  {marker} {i + 1:>2}. {label}")
        print()
        try:
            val = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
            if not val:
                break
            idx = int(val) - 1
            if 0 <= idx < len(labels):
                if idx in selected:
                    selected.discard(idx)
                else:
                    selected.add(idx)
        except (ValueError, KeyboardInterrupt, EOFError):
            return enabled
        print()

    return {CONFIGURABLE_TOOLSETS[i][0] for i in selected}


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


def _configure_tool_category(ts_key: str, cat: dict, config: dict):
    """Configure a tool category with provider selection."""
    icon = cat.get("icon", "")
    name = cat["name"]
    providers = cat["providers"]

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
        _configure_provider(provider, config)
    else:
        # Multiple providers - let user choose
        print()
        print(color(f"  --- {icon} {name} - Choose a provider ---", Colors.CYAN))
        print()

        # Plain text labels only (no ANSI codes in menu items)
        provider_choices = []
        for p in providers:
            tag = f" ({p['tag']})" if p.get("tag") else ""
            configured = ""
            env_vars = p.get("env_vars", [])
            if not env_vars or all(get_env_value(v["key"]) for v in env_vars):
                if p.get("tts_provider") and config.get("tts", {}).get("provider") == p["tts_provider"]:
                    configured = " [active]"
                elif not env_vars:
                    configured = " [active]" if config.get("tts", {}).get("provider", "edge") == p.get("tts_provider", "") else ""
                else:
                    configured = " [configured]"
            provider_choices.append(f"{p['name']}{tag}{configured}")

        # Detect current provider as default
        default_idx = 0
        for i, p in enumerate(providers):
            if p.get("tts_provider") and config.get("tts", {}).get("provider") == p["tts_provider"]:
                default_idx = i
                break
            env_vars = p.get("env_vars", [])
            if env_vars and all(get_env_value(v["key"]) for v in env_vars):
                default_idx = i
                break

        provider_idx = _prompt_choice("  Select provider:", provider_choices, default_idx)
        _configure_provider(providers[provider_idx], config)


def _configure_provider(provider: dict, config: dict):
    """Configure a single provider - prompt for API keys and set config."""
    env_vars = provider.get("env_vars", [])

    # Set TTS provider in config if applicable
    if provider.get("tts_provider"):
        config.setdefault("tts", {})["provider"] = provider["tts_provider"]

    if not env_vars:
        _print_success(f"  {provider['name']} - no configuration needed!")
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
                _print_success(f"    Saved")
            else:
                _print_warning(f"    Skipped")
                all_configured = False

    # Run post-setup hooks if needed
    if provider.get("post_setup") and all_configured:
        _run_post_setup(provider["post_setup"])

    if all_configured:
        _print_success(f"  {provider['name']} configured!")


def _configure_simple_requirements(ts_key: str):
    """Simple fallback for toolsets that just need env vars (no provider selection)."""
    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return

    missing = [(var, url) for var, url in requirements if not get_env_value(var)]
    if not missing:
        return

    ts_label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts_key), ts_key)
    print()
    print(color(f"  {ts_label} requires configuration:", Colors.YELLOW))

    for var, url in missing:
        if url:
            _print_info(f"  Get key at: {url}")
        value = _prompt(f"    {var}", password=True)
        if value and value.strip():
            save_env_value(var, value.strip())
            _print_success(f"    Saved")
        else:
            _print_warning(f"    Skipped")


def _reconfigure_tool(config: dict):
    """Let user reconfigure an existing tool's provider or API key."""
    # Build list of configurable tools that are currently set up
    configurable = []
    for ts_key, ts_label, _ in CONFIGURABLE_TOOLSETS:
        cat = TOOL_CATEGORIES.get(ts_key)
        reqs = TOOLSET_ENV_REQUIREMENTS.get(ts_key)
        if cat or reqs:
            if _toolset_has_keys(ts_key):
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
    providers = cat["providers"]

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
            tag = f" ({p['tag']})" if p.get("tag") else ""
            configured = ""
            env_vars = p.get("env_vars", [])
            if not env_vars or all(get_env_value(v["key"]) for v in env_vars):
                if p.get("tts_provider") and config.get("tts", {}).get("provider") == p["tts_provider"]:
                    configured = " [active]"
                elif not env_vars:
                    configured = ""
                else:
                    configured = " [configured]"
            provider_choices.append(f"{p['name']}{tag}{configured}")

        default_idx = 0
        for i, p in enumerate(providers):
            if p.get("tts_provider") and config.get("tts", {}).get("provider") == p["tts_provider"]:
                default_idx = i
                break
            env_vars = p.get("env_vars", [])
            if env_vars and all(get_env_value(v["key"]) for v in env_vars):
                default_idx = i
                break

        provider_idx = _prompt_choice("  Select provider:", provider_choices, default_idx)
        _reconfigure_provider(providers[provider_idx], config)


def _reconfigure_provider(provider: dict, config: dict):
    """Reconfigure a provider - update API keys."""
    env_vars = provider.get("env_vars", [])

    if provider.get("tts_provider"):
        config.setdefault("tts", {})["provider"] = provider["tts_provider"]
        _print_success(f"  TTS provider set to: {provider['tts_provider']}")

    if not env_vars:
        _print_success(f"  {provider['name']} - no configuration needed!")
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
            _print_success(f"    Updated")
        else:
            _print_info(f"    Kept current")


def _reconfigure_simple_requirements(ts_key: str):
    """Reconfigure simple env var requirements."""
    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return

    ts_label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts_key), ts_key)
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
            _print_success(f"    Updated")
        else:
            _print_info(f"    Kept current")


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def tools_command(args=None):
    """Entry point for `hermes tools` and `hermes setup tools`."""
    config = load_config()
    enabled_platforms = _get_enabled_platforms()

    print()
    print(color("⚕ Hermes Tool Configuration", Colors.CYAN, Colors.BOLD))
    print(color("  Enable or disable tools per platform.", Colors.DIM))
    print(color("  Tools that need API keys will be configured when enabled.", Colors.DIM))
    print()

    # Build platform choices
    platform_choices = []
    platform_keys = []
    for pkey in enabled_platforms:
        pinfo = PLATFORMS[pkey]
        current = _get_platform_tools(config, pkey)
        count = len(current)
        total = len(CONFIGURABLE_TOOLSETS)
        platform_choices.append(f"Configure {pinfo['label']}  ({count}/{total} enabled)")
        platform_keys.append(pkey)

    platform_choices.append("Reconfigure an existing tool's provider or API key")
    platform_choices.append("Done")

    while True:
        idx = _prompt_choice("Select an option:", platform_choices, default=0)

        # "Done" selected
        if idx == len(platform_keys) + 1:
            break

        # "Reconfigure" selected
        if idx == len(platform_keys):
            _reconfigure_tool(config)
            print()
            continue

        pkey = platform_keys[idx]
        pinfo = PLATFORMS[pkey]

        # Get current enabled toolsets for this platform
        current_enabled = _get_platform_tools(config, pkey)

        # Show checklist
        new_enabled = _prompt_toolset_checklist(pinfo["label"], current_enabled)

        # Detect first-time configuration (no saved toolsets for this platform yet)
        is_first_config = pkey not in config.get("platform_toolsets", {})

        if new_enabled != current_enabled or is_first_config:
            added = new_enabled - current_enabled
            removed = current_enabled - new_enabled

            if added:
                for ts in sorted(added):
                    label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts), ts)
                    print(color(f"  + {label}", Colors.GREEN))
            if removed:
                for ts in sorted(removed):
                    label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts), ts)
                    print(color(f"  - {label}", Colors.RED))

            # Determine which tools need API key configuration.
            # On first-time setup, check ALL enabled tools (the defaults
            # include everything, so "added" would be empty and no API key
            # prompts would ever appear).  For returning users, only
            # prompt for newly added tools.
            tools_to_configure = new_enabled if is_first_config else added

            unconfigured = [
                ts_key for ts_key in sorted(tools_to_configure)
                if (TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key))
                and not _toolset_has_keys(ts_key)
            ]

            if unconfigured:
                print()
                print(color(f"  {len(unconfigured)} tool(s) need API keys to be configured:", Colors.YELLOW))
                for ts_key in unconfigured:
                    label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts_key), ts_key)
                    print(color(f"    • {label}", Colors.DIM))
                print()
                for ts_key in unconfigured:
                    _configure_toolset(ts_key, config)

            _save_platform_tools(config, pkey, new_enabled)
            save_config(config)
            print(color(f"  ✓ Saved {pinfo['label']} configuration", Colors.GREEN))
        else:
            print(color(f"  No changes to {pinfo['label']}", Colors.DIM))

        print()

        # Update the choice label with new count
        new_count = len(_get_platform_tools(config, pkey))
        total = len(CONFIGURABLE_TOOLSETS)
        platform_choices[idx] = f"Configure {pinfo['label']}  ({new_count}/{total} enabled)"

    print()
    print(color("  Tool configuration saved to ~/.hermes/config.yaml", Colors.DIM))
    print(color("  Changes take effect on next 'hermes' or gateway restart.", Colors.DIM))
    print()
