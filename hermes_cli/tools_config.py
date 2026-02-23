"""
Interactive tool configuration for Hermes Agent.

`hermes tools` — select a platform, then toggle toolsets on/off via checklist.
Saves per-platform tool configuration to ~/.hermes/config.yaml under
the `platform_toolsets` key.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

from hermes_cli.config import load_config, save_config, get_env_value
from hermes_cli.colors import Colors, color

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
]

# Platform display config
PLATFORMS = {
    "cli":      {"label": "🖥️  CLI",       "default_toolset": "hermes-cli"},
    "telegram": {"label": "📱 Telegram",   "default_toolset": "hermes-telegram"},
    "discord":  {"label": "💬 Discord",    "default_toolset": "hermes-discord"},
    "slack":    {"label": "💼 Slack",      "default_toolset": "hermes-slack"},
    "whatsapp": {"label": "📱 WhatsApp",   "default_toolset": "hermes-whatsapp"},
}


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

    if not toolset_names or not isinstance(toolset_names, list):
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


def _prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu (arrow keys)."""
    print(color(question, Colors.YELLOW))

    try:
        from simple_term_menu import TerminalMenu
        menu = TerminalMenu(
            [f"  {c}" for c in choices],
            cursor_index=default,
            menu_cursor="→ ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
        )
        idx = menu.show()
        if idx is None:
            sys.exit(0)
        print()
        return idx
    except ImportError:
        for i, c in enumerate(choices):
            marker = "●" if i == default else "○"
            style = Colors.GREEN if i == default else ""
            print(color(f"  {marker} {c}", style) if style else f"  {marker} {c}")
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
                sys.exit(0)


def _prompt_toolset_checklist(platform_label: str, enabled: Set[str]) -> Set[str]:
    """Multi-select checklist of toolsets. Returns set of selected toolset keys."""
    title = color(f"Tools for {platform_label}", Colors.YELLOW)
    hint = color("  Press SPACE to toggle, ENTER on Continue when done.", Colors.DIM)
    print(title)
    print(hint)
    print()

    labels = []
    for ts_key, ts_label, ts_desc in CONFIGURABLE_TOOLSETS:
        labels.append(f"{ts_label}  {color(ts_desc, Colors.DIM)}")

    pre_selected_indices = [
        i for i, (ts_key, _, _) in enumerate(CONFIGURABLE_TOOLSETS)
        if ts_key in enabled
    ]

    try:
        from simple_term_menu import TerminalMenu

        menu_items = [f"  {label}" for label in labels] + ["  Continue →"]
        preselected = [menu_items[i] for i in pre_selected_indices if i < len(menu_items)]

        menu = TerminalMenu(
            menu_items,
            multi_select=True,
            show_multi_select_hint=False,
            multi_select_cursor="[✓] ",
            multi_select_select_on_accept=False,
            multi_select_empty_ok=True,
            preselected_entries=preselected if preselected else None,
            menu_cursor="→ ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
        )

        menu.show()

        if menu.chosen_menu_entries is None:
            return enabled  # Escape pressed, keep current

        continue_idx = len(CONFIGURABLE_TOOLSETS)
        selected_indices = [i for i in (menu.chosen_menu_indices or []) if i != continue_idx]

        return {CONFIGURABLE_TOOLSETS[i][0] for i in selected_indices}

    except ImportError:
        # Fallback: numbered toggle
        selected = set(pre_selected_indices)
        while True:
            for i, label in enumerate(labels):
                marker = color("[✓]", Colors.GREEN) if i in selected else "[ ]"
                print(f"  {marker} {i + 1}. {label}")
            print(f"      {len(labels) + 1}. {color('Continue →', Colors.GREEN)}")
            print()
            try:
                val = input(color("  Toggle # (or Enter to continue): ", Colors.DIM)).strip()
                if not val:
                    break
                idx = int(val) - 1
                if idx == len(labels):
                    break
                if 0 <= idx < len(labels):
                    if idx in selected:
                        selected.discard(idx)
                    else:
                        selected.add(idx)
            except (ValueError, KeyboardInterrupt, EOFError):
                return enabled
            print()

        return {CONFIGURABLE_TOOLSETS[i][0] for i in selected}


def tools_command(args):
    """Entry point for `hermes tools`."""
    config = load_config()
    enabled_platforms = _get_enabled_platforms()

    print()
    print(color("⚕ Hermes Tool Configuration", Colors.CYAN, Colors.BOLD))
    print(color("  Enable or disable tools per platform.", Colors.DIM))
    print()

    # Build platform choices
    platform_choices = []
    platform_keys = []
    for pkey in enabled_platforms:
        pinfo = PLATFORMS[pkey]
        # Count currently enabled toolsets
        current = _get_platform_tools(config, pkey)
        count = len(current)
        total = len(CONFIGURABLE_TOOLSETS)
        count_str = color(f"({count}/{total} enabled)", Colors.DIM)
        platform_choices.append(f"Configure {pinfo['label']}  {count_str}")
        platform_keys.append(pkey)

    platform_choices.append(f"{color('Done — save and exit', Colors.GREEN)}")

    while True:
        idx = _prompt_choice("Select a platform to configure:", platform_choices, default=0)

        # "Done" selected
        if idx == len(platform_keys):
            break

        pkey = platform_keys[idx]
        pinfo = PLATFORMS[pkey]

        # Get current enabled toolsets for this platform
        current_enabled = _get_platform_tools(config, pkey)

        # Show checklist
        new_enabled = _prompt_toolset_checklist(pinfo["label"], current_enabled)

        if new_enabled != current_enabled:
            _save_platform_tools(config, pkey, new_enabled)

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

            print(color(f"  ✓ Saved {pinfo['label']} configuration", Colors.GREEN))
        else:
            print(color(f"  No changes to {pinfo['label']}", Colors.DIM))

        print()

        # Update the choice label with new count
        new_count = len(_get_platform_tools(config, pkey))
        total = len(CONFIGURABLE_TOOLSETS)
        count_str = color(f"({new_count}/{total} enabled)", Colors.DIM)
        platform_choices[idx] = f"Configure {pinfo['label']}  {count_str}"

    print()
    print(color("  Tool configuration saved to ~/.hermes/config.yaml", Colors.DIM))
    print(color("  Changes take effect on next 'hermes' or gateway restart.", Colors.DIM))
    print()
