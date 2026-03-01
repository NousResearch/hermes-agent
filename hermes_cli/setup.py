"""
Interactive setup wizard for Hermes Agent.

Guides users through:
1. Installation directory confirmation
2. API key configuration
3. Model selection  
4. Terminal backend selection
5. Messaging platform setup
6. Optional features

Config files are stored in ~/.hermes/ for easy access.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Import config helpers
from hermes_cli.config import (
    get_hermes_home, get_config_path, get_env_path,
    load_config, save_config, save_env_value, get_env_value,
    ensure_hermes_home, DEFAULT_CONFIG
)

from hermes_cli.colors import Colors, color

def print_header(title: str):
    """Print a section header."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))

def print_info(text: str):
    """Print info text."""
    print(color(f"  {text}", Colors.DIM))

def print_success(text: str):
    """Print success message."""
    print(color(f"✓ {text}", Colors.GREEN))

def print_warning(text: str):
    """Print warning message."""
    print(color(f"⚠ {text}", Colors.YELLOW))

def print_error(text: str):
    """Print error message."""
    print(color(f"✗ {text}", Colors.RED))

def prompt(question: str, default: str = None, password: bool = False) -> str:
    """Prompt for input with optional default."""
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
        sys.exit(1)

def prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Prompt for a choice from a list with arrow key navigation."""
    print(color(question, Colors.YELLOW))
    
    # Try to use interactive menu if available
    try:
        from simple_term_menu import TerminalMenu
        
        # Add visual indicators
        menu_choices = [f"  {choice}" for choice in choices]
        
        terminal_menu = TerminalMenu(
            menu_choices,
            cursor_index=default,
            menu_cursor="→ ",
            menu_cursor_style=("fg_green", "bold"),
            menu_highlight_style=("fg_green",),
            cycle_cursor=True,
            clear_screen=False,
        )
        
        idx = terminal_menu.show()
        if idx is None:  # User pressed Escape or Ctrl+C
            print()
            sys.exit(1)
        print()  # Add newline after selection
        return idx
        
    except (ImportError, NotImplementedError):
        # Fallback to number-based selection (simple_term_menu doesn't support Windows)
        for i, choice in enumerate(choices):
            marker = "●" if i == default else "○"
            if i == default:
                print(color(f"  {marker} {choice}", Colors.GREEN))
            else:
                print(f"  {marker} {choice}")
        
        while True:
            try:
                value = input(color(f"  Select [1-{len(choices)}] ({default + 1}): ", Colors.DIM))
                if not value:
                    return default
                idx = int(value) - 1
                if 0 <= idx < len(choices):
                    return idx
                print_error(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print_error("Please enter a number")
            except (KeyboardInterrupt, EOFError):
                print()
                sys.exit(1)

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no."""
    default_str = "Y/n" if default else "y/N"
    
    while True:
        value = input(color(f"{question} [{default_str}]: ", Colors.YELLOW)).strip().lower()
        
        if not value:
            return default
        if value in ('y', 'yes'):
            return True
        if value in ('n', 'no'):
            return False
        print_error("Please enter 'y' or 'n'")


def prompt_checklist(title: str, items: list, pre_selected: list = None) -> list:
    """
    Display a multi-select checklist and return the indices of selected items.
    
    Each item in `items` is a display string. `pre_selected` is a list of
    indices that should be checked by default. A "Continue →" option is
    appended at the end — the user toggles items with Space and confirms
    with Enter on "Continue →".
    
    Falls back to a numbered toggle interface when simple_term_menu is
    unavailable.
    
    Returns:
        List of selected indices (not including the Continue option).
    """
    if pre_selected is None:
        pre_selected = []
    
    print(color(title, Colors.YELLOW))
    print_info("SPACE to toggle, ENTER to confirm.")
    print()
    
    try:
        from simple_term_menu import TerminalMenu
        import re
        
        # Strip emoji characters from menu labels — simple_term_menu miscalculates
        # visual width of emojis on macOS, causing duplicated/garbled lines.
        _emoji_re = re.compile(
            "[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0000fe00-\U0000fe0f"
            "\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff\u200d]+", flags=re.UNICODE
        )
        menu_items = [f"  {_emoji_re.sub('', item).strip()}" for item in items]
        
        # Map pre-selected indices to the actual menu entry strings
        preselected = [menu_items[i] for i in pre_selected if i < len(menu_items)]
        
        terminal_menu = TerminalMenu(
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
        
        terminal_menu.show()
        
        if terminal_menu.chosen_menu_entries is None:
            return []
        
        selected = list(terminal_menu.chosen_menu_indices or [])
        return selected
        
    except (ImportError, NotImplementedError):
        # Fallback: numbered toggle interface (simple_term_menu doesn't support Windows)
        selected = set(pre_selected)
        
        while True:
            for i, item in enumerate(items):
                marker = color("[✓]", Colors.GREEN) if i in selected else "[ ]"
                print(f"  {marker} {i + 1}. {item}")
            print()
            
            try:
                value = input(color("  Toggle # (or Enter to confirm): ", Colors.DIM)).strip()
                if not value:
                    break
                idx = int(value) - 1
                if 0 <= idx < len(items):
                    if idx in selected:
                        selected.discard(idx)
                    else:
                        selected.add(idx)
                else:
                    print_error(f"Enter a number between 1 and {len(items) + 1}")
            except ValueError:
                print_error("Enter a number")
            except (KeyboardInterrupt, EOFError):
                print()
                return []
            
            # Clear and redraw (simple approach)
            print()
        
        return sorted(selected)


def _prompt_api_key(var: dict):
    """Display a nicely formatted API key input screen for a single env var."""
    tools = var.get("tools", [])
    tools_str = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_str += f", +{len(tools) - 3} more"

    print()
    print(color(f"  ─── {var.get('description', var['name'])} ───", Colors.CYAN))
    print()
    if tools_str:
        print_info(f"  Enables: {tools_str}")
    if var.get("url"):
        print_info(f"  Get your key at: {var['url']}")
    print()

    if var.get("password"):
        value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
    else:
        value = prompt(f"  {var.get('prompt', var['name'])}")

    if value:
        save_env_value(var["name"], value)
        print_success(f"  ✓ Saved")
    else:
        print_warning(f"  Skipped (configure later with 'hermes setup')")


def _print_setup_summary(
    config: dict,
    hermes_home,
    selected_summary: dict = None,
    api_key_alerts: list = None,
):
    """Print a concise setup completion summary."""
    optional_tools = [
        ("Vision/MoA (OpenRouter)", bool(get_env_value("OPENROUTER_API_KEY"))),
        ("Web Search (Firecrawl)", bool(get_env_value("FIRECRAWL_API_KEY"))),
        ("Browser Automation (Browserbase)", bool(get_env_value("BROWSERBASE_API_KEY"))),
        ("Image Generation (FAL)", bool(get_env_value("FAL_KEY"))),
        ("Voice Tools (OpenAI)", bool(get_env_value("VOICE_TOOLS_OPENAI_KEY"))),
        ("Premium TTS (ElevenLabs)", bool(get_env_value("ELEVENLABS_API_KEY"))),
        ("RL Training (Tinker+WandB)", bool(get_env_value("TINKER_API_KEY") and get_env_value("WANDB_API_KEY"))),
        ("Skills Hub (GitHub)", bool(get_env_value("GITHUB_TOKEN"))),
    ]
    enabled_count = sum(1 for _, enabled in optional_tools if enabled)
    missing_tool_names = [name for name, enabled in optional_tools if not enabled]

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN))
    print(color("│                 ✓ Setup Complete!                       │", Colors.GREEN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN))
    print()

    print(color("📁 Configuration", Colors.CYAN, Colors.BOLD))
    print(f"   {color('Settings:', Colors.YELLOW)} {get_config_path()}")
    print(f"   {color('API Keys:', Colors.YELLOW)} {get_env_path()}")
    print(f"   {color('Data:', Colors.YELLOW)}     {hermes_home}/cron/, sessions/, logs/")
    print()

    if selected_summary:
        print(color("🧭 You chose", Colors.CYAN, Colors.BOLD))
        for label, value in selected_summary.items():
            print(f"   {color(label + ':', Colors.YELLOW)} {value}")
        print()

    print(color("🚀 Get started", Colors.CYAN, Colors.BOLD))
    print(f"   {color('hermes', Colors.GREEN)}              Start chatting")
    print(f"   {color('hermes setup', Colors.GREEN)}        Re-run setup")
    print(f"   {color('hermes gateway', Colors.GREEN)}      Start messaging gateway")
    print(f"   {color('hermes doctor', Colors.GREEN)}       Check for issues")
    print()

    print(color("🔧 Optional tools", Colors.CYAN, Colors.BOLD))
    print_info(f"Enabled: {enabled_count}/{len(optional_tools)}")
    if missing_tool_names:
        print_info(f"Not enabled: {', '.join(missing_tool_names)}")
        print_info("Run 'hermes setup' again anytime to add them.")

    if api_key_alerts:
        print()
        print(color("🔑 API key checks", Colors.CYAN, Colors.BOLD))
        for integration, missing_keys in api_key_alerts:
            print_warning(f"{integration}: missing/empty {', '.join(missing_keys)}")


def run_setup_wizard(args):
    """Run the interactive setup wizard."""
    ensure_hermes_home()
    config = load_config()
    hermes_home = get_hermes_home()

    from hermes_cli.config import (
        get_missing_env_vars,
        get_missing_config_fields,
        check_config_version,
        migrate_config,
    )

    missing_required = [v for v in get_missing_env_vars(required_only=False) if v.get("is_required")]
    missing_optional = [v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()

    is_existing = (
        get_env_value("OPENROUTER_API_KEY") is not None
        or get_env_value("OPENAI_BASE_URL") is not None
        or get_config_path().exists()
    )
    has_missing = bool(missing_required or missing_optional or missing_config or current_ver < latest_ver)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│             ⚕ Hermes Agent Setup Wizard                │", Colors.MAGENTA))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("│  Configure Hermes in a few guided steps.               │", Colors.MAGENTA))
    print(color("│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))

    selected_summary = {}
    selected_tool_keys_all = set()
    selected_platforms_all = []
    selected_provider_name = None

    def set_summary(label: str, value):
        if value is None:
            return
        value_str = str(value).strip()
        if not value_str:
            return
        selected_summary[label] = value_str

    def key_present(env_key: str) -> bool:
        value = get_env_value(env_key)
        return bool(value and str(value).strip())

    def build_api_key_alerts():
        alerts = []

        def add_alert_if_missing(integration: str, required_keys: list, active: bool):
            if not active:
                return
            missing = [k for k in required_keys if not key_present(k)]
            if missing:
                alerts.append((integration, missing))

        add_alert_if_missing(
            "OpenRouter (provider/tools)",
            ["OPENROUTER_API_KEY"],
            selected_provider_name in ("openrouter", "nous"),
        )
        add_alert_if_missing(
            "Firecrawl",
            ["FIRECRAWL_API_KEY"],
            "firecrawl" in selected_tool_keys_all or key_present("FIRECRAWL_API_KEY"),
        )
        add_alert_if_missing(
            "Browserbase",
            ["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
            "browserbase" in selected_tool_keys_all
            or key_present("BROWSERBASE_API_KEY")
            or key_present("BROWSERBASE_PROJECT_ID"),
        )
        add_alert_if_missing(
            "FAL",
            ["FAL_KEY"],
            "fal" in selected_tool_keys_all or key_present("FAL_KEY"),
        )
        add_alert_if_missing(
            "OpenAI Voice Tools",
            ["VOICE_TOOLS_OPENAI_KEY"],
            "openai_voice" in selected_tool_keys_all or key_present("VOICE_TOOLS_OPENAI_KEY"),
        )
        add_alert_if_missing(
            "ElevenLabs",
            ["ELEVENLABS_API_KEY"],
            "elevenlabs" in selected_tool_keys_all or key_present("ELEVENLABS_API_KEY"),
        )
        add_alert_if_missing(
            "RL Training (Tinker+WandB)",
            ["TINKER_API_KEY", "WANDB_API_KEY"],
            "rl_training" in selected_tool_keys_all
            or key_present("TINKER_API_KEY")
            or key_present("WANDB_API_KEY"),
        )
        add_alert_if_missing(
            "Skills Hub",
            ["GITHUB_TOKEN"],
            "github" in selected_tool_keys_all or key_present("GITHUB_TOKEN"),
        )
        add_alert_if_missing(
            "Telegram",
            ["TELEGRAM_BOT_TOKEN"],
            "Telegram" in selected_platforms_all or key_present("TELEGRAM_BOT_TOKEN"),
        )
        add_alert_if_missing(
            "Discord",
            ["DISCORD_BOT_TOKEN"],
            "Discord" in selected_platforms_all or key_present("DISCORD_BOT_TOKEN"),
        )
        add_alert_if_missing(
            "Slack",
            ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
            "Slack" in selected_platforms_all
            or key_present("SLACK_BOT_TOKEN")
            or key_present("SLACK_APP_TOKEN"),
        )

        return alerts

    def save_checkpoint():
        save_config(config)

    def print_step(step_num: int, total_steps: int, title: str):
        print()
        print(color(f"[{step_num}/{total_steps}] {title}", Colors.CYAN, Colors.BOLD))

    def normalize_path(path_value: str) -> str:
        if path_value.startswith("~"):
            return str(Path.home()) + path_value[1:]
        return path_value

    def format_minutes(minutes: int) -> str:
        total = max(0, int(minutes))
        hours, mins = divmod(total, 60)
        if hours and mins:
            return f"{hours}h{mins}m"
        if hours:
            return f"{hours}h"
        return f"{mins}m"

    def parse_duration_minutes(raw: str, default: int) -> int:
        value = (raw or "").strip().lower()
        if not value:
            return default
        if value.isdigit():
            parsed = int(value)
            if parsed <= 0:
                raise ValueError
            return parsed

        import re

        compact = value.replace(" ", "")
        matches = list(re.finditer(r"(\d+)([hm])", compact))
        if not matches or "".join(m.group(0) for m in matches) != compact:
            raise ValueError

        total = 0
        for match in matches:
            amount = int(match.group(1))
            unit = match.group(2)
            total += amount * 60 if unit == "h" else amount
        if total <= 0:
            raise ValueError
        return total

    def compression_label(threshold: float) -> str:
        if threshold <= 0.79:
            return "low"
        if threshold >= 0.90:
            return "high"
        return "balanced"

    def reset_label(reset_cfg: dict) -> str:
        mode = reset_cfg.get("mode", "both")
        idle = reset_cfg.get("idle_minutes", 1440)
        hour = reset_cfg.get("at_hour", 4)
        if mode == "both":
            return f"inactivity + daily ({format_minutes(idle)}, {hour}:00)"
        if mode == "idle":
            return f"inactivity only ({format_minutes(idle)})"
        if mode == "daily":
            return f"daily only ({hour}:00)"
        return "off"

    def validate_openrouter_key(api_key: str):
        if not api_key:
            return
        try:
            import httpx
            response = httpx.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            if response.status_code == 401:
                print_warning("OpenRouter key looks invalid (401 Unauthorized).")
            elif response.status_code >= 400:
                print_warning(f"OpenRouter key check returned HTTP {response.status_code}. Saved anyway.")
            else:
                print_success("OpenRouter API key verified")
        except Exception:
            print_warning("Could not verify OpenRouter key (network/API issue). Saved anyway.")

    def validate_docker_backend() -> bool:
        import shutil
        import subprocess

        if not shutil.which("docker"):
            print_warning("Docker is not installed.")
            print_info("Install Docker: https://docs.docker.com/get-docker/")
            return False
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=8)
            if result.returncode != 0:
                print_warning("Docker is installed but the daemon is not running.")
                return False
        except Exception:
            print_warning("Could not validate Docker daemon status.")
            return False
        return True

    def validate_singularity_backend() -> bool:
        import shutil
        import subprocess

        executable = shutil.which("apptainer") or shutil.which("singularity")
        if not executable:
            print_warning("Neither apptainer nor singularity was found.")
            return False
        try:
            result = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=8)
            if result.returncode != 0:
                print_warning(f"{executable} is installed but failed to respond.")
                return False
        except Exception:
            print_warning("Could not validate Singularity/Apptainer installation.")
            return False
        return True

    def validate_ssh_connection(host: str, user: str, port: str, key_path: str):
        import subprocess

        if not host:
            return
        target = f"{user}@{host}" if user else host
        cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
        if port and port != "22":
            cmd.extend(["-p", port])
        if key_path:
            cmd.extend(["-i", normalize_path(key_path)])
        cmd.extend([target, "echo ok"])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_success(f"SSH connectivity check passed ({target})")
            else:
                print_warning(f"Could not connect to {target}. Check SSH settings.")
        except Exception:
            print_warning(f"Could not test SSH connectivity for {target}.")

    def setup_telegram():
        print()
        print_header("Telegram")
        existing = get_env_value("TELEGRAM_BOT_TOKEN")
        if existing and not prompt_yes_no("Telegram is already configured. Update it?", False):
            return
        token = prompt("Telegram bot token", password=True)
        if token:
            save_env_value("TELEGRAM_BOT_TOKEN", token)
            print_success("Telegram token saved")
        allowed_default = get_env_value("TELEGRAM_ALLOWED_USERS") or ""
        allowed = prompt("Allowed Telegram user IDs (comma-separated, optional)", allowed_default)
        if allowed:
            save_env_value("TELEGRAM_ALLOWED_USERS", allowed.replace(" ", ""))
        home_default = get_env_value("TELEGRAM_HOME_CHANNEL") or ""
        home = prompt("Telegram home channel/user ID (optional)", home_default)
        if home:
            save_env_value("TELEGRAM_HOME_CHANNEL", home)

    def setup_discord():
        print()
        print_header("Discord")
        existing = get_env_value("DISCORD_BOT_TOKEN")
        if existing and not prompt_yes_no("Discord is already configured. Update it?", False):
            return
        token = prompt("Discord bot token", password=True)
        if token:
            save_env_value("DISCORD_BOT_TOKEN", token)
            print_success("Discord token saved")
        allowed_default = get_env_value("DISCORD_ALLOWED_USERS") or ""
        allowed = prompt("Allowed Discord IDs/usernames (comma-separated, optional)", allowed_default)
        if allowed:
            save_env_value("DISCORD_ALLOWED_USERS", allowed.replace(" ", ""))
        home_default = get_env_value("DISCORD_HOME_CHANNEL") or ""
        home = prompt("Discord home channel ID (optional)", home_default)
        if home:
            save_env_value("DISCORD_HOME_CHANNEL", home)

    def setup_slack():
        print()
        print_header("Slack")
        existing = get_env_value("SLACK_BOT_TOKEN")
        if existing and not prompt_yes_no("Slack is already configured. Update it?", False):
            return
        bot_token = prompt("Slack bot token (xoxb-...)", password=True)
        if bot_token:
            save_env_value("SLACK_BOT_TOKEN", bot_token)
        app_token = prompt("Slack app token (xapp-...)", password=True)
        if app_token:
            save_env_value("SLACK_APP_TOKEN", app_token)
        if bot_token or app_token:
            print_success("Slack tokens saved")
        allowed_default = get_env_value("SLACK_ALLOWED_USERS") or ""
        allowed = prompt("Allowed Slack user IDs (comma-separated, optional)", allowed_default)
        if allowed:
            save_env_value("SLACK_ALLOWED_USERS", allowed.replace(" ", ""))
        home_default = get_env_value("SLACK_HOME_CHANNEL") or ""
        home = prompt("Slack home channel ID (optional)", home_default)
        if home:
            save_env_value("SLACK_HOME_CHANNEL", home)

    def setup_whatsapp():
        print()
        print_header("WhatsApp")
        enabled_default = (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true"
        enabled = prompt_yes_no("Enable WhatsApp bridge?", enabled_default or True)
        save_env_value("WHATSAPP_ENABLED", "true" if enabled else "false")
        if not enabled:
            return
        allowed_default = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
        allowed = prompt("Allowed phone numbers (comma-separated, optional)", allowed_default)
        if allowed:
            save_env_value("WHATSAPP_ALLOWED_USERS", allowed.replace(" ", ""))

    def configure_selected_messaging(platforms):
        for platform_name in platforms:
            if platform_name == "Telegram":
                setup_telegram()
            elif platform_name == "Discord":
                setup_discord()
            elif platform_name == "Slack":
                setup_slack()
            elif platform_name == "WhatsApp":
                setup_whatsapp()

    quick_mode = False
    if is_existing and has_missing:
        print()
        print_header("Existing Installation Detected")
        print_info("Choose quick setup to configure only missing items.")
        setup_choices = [
            "Quick setup - configure missing items",
            "Full setup - reconfigure everything",
            "Skip - exit setup",
        ]
        choice = prompt_choice("What would you like to do?", setup_choices, 0)
        if choice == 0:
            quick_mode = True
        elif choice == 2:
            print()
            print_info("Exiting. Run 'hermes setup' again when ready.")
            return
    elif is_existing and not has_missing:
        print()
        print_header("Configuration Status")
        print_success("Your configuration is complete.")
        if not prompt_yes_no("Reconfigure anyway?", False):
            print_info(f"Config: {get_config_path()}")
            print_info(f"Secrets: {get_env_path()}")
            return

    if quick_mode:
        set_summary("Setup mode", "Quick (missing items)")
        print()
        print_header("Quick Setup")
        for var in missing_required:
            _prompt_api_key(var)

        missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
        if missing_tools:
            tool_labels = [v.get("description", v["name"]) for v in missing_tools]
            selected = prompt_checklist("Which optional tool keys would you like to set up?", tool_labels)
            selected_tool_labels = [tool_labels[i] for i in selected]
            quick_tool_key_map = {
                "FIRECRAWL_API_KEY": "firecrawl",
                "BROWSERBASE_API_KEY": "browserbase",
                "BROWSERBASE_PROJECT_ID": "browserbase",
                "FAL_KEY": "fal",
                "VOICE_TOOLS_OPENAI_KEY": "openai_voice",
                "ELEVENLABS_API_KEY": "elevenlabs",
                "TINKER_API_KEY": "rl_training",
                "WANDB_API_KEY": "rl_training",
                "GITHUB_TOKEN": "github",
            }
            for idx in selected:
                _prompt_api_key(missing_tools[idx])
            if selected_tool_labels:
                set_summary("Optional tools", ", ".join(selected_tool_labels))
            selected_tool_keys_all.update(
                quick_tool_key_map.get(missing_tools[idx].get("name", ""), "") for idx in selected
            )
            selected_tool_keys_all.discard("")

        missing_messaging = [v for v in missing_optional if v.get("category") == "messaging" and not v.get("advanced")]
        platforms = []
        for var in missing_messaging:
            name = var["name"]
            if "TELEGRAM" in name:
                platforms.append("Telegram")
            elif "DISCORD" in name:
                platforms.append("Discord")
            elif "SLACK" in name:
                platforms.append("Slack")
            elif "WHATSAPP" in name:
                platforms.append("WhatsApp")
        platforms = list(dict.fromkeys(platforms))
        if platforms:
            selected = prompt_checklist("Which messaging platforms would you like to configure?", platforms)
            selected_platforms = [platforms[i] for i in selected]
            configure_selected_messaging(selected_platforms)
            if selected_platforms:
                set_summary("Messaging", ", ".join(selected_platforms))
                selected_platforms_all.extend(selected_platforms)
            if selected_platforms:
                current_cwd = get_env_value("MESSAGING_CWD") or str(Path.home())
                cwd_input = prompt("Messaging working directory", current_cwd)
                save_env_value("MESSAGING_CWD", normalize_path(cwd_input))
                set_summary("Messaging CWD", normalize_path(cwd_input))

        config["_config_version"] = latest_ver
        save_checkpoint()
        migrate_config(interactive=False, quiet=True)
        _print_setup_summary(
            load_config(),
            hermes_home,
            selected_summary,
            build_api_key_alerts(),
        )
        return

    express_mode = False
    if not is_existing:
        print()
        setup_mode = prompt_choice(
            "How would you like to set up Hermes?",
            [
                "Express setup (API key + model, ~1 minute)",
                "Full setup (all options, ~5 minutes)",
            ],
            0,
        )
        express_mode = setup_mode == 0
    set_summary("Setup mode", "Express" if express_mode else "Full")

    total_steps = 2 if express_mode else 5
    step_num = 1

    # Step 1: Provider
    print_step(step_num, total_steps, "Inference Provider")
    print_info("Choose your primary model provider.")

    from hermes_cli.auth import (
        get_active_provider,
        PROVIDER_REGISTRY,
        fetch_nous_models,
        resolve_nous_runtime_credentials,
    )

    existing_custom = get_env_value("OPENAI_BASE_URL")
    existing_or = get_env_value("OPENROUTER_API_KEY")
    active_oauth = get_active_provider()
    has_any_provider = bool(active_oauth or existing_custom or existing_or)

    if active_oauth and active_oauth in PROVIDER_REGISTRY:
        keep_label = f"Keep current ({PROVIDER_REGISTRY[active_oauth].name})"
    elif existing_custom:
        keep_label = f"Keep current (Custom: {existing_custom})"
    elif existing_or:
        keep_label = "Keep current (OpenRouter)"
    else:
        keep_label = None

    provider_choices = [
        "Nous Portal login",
        "OpenRouter API key",
        "Custom OpenAI-compatible endpoint",
    ]
    if keep_label:
        provider_choices.append(keep_label)

    default_provider = len(provider_choices) - 1 if has_any_provider else 1
    provider_idx = prompt_choice("Select provider:", provider_choices, default_provider)
    provider_summary = {
        0: "Nous Portal",
        1: "OpenRouter",
        2: "Custom endpoint",
    }.get(provider_idx, "Keep current")
    set_summary("Provider", provider_summary)

    selected_provider = None
    nous_models = []
    custom_model_set = False

    if provider_idx == 0:
        selected_provider = "nous"
        selected_provider_name = "nous"
        print_info("Opening browser for Nous Portal login.")
        try:
            import argparse
            from hermes_cli.auth import _login_nous

            mock_args = argparse.Namespace(
                portal_url=None,
                inference_url=None,
                client_id=None,
                scope=None,
                no_browser=False,
                timeout=15.0,
                ca_bundle=None,
                insecure=False,
            )
            _login_nous(mock_args, PROVIDER_REGISTRY["nous"])
            try:
                creds = resolve_nous_runtime_credentials(
                    min_key_ttl_seconds=5 * 60,
                    timeout_seconds=15.0,
                )
                nous_models = fetch_nous_models(
                    inference_base_url=creds.get("base_url", ""),
                    api_key=creds.get("api_key", ""),
                )
            except Exception as exc:
                logger.debug("Could not fetch Nous models after login: %s", exc)
        except SystemExit:
            print_warning("Nous login cancelled.")
            selected_provider = None
        except Exception as exc:
            print_error(f"Nous login failed: {exc}")
            selected_provider = None
    elif provider_idx == 1:
        selected_provider = "openrouter"
        selected_provider_name = "openrouter"
        if existing_or:
            print_info("OpenRouter key is already configured.")
            if prompt_yes_no("Update OpenRouter API key?", False):
                api_key = prompt("OpenRouter API key", password=True)
                if api_key:
                    save_env_value("OPENROUTER_API_KEY", api_key)
                    validate_openrouter_key(api_key)
        else:
            api_key = prompt("OpenRouter API key", password=True)
            if api_key:
                save_env_value("OPENROUTER_API_KEY", api_key)
                validate_openrouter_key(api_key)
            else:
                print_warning("No key entered. Hermes needs a provider key to chat.")
        if existing_custom:
            save_env_value("OPENAI_BASE_URL", "")
            save_env_value("OPENAI_API_KEY", "")
    elif provider_idx == 2:
        selected_provider = "custom"
        selected_provider_name = "custom"
        current_url = get_env_value("OPENAI_BASE_URL") or ""
        current_model = config.get("model", "")
        base_url = prompt("API base URL", current_url)
        api_key = prompt("API key", password=True)
        model_name = prompt("Model name", current_model)
        if base_url:
            save_env_value("OPENAI_BASE_URL", base_url)
        if api_key:
            save_env_value("OPENAI_API_KEY", api_key)
        if model_name:
            config["model"] = model_name
            save_env_value("LLM_MODEL", model_name)
            custom_model_set = True

    if selected_provider in ("nous", "custom") and not get_env_value("OPENROUTER_API_KEY"):
        print_info("Optional: add OpenRouter key for vision/web/MoA tools.")
        tools_key = prompt("OpenRouter API key for tools (optional)", password=True)
        if tools_key:
            save_env_value("OPENROUTER_API_KEY", tools_key)
            validate_openrouter_key(tools_key)

    save_checkpoint()

    # Step 2: Model
    step_num += 1
    print_step(step_num, total_steps, "Default Model")
    current_model = config.get("model", "anthropic/claude-opus-4.6")
    print_info(f"Current model: {current_model}")

    if not custom_model_set:
        if selected_provider == "nous" and nous_models:
            model_choices = [f"{m}" for m in nous_models] + ["Custom model", f"Keep current ({current_model})"]
            model_idx = prompt_choice("Select default model:", model_choices, len(model_choices) - 1)
            if model_idx < len(nous_models):
                config["model"] = nous_models[model_idx]
                save_env_value("LLM_MODEL", nous_models[model_idx])
            elif model_idx == len(nous_models):
                custom = prompt("Custom model name")
                if custom:
                    config["model"] = custom
                    save_env_value("LLM_MODEL", custom)
        else:
            from hermes_cli.models import model_ids, menu_labels

            ids = model_ids()
            model_choices = menu_labels() + ["Custom model", f"Keep current ({current_model})"]
            model_idx = prompt_choice("Select default model:", model_choices, len(model_choices) - 1)
            if model_idx < len(ids):
                config["model"] = ids[model_idx]
                save_env_value("LLM_MODEL", ids[model_idx])
            elif model_idx == len(ids):
                custom = prompt("Custom model name")
                if custom:
                    config["model"] = custom
                    save_env_value("LLM_MODEL", custom)

    save_checkpoint()
    set_summary("Model", config.get("model", "not set"))

    if express_mode:
        config.setdefault("terminal", {})
        if not config["terminal"].get("backend"):
            config["terminal"]["backend"] = "local"
            save_env_value("TERMINAL_ENV", "local")
        set_summary("Terminal backend", config["terminal"].get("backend", "local"))
        config.setdefault("display", {})
        config["display"].setdefault("tool_progress", "all")
        set_summary("Tool progress", config["display"].get("tool_progress", "all"))
        config.setdefault("compression", {})
        config["compression"]["enabled"] = True
        config["compression"].setdefault("threshold", 0.85)
        set_summary(
            "Compression",
            f"{compression_label(float(config['compression'].get('threshold', 0.85)))} "
            f"({float(config['compression'].get('threshold', 0.85)):.2f})",
        )
        config.setdefault("session_reset", {})
        config["session_reset"].setdefault("mode", "both")
        config["session_reset"].setdefault("idle_minutes", 1440)
        config["session_reset"].setdefault("at_hour", 4)
        set_summary("Session reset", reset_label(config["session_reset"]))
        if not get_env_value("HERMES_MAX_ITERATIONS"):
            max_turns = str(config.get("max_turns", 60))
            save_env_value("HERMES_MAX_ITERATIONS", max_turns)
        set_summary("Max iterations", get_env_value("HERMES_MAX_ITERATIONS") or str(config.get("max_turns", 60)))
        config["_config_version"] = latest_ver
        save_checkpoint()
        migrate_config(interactive=False, quiet=True)
        _print_setup_summary(
            load_config(),
            hermes_home,
            selected_summary,
            build_api_key_alerts(),
        )
        return

    # Step 3: Terminal backend
    step_num += 1
    print_step(step_num, total_steps, "Terminal Backend")
    print_info("Select where terminal commands run.")

    import platform

    is_linux = platform.system() == "Linux"
    current_backend = config.get("terminal", {}).get("backend", "local")
    terminal_choices = [
        "Local (runs on this machine)",
        "Docker (isolated container)",
    ]
    if is_linux:
        terminal_choices.append("Singularity/Apptainer (HPC)")
    terminal_choices.extend([
        "Modal (cloud execution)",
        "SSH (remote server)",
        f"Keep current ({current_backend})",
    ])

    if is_linux:
        idx_to_backend = {0: "local", 1: "docker", 2: "singularity", 3: "modal", 4: "ssh"}
        keep_idx = 5
    else:
        idx_to_backend = {0: "local", 1: "docker", 2: "modal", 3: "ssh"}
        keep_idx = 4

    selected_backend = None
    while True:
        terminal_idx = prompt_choice("Select terminal backend:", terminal_choices, keep_idx)
        selected_backend = idx_to_backend.get(terminal_idx)
        if selected_backend is None:
            break
        if selected_backend == "docker" and not validate_docker_backend():
            if not prompt_yes_no("Continue with Docker backend anyway?", False):
                continue
        if selected_backend == "singularity" and not validate_singularity_backend():
            if not prompt_yes_no("Continue with Singularity backend anyway?", False):
                continue
        break

    effective_backend = selected_backend or current_backend

    if effective_backend == "local":
        if selected_backend == "local":
            config.setdefault("terminal", {})["backend"] = "local"

        has_existing_sudo = bool(get_env_value("SUDO_PASSWORD"))
        if prompt_yes_no("Enable sudo support?", has_existing_sudo):
            print_warning("Sudo password is stored in plaintext in ~/.hermes/.env")
            sudo_pass = prompt("Sudo password (leave empty to keep current)", password=True)
            if sudo_pass:
                save_env_value("SUDO_PASSWORD", sudo_pass)
        elif has_existing_sudo and prompt_yes_no("Disable stored sudo password?", False):
            save_env_value("SUDO_PASSWORD", "")
            print_info("Sudo password cleared.")
    elif selected_backend == "docker":
        config.setdefault("terminal", {})["backend"] = "docker"
        default_docker = config.get("terminal", {}).get("docker_image", "nikolaik/python-nodejs:python3.11-nodejs20")
        config["terminal"]["docker_image"] = prompt("Docker image", default_docker)
    elif selected_backend == "singularity":
        config.setdefault("terminal", {})["backend"] = "singularity"
        default_img = config.get("terminal", {}).get("singularity_image", "docker://nikolaik/python-nodejs:python3.11-nodejs20")
        config["terminal"]["singularity_image"] = prompt("Singularity image", default_img)
    elif selected_backend == "modal":
        config.setdefault("terminal", {})["backend"] = "modal"
        default_modal = config.get("terminal", {}).get("modal_image", "nikolaik/python-nodejs:python3.11-nodejs20")
        config["terminal"]["modal_image"] = prompt("Modal image", default_modal)
        token_id = prompt("Modal token ID", get_env_value("MODAL_TOKEN_ID") or "")
        token_secret = prompt("Modal token secret", password=True)
        if token_id:
            save_env_value("MODAL_TOKEN_ID", token_id)
        if token_secret:
            save_env_value("MODAL_TOKEN_SECRET", token_secret)
    elif selected_backend == "ssh":
        config.setdefault("terminal", {})["backend"] = "ssh"
        ssh_host = prompt("SSH host", get_env_value("TERMINAL_SSH_HOST") or "")
        ssh_user = prompt("SSH user", get_env_value("TERMINAL_SSH_USER") or os.getenv("USER", ""))
        ssh_port = prompt("SSH port", get_env_value("TERMINAL_SSH_PORT") or "22")
        ssh_key = prompt("SSH key path (optional)", get_env_value("TERMINAL_SSH_KEY") or "~/.ssh/id_rsa")
        if ssh_host:
            save_env_value("TERMINAL_SSH_HOST", ssh_host)
        if ssh_user:
            save_env_value("TERMINAL_SSH_USER", ssh_user)
        if ssh_port:
            save_env_value("TERMINAL_SSH_PORT", ssh_port)
        if ssh_key:
            save_env_value("TERMINAL_SSH_KEY", ssh_key)
        validate_ssh_connection(ssh_host, ssh_user, ssh_port, ssh_key)

    if selected_backend:
        save_env_value("TERMINAL_ENV", selected_backend)
        docker_image = config.get("terminal", {}).get("docker_image")
        if docker_image:
            save_env_value("TERMINAL_DOCKER_IMAGE", docker_image)
    set_summary("Terminal backend", effective_backend)

    save_checkpoint()

    # Step 4: Agent settings
    step_num += 1
    print_step(step_num, total_steps, "Agent Settings")
    config.setdefault("display", {})
    config.setdefault("compression", {})
    config.setdefault("session_reset", {})
    current_max = get_env_value("HERMES_MAX_ITERATIONS") or str(config.get("max_turns", 60))
    current_progress = config["display"].get("tool_progress", "all")
    current_threshold = config["compression"].get("threshold", 0.85)
    current_reset = config["session_reset"].get("mode", "both")
    try:
        current_threshold_float = float(current_threshold)
    except (TypeError, ValueError):
        current_threshold_float = 0.85
    print_info(f"Max iterations: {current_max} (recommended: 30-60)")
    print_info(f"Tool progress: {current_progress}")
    print_info(
        f"Context compression: {compression_label(current_threshold_float)} "
        f"({current_threshold_float:.2f})"
    )
    print_info(f"Session reset: {reset_label(config['session_reset'])}")

    if prompt_yes_no("Customize advanced settings?", False):
        max_iter_str = prompt("Max iterations", current_max)
        try:
            max_iter = int(max_iter_str)
            if max_iter > 0:
                save_env_value("HERMES_MAX_ITERATIONS", str(max_iter))
                config["max_turns"] = max_iter
        except ValueError:
            print_warning("Invalid max iterations, keeping current value.")

        progress_labels = [
            "off (silent)",
            "new (show when tool changes)",
            "all (show every tool call)",
            "verbose (full debug output)",
        ]
        progress_map = ["off", "new", "all", "verbose"]
        default_progress = progress_map.index(current_progress) if current_progress in progress_map else 2
        progress_idx = prompt_choice("Tool progress display:", progress_labels, default_progress)
        config["display"]["tool_progress"] = progress_map[progress_idx]

        compression_choices = [
            "Low (0.75) - compress sooner",
            "Balanced (0.85) - recommended",
            "High (0.92) - compress later",
            "Custom value (0.50-0.95)",
            f"Keep current ({current_threshold_float:.2f})",
        ]
        default_compression_idx = 1
        if current_threshold_float <= 0.79:
            default_compression_idx = 0
        elif current_threshold_float >= 0.90:
            default_compression_idx = 2
        compression_idx = prompt_choice("Context compression:", compression_choices, default_compression_idx)
        config["compression"]["enabled"] = True
        if compression_idx == 0:
            config["compression"]["threshold"] = 0.75
        elif compression_idx == 1:
            config["compression"]["threshold"] = 0.85
        elif compression_idx == 2:
            config["compression"]["threshold"] = 0.92
        elif compression_idx == 3:
            custom_threshold = prompt("Custom compression threshold (0.50-0.95)", f"{current_threshold_float:.2f}")
            try:
                custom_value = float(custom_threshold)
                if 0.50 <= custom_value <= 0.95:
                    config["compression"]["threshold"] = round(custom_value, 2)
                else:
                    print_warning("Custom threshold out of range, keeping current value.")
            except ValueError:
                print_warning("Invalid custom threshold, keeping current value.")

        reset_choices = [
            "Inactivity + daily reset (recommended)",
            "Inactivity only",
            "Daily only",
            "Never auto-reset",
            "Keep current",
        ]
        default_reset = {"both": 0, "idle": 1, "daily": 2, "none": 3}.get(current_reset, 0)
        reset_idx = prompt_choice("Session reset mode:", reset_choices, default_reset)
        if reset_idx == 0:
            config["session_reset"]["mode"] = "both"
            current_idle_default = int(config["session_reset"].get("idle_minutes", 1440))
            idle = prompt(
                "Inactivity timeout (e.g. 90m, 1h30m, 2h)",
                format_minutes(current_idle_default),
            )
            hour = prompt("Daily reset hour (0-23)", str(config["session_reset"].get("at_hour", 4)))
            try:
                idle_val = parse_duration_minutes(idle, current_idle_default)
                config["session_reset"]["idle_minutes"] = idle_val
            except ValueError:
                print_warning("Invalid inactivity duration, keeping current value.")
            try:
                hour_val = int(hour)
                if 0 <= hour_val <= 23:
                    config["session_reset"]["at_hour"] = hour_val
            except ValueError:
                pass
        elif reset_idx == 1:
            config["session_reset"]["mode"] = "idle"
            current_idle_default = int(config["session_reset"].get("idle_minutes", 1440))
            idle = prompt(
                "Inactivity timeout (e.g. 90m, 1h30m, 2h)",
                format_minutes(current_idle_default),
            )
            try:
                idle_val = parse_duration_minutes(idle, current_idle_default)
                config["session_reset"]["idle_minutes"] = idle_val
            except ValueError:
                print_warning("Invalid inactivity duration, keeping current value.")
        elif reset_idx == 2:
            config["session_reset"]["mode"] = "daily"
            hour = prompt("Daily reset hour (0-23)", str(config["session_reset"].get("at_hour", 4)))
            try:
                hour_val = int(hour)
                if 0 <= hour_val <= 23:
                    config["session_reset"]["at_hour"] = hour_val
            except ValueError:
                pass
        elif reset_idx == 3:
            config["session_reset"]["mode"] = "none"
    else:
        config["display"].setdefault("tool_progress", "all")
        config["compression"]["enabled"] = True
        config["compression"].setdefault("threshold", 0.85)
        config["session_reset"].setdefault("mode", "both")
        config["session_reset"].setdefault("idle_minutes", 1440)
        config["session_reset"].setdefault("at_hour", 4)
        if not get_env_value("HERMES_MAX_ITERATIONS"):
            save_env_value("HERMES_MAX_ITERATIONS", current_max)

    save_checkpoint()
    set_summary("Max iterations", get_env_value("HERMES_MAX_ITERATIONS") or str(config.get("max_turns", 60)))
    set_summary("Tool progress", config["display"].get("tool_progress", "all"))
    threshold_value = float(config["compression"].get("threshold", 0.85))
    set_summary("Compression", f"{compression_label(threshold_value)} ({threshold_value:.2f})")
    set_summary("Session reset", reset_label(config["session_reset"]))

    # Step 5: Tools & integrations
    step_num += 1
    print_step(step_num, total_steps, "Tools & Integrations")
    print_info("Select messaging platforms and optional tool keys.")

    messaging_platforms = [
        ("Telegram", "TELEGRAM_BOT_TOKEN"),
        ("Discord", "DISCORD_BOT_TOKEN"),
        ("Slack", "SLACK_BOT_TOKEN"),
        ("WhatsApp", "WHATSAPP_ENABLED"),
    ]
    pre_selected_messaging = []
    for idx, (_, env_var) in enumerate(messaging_platforms):
        val = get_env_value(env_var)
        if env_var == "WHATSAPP_ENABLED":
            if (val or "").lower() == "true":
                pre_selected_messaging.append(idx)
        elif val:
            pre_selected_messaging.append(idx)

    selected_msg_idx = prompt_checklist(
        "Which messaging platforms would you like to configure?",
        [name for name, _ in messaging_platforms],
        pre_selected=pre_selected_messaging,
    )
    selected_platforms = [messaging_platforms[i][0] for i in selected_msg_idx]
    selected_platforms_all = list(selected_platforms)
    configure_selected_messaging(selected_platforms)
    if selected_platforms:
        set_summary("Messaging", ", ".join(selected_platforms))

    if selected_platforms:
        current_cwd = get_env_value("MESSAGING_CWD") or str(Path.home())
        cwd_input = prompt("Messaging working directory", current_cwd)
        save_env_value("MESSAGING_CWD", normalize_path(cwd_input))
        set_summary("Messaging CWD", normalize_path(cwd_input))

    tool_categories = [
        {"label": "Web Search/Scraping (Firecrawl)", "key": "firecrawl", "check": ["FIRECRAWL_API_KEY"]},
        {"label": "Browser Automation (Browserbase)", "key": "browserbase", "check": ["BROWSERBASE_API_KEY"]},
        {"label": "Image Generation (FAL)", "key": "fal", "check": ["FAL_KEY"]},
        {"label": "Voice Tools (OpenAI)", "key": "openai_voice", "check": ["VOICE_TOOLS_OPENAI_KEY"]},
        {"label": "Premium TTS (ElevenLabs)", "key": "elevenlabs", "check": ["ELEVENLABS_API_KEY"]},
        {"label": "RL Training (Tinker + WandB)", "key": "rl_training", "check": ["TINKER_API_KEY", "WANDB_API_KEY"]},
        {"label": "Skills Hub (GitHub token)", "key": "github", "check": ["GITHUB_TOKEN"]},
    ]
    pre_selected_tools = [i for i, cat in enumerate(tool_categories) if all(get_env_value(k) for k in cat["check"])]
    selected_tools_idx = prompt_checklist(
        "Which optional tools would you like to configure?",
        [cat["label"] for cat in tool_categories],
        pre_selected=pre_selected_tools,
    )
    selected_tool_keys = {tool_categories[i]["key"] for i in selected_tools_idx}
    selected_tool_keys_all = set(selected_tool_keys)
    if selected_tool_keys:
        selected_tool_labels = [cat["label"] for cat in tool_categories if cat["key"] in selected_tool_keys]
        set_summary("Optional tools", ", ".join(selected_tool_labels))

    if "firecrawl" in selected_tool_keys:
        key = prompt("Firecrawl API key", password=True)
        if key:
            save_env_value("FIRECRAWL_API_KEY", key)
    if "browserbase" in selected_tool_keys:
        key = prompt("Browserbase API key", password=True)
        project_id = prompt("Browserbase project ID", get_env_value("BROWSERBASE_PROJECT_ID") or "")
        if key:
            save_env_value("BROWSERBASE_API_KEY", key)
        if project_id:
            save_env_value("BROWSERBASE_PROJECT_ID", project_id)
    if "fal" in selected_tool_keys:
        key = prompt("FAL API key", password=True)
        if key:
            save_env_value("FAL_KEY", key)
    if "openai_voice" in selected_tool_keys:
        key = prompt("OpenAI API key (voice tools)", password=True)
        if key:
            save_env_value("VOICE_TOOLS_OPENAI_KEY", key)
    if "elevenlabs" in selected_tool_keys:
        key = prompt("ElevenLabs API key", password=True)
        if key:
            save_env_value("ELEVENLABS_API_KEY", key)
    if "rl_training" in selected_tool_keys:
        tinker = prompt("Tinker API key", password=True)
        wandb = prompt("WandB API key", password=True)
        if tinker:
            save_env_value("TINKER_API_KEY", tinker)
        if wandb:
            save_env_value("WANDB_API_KEY", wandb)
    if "github" in selected_tool_keys:
        token = prompt("GitHub token", password=True)
        if token:
            save_env_value("GITHUB_TOKEN", token)

    config["_config_version"] = latest_ver
    save_checkpoint()
    migrate_config(interactive=False, quiet=True)
    _print_setup_summary(
        load_config(),
        hermes_home,
        selected_summary,
        build_api_key_alerts(),
    )
