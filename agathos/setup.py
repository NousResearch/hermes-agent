"""
Interactive setup for Agathos - Agent Resource Guardian & Unified Supervisor.

Follows the Hermes CLI pattern with:
  - Color-coded UI (Colors class)
  - prompt_choice() / prompt_yes_no() helpers
  - Quick vs Full setup modes
  - Section-specific configuration

Usage (via CLI):
  argus setup              # Interactive setup
  argus setup quick        # Quick setup (essential settings)
  argus setup core         # Core watcher settings
  argus setup modules      # Monitoring modules
  argus setup alerts       # Notification settings
  argus setup full         # Full reconfiguration
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add hermes-agent to path for hermes_constants
HERMES_AGENT_PATH = Path("~/hermes/hermes-agent").expanduser()
if str(HERMES_AGENT_PATH) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_PATH))

try:
    from hermes_constants import get_hermes_home
    _HERMES_HOME = get_hermes_home()
except ImportError:
    _HERMES_HOME = Path(os.path.expanduser("~/.hermes"))

_ARGUS_HOME = Path(os.path.expanduser("~/hermes"))

# === UI Colors (same as hermes_cli style) ===
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def color(text: str, c: str) -> str:
    """Wrap text with color codes."""
    return f"{c}{text}{Colors.RESET}"


# === UI Helpers (same pattern as hermes_cli/setup.py) ===
def prompt_choice(question: str, choices: List[str], default: int = 0) -> int:
    """Show numbered choices and return selected index."""
    print()
    print(color(f"  {question}", Colors.CYAN))
    for i, choice in enumerate(choices):
        marker = color(">", Colors.GREEN) if i == default else " "
        print(f"    {marker} [{i + 1}] {choice}")
    print()

    while True:
        try:
            inp = input(color("  Enter choice: ", Colors.CYAN)).strip()
            if not inp:
                return default
            idx = int(inp) - 1
            if 0 <= idx < len(choices):
                return idx
            print(color(f"  Please enter 1-{len(choices)}", Colors.RED))
        except ValueError:
            print(color("  Please enter a number", Colors.RED))
        except KeyboardInterrupt:
            print()
            print(color("  Setup cancelled.", Colors.YELLOW))
            sys.exit(0)


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Ask yes/no question."""
    default_str = "Y/n" if default else "y/N"
    prompt = color(f"  {question} [{default_str}]: ", Colors.CYAN)

    while True:
        try:
            inp = input(prompt).strip().lower()
            if not inp:
                return default
            if inp in ("y", "yes"):
                return True
            if inp in ("n", "no"):
                return False
            print(color("  Please enter 'y' or 'n'", Colors.RED))
        except KeyboardInterrupt:
            print()
            print(color("  Setup cancelled.", Colors.YELLOW))
            sys.exit(0)


def prompt_input(question: str, default: str = "") -> str:
    """Get text input with optional default."""
    if default:
        prompt = color(f"  {question} [{default}]: ", Colors.CYAN)
    else:
        prompt = color(f"  {question}: ", Colors.CYAN)

    try:
        inp = input(prompt).strip()
        return inp if inp else default
    except KeyboardInterrupt:
        print()
        print(color("  Setup cancelled.", Colors.YELLOW))
        sys.exit(0)


def print_header(title: str):
    """Print a section header."""
    print()
    print(color(f"  ┌─ {title} ", Colors.MAGENTA) + color("─" * (50 - len(title)), Colors.MAGENTA))


def print_success(msg: str):
    """Print success message."""
    print(color(f"  ✓ {msg}", Colors.GREEN))


def print_info(msg: str):
    """Print info message."""
    print(color(f"  ℹ {msg}", Colors.BLUE))


def print_warning(msg: str):
    """Print warning message."""
    print(color(f"  ⚠ {msg}", Colors.YELLOW))


def print_error(msg: str):
    """Print error message."""
    print(color(f"  ✗ {msg}", Colors.RED))


# === Config Management ===
def get_agathos_config_path() -> Path:
    """Get the Agathos config file path."""
    return _ARGUS_HOME / "config" / "argus.yaml"


def get_default_config() -> Dict[str, Any]:
    """Get default Agathos configuration matching argus.py defaults."""
    return {
        # Paths (using Hermes conventions)
        "agathos_db_path": str(_ARGUS_HOME / "data" / "watcher" / "agathos.db"),
        "state_db_path": str(_HERMES_HOME / "state.db"),
        "holographic_db_path": str(_HERMES_HOME / "holographic_memory.db"),
        
        # Feature toggles (all enabled by default for safety)
        "enabled": True,
        "wal_monitor_enabled": True,
        "metrics_enabled": True,
        "entropy_detection_enabled": True,
        "prime_directives_enabled": True,
        "actions_enabled": True,
        "resource_checks_enabled": True,
        "drift_detection_enabled": True,
        "provider_health_enabled": True,
        "cleanup_enabled": True,
        "cost_monitoring_enabled": False,  # Disabled by default (needs setup)
        "audit_trail_enabled": True,
        "notifications_enabled": False,  # Disabled by default (needs Telegram)
        "ml_data_enabled": False,  # Optional ML training data export
        
        # Polling interval (seconds)
        "poll_interval": 30,
        
        # Thresholds
        "entropy_threshold": 0.7,
        "directive_threshold": 0.5,
        "resource_threshold": 0.8,
        
        # Notifications (user fills these in)
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        
        # Circuit breaker (nested config)
        "circuit_breaker": {
            "enabled": False,
            "failure_threshold": 5,
            "recovery_timeout": 300,
        },
    }


def load_agathos_config() -> Dict[str, Any]:
    """Load existing config or return defaults."""
    config_path = get_agathos_config_path()
    if config_path.exists():
        try:
            # Try YAML first
            import yaml
            with open(config_path) as f:
                loaded = yaml.safe_load(f) or {}
                config = get_default_config()
                config.update(loaded)
                return config
        except ImportError:
            # Fall back to JSON
            try:
                json_path = config_path.with_suffix(".json")
                if json_path.exists():
                    with open(json_path) as f:
                        loaded = json.load(f)
                        config = get_default_config()
                        config.update(loaded)
                        return config
            except Exception:
                pass
    return get_default_config()


def save_agathos_config(config: Dict[str, Any]):
    """Save config to YAML file."""
    config_path = get_agathos_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fall back to JSON
        json_path = config_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)
        print_warning(f"YAML not available, saved to {json_path}")


# === Setup Sections ===
def setup_core_settings(config: Dict[str, Any]):
    """Configure core watcher settings."""
    print_header("Core Watcher Settings")
    
    print_info("ARGUS monitors your Hermes agent sessions for entropy and failures.")
    print()
    
    # Enable/disable
    config["enabled"] = prompt_yes_no("Enable ARGUS watcher?", config.get("enabled", True))
    if not config["enabled"]:
        print_warning("ARGUS will be disabled. Run setup again to enable.")
        return
    
    # Poll interval
    current_interval = config.get("poll_interval", 30)
    print_info(f"Current poll interval: {current_interval}s")
    interval_choice = prompt_choice(
        "How often should ARGUS check sessions?",
        ["30 seconds (default)", "60 seconds (slower, less CPU)", "10 seconds (faster detection)"],
        0 if current_interval == 30 else (1 if current_interval == 60 else 2)
    )
    config["poll_interval"] = [30, 60, 10][interval_choice]
    
    # WAL monitor (real-time detection)
    print()
    print_info("WAL Monitor: Real-time detection of stuck loops and repeat tool calls")
    config["wal_monitor_enabled"] = prompt_yes_no(
        "Enable WAL Monitor for real-time detection?",
        config.get("wal_monitor_enabled", True)
    )
    
    # Entropy detection
    print()
    print_info("Entropy Detection: Identifies low-quality reasoning and failures")
    config["entropy_detection_enabled"] = prompt_yes_no(
        "Enable entropy detection?",
        config.get("entropy_detection_enabled", True)
    )
    
    # Prime directives
    print()
    print_info("Prime Directives: Enforces ML data quality rules")
    config["prime_directives_enabled"] = prompt_yes_no(
        "Enable prime directive checks?",
        config.get("prime_directives_enabled", True)
    )
    
    # Actions (restart/kill)
    print()
    print_info("Automated Actions: Restart or kill sessions showing entropy")
    config["actions_enabled"] = prompt_yes_no(
        "Enable automated restart/kill actions?",
        config.get("actions_enabled", True)
    )
    
    print_success("Core settings configured")


def setup_monitoring_modules(config: Dict[str, Any]):
    """Configure monitoring modules."""
    print_header("Monitoring Modules")
    
    print_info("These modules run periodic checks every 10 cycles (~5 minutes at 30s interval)")
    print()
    
    # Resource monitoring
    config["resource_checks_enabled"] = prompt_yes_no(
        "Monitor system resources (CPU, memory, disk)?",
        config.get("resource_checks_enabled", True)
    )
    
    # Config drift
    config["drift_detection_enabled"] = prompt_yes_no(
        "Detect configuration drift in Hermes files?",
        config.get("drift_detection_enabled", True)
    )
    
    # Provider health
    config["provider_health_enabled"] = prompt_yes_no(
        "Monitor AI provider health (OpenRouter, etc.)?",
        config.get("provider_health_enabled", True)
    )
    
    # Cleanup
    config["cleanup_enabled"] = prompt_yes_no(
        "Clean up orphaned sessions and old records?",
        config.get("cleanup_enabled", True)
    )
    
    # Cost monitoring (optional)
    print()
    print_info("Cost monitoring tracks token usage and API spend (requires additional setup)")
    config["cost_monitoring_enabled"] = prompt_yes_no(
        "Enable cost monitoring?",
        config.get("cost_monitoring_enabled", False)
    )
    
    # ML data export (optional)
    print()
    print_info("ML data export creates training data for quality prediction models")
    config["ml_data_enabled"] = prompt_yes_no(
        "Enable ML training data export?",
        config.get("ml_data_enabled", False)
    )
    
    print_success("Monitoring modules configured")


def setup_notifications(config: Dict[str, Any]):
    """Configure Telegram notifications."""
    print_header("Notifications (Telegram)")
    
    print_info("ARGUS can send alerts via Telegram when sessions are restarted or killed.")
    print()
    
    enable_notifications = prompt_yes_no(
        "Enable Telegram notifications?",
        config.get("notifications_enabled", False)
    )
    
    if not enable_notifications:
        config["notifications_enabled"] = False
        config["telegram_bot_token"] = ""
        config["telegram_chat_id"] = ""
        print_info("Notifications disabled")
        return
    
    # Get bot token
    current_token = config.get("telegram_bot_token", "")
    if current_token:
        masked = current_token[:10] + "..." + current_token[-5:] if len(current_token) > 15 else "***"
        print_info(f"Current bot token: {masked}")
    
    print()
    print_info("Create a bot via @BotFather on Telegram to get a token")
    new_token = prompt_input("Telegram Bot Token", current_token)
    if new_token:
        config["telegram_bot_token"] = new_token
    
    # Get chat ID
    print()
    current_chat = config.get("telegram_chat_id", "")
    print_info("Chat ID can be your user ID (for DMs) or group chat ID")
    print_info("Use @userinfobot on Telegram to find your user ID")
    new_chat = prompt_input("Telegram Chat ID", str(current_chat) if current_chat else "")
    if new_chat:
        config["telegram_chat_id"] = new_chat
    
    # Validate and enable
    if config.get("telegram_bot_token") and config.get("telegram_chat_id"):
        config["notifications_enabled"] = True
        print_success("Telegram notifications configured")
    else:
        print_warning("Token or Chat ID missing - notifications will be disabled")
        config["notifications_enabled"] = False


def setup_audit_trail(config: Dict[str, Any]):
    """Configure audit logging."""
    print_header("Audit Trail")
    
    print_info("ARGUS records all decisions and actions to the database for review.")
    print()
    
    config["audit_trail_enabled"] = prompt_yes_no(
        "Enable audit trail logging?",
        config.get("audit_trail_enabled", True)
    )
    
    print_success("Audit trail configured")


def setup_launchd_integration(config: Dict[str, Any]):
    """Configure system service integration (macOS launchd)."""
    # Check platform
    _is_macos = sys.platform == 'darwin'

    if _is_macos:
        print_header("System Service (macOS launchd)")
        print_info("ARGUS can run as a background service via macOS launchd.")
    else:
        print_header("System Service")
        print_info(f"Platform detected: {sys.platform}")
        print_info("Service integration is available on macOS and Linux (systemd).")
        print_info("Windows support is planned.")
        print()
        print_info("You can run ARGUS manually: python -m agathos")
        return

    print()

    install_service = prompt_yes_no(
        "Install ARGUS as a system service?",
        False
    )

    if install_service:
        try:
            from .daemon_mgmt import agathos_launchd_install
            if agathos_launchd_install():
                print_success("ARGUS service installed")
                print_info("Use 'launchctl list com.hermes.agathos' to check status")
            else:
                print_error("Failed to install service")
        except Exception as e:
            print_error(f"Service setup failed: {e}")
    else:
        print_info("System service not installed")
        print_info("You can run ARGUS manually: python -m agathos")


# === Main Wizard ===
def print_banner():
    """Print the setup banner."""
    print()
    print(color("  ┌─────────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("  │           ⚔ ARGUS Setup                                     │", Colors.MAGENTA))
    print(color("  │     Agent Resource Guardian & Unified Supervisor            │", Colors.MAGENTA))
    print(color("  ├─────────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("  │  Configure your Agathos monitoring and recovery system.       │", Colors.CYAN))
    print(color("  │  Press Ctrl+C at any time to exit.                          │", Colors.CYAN))
    print(color("  └─────────────────────────────────────────────────────────────┘", Colors.MAGENTA))


def print_summary(config: Dict[str, Any]):
    """Print configuration summary."""
    print()
    print(color("  ┌─ Configuration Summary ", Colors.MAGENTA) + color("─" * 32, Colors.MAGENTA))
    
    status = "enabled" if config.get("enabled") else "disabled"
    status_color = Colors.GREEN if config.get("enabled") else Colors.RED
    print(f"    ARGUS Watcher:      {color(status, status_color)}")
    
    print(f"    Poll Interval:      {config.get('poll_interval', 30)}s")
    print(f"    Database:           {config.get('agathos_db_path', 'N/A')}")
    
    # Features
    features = []
    if config.get("wal_monitor_enabled"):
        features.append("WAL")
    if config.get("entropy_detection_enabled"):
        features.append("entropy")
    if config.get("actions_enabled"):
        features.append("actions")
    if config.get("notifications_enabled"):
        features.append("notifications")
    
    print(f"    Active Features:    {', '.join(features) if features else 'none'}")
    
    # Config file location
    config_path = get_agathos_config_path()
    print(f"    Config File:        {config_path}")
    
    print(color("  └" + "─" * 57, Colors.MAGENTA))
    print()


def run_quick_setup(config: Dict[str, Any]):
    """Run quick setup - essential settings only."""
    print_header("Quick Setup")
    print_info("Configuring essential ARGUS settings...")
    print()
    
    # Core enable
    config["enabled"] = prompt_yes_no("Enable ARGUS watcher?", True)
    if not config["enabled"]:
        save_agathos_config(config)
        return
    
    # Key features
    config["wal_monitor_enabled"] = prompt_yes_no("Enable real-time WAL monitoring?", True)
    config["entropy_detection_enabled"] = prompt_yes_no("Enable entropy detection?", True)
    config["actions_enabled"] = prompt_yes_no("Enable auto-restart/kill?", True)
    
    # Notifications
    setup_notifications(config)
    
    print_success("Quick setup complete!")


def run_full_setup(config: Dict[str, Any]):
    """Run full setup - all sections."""
    # Section 1: Core Settings
    setup_core_settings(config)
    if not config.get("enabled"):
        save_agathos_config(config)
        print_summary(config)
        return
    
    # Section 2: Monitoring Modules
    setup_monitoring_modules(config)
    
    # Section 3: Audit Trail
    setup_audit_trail(config)
    
    # Section 4: Notifications
    setup_notifications(config)
    
    # Section 5: Launchd (optional)
    if sys.platform == "darwin":
        setup_launchd_integration(config)
    
    print_summary(config)
    print_success("Full setup complete!")


def main():
    """Main entry point for the setup wizard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARGUS Setup Wizard")
    parser.add_argument(
        "section",
        nargs="?",
        choices=["quick", "core", "modules", "alerts", "full"],
        help="Setup section to run (default: full with menu)"
    )
    args = parser.parse_args()
    
    # Load existing config
    config = load_agathos_config()
    
    # Print banner
    print_wizard_banner()
    
    # Check if this is existing install
    config_exists = get_agathos_config_path().exists()
    
    if args.section == "quick":
        run_quick_setup(config)
        save_agathos_config(config)
        
    elif args.section == "core":
        setup_core_settings(config)
        save_agathos_config(config)
        
    elif args.section == "modules":
        setup_monitoring_modules(config)
        save_agathos_config(config)
        
    elif args.section == "alerts":
        setup_notifications(config)
        save_agathos_config(config)
        
    else:
        # Full setup with menu if config exists
        if config_exists and not args.section:
            print()
            print_header("Welcome Back!")
            print_success("You have an existing ARGUS configuration.")
            print()
            
            choice = prompt_choice(
                "What would you like to do?",
                [
                    "Quick Setup - essential settings only",
                    "Full Setup - reconfigure everything",
                    "Core Settings - watcher enable/disable",
                    "Monitoring Modules - periodic checks",
                    "Notifications - Telegram alerts",
                    "Exit without changes",
                ],
                0
            )
            
            if choice == 0:
                run_quick_setup(config)
                save_agathos_config(config)
            elif choice == 1:
                run_full_setup(config)
                save_agathos_config(config)
            elif choice == 2:
                setup_core_settings(config)
                save_agathos_config(config)
            elif choice == 3:
                setup_monitoring_modules(config)
                save_agathos_config(config)
            elif choice == 4:
                setup_notifications(config)
                save_agathos_config(config)
            else:
                print_info("Exiting without changes.")
                return
        else:
            # First-time or explicit full
            run_full_setup(config)
            save_agathos_config(config)
    
    # Print next steps
    print()
    print_header("Next Steps")
    print_info("Run ARGUS:")
    print("    python -m agathos")
    print()

    # Platform-specific status check hints
    _is_macos = sys.platform == 'darwin'
    print_info("Check status:")
    if _is_macos:
        print("    launchctl list com.hermes.agathos  # if service installed")
    print("    python -m agathos.cli status       # manual daemon status")
    print()

    print_info("Edit config directly:")
    print(f"    {get_agathos_config_path()}")
    print()


if __name__ == "__main__":
    main()
