"""
Gateway subcommand for hermes CLI.

Handles: hermes gateway [run|start|stop|restart|status|install|uninstall|setup]
"""

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.config import get_env_value, save_env_value
from hermes_cli.setup import (
    print_header, print_info, print_success, print_warning, print_error,
    prompt, prompt_choice, prompt_yes_no,
)
from hermes_cli.colors import Colors, color


# =============================================================================
# Process Management (for manual gateway runs)
# =============================================================================

def find_gateway_pids() -> list:
    """Find PIDs of running gateway processes."""
    pids = []
    patterns = [
        "hermes_cli.main gateway",
        "hermes gateway",
        "gateway/run.py",
    ]

    try:
        if is_windows():
            # Windows: use wmic to search command lines
            result = subprocess.run(
                ["wmic", "process", "get", "ProcessId,CommandLine", "/FORMAT:LIST"],
                capture_output=True, text=True
            )
            # Parse WMIC LIST output: blocks of "CommandLine=...\nProcessId=...\n"
            current_cmd = ""
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith("CommandLine="):
                    current_cmd = line[len("CommandLine="):]
                elif line.startswith("ProcessId="):
                    pid_str = line[len("ProcessId="):]
                    if any(p in current_cmd for p in patterns):
                        try:
                            pid = int(pid_str)
                            if pid != os.getpid() and pid not in pids:
                                pids.append(pid)
                        except ValueError:
                            pass
                    current_cmd = ""
        else:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                # Skip grep and current process
                if 'grep' in line or str(os.getpid()) in line:
                    continue
                for pattern in patterns:
                    if pattern in line:
                        parts = line.split()
                        if len(parts) > 1:
                            try:
                                pid = int(parts[1])
                                if pid not in pids:
                                    pids.append(pid)
                            except ValueError:
                                continue
                        break
    except Exception:
        pass

    return pids


def kill_gateway_processes(force: bool = False) -> int:
    """Kill any running gateway processes. Returns count killed."""
    pids = find_gateway_pids()
    killed = 0
    
    for pid in pids:
        try:
            if force and not is_windows():
                os.kill(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGTERM)
            killed += 1
        except ProcessLookupError:
            # Process already gone
            pass
        except PermissionError:
            print(f"⚠ Permission denied to kill PID {pid}")
    
    return killed


def is_linux() -> bool:
    return sys.platform.startswith('linux')

def is_macos() -> bool:
    return sys.platform == 'darwin'

def is_windows() -> bool:
    return sys.platform == 'win32'


# =============================================================================
# Service Configuration
# =============================================================================

SERVICE_NAME = "hermes-gateway"
SERVICE_DESCRIPTION = "Hermes Agent Gateway - Messaging Platform Integration"

def get_systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service"

def get_launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"

def get_python_path() -> str:
    if is_windows():
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

def get_hermes_cli_path() -> str:
    """Get the path to the hermes CLI."""
    # Check if installed via pip
    import shutil
    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return hermes_bin
    
    # Fallback to direct module execution
    return f"{get_python_path()} -m hermes_cli.main"


# =============================================================================
# Systemd (Linux)
# =============================================================================

def generate_systemd_unit() -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main gateway run
WorkingDirectory={working_dir}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""

def systemd_install(force: bool = False):
    unit_path = get_systemd_unit_path()
    
    if unit_path.exists() and not force:
        print(f"Service already installed at: {unit_path}")
        print("Use --force to reinstall")
        return
    
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing systemd service to: {unit_path}")
    unit_path.write_text(generate_systemd_unit())
    
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", SERVICE_NAME], check=True)
    
    print()
    print("✓ Service installed and enabled!")
    print()
    print("Next steps:")
    print(f"  hermes gateway start              # Start the service")
    print(f"  hermes gateway status             # Check status")
    print(f"  journalctl --user -u {SERVICE_NAME} -f  # View logs")
    print()
    print("To enable lingering (keeps running after logout):")
    print("  sudo loginctl enable-linger $USER")

def systemd_uninstall():
    subprocess.run(["systemctl", "--user", "stop", SERVICE_NAME], check=False)
    subprocess.run(["systemctl", "--user", "disable", SERVICE_NAME], check=False)
    
    unit_path = get_systemd_unit_path()
    if unit_path.exists():
        unit_path.unlink()
        print(f"✓ Removed {unit_path}")
    
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    print("✓ Service uninstalled")

def systemd_start():
    subprocess.run(["systemctl", "--user", "start", SERVICE_NAME], check=True)
    print("✓ Service started")

def systemd_stop():
    subprocess.run(["systemctl", "--user", "stop", SERVICE_NAME], check=True)
    print("✓ Service stopped")

def systemd_restart():
    subprocess.run(["systemctl", "--user", "restart", SERVICE_NAME], check=True)
    print("✓ Service restarted")

def systemd_status(deep: bool = False):
    # Check if service unit file exists
    unit_path = get_systemd_unit_path()
    if not unit_path.exists():
        print("✗ Gateway service is not installed")
        print("  Run: hermes gateway install")
        return
    
    # Show detailed status first
    subprocess.run(
        ["systemctl", "--user", "status", SERVICE_NAME, "--no-pager"],
        capture_output=False
    )
    
    # Check if service is active
    result = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        capture_output=True,
        text=True
    )
    
    status = result.stdout.strip()
    
    if status == "active":
        print("✓ Gateway service is running")
    else:
        print("✗ Gateway service is stopped")
        print("  Run: hermes gateway start")
    
    if deep:
        print()
        print("Recent logs:")
        subprocess.run([
            "journalctl", "--user", "-u", SERVICE_NAME,
            "-n", "20", "--no-pager"
        ])


# =============================================================================
# Launchd (macOS)
# =============================================================================

def generate_launchd_plist() -> str:
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    log_dir = Path.home() / ".hermes" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.hermes.gateway</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hermes_cli.main</string>
        <string>gateway</string>
        <string>run</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/gateway.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/gateway.error.log</string>
</dict>
</plist>
"""

def launchd_install(force: bool = False):
    plist_path = get_launchd_plist_path()
    
    if plist_path.exists() and not force:
        print(f"Service already installed at: {plist_path}")
        print("Use --force to reinstall")
        return
    
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Installing launchd service to: {plist_path}")
    plist_path.write_text(generate_launchd_plist())
    
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    
    print()
    print("✓ Service installed and loaded!")
    print()
    print("Next steps:")
    print("  hermes gateway status             # Check status")
    print("  tail -f ~/.hermes/logs/gateway.log  # View logs")

def launchd_uninstall():
    plist_path = get_launchd_plist_path()
    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    
    if plist_path.exists():
        plist_path.unlink()
        print(f"✓ Removed {plist_path}")
    
    print("✓ Service uninstalled")

def launchd_start():
    subprocess.run(["launchctl", "start", "ai.hermes.gateway"], check=True)
    print("✓ Service started")

def launchd_stop():
    subprocess.run(["launchctl", "stop", "ai.hermes.gateway"], check=True)
    print("✓ Service stopped")

def launchd_restart():
    launchd_stop()
    launchd_start()

def launchd_status(deep: bool = False):
    result = subprocess.run(
        ["launchctl", "list", "ai.hermes.gateway"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Gateway service is loaded")
        print(result.stdout)
    else:
        print("✗ Gateway service is not loaded")
    
    if deep:
        log_file = Path.home() / ".hermes" / "logs" / "gateway.log"
        if log_file.exists():
            print()
            print("Recent logs:")
            subprocess.run(["tail", "-20", str(log_file)])


# =============================================================================
# Gateway Runner
# =============================================================================

def run_gateway(verbose: bool = False):
    """Run the gateway in foreground."""
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from gateway.run import start_gateway
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│           ⚕ Hermes Gateway Starting...                 │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│  Messaging platforms + cron scheduler                    │")
    print("│  Press Ctrl+C to stop                                   │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    # Exit with code 1 if gateway fails to connect any platform,
    # so systemd Restart=on-failure will retry on transient errors
    success = asyncio.run(start_gateway())
    if not success:
        sys.exit(1)


# =============================================================================
# Gateway Setup (Interactive Messaging Platform Configuration)
# =============================================================================

# Per-platform config: each entry defines the env vars, setup instructions,
# and prompts needed to configure a messaging platform.
_PLATFORMS = [
    {
        "key": "telegram",
        "label": "Telegram",
        "emoji": "📱",
        "token_var": "TELEGRAM_BOT_TOKEN",
        "vars": [
            {"name": "TELEGRAM_BOT_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Create a bot via @BotFather on Telegram to get a token."},
            {"name": "TELEGRAM_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated, or empty for open access)", "password": False,
             "help": "To find your user ID: message @userinfobot on Telegram."},
            {"name": "TELEGRAM_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "For DMs, this is your user ID. You can set it later by typing /set-home in chat."},
        ],
    },
    {
        "key": "discord",
        "label": "Discord",
        "emoji": "💬",
        "token_var": "DISCORD_BOT_TOKEN",
        "vars": [
            {"name": "DISCORD_BOT_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Create a bot at https://discord.com/developers/applications"},
            {"name": "DISCORD_ALLOWED_USERS", "prompt": "Allowed user IDs or usernames (comma-separated, or empty for open access)", "password": False,
             "help": "Enable Developer Mode in Discord settings, then right-click your name → Copy ID."},
            {"name": "DISCORD_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "Right-click a channel → Copy Channel ID (requires Developer Mode)."},
        ],
    },
    {
        "key": "slack",
        "label": "Slack",
        "emoji": "💼",
        "token_var": "SLACK_BOT_TOKEN",
        "vars": [
            {"name": "SLACK_BOT_TOKEN", "prompt": "Bot Token (xoxb-...)", "password": True,
             "help": "Go to https://api.slack.com/apps → Create New App → OAuth & Permissions → Install to Workspace."},
            {"name": "SLACK_APP_TOKEN", "prompt": "App Token (xapp-...)", "password": True,
             "help": "App Settings → Basic Information → App-Level Tokens → Generate (with connections:write scope)."},
            {"name": "SLACK_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated, or empty for open access)", "password": False,
             "help": "Find Slack user IDs in your profile or via the Slack API."},
        ],
    },
    {
        "key": "whatsapp",
        "label": "WhatsApp",
        "emoji": "📲",
        "token_var": "WHATSAPP_ENABLED",
    },
]


def _platform_status(platform: dict) -> str:
    """Return a plain-text status string for a platform.

    Returns uncolored text so it can safely be embedded in
    simple_term_menu items (ANSI codes break width calculation).
    """
    token_var = platform["token_var"]
    val = get_env_value(token_var)
    if token_var == "WHATSAPP_ENABLED":
        if val and val.lower() == "true":
            session_file = Path.home() / ".hermes" / "whatsapp" / "session" / "creds.json"
            if session_file.exists():
                return "configured + paired"
            return "enabled, not paired"
        return "not configured"
    if val:
        return "configured"
    return "not configured"


def _setup_standard_platform(platform: dict):
    """Interactive setup for Telegram, Discord, or Slack."""
    emoji = platform["emoji"]
    label = platform["label"]
    token_var = platform["token_var"]

    print()
    print(color(f"  ─── {emoji} {label} Setup ───", Colors.CYAN))

    existing_token = get_env_value(token_var)
    if existing_token:
        print()
        print_success(f"{label} is already configured.")
        if not prompt_yes_no(f"  Reconfigure {label}?", False):
            return

    for var in platform["vars"]:
        print()
        print_info(f"  {var['help']}")
        existing = get_env_value(var["name"])
        if existing and var["name"] != token_var:
            print_info(f"  Current: {existing}")

        value = prompt(f"  {var['prompt']}", password=var.get("password", False))
        if value:
            cleaned = value.replace(" ", "") if "user" in var["name"].lower() else value
            save_env_value(var["name"], cleaned)
            print_success(f"  Saved {var['name']}")
        elif var["name"] == token_var:
            print_warning(f"  Skipped — {label} won't work without this.")
            return
        else:
            print_info(f"  Skipped (can configure later)")

    # If the first allowed-user value was set and home channel wasn't,
    # offer to reuse it (common for Telegram DMs).
    allowed_var = f"{label.upper()}_ALLOWED_USERS"
    home_var = f"{label.upper()}_HOME_CHANNEL"
    allowed_val = get_env_value(allowed_var)
    home_val = get_env_value(home_var)
    if allowed_val and not home_val and label == "Telegram":
        first_id = allowed_val.split(",")[0].strip()
        if first_id and prompt_yes_no(f"  Use your user ID ({first_id}) as the home channel?", True):
            save_env_value(home_var, first_id)
            print_success(f"  Home channel set to {first_id}")

    print()
    print_success(f"{emoji} {label} configured!")


def _setup_whatsapp():
    """Delegate to the existing WhatsApp setup flow."""
    from hermes_cli.main import cmd_whatsapp
    import argparse
    cmd_whatsapp(argparse.Namespace())


def _is_service_installed() -> bool:
    """Check if the gateway is installed as a system service."""
    if is_linux():
        return get_systemd_unit_path().exists()
    elif is_macos():
        return get_launchd_plist_path().exists()
    return False


def _is_service_running() -> bool:
    """Check if the gateway service is currently running."""
    if is_linux() and get_systemd_unit_path().exists():
        result = subprocess.run(
            ["systemctl", "--user", "is-active", SERVICE_NAME],
            capture_output=True, text=True
        )
        return result.stdout.strip() == "active"
    elif is_macos() and get_launchd_plist_path().exists():
        result = subprocess.run(
            ["launchctl", "list", "ai.hermes.gateway"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    # Check for manual processes
    return len(find_gateway_pids()) > 0


def gateway_setup():
    """Interactive setup for messaging platforms + gateway service."""

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│             ⚕ Gateway Setup                            │", Colors.MAGENTA))
    print(color("├─────────────────────────────────────────────────────────┤", Colors.MAGENTA))
    print(color("│  Configure messaging platforms and the gateway service. │", Colors.MAGENTA))
    print(color("│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))

    # ── Gateway service status ──
    print()
    service_installed = _is_service_installed()
    service_running = _is_service_running()

    if service_installed and service_running:
        print_success("Gateway service is installed and running.")
    elif service_installed:
        print_warning("Gateway service is installed but not running.")
        if prompt_yes_no("  Start it now?", True):
            try:
                if is_linux():
                    systemd_start()
                elif is_macos():
                    launchd_start()
            except subprocess.CalledProcessError as e:
                print_error(f"  Failed to start: {e}")
    else:
        print_info("Gateway service is not installed.")
        print_info("You can install it after configuring platforms: hermes gateway install")

    # ── Platform configuration loop ──
    while True:
        print()
        print_header("Messaging Platforms")

        menu_items = []
        for plat in _PLATFORMS:
            status = _platform_status(plat)
            menu_items.append(f"{plat['label']}  ({status})")
        menu_items.append("Done")

        choice = prompt_choice("Select a platform to configure:", menu_items, len(menu_items) - 1)

        if choice == len(_PLATFORMS):
            break

        platform = _PLATFORMS[choice]

        if platform["key"] == "whatsapp":
            _setup_whatsapp()
        else:
            _setup_standard_platform(platform)

    # ── Post-setup: offer to install/restart gateway ──
    any_configured = any(
        bool(get_env_value(p["token_var"]))
        for p in _PLATFORMS
        if p["key"] != "whatsapp"
    ) or (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true"

    if any_configured:
        print()
        print(color("─" * 58, Colors.DIM))
        service_installed = _is_service_installed()
        service_running = _is_service_running()

        if service_running:
            if prompt_yes_no("  Restart the gateway to pick up changes?", True):
                try:
                    if is_linux():
                        systemd_restart()
                    elif is_macos():
                        launchd_restart()
                    else:
                        kill_gateway_processes()
                        print_info("Start manually: hermes gateway")
                except subprocess.CalledProcessError as e:
                    print_error(f"  Restart failed: {e}")
        elif service_installed:
            if prompt_yes_no("  Start the gateway service?", True):
                try:
                    if is_linux():
                        systemd_start()
                    elif is_macos():
                        launchd_start()
                except subprocess.CalledProcessError as e:
                    print_error(f"  Start failed: {e}")
        else:
            print()
            print_info("Next steps:")
            print_info("  hermes gateway              Run in foreground")
            print_info("  hermes gateway install      Install as background service")
    else:
        print()
        print_info("No platforms configured. Run 'hermes gateway setup' when ready.")

    print()


# =============================================================================
# Main Command Handler
# =============================================================================

def gateway_command(args):
    """Handle gateway subcommands."""
    subcmd = getattr(args, 'gateway_command', None)
    
    # Default to run if no subcommand
    if subcmd is None or subcmd == "run":
        verbose = getattr(args, 'verbose', False)
        run_gateway(verbose)
        return

    if subcmd == "setup":
        gateway_setup()
        return

    # Service management commands
    if subcmd == "install":
        force = getattr(args, 'force', False)
        if is_linux():
            systemd_install(force)
        elif is_macos():
            launchd_install(force)
        else:
            print("Service installation not supported on this platform.")
            print("Run manually: hermes gateway run")
            sys.exit(1)
    
    elif subcmd == "uninstall":
        if is_linux():
            systemd_uninstall()
        elif is_macos():
            launchd_uninstall()
        else:
            print("Not supported on this platform.")
            sys.exit(1)
    
    elif subcmd == "start":
        if is_linux():
            systemd_start()
        elif is_macos():
            launchd_start()
        else:
            print("Not supported on this platform.")
            sys.exit(1)
    
    elif subcmd == "stop":
        # Try service first, fall back to killing processes directly
        service_available = False
        
        if is_linux() and get_systemd_unit_path().exists():
            try:
                systemd_stop()
                service_available = True
            except subprocess.CalledProcessError:
                pass  # Fall through to process kill
        elif is_macos() and get_launchd_plist_path().exists():
            try:
                launchd_stop()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        
        if not service_available:
            # Kill gateway processes directly
            killed = kill_gateway_processes()
            if killed:
                print(f"✓ Stopped {killed} gateway process(es)")
            else:
                print("✗ No gateway processes found")
    
    elif subcmd == "restart":
        # Try service first, fall back to killing and restarting
        service_available = False
        
        if is_linux() and get_systemd_unit_path().exists():
            try:
                systemd_restart()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        elif is_macos() and get_launchd_plist_path().exists():
            try:
                launchd_restart()
                service_available = True
            except subprocess.CalledProcessError:
                pass
        
        if not service_available:
            # Manual restart: kill existing processes
            killed = kill_gateway_processes()
            if killed:
                print(f"✓ Stopped {killed} gateway process(es)")
            
            import time
            time.sleep(2)
            
            # Start fresh
            print("Starting gateway...")
            run_gateway(verbose=False)
    
    elif subcmd == "status":
        deep = getattr(args, 'deep', False)
        
        # Check for service first
        if is_linux() and get_systemd_unit_path().exists():
            systemd_status(deep)
        elif is_macos() and get_launchd_plist_path().exists():
            launchd_status(deep)
        else:
            # Check for manually running processes
            pids = find_gateway_pids()
            if pids:
                print(f"✓ Gateway is running (PID: {', '.join(map(str, pids))})")
                print("  (Running manually, not as a system service)")
                print()
                print("To install as a service:")
                print("  hermes gateway install")
            else:
                print("✗ Gateway is not running")
                print()
                print("To start:")
                print("  hermes gateway          # Run in foreground")
                print("  hermes gateway install  # Install as service")
