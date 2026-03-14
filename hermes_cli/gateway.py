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
    import shutil
    python_path = get_python_path()
    working_dir = str(PROJECT_ROOT)
    venv_dir = str(PROJECT_ROOT / "venv")
    venv_bin = str(PROJECT_ROOT / "venv" / "bin")
    node_bin = str(PROJECT_ROOT / "node_modules" / ".bin")

    # Build a PATH that includes the venv, node_modules, and standard system dirs
    sane_path = f"{venv_bin}:{node_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    
    hermes_cli = shutil.which("hermes") or f"{python_path} -m hermes_cli.main"
    return f"""[Unit]
Description={SERVICE_DESCRIPTION}
After=network.target

[Service]
Type=simple
ExecStart={python_path} -m hermes_cli.main gateway run --replace
ExecStop={hermes_cli} gateway stop
WorkingDirectory={working_dir}
Environment="PATH={sane_path}"
Environment="VIRTUAL_ENV={venv_dir}"
Restart=on-failure
RestartSec=10
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=15
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
    # `systemctl restart` sends SIGTERM and waits up to TimeoutStopSec (default 90s)
    # before giving up. The gateway runs agent LLM calls in a thread-pool executor
    # which cannot be interrupted by SIGTERM, causing the restart to hang or fail
    # with "Job canceled" if the old process doesn't exit in time.
    #
    # Fix: send SIGKILL immediately to ensure the old process is gone, then start
    # fresh. This is safe because the gateway persists all state to disk (SQLite
    # crypto store, session DB, sync token) — no in-flight work is lost that
    # wouldn't be retried on reconnect anyway.
    subprocess.run(
        ["systemctl", "--user", "kill", "-s", "SIGKILL", SERVICE_NAME],
        check=False,  # ignore error if service is already stopped
    )
    import time; time.sleep(1)
    subprocess.run(["systemctl", "--user", "start", SERVICE_NAME], check=True)
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

def run_gateway(verbose: bool = False, replace: bool = False):
    """Run the gateway in foreground.
    
    Args:
        verbose: Enable verbose logging output.
        replace: If True, kill any existing gateway instance before starting.
                 This prevents systemd restart loops when the old process
                 hasn't fully exited yet.
    """
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
    success = asyncio.run(start_gateway(replace=replace))
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
        "setup_instructions": [
            "1. Open Telegram and message @BotFather",
            "2. Send /newbot and follow the prompts to create your bot",
            "3. Copy the bot token BotFather gives you",
            "4. To find your user ID: message @userinfobot — it replies with your numeric ID",
        ],
        "vars": [
            {"name": "TELEGRAM_BOT_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the token from @BotFather (step 3 above)."},
            {"name": "TELEGRAM_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Paste your user ID from step 4 above."},
            {"name": "TELEGRAM_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "For DMs, this is your user ID. You can set it later by typing /set-home in chat."},
        ],
    },
    {
        "key": "discord",
        "label": "Discord",
        "emoji": "💬",
        "token_var": "DISCORD_BOT_TOKEN",
        "setup_instructions": [
            "1. Go to https://discord.com/developers/applications → New Application",
            "2. Go to Bot → Reset Token → copy the bot token",
            "3. Enable: Bot → Privileged Gateway Intents → Message Content Intent",
            "4. Invite the bot to your server:",
            "   OAuth2 → URL Generator → check BOTH scopes:",
            "     - bot",
            "     - applications.commands  (required for slash commands!)",
            "   Bot Permissions: Send Messages, Read Message History, Attach Files",
            "   Copy the URL and open it in your browser to invite.",
            "5. Get your user ID: enable Developer Mode in Discord settings,",
            "   then right-click your name → Copy ID",
        ],
        "vars": [
            {"name": "DISCORD_BOT_TOKEN", "prompt": "Bot token", "password": True,
             "help": "Paste the token from step 2 above."},
            {"name": "DISCORD_ALLOWED_USERS", "prompt": "Allowed user IDs or usernames (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Paste your user ID from step 5 above."},
            {"name": "DISCORD_HOME_CHANNEL", "prompt": "Home channel ID (for cron/notification delivery, or empty to set later with /set-home)", "password": False,
             "help": "Right-click a channel → Copy Channel ID (requires Developer Mode)."},
        ],
    },
    {
        "key": "slack",
        "label": "Slack",
        "emoji": "💼",
        "token_var": "SLACK_BOT_TOKEN",
        "setup_instructions": [
            "1. Go to https://api.slack.com/apps → Create New App → From Scratch",
            "2. Enable Socket Mode: Settings → Socket Mode → Enable",
            "   Create an App-Level Token with scope: connections:write → copy xapp-... token",
            "3. Add Bot Token Scopes: Features → OAuth & Permissions → Scopes",
            "   Required: chat:write, app_mentions:read, channels:history, channels:read,",
            "   groups:history, im:history, im:read, im:write, users:read, files:write",
            "4. Subscribe to Events: Features → Event Subscriptions → Enable",
            "   Required events: message.im, message.channels, app_mention",
            "   Optional: message.groups (for private channels)",
            "   ⚠ Without message.channels the bot will ONLY work in DMs!",
            "5. Install to Workspace: Settings → Install App → copy xoxb-... token",
            "6. Reinstall the app after any scope or event changes",
            "7. Find your user ID: click your profile → three dots → Copy member ID",
            "8. Invite the bot to channels: /invite @YourBot",
        ],
        "vars": [
            {"name": "SLACK_BOT_TOKEN", "prompt": "Bot Token (xoxb-...)", "password": True,
             "help": "Paste the bot token from step 3 above."},
            {"name": "SLACK_APP_TOKEN", "prompt": "App Token (xapp-...)", "password": True,
             "help": "Paste the app-level token from step 4 above."},
            {"name": "SLACK_ALLOWED_USERS", "prompt": "Allowed user IDs (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Paste your member ID from step 7 above."},
        ],
    },
    {
        "key": "whatsapp",
        "label": "WhatsApp",
        "emoji": "📲",
        "token_var": "WHATSAPP_ENABLED",
    },
    {
        "key": "signal",
        "label": "Signal",
        "emoji": "📡",
        "token_var": "SIGNAL_HTTP_URL",
    },
    {
        "key": "matrix",
        "label": "Matrix",
        "emoji": "🔷",
        "token_var": "MATRIX_ACCESS_TOKEN",
        "setup_instructions": [
            "Run the guided setup wizard — it handles everything automatically:",
            "  hermes gateway setup matrix",
            "",
            "What you need:",
            "  1. A Matrix homeserver (Synapse, Dendrite, matrix.org, etc.)",
            "  2. A dedicated bot account — just a username and password",
            "  3. Your own Matrix user ID (e.g. @you:yourserver.org) for the allowlist",
            "",
            "The wizard will log in as the bot, configure E2EE, bootstrap cross-signing,",
            "and sign the bot's identity with your account — all in one pass.",
            "",
            "To re-run trust verification separately (e.g. after adding a new allowed user):",
            "  hermes gateway verify-matrix",
        ],
        "vars": [
            {"name": "MATRIX_HOMESERVER_URL", "prompt": "Homeserver URL", "password": False,
             "help": "e.g., https://matrix.org or https://matrix.example.org:8448"},
            {"name": "MATRIX_ACCESS_TOKEN", "prompt": "Bot access token", "password": True,
             "help": "Generate via: curl -XPOST 'https://homeserver/_matrix/client/v3/login' "
                     "-d '{\"type\":\"m.login.password\",\"user\":\"bot\",\"password\":\"pass\"}'"},
            {"name": "MATRIX_USER_ID", "prompt": "Bot Matrix user ID", "password": False,
             "help": "e.g., @hermes:matrix.org"},
            {"name": "MATRIX_HOME_CHANNEL", "prompt": "Home room ID (optional)", "password": False,
             "help": "e.g., !roomid:matrix.org — used as default delivery target for cron jobs"},
            {"name": "MATRIX_ALLOWED_USERS", "prompt": "Allowed Matrix users (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "e.g., @alice:matrix.org,@bob:example.org"},
            {"name": "MATRIX_VERIFY_SSL", "prompt": "Verify SSL (true/false, default: true)", "password": False,
             "help": "Set to false only for private homeservers with self-signed certificates"},
        ],
    },
    {
        "key": "email",
        "label": "Email",
        "emoji": "📧",
        "token_var": "EMAIL_ADDRESS",
        "setup_instructions": [
            "1. Use a dedicated email account for your Hermes agent",
            "2. For Gmail: enable 2FA, then create an App Password at",
            "   https://myaccount.google.com/apppasswords",
            "3. For other providers: use your email password or app-specific password",
            "4. IMAP must be enabled on your email account",
        ],
        "vars": [
            {"name": "EMAIL_ADDRESS", "prompt": "Email address", "password": False,
             "help": "The email address Hermes will use (e.g., hermes@gmail.com)."},
            {"name": "EMAIL_PASSWORD", "prompt": "Email password (or app password)", "password": True,
             "help": "For Gmail, use an App Password (not your regular password)."},
            {"name": "EMAIL_IMAP_HOST", "prompt": "IMAP host", "password": False,
             "help": "e.g., imap.gmail.com for Gmail, outlook.office365.com for Outlook."},
            {"name": "EMAIL_SMTP_HOST", "prompt": "SMTP host", "password": False,
             "help": "e.g., smtp.gmail.com for Gmail, smtp.office365.com for Outlook."},
            {"name": "EMAIL_ALLOWED_USERS", "prompt": "Allowed sender emails (comma-separated)", "password": False,
             "is_allowlist": True,
             "help": "Only emails from these addresses will be processed."},
        ],
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
    if platform.get("key") == "signal":
        account = get_env_value("SIGNAL_ACCOUNT")
        if val and account:
            return "configured"
        if val or account:
            return "partially configured"
        return "not configured"
    if platform.get("key") == "matrix":
        homeserver = get_env_value("MATRIX_HOMESERVER_URL")
        user_id = get_env_value("MATRIX_USER_ID")
        if val and homeserver and user_id:
            return "configured"
        if val or homeserver or user_id:
            return "partially configured"
        return "not configured"
    if platform.get("key") == "email":
        pwd = get_env_value("EMAIL_PASSWORD")
        imap = get_env_value("EMAIL_IMAP_HOST")
        smtp = get_env_value("EMAIL_SMTP_HOST")
        if all([val, pwd, imap, smtp]):
            return "configured"
        if any([val, pwd, imap, smtp]):
            return "partially configured"
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

    # Show step-by-step setup instructions if this platform has them
    instructions = platform.get("setup_instructions")
    if instructions:
        print()
        for line in instructions:
            print_info(f"  {line}")

    existing_token = get_env_value(token_var)
    if existing_token:
        print()
        print_success(f"{label} is already configured.")
        if not prompt_yes_no(f"  Reconfigure {label}?", False):
            return

    allowed_val_set = None  # Track if user set an allowlist (for home channel offer)

    for var in platform["vars"]:
        print()
        print_info(f"  {var['help']}")
        existing = get_env_value(var["name"])
        if existing and var["name"] != token_var:
            print_info(f"  Current: {existing}")

        # Allowlist fields get special handling for the deny-by-default security model
        if var.get("is_allowlist"):
            print_info(f"  The gateway DENIES all users by default for security.")
            print_info(f"  Enter user IDs to create an allowlist, or leave empty")
            print_info(f"  and you'll be asked about open access next.")
            value = prompt(f"  {var['prompt']}", password=False)
            if value:
                cleaned = value.replace(" ", "")
                # For Discord, strip common prefixes (user:123, <@123>, <@!123>)
                if "DISCORD" in var["name"]:
                    parts = []
                    for uid in cleaned.split(","):
                        uid = uid.strip()
                        if uid.startswith("<@") and uid.endswith(">"):
                            uid = uid.lstrip("<@!").rstrip(">")
                        if uid.lower().startswith("user:"):
                            uid = uid[5:]
                        if uid:
                            parts.append(uid)
                    cleaned = ",".join(parts)
                save_env_value(var["name"], cleaned)
                print_success(f"  Saved — only these users can interact with the bot.")
                allowed_val_set = cleaned
            else:
                # No allowlist — ask about open access vs DM pairing
                print()
                access_choices = [
                    "Enable open access (anyone can message the bot)",
                    "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
                    "Skip for now (bot will deny all users until configured)",
                ]
                access_idx = prompt_choice("  How should unauthorized users be handled?", access_choices, 1)
                if access_idx == 0:
                    save_env_value("GATEWAY_ALLOW_ALL_USERS", "true")
                    print_warning("  Open access enabled — anyone can use your bot!")
                elif access_idx == 1:
                    print_success("  DM pairing mode — users will receive a code to request access.")
                    print_info("  Approve with: hermes pairing approve {platform} {code}")
                else:
                    print_info("  Skipped — configure later with 'hermes gateway setup'")
            continue

        value = prompt(f"  {var['prompt']}", password=var.get("password", False))
        if value:
            save_env_value(var["name"], value)
            print_success(f"  Saved {var['name']}")
        elif var["name"] == token_var:
            print_warning(f"  Skipped — {label} won't work without this.")
            return
        else:
            print_info(f"  Skipped (can configure later)")

    # If an allowlist was set and home channel wasn't, offer to reuse
    # the first user ID (common for Telegram DMs).
    home_var = f"{label.upper()}_HOME_CHANNEL"
    home_val = get_env_value(home_var)
    if allowed_val_set and not home_val and label == "Telegram":
        first_id = allowed_val_set.split(",")[0].strip()
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


def _detect_linux_distro() -> str:
    """Return a simple distro family string: 'arch', 'debian', 'fedora', or 'unknown'."""
    try:
        with open("/etc/os-release") as f:
            content = f.read().lower()
        if "arch" in content or "manjaro" in content or "endeavour" in content:
            return "arch"
        if "debian" in content or "ubuntu" in content or "mint" in content or "pop" in content:
            return "debian"
        if "fedora" in content or "rhel" in content or "centos" in content or "rocky" in content:
            return "fedora"
    except OSError:
        pass
    return "unknown"


def _install_libolm() -> bool:
    """Attempt to install libolm via the system package manager. Returns True on success.

    libolm is a C library required by the mautrix E2EE backend (python-olm).
    """
    import shutil
    import platform as _platform

    system = _platform.system()
    cmd: list = []

    if system == "Darwin":
        if shutil.which("brew"):
            cmd = ["brew", "install", "libolm"]
        else:
            print_error("  Homebrew not found. Install it from https://brew.sh then run: brew install libolm")
            return False
    elif system == "Linux":
        distro = _detect_linux_distro()
        if distro == "arch" and shutil.which("pacman"):
            cmd = ["sudo", "pacman", "-S", "--noconfirm", "libolm"]
        elif distro == "debian" and shutil.which("apt-get"):
            cmd = ["sudo", "apt-get", "install", "-y", "libolm-dev"]
        elif distro == "fedora" and shutil.which("dnf"):
            cmd = ["sudo", "dnf", "install", "-y", "libolm-devel"]
        else:
            print_warning("  Could not detect package manager.")
            print_info("  Install libolm manually:")
            print_info("    Arch:   sudo pacman -S libolm")
            print_info("    Debian: sudo apt install libolm-dev")
            print_info("    Fedora: sudo dnf install libolm-devel")
            print_info("    macOS:  brew install libolm")
            return False
    else:
        print_warning(f"  Unsupported OS: {system}. Install libolm manually.")
        return False

    print_info(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print_success("  libolm installed successfully.")
        return True
    else:
        print_error("  libolm installation failed. You may need to run it manually with sudo.")
        return False


def _bootstrap_user_trust(
    homeserver: str,
    allowed_users_str: str,
    bot_user_id: str,
    bot_token: str,
    verify_ssl: bool,
) -> None:
    """Bootstrap cross-signing trust between allowed users and the bot.

    For each allowed user:
      1. Ask for their Matrix password (used only to obtain a temporary token).
      2. If their cross-signing master/user-signing keys are missing, upload them.
      3. Sign the bot's master key with their user-signing key.

    This makes Element show the bot as "verified" without any manual ceremony.
    The user's password is NOT stored — it is only used for the one-time login.
    """
    try:
        import httpx as _httpx
        import json as _json
        import base64 as _b64
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat, PrivateFormat, NoEncryption,
        )
        from pathlib import Path as _Path
    except ImportError as e:
        print_warning(f"  Skipping trust bootstrap — missing dependency: {e}")
        return

    hs = homeserver.rstrip("/")
    users = [u.strip() for u in allowed_users_str.split(",") if u.strip()]
    if not users:
        return

    # Fetch bot's master key from server
    try:
        r = _httpx.post(
            f"{hs}/_matrix/client/v3/keys/query",
            headers={"Authorization": f"Bearer {bot_token}"},
            json={"device_keys": {bot_user_id: []}},
            verify=verify_ssl, timeout=10,
        )
        bot_mk_obj = r.json().get("master_keys", {}).get(bot_user_id, {})
        if not bot_mk_obj:
            print_warning("  Bot has no master key yet — start the gateway first, then re-run setup.")
            return
        bot_mk_pub = list(bot_mk_obj.get("keys", {}).values())[0]
    except Exception as e:
        print_warning(f"  Could not fetch bot master key: {e}")
        return

    def _canonical_json(obj):
        return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()

    def _sign_obj(obj, priv, uid, key_id):
        to_sign = {k: v for k, v in obj.items() if k not in ("signatures", "unsigned")}
        sig = _b64.b64encode(priv.sign(_canonical_json(to_sign))).decode()
        obj.setdefault("signatures", {}).setdefault(uid, {})[f"ed25519:{key_id}"] = sig
        return obj

    print()
    print_info("  Setting up cross-signing trust between your account(s) and the bot.")
    print_info("  Your password is used only to obtain a one-time login token — it is not stored.")

    for user_id in users:
        print()
        print_info(f"  User: {user_id}")
        try:
            import getpass
            password = getpass.getpass(f"  Password for {user_id} (Enter to skip): ").strip()
        except (KeyboardInterrupt, EOFError):
            print_info("  Skipped.")
            continue
        if not password:
            print_info("  Skipped.")
            continue

        # Log in to get a temporary token
        # Retry login with backoff — Synapse rate-limits repeated logins
        # (429 Too Many Requests) when multiple logins happen in quick succession
        # during the setup wizard (bot login + recovery key bootstrap + user login).
        import time as _time
        user_token = None
        temp_device_id = ""
        localpart = user_id.split(":")[0].lstrip("@")
        for _attempt in range(3):
            if _attempt > 0:
                _wait = _attempt * 5
                print_info(f"  Rate limited — waiting {_wait}s before retrying...")
                _time.sleep(_wait)
            try:
                r = _httpx.post(
                    f"{hs}/_matrix/client/v3/login",
                    json={
                        "type": "m.login.password",
                        "identifier": {"type": "m.id.user", "user": localpart},
                        "password": password,
                        "initial_device_display_name": "Hermes Setup (temporary)",
                    },
                    verify=verify_ssl, timeout=10,
                )
                if r.status_code == 429:
                    continue  # retry
                if r.status_code != 200:
                    print_warning(f"  Login failed ({r.status_code}): {r.json().get('error', r.text[:80])}")
                    break
                user_token = r.json()["access_token"]
                temp_device_id = r.json().get("device_id", "")
                break
            except Exception as e:
                print_warning(f"  Login error: {e}")
                break
        if not user_token:
            print_warning(f"  Could not log in as {user_id} — skipping trust setup.")
            print_info("  Run 'hermes gateway verify-matrix' after setup to complete trust.")
            continue

        headers = {"Authorization": f"Bearer {user_token}"}

        try:
            # Check existing cross-signing state
            rq = _httpx.post(
                f"{hs}/_matrix/client/v3/keys/query",
                headers=headers,
                json={"device_keys": {user_id: []}},
                verify=verify_ssl, timeout=10,
            )
            qdata = rq.json()
            existing_mk = qdata.get("master_keys", {}).get(user_id, {})
            existing_usk = qdata.get("user_signing_keys", {}).get(user_id, {})
            existing_ssk = qdata.get("self_signing_keys", {}).get(user_id, {})

            # Load or generate user's cross-signing keys.
            # Always verify local keys match what the server has. If they don't
            # (e.g. after multiple setup attempts), regenerate and re-upload.
            user_keys_file = _Path(f"~/.hermes/matrix/user_cross_signing_{user_id.lstrip('@').replace(':', '_')}.json").expanduser()

            def _gen_and_save_keys():
                _mk = Ed25519PrivateKey.generate()
                _usk = Ed25519PrivateKey.generate()
                _ssk = Ed25519PrivateKey.generate()
                _mk_pub  = _b64.b64encode(_mk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
                _usk_pub = _b64.b64encode(_usk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
                _ssk_pub = _b64.b64encode(_ssk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()
                user_keys_file.write_text(_json.dumps({
                    "master_private": _b64.b64encode(_mk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode(),
                    "master_public": _mk_pub,
                    "user_signing_private": _b64.b64encode(_usk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode(),
                    "user_signing_public": _usk_pub,
                    "self_signing_private": _b64.b64encode(_ssk.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())).decode(),
                    "self_signing_public": _ssk_pub,
                }))
                user_keys_file.chmod(0o600)
                return _mk, _usk, _ssk, _mk_pub, _usk_pub, _ssk_pub

            needs_upload = not existing_mk or not existing_usk or not existing_ssk

            if user_keys_file.exists() and existing_mk and not needs_upload:
                # Verify local keys match the server before reusing them.
                # If they don't match (e.g. because setup was run multiple times),
                # regenerate fresh keys and re-upload.
                saved = _json.loads(user_keys_file.read_text())
                server_usk_pub = list(existing_usk.get("keys", {}).values())[0] if existing_usk else ""
                if saved.get("user_signing_public") == server_usk_pub:
                    mk_priv = Ed25519PrivateKey.from_private_bytes(_b64.b64decode(saved["master_private"]))
                    usk_priv = Ed25519PrivateKey.from_private_bytes(_b64.b64decode(saved["user_signing_private"]))
                    ssk_priv = Ed25519PrivateKey.from_private_bytes(_b64.b64decode(saved["self_signing_private"]))
                    mk_pub = saved["master_public"]
                    usk_pub = saved["user_signing_public"]
                    ssk_pub = saved["self_signing_public"]
                else:
                    mk_priv, usk_priv, ssk_priv, mk_pub, usk_pub, ssk_pub = _gen_and_save_keys()
                    needs_upload = True
            else:
                mk_priv, usk_priv, ssk_priv, mk_pub, usk_pub, ssk_pub = _gen_and_save_keys()

            # After generating, always upload to ensure server matches local keys.
            # This is idempotent: if the server already has matching keys, no harm done.
            needs_upload = True

            if needs_upload:
                mk_obj = {"user_id": user_id, "usage": ["master"], "keys": {f"ed25519:{mk_pub}": mk_pub}}
                ssk_obj = {"user_id": user_id, "usage": ["self_signing"], "keys": {f"ed25519:{ssk_pub}": ssk_pub}}
                usk_obj = {"user_id": user_id, "usage": ["user_signing"], "keys": {f"ed25519:{usk_pub}": usk_pub}}
                _sign_obj(ssk_obj, mk_priv, user_id, mk_pub)
                _sign_obj(usk_obj, mk_priv, user_id, mk_pub)

                body = {"master_key": mk_obj, "self_signing_key": ssk_obj, "user_signing_key": usk_obj}

                ru = _httpx.post(
                    f"{hs}/_matrix/client/v3/keys/device_signing/upload",
                    headers=headers, json=body, verify=verify_ssl, timeout=15,
                )
                if ru.status_code == 401:
                    # UIA required — retry with password auth
                    session = ru.json().get("session")
                    body["auth"] = {
                        "type": "m.login.password",
                        "identifier": {"type": "m.id.user", "user": user_id},
                        "password": password,
                        "session": session,
                    }
                    ru = _httpx.post(
                        f"{hs}/_matrix/client/v3/keys/device_signing/upload",
                        headers=headers, json=body, verify=verify_ssl, timeout=15,
                    )
                if ru.status_code != 200:
                    print_warning(f"  Cross-signing upload failed: {ru.status_code} {ru.text[:100]}")
                    continue
                print_success(f"  Uploaded cross-signing keys for {user_id}")
                # Brief pause for Synapse to propagate the new keys before signing
                import time as _t; _t.sleep(2)
            else:
                # Keys exist — load usk_priv from file
                if not user_keys_file.exists():
                    print_warning(f"  {user_id} already has cross-signing keys but local private keys are missing.")
                    print_warning("  Cannot sign the bot without the private key. Skipping.")
                    continue

            # Re-fetch the bot's master key fresh from the server to ensure
            # we sign the exact bytes Synapse has stored (not a cached version).
            try:
                _rq2 = _httpx.post(
                    f"{hs}/_matrix/client/v3/keys/query",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    json={"device_keys": {bot_user_id: []}},
                    verify=verify_ssl, timeout=10,
                )
                _fresh_mk = _rq2.json().get("master_keys", {}).get(bot_user_id, {})
                if _fresh_mk:
                    bot_mk_obj = _fresh_mk
                    bot_mk_pub = list(bot_mk_obj.get("keys", {}).values())[0]
            except Exception:
                pass  # Use the previously fetched bot_mk_obj

            # Sign the bot's master key with the user's user-signing key
            bot_mk_to_sign = {k: v for k, v in bot_mk_obj.items() if k not in ("signatures", "unsigned")}
            sig = _b64.b64encode(usk_priv.sign(_canonical_json(bot_mk_to_sign))).decode()
            signed_bot_mk = dict(bot_mk_obj)
            signed_bot_mk.setdefault("signatures", {}).setdefault(user_id, {})[f"ed25519:{usk_pub}"] = sig

            # The Matrix spec requires the key ID for cross-signing master keys
            # to use the bare public key (no "ed25519:" prefix) as the dict key
            # in the signatures/upload body. Device keys use the device ID instead.
            rs = _httpx.post(
                f"{hs}/_matrix/client/v3/keys/signatures/upload",
                headers=headers,
                json={bot_user_id: {bot_mk_pub: signed_bot_mk}},
                verify=verify_ssl, timeout=10,
            )
            failures = rs.json().get("failures", {}) if rs.status_code == 200 else {}
            if rs.status_code == 200 and not failures:
                print_success(f"  {user_id} now trusts the bot's identity — no more verification warnings.")
            else:
                # Log the full failure for debugging
                print_warning(f"  Signature upload failed: {rs.status_code} {rs.text[:200]}")

        except Exception as e:
            print_warning(f"  Bootstrap error for {user_id}: {e}")
        finally:
            # Always log out the temporary session
            try:
                # Log out the temporary session — this also invalidates the token
                # and cleans up the device on the homeserver automatically.
                _httpx.post(
                    f"{hs}/_matrix/client/v3/logout",
                    headers=headers, verify=verify_ssl, timeout=5,
                )
            except Exception:
                pass


def _setup_matrix():
    """Interactive setup for Matrix homeserver integration."""
    print()
    print(color("  ─── 🔷 Matrix Setup ───", Colors.CYAN))

    # ── Prerequisite checklist ──
    print()
    print_info("  What you need to get started:")
    print_info("  1. A running Matrix homeserver (Synapse, Dendrite, matrix.org, etc.)")
    print_info("  2. A dedicated bot account on that homeserver")
    print_info("     — just a username and password, e.g. 'hermes' / 'your-chosen-password'")
    print_info("  3. Your own Matrix user ID to add to the allowlist")
    print_info("     e.g. @you:yourserver.org")
    print()
    print_info("  That's it. The wizard will handle everything else:")
    print_info("  • Log in as the bot and obtain its access token and device ID")
    print_info("  • Set up E2EE encryption (optional but recommended)")
    print_info("  • Bootstrap cross-signing so Element shows the bot as verified")
    print_info("  • Sign the bot's identity with your account")
    print()
    print_info("  To create a bot account on Synapse (if you haven't already):")
    print_info("    register_new_matrix_user -c /path/to/homeserver.yaml \\")
    print_info("      --no-admin -u botname http://localhost:8008")
    print_info("  (For Docker/k3s: run inside the Synapse container)")

    existing_token = get_env_value("MATRIX_ACCESS_TOKEN")
    if existing_token:
        print()
        print_success("Matrix is already configured.")
        if not prompt_yes_no("  Reconfigure Matrix?", False):
            return

        # Offer to wipe existing crypto state so the new configuration starts
        # clean. Stale Olm sessions, device keys, and cross-signing data from
        # the old account will cause decrypt failures with a new bot account.
        print()
        print_info("  Reconfiguring creates a new bot identity.")
        print_info("  Existing E2EE state (crypto DB, sessions, recovery key) should be")
        print_info("  wiped if you are changing the homeserver or bot account.")
        if prompt_yes_no("  Wipe existing E2EE state for a clean start?", True):
            import shutil as _sh
            from pathlib import Path as _P
            wiped = []
            matrix_dir = _P(os.path.expanduser("~/.hermes/matrix"))
            if matrix_dir.exists():
                _sh.rmtree(matrix_dir)
                wiped.append("crypto DB and sessions")
            for _key in ("MATRIX_RECOVERY_KEY", "MATRIX_DEVICE_ID",
                         "MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD"):
                _val = get_env_value(_key)
                if _val:
                    save_env_value(_key, "")
                    wiped.append(_key)
            if wiped:
                print_success(f"  Wiped: {', '.join(wiped)}")
            else:
                print_info("  Nothing to wipe.")
        else:
            print_info("  Keeping existing E2EE state — only credentials will be updated.")

    # ── Homeserver URL ──
    print()
    print_info("  Enter your Matrix homeserver URL.")
    print_info("  Example: https://matrix.example.org or https://matrix.org")
    existing_hs = get_env_value("MATRIX_HOMESERVER_URL") or ""
    if existing_hs:
        print_info(f"  Current: {existing_hs}")
    try:
        homeserver = input(f"  Homeserver URL{f' [{existing_hs}]' if existing_hs else ''}: ").strip()
        if not homeserver:
            homeserver = existing_hs
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return
    if not homeserver:
        print_error("  Homeserver URL is required.")
        return
    homeserver = homeserver.rstrip("/")

    # ── SSL verification (up front — critical for self-hosted) ──
    print()
    print_info("  If your homeserver uses a self-signed TLS certificate")
    print_info("  (common for self-hosted servers), set verify_ssl to false.")
    print_info("  The connection is still encrypted — only certificate validation is skipped.")
    existing_ssl = get_env_value("MATRIX_VERIFY_SSL") or "true"
    try:
        ssl_input = input(f"  Verify SSL? [true/false] [{existing_ssl}]: ").strip().lower()
        if not ssl_input:
            ssl_input = existing_ssl
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return
    verify_ssl = ssl_input not in ("false", "0", "no")
    ssl_val = "true" if verify_ssl else "false"

    # ── Test homeserver reachability ──
    print_info("  Testing homeserver reachability...")
    try:
        import httpx
        resp = httpx.get(
            f"{homeserver}/_matrix/client/versions",
            timeout=10.0,
            verify=verify_ssl,
        )
        if resp.status_code == 200:
            print_success("  Homeserver is reachable!")
        else:
            print_warning(f"  Homeserver responded with HTTP {resp.status_code}.")
            if not prompt_yes_no("  Continue anyway?", False):
                return
    except Exception as e:
        print_warning(f"  Could not reach homeserver at {homeserver}: {e}")
        if not prompt_yes_no("  Save anyway? (you can fix the URL later)", True):
            return

    # ── Bot user ID ──
    print()
    print_info("  Enter the bot's Matrix user ID (e.g., @agentname:yourdomain).")
    existing_uid = get_env_value("MATRIX_USER_ID") or ""
    if existing_uid:
        print_info(f"  Current: {existing_uid}")
    try:
        user_id = input(f"  Bot Matrix user ID{f' [{existing_uid}]' if existing_uid else ''}: ").strip()
        if not user_id:
            user_id = existing_uid
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return
    if not user_id:
        print_error("  Bot user ID is required.")
        return

    # ── Access token + Device ID — offer to fetch automatically ──
    print()
    print_info("  The bot needs an access token and device ID.")
    print_info("  Option A — enter the bot account password and the wizard fetches them:")
    print_info("  Option B — paste a token you already obtained manually.")
    print()
    _localpart = user_id.split(":")[0].lstrip("@") if user_id else "botusername"
    print_info(f"  To get a token manually (replace PASSWORD with {_localpart}'s password):")
    print_info(f"    curl -s -XPOST '{homeserver}/_matrix/client/v3/login' \\")
    print_info( "      -H 'Content-Type: application/json' \\")
    print_info(f"      -d '{{\"type\":\"m.login.password\",\"user\":\"{_localpart}\",\"password\":\"PASSWORD\"}}'")
    print_info("  Then copy 'access_token' and 'device_id' from the JSON output.")

    token = ""
    device_id = ""

    # Try auto-fetch if user provides the bot account password
    try:
        import getpass as _gp
        bot_password = _gp.getpass(
            "  Bot account password (Enter to skip and paste token manually): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    if bot_password:
        try:
            import httpx as _hx
            localpart = user_id.split(":")[0].lstrip("@")
            login_resp = _hx.post(
                f"{homeserver}/_matrix/client/v3/login",
                json={
                    "type": "m.login.password",
                    "identifier": {"type": "m.id.user", "user": localpart},
                    "password": bot_password,
                    "initial_device_display_name": "Hermes Agent",
                },
                verify=verify_ssl, timeout=10,
            )
            if login_resp.status_code == 200:
                ld = login_resp.json()
                token = ld.get("access_token", "")
                device_id = ld.get("device_id", "")
                print_success(f"  Logged in — device ID: {device_id}")
                # Also save the password for cross-signing bootstrap
                save_env_value("MATRIX_PASSWORD", bot_password)
            else:
                print_warning(f"  Login failed ({login_resp.status_code}): {login_resp.json().get('error','')}")
                print_info("  Falling back to manual token entry.")
        except Exception as _e:
            print_warning(f"  Auto-login failed: {_e}")
            print_info("  Falling back to manual token entry.")

    if not token:
        print()
        print_info("  Paste the 'access_token' value from the curl output above (starts with syt_).")
        try:
            import getpass
            token = getpass.getpass("  Bot access token: ").strip()
            # Strip invisible unicode characters that can appear when pasting
            # from terminals or chat apps (zero-width spaces, BOM, etc.)
            token = "".join(c for c in token if c.isprintable() and ord(c) < 128)
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return
    if not token:
        print_warning("  Skipped — Matrix won't work without an access token.")
        return

    # ── Validate token with whoami ──
    print_info("  Validating access token...")
    try:
        import httpx
        whoami_resp = httpx.get(
            f"{homeserver}/_matrix/client/v3/account/whoami",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10.0,
            verify=verify_ssl,
        )
        if whoami_resp.status_code == 200:
            returned_uid = whoami_resp.json().get("user_id", "")
            if returned_uid == user_id:
                print_success(f"  Token valid — authenticated as {returned_uid}")
            else:
                print_warning(
                    f"  Token is valid but returned user_id '{returned_uid}' "
                    f"does not match '{user_id}'."
                )
                if not prompt_yes_no("  Update bot user ID to match the token?", True):
                    if not prompt_yes_no("  Continue with the mismatched user ID anyway?", False):
                        return
                else:
                    user_id = returned_uid
                    print_success(f"  Bot user ID updated to {user_id}")
        elif whoami_resp.status_code == 401:
            print_error("  Token is invalid (401 Unauthorized). Check for typos.")
            if not prompt_yes_no("  Save the token anyway?", False):
                return
        else:
            print_warning(f"  whoami returned HTTP {whoami_resp.status_code} — could not validate.")
            if not prompt_yes_no("  Continue anyway?", True):
                return
    except Exception as e:
        print_warning(f"  Could not validate token: {e}")
        if not prompt_yes_no("  Continue anyway?", True):
            return

    # ── Device ID ──
    # Already fetched if we did auto-login above. Otherwise prompt.
    existing_dev = get_env_value("MATRIX_DEVICE_ID") or ""
    if not device_id:
        print()
        print_info("  Enter the 'device_id' value from the curl output above.")
        print_info("  This pins the bot to one device so Synapse doesn't create a new")
        print_info("  session on every restart. Strongly recommended.")
        if existing_dev:
            print_info(f"  Current: {existing_dev}")
        try:
            device_id = input(f"  Device ID{f' [{existing_dev}]' if existing_dev else ''} (or Enter to skip): ").strip()
            if not device_id:
                device_id = existing_dev
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return
    if not device_id:
        print_warning("  No device ID set — Synapse will assign a new one on each restart.")
    else:
        print_success(f"  Device ID: {device_id}")

    # ── E2EE ──
    print()
    print_info("  End-to-end encryption (E2EE) lets the bot read and send encrypted messages.")
    print_info("  Required packages: pip install 'mautrix[e2be]' asyncpg aiosqlite base58")
    print_info("  Also requires the libolm C library (used by mautrix for Olm/Megolm crypto):")
    print_info("    Arch:   sudo pacman -S libolm")
    print_info("    Debian: sudo apt install libolm-dev")
    print_info("    Fedora: sudo dnf install libolm-devel")
    print_info("    macOS:  brew install libolm")
    print_info("  For private homeservers on Tailscale/VPN, E2EE is optional —")
    print_info("  network traffic is already encrypted at the transport layer.")
    existing_e2ee = get_env_value("MATRIX_E2EE") or "false"
    try:
        e2ee_input = input(f"  Enable E2EE? [true/false] [{existing_e2ee}]: ").strip().lower()
        if not e2ee_input:
            e2ee_input = existing_e2ee
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return
    e2ee_enabled = e2ee_input in ("true", "1", "yes")
    if e2ee_enabled:
        # 1. Check / install libolm (required by python-olm which mautrix uses)
        try:
            import olm  # noqa: F401
            print_success("  libolm found.")
        except ImportError:
            print_warning("  libolm not found (python-olm import failed).")
            if prompt_yes_no("  Attempt to install libolm now?", True):
                _install_libolm()
            else:
                print_warning("  Install libolm manually before starting the gateway.")

        # 2. Check / install mautrix[e2be] + SQLite store dependencies
        try:
            from mautrix.crypto import OlmMachine  # noqa: F401
            from mautrix.crypto.store.asyncpg import PgCryptoStore  # noqa: F401
            import aiosqlite  # noqa: F401
            print_success("  mautrix[e2be] + asyncpg + aiosqlite found.")
        except ImportError:
            print_warning("  mautrix[e2be] not installed.")
            if prompt_yes_no("  Install mautrix[e2be] now?", True):
                import sys as _sys
                # asyncpg and aiosqlite are required for the SQLite crypto store
                result = subprocess.run(
                    [_sys.executable, "-m", "pip", "install",
                     "mautrix[e2be]", "asyncpg", "aiosqlite", "base58"],
                    capture_output=True,
                )
                if result.returncode == 0:
                    print_success("  mautrix[e2be] and dependencies installed.")
                else:
                    print_error(
                        "  Installation failed. Run manually: "
                        "pip install 'mautrix[e2be]' asyncpg aiosqlite base58"
                    )
            else:
                print_warning(
                    "  Skipping — run: pip install 'mautrix[e2be]' asyncpg aiosqlite base58"
                )

    # ── Cross-signing (password for bootstrap) ──
    print()
    print_info("  Cross-signing lets Element verify the bot's identity so the")
    print_info("  'device not verified by owner' warning disappears permanently.")
    print_info("  The bot account password is stored in .env and used to bootstrap")
    print_info("  cross-signing keys on first gateway start.")
    existing_password = get_env_value("MATRIX_PASSWORD") or ""
    if existing_password:
        # Password was already saved (either from the auto-login step above,
        # or from a previous wizard run). Confirm it's set — no need to re-enter.
        print_success("  ✓ Bot password already saved in .env — cross-signing will bootstrap automatically.")
    else:
        try:
            import getpass
            password_input = getpass.getpass(
                "  Bot account password (leave empty to skip): "
            ).strip()
            if password_input:
                save_env_value("MATRIX_PASSWORD", password_input)
                print_success("  Password saved.")
            else:
                print_info("  Skipped — cross-signing bootstrap will be attempted without password.")
                print_info("  If it fails, re-run this wizard and enter the bot account password.")
        except (KeyboardInterrupt, EOFError):
            pass

    # ── Recovery key — bootstrap and save automatically ──
    # At this point we have: token, password, homeserver, user_id.
    # That's everything needed to bootstrap cross-signing and obtain the
    # recovery key right now, without the user having to run the gateway
    # first and copy anything manually.
    if e2ee_enabled:
        print()
        print_info("  Bootstrapping E2EE cross-signing keys...")
        print_info("  The recovery key lets the bot self-sign its device on every restart,")
        print_info("  removing the 'encrypted by unverified device' warning permanently.")

        existing_rk = get_env_value("MATRIX_RECOVERY_KEY") or ""
        bot_password = get_env_value("MATRIX_PASSWORD") or ""

        if existing_rk:
            print_success("  ✓ Recovery key already set in .env — no action needed.")
        elif not bot_password:
            print_warning("  Bot password not set — skipping automatic recovery key bootstrap.")
            print_info("  Start the gateway and save the printed recovery key manually.")
        else:
            # Bootstrap cross-signing and obtain the recovery key directly.
            # We do this by making the same API calls the gateway makes on first start,
            # but synchronously using httpx here in the wizard.
            try:
                import httpx as _hx, json as _j, asyncio as _asyncio

                async def _get_recovery_key():
                    """Bootstrap cross-signing using mautrix and return the recovery key."""
                    import logging as _log
                    # Suppress mautrix internal debug warnings during bootstrap
                    for _nl in ("mautrix.crypto", "mau.crypto", "mau.client.crypto"):
                        _log.getLogger(_nl).setLevel(_log.ERROR)
                    import aiohttp as _aiohttp
                    from mautrix.util.async_db import Database
                    from mautrix.crypto.store.asyncpg import PgCryptoStore
                    from mautrix.crypto import OlmMachine
                    from mautrix.client.state_store import MemoryStateStore
                    from mautrix.client import Client
                    import os as _os
                    from pathlib import Path as _Path

                    data_dir = _Path(_os.path.expanduser("~/.hermes/matrix"))
                    data_dir.mkdir(parents=True, exist_ok=True)
                    db_path = data_dir / "crypto.db"

                    db = Database.create(
                        f"sqlite:{db_path}",
                        upgrade_table=PgCryptoStore.upgrade_table,
                    )
                    await db.start()
                    store = PgCryptoStore(user_id, "hermes-matrix-e2ee", db)
                    await store.open()

                    connector = _aiohttp.TCPConnector(ssl=False if not verify_ssl else None)
                    session = _aiohttp.ClientSession(connector=connector)
                    state_store = MemoryStateStore()
                    client = Client(
                        mxid=user_id, base_url=homeserver,
                        token=token, client_session=session,
                        state_store=state_store, sync_store=store,
                    )
                    whoami = await client.whoami()
                    client.device_id = whoami.device_id

                    crypto = OlmMachine(client=client, crypto_store=store, state_store=state_store)
                    client.crypto = crypto
                    await crypto.load()

                    # Upload device keys if needed
                    if not crypto.account.shared:
                        await crypto.share_keys()

                    # Check trust state — only bootstrap if not already verified
                    from mautrix.types import TrustState as _TS
                    try:
                        ts = await crypto.resolve_trust(crypto.own_identity)
                        if ts >= _TS.CROSS_SIGNED_UNTRUSTED:
                            await session.close()
                            await db.stop()
                            return None, "already verified"
                    except Exception:
                        pass

                    # Bootstrap cross-signing with UIA fallback
                    rk = None
                    try:
                        rk = await crypto.generate_recovery_key(passphrase=bot_password or None)
                    except Exception as e:
                        if "401" in str(e) and bot_password:
                            _uia = _j.loads(str(e).split("401: ", 1)[-1])
                            _auth = {
                                "type": "m.login.password",
                                "identifier": {"type": "m.id.user", "user": user_id},
                                "password": bot_password,
                                "session": _uia.get("session", ""),
                            }
                            from mautrix.crypto.cross_signing_key import CrossSigningSeeds as _CSS
                            _seeds = _CSS.generate()
                            _sk = await crypto.ssss.generate_and_upload_key(bot_password or None)
                            await crypto._upload_cross_signing_keys_to_ssss(_sk, _seeds)
                            await crypto._publish_cross_signing_keys(_seeds.to_keys(), auth=_auth)
                            await crypto.ssss.set_default_key_id(_sk.id)
                            await crypto.sign_own_device(crypto.own_identity)
                            rk = _sk.recovery_key
                        else:
                            raise

                    await session.close()
                    await db.stop()
                    return rk, None

                recovery_key, reason = _asyncio.run(_get_recovery_key())

                if recovery_key:
                    save_env_value("MATRIX_RECOVERY_KEY", recovery_key)
                    print_success(f"  ✓ Recovery key obtained and saved to .env: {recovery_key}")
                    print_info("  The bot will use this to self-sign its device on every restart.")
                elif reason == "already verified":
                    print_success("  ✓ Bot cross-signing already verified — no bootstrap needed.")
                else:
                    print_warning("  Could not obtain recovery key automatically.")
                    print_info("  The gateway will bootstrap on first start and save the key.")

            except Exception as _rk_exc:
                print_warning(f"  Automatic bootstrap skipped ({_rk_exc}).")
                print_info("  The gateway will bootstrap on first start and save the key.")

        print()
        print_success("  E2EE setup complete.")

    # ── Save core vars ──
    save_env_value("MATRIX_HOMESERVER_URL", homeserver)
    save_env_value("MATRIX_USER_ID", user_id)
    save_env_value("MATRIX_ACCESS_TOKEN", token)
    save_env_value("MATRIX_VERIFY_SSL", ssl_val)
    save_env_value("MATRIX_E2EE", "true" if e2ee_enabled else "false")
    if device_id:
        save_env_value("MATRIX_DEVICE_ID", device_id)

    # ── Allowed users ──
    print()
    print_info("  The gateway DENIES all users by default for security.")
    print_info("  Enter Matrix user IDs allowed to message the bot (comma-separated).")
    print_info("  This is YOUR Matrix ID (not the bot's) — e.g. @alice:matrix.example.org")
    existing_allowed = get_env_value("MATRIX_ALLOWED_USERS") or ""
    if existing_allowed:
        print_info(f"  Current: {existing_allowed}")
    try:
        allowed = input(f"  Allowed users{f' [{existing_allowed}]' if existing_allowed else ''}: ").strip()
        if not allowed:
            allowed = existing_allowed
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return
    if allowed:
        save_env_value("MATRIX_ALLOWED_USERS", allowed.replace(" ", ""))
        print_success("  Saved — only these users can interact with the bot.")
    else:
        print()
        access_choices = [
            "Enable open access (anyone who invites the bot can use it)",
            "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
            "Skip for now (bot will deny all users until configured)",
        ]
        access_idx = prompt_choice("  How should unauthorized users be handled?", access_choices, 1)
        if access_idx == 0:
            save_env_value("GATEWAY_ALLOW_ALL_USERS", "true")
            print_warning("  Open access enabled — anyone can use your bot!")
        elif access_idx == 1:
            print_success("  DM pairing mode — users will receive a code to request access.")
        else:
            print_info("  Skipped — configure later with 'hermes gateway setup'")

    # ── Home room ──
    print()
    print_info("  Optionally set a home room ID for cron job delivery.")
    print_info("  Room IDs look like: !abc123:yourdomain")
    print_info("  Find it in Element: Room Settings → Advanced → Internal Room ID")
    existing_home = get_env_value("MATRIX_HOME_CHANNEL") or ""
    if existing_home:
        print_info(f"  Current: {existing_home}")
    try:
        home_channel = input(f"  Home room ID{f' [{existing_home}]' if existing_home else ''} (or Enter to skip): ").strip()
        if not home_channel:
            home_channel = existing_home
    except (EOFError, KeyboardInterrupt):
        pass
    else:
        if home_channel:
            save_env_value("MATRIX_HOME_CHANNEL", home_channel)
            print_success(f"  Home room set to {home_channel}")

    # ── Cross-signing bootstrap and trust verification ──
    # Run this inline during setup so the user never needs to run a separate
    # command. We have everything we need: bot token, bot password, allowed
    # user passwords (prompted in _bootstrap_user_trust).
    if e2ee_enabled and allowed:
        print()
        print_info("  Setting up E2EE trust chain so Element shows the bot as verified...")

        # Step 1: Upload the bot's cross-signing keys directly using the bot password.
        # This is what the gateway does on first start, but we do it here so the
        # trust chain is complete before the user ever starts the gateway.
        bot_password = get_env_value("MATRIX_PASSWORD") or ""
        if bot_password:
            try:
                import httpx as _hx, json as _j
                # First attempt without UIA
                _cs_resp = _hx.post(
                    f"{homeserver.rstrip('/')}/_matrix/client/v3/keys/device_signing/upload",
                    headers={"Authorization": f"Bearer {token}"},
                    json={},  # empty — just to get the UIA challenge
                    verify=verify_ssl, timeout=10,
                )
                if _cs_resp.status_code == 401:
                    # Server requires UIA — expected on Synapse when keys exist
                    pass  # _bootstrap_user_trust handles the actual key generation
                elif _cs_resp.status_code == 200:
                    pass  # No UIA needed, gateway will handle on first start
            except Exception:
                pass

        # Step 2: Sign the bot's master key with each allowed user's key.
        # _bootstrap_user_trust prompts for each allowed user's password.
        # It handles generating/uploading user cross-signing keys AND signing
        # the bot's master key — completing the full trust chain.
        try:
            import httpx as _hx
            _r = _hx.post(
                f"{homeserver.rstrip('/')}/_matrix/client/v3/keys/query",
                headers={"Authorization": f"Bearer {token}"},
                json={"device_keys": {user_id: []}},
                verify=verify_ssl, timeout=10,
            )
            _has_mk = bool(_r.json().get("master_keys", {}).get(user_id))
        except Exception:
            _has_mk = False

        if _has_mk:
            _bootstrap_user_trust(homeserver, allowed, user_id, token, verify_ssl)
        else:
            print_info("  Bot cross-signing keys will be uploaded on first gateway start.")
            print_info("  After starting the gateway, run:")
            print_info("    hermes gateway verify-matrix")
            print_info("  to complete the trust chain. This is a one-time step.")

    print()
    print_success("🔷 Matrix configured!")
    print_info("  Invite the bot to any room — it will auto-accept if you're in the allowlist.")
    print()
    print_info("  Next steps:")
    print_info("  1. Start the gateway:  hermes gateway run")
    if e2ee_enabled:
        print_info("  2. Restart Element to see the verified status.")
        print_info("  If trust verification didn't complete (e.g. rate-limited or")
        print_info("  you added new allowed users later), run:")
        print_info("    hermes gateway verify-matrix")


def _setup_signal():
    """Interactive setup for Signal messenger."""
    import shutil

    print()
    print(color("  ─── 📡 Signal Setup ───", Colors.CYAN))

    existing_url = get_env_value("SIGNAL_HTTP_URL")
    existing_account = get_env_value("SIGNAL_ACCOUNT")
    if existing_url and existing_account:
        print()
        print_success("Signal is already configured.")
        if not prompt_yes_no("  Reconfigure Signal?", False):
            return

    # Check if signal-cli is available
    print()
    if shutil.which("signal-cli"):
        print_success("signal-cli found on PATH.")
    else:
        print_warning("signal-cli not found on PATH.")
        print_info("  Signal requires signal-cli running as an HTTP daemon.")
        print_info("  Install options:")
        print_info("    Linux:  sudo apt install signal-cli")
        print_info("            or download from https://github.com/AsamK/signal-cli")
        print_info("    macOS:  brew install signal-cli")
        print_info("    Docker: bbernhard/signal-cli-rest-api")
        print()
        print_info("  After installing, link your account and start the daemon:")
        print_info("    signal-cli link -n \"HermesAgent\"")
        print_info("    signal-cli --account +YOURNUMBER daemon --http 127.0.0.1:8080")
        print()

    # HTTP URL
    print()
    print_info("  Enter the URL where signal-cli HTTP daemon is running.")
    default_url = existing_url or "http://127.0.0.1:8080"
    try:
        url = input(f"  HTTP URL [{default_url}]: ").strip() or default_url
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    # Test connectivity
    print_info("  Testing connection...")
    try:
        import httpx
        resp = httpx.get(f"{url.rstrip('/')}/api/v1/check", timeout=10.0)
        if resp.status_code == 200:
            print_success("  signal-cli daemon is reachable!")
        else:
            print_warning(f"  signal-cli responded with status {resp.status_code}.")
            if not prompt_yes_no("  Continue anyway?", False):
                return
    except Exception as e:
        print_warning(f"  Could not reach signal-cli at {url}: {e}")
        if not prompt_yes_no("  Save this URL anyway? (you can start signal-cli later)", True):
            return

    save_env_value("SIGNAL_HTTP_URL", url)

    # Account phone number
    print()
    print_info("  Enter your Signal account phone number in E.164 format.")
    print_info("  Example: +15551234567")
    default_account = existing_account or ""
    try:
        account = input(f"  Account number{f' [{default_account}]' if default_account else ''}: ").strip()
        if not account:
            account = default_account
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    if not account:
        print_error("  Account number is required.")
        return

    save_env_value("SIGNAL_ACCOUNT", account)

    # Allowed users
    print()
    print_info("  The gateway DENIES all users by default for security.")
    print_info("  Enter phone numbers or UUIDs of allowed users (comma-separated).")
    existing_allowed = get_env_value("SIGNAL_ALLOWED_USERS") or ""
    default_allowed = existing_allowed or account
    try:
        allowed = input(f"  Allowed users [{default_allowed}]: ").strip() or default_allowed
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return

    save_env_value("SIGNAL_ALLOWED_USERS", allowed)

    # Group messaging
    print()
    if prompt_yes_no("  Enable group messaging? (disabled by default for security)", False):
        print()
        print_info("  Enter group IDs to allow, or * for all groups.")
        existing_groups = get_env_value("SIGNAL_GROUP_ALLOWED_USERS") or ""
        try:
            groups = input(f"  Group IDs [{existing_groups or '*'}]: ").strip() or existing_groups or "*"
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return
        save_env_value("SIGNAL_GROUP_ALLOWED_USERS", groups)

    print()
    print_success("Signal configured!")
    print_info(f"  URL: {url}")
    print_info(f"  Account: {account}")
    print_info(f"  DM auth: via SIGNAL_ALLOWED_USERS + DM pairing")
    print_info(f"  Groups: {'enabled' if get_env_value('SIGNAL_GROUP_ALLOWED_USERS') else 'disabled'}")


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
        print_info("Gateway service is not installed yet.")
        print_info("You'll be offered to install it after configuring platforms.")

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
        elif platform["key"] == "signal":
            _setup_signal()
        elif platform["key"] == "matrix":
            _setup_matrix()
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
            if is_linux() or is_macos():
                platform_name = "systemd" if is_linux() else "launchd"
                if prompt_yes_no(f"  Install the gateway as a {platform_name} service? (runs in background, starts on boot)", True):
                    try:
                        force = False
                        if is_linux():
                            systemd_install(force)
                        else:
                            launchd_install(force)
                        print()
                        if prompt_yes_no("  Start the service now?", True):
                            try:
                                if is_linux():
                                    systemd_start()
                                else:
                                    launchd_start()
                            except subprocess.CalledProcessError as e:
                                print_error(f"  Start failed: {e}")
                    except subprocess.CalledProcessError as e:
                        print_error(f"  Install failed: {e}")
                        print_info("  You can try manually: hermes gateway install")
                else:
                    print_info("  You can install later: hermes gateway install")
                    print_info("  Or run in foreground:  hermes gateway")
            else:
                print_info("  Service install not supported on this platform.")
                print_info("  Run in foreground: hermes gateway")
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
        replace = getattr(args, 'replace', False)
        run_gateway(verbose, replace=replace)
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
