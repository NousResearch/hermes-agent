"""
Status command for hermes CLI.

Shows the status of all Hermes Agent components.
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"

def color(text: str, *codes) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET

def check_mark(ok: bool) -> str:
    if ok:
        return color("✓", Colors.GREEN)
    return color("✗", Colors.RED)

def redact_key(key: str) -> str:
    """Redact an API key for display."""
    if not key:
        return "(not set)"
    if len(key) < 12:
        return "***"
    return key[:4] + "..." + key[-4:]


def show_status(args):
    """Show status of all Hermes Agent components."""
    show_all = getattr(args, 'all', False)
    deep = getattr(args, 'deep', False)
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 🦋 Hermes Agent Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    
    # =========================================================================
    # Environment
    # =========================================================================
    print()
    print(color("◆ Environment", Colors.CYAN, Colors.BOLD))
    print(f"  Project:      {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")
    
    env_path = PROJECT_ROOT / '.env'
    print(f"  .env file:    {check_mark(env_path.exists())} {'exists' if env_path.exists() else 'not found'}")
    
    # =========================================================================
    # API Keys
    # =========================================================================
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))
    
    keys = {
        "OpenRouter": "OPENROUTER_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "OpenAI": "OPENAI_API_KEY",
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Browserbase": "BROWSERBASE_API_KEY",
        "FAL": "FAL_KEY",
        "Tinker": "TINKER_API_KEY",
        "WandB": "WANDB_API_KEY",
    }
    
    for name, env_var in keys.items():
        value = os.getenv(env_var, "")
        has_key = bool(value)
        display = redact_key(value) if not show_all else value
        print(f"  {name:<12}  {check_mark(has_key)} {display}")
    
    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color("◆ Terminal Backend", Colors.CYAN, Colors.BOLD))
    
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    print(f"  Backend:      {terminal_env}")
    
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  SSH Host:     {ssh_host or '(not set)'}")
        print(f"  SSH User:     {ssh_user or '(not set)'}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  Docker Image: {docker_image}")
    
    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")
    
    # =========================================================================
    # Messaging Platforms
    # =========================================================================
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))
    
    platforms = {
        "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
        "Discord": ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"),
        "WhatsApp": ("WHATSAPP_ENABLED", None),
    }
    
    for name, (token_var, home_var) in platforms.items():
        token = os.getenv(token_var, "")
        has_token = bool(token)
        
        home_channel = ""
        if home_var:
            home_channel = os.getenv(home_var, "")
        
        status = "configured" if has_token else "not configured"
        if home_channel:
            status += f" (home: {home_channel})"
        
        print(f"  {name:<12}  {check_mark(has_token)} {status}")
    
    # =========================================================================
    # Gateway Status
    # =========================================================================
    print()
    print(color("◆ Gateway Service", Colors.CYAN, Colors.BOLD))
    
    if sys.platform.startswith('linux'):
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "hermes-gateway"],
            capture_output=True,
            text=True
        )
        is_active = result.stdout.strip() == "active"
        print(f"  Status:       {check_mark(is_active)} {'running' if is_active else 'stopped'}")
        print(f"  Manager:      systemd (user)")
        
    elif sys.platform == 'darwin':
        result = subprocess.run(
            ["launchctl", "list", "ai.hermes.gateway"],
            capture_output=True,
            text=True
        )
        is_loaded = result.returncode == 0
        print(f"  Status:       {check_mark(is_loaded)} {'loaded' if is_loaded else 'not loaded'}")
        print(f"  Manager:      launchd")
    else:
        print(f"  Status:       {color('N/A', Colors.DIM)}")
        print(f"  Manager:      (not supported on this platform)")
    
    # =========================================================================
    # Cron Jobs
    # =========================================================================
    print()
    print(color("◆ Scheduled Jobs", Colors.CYAN, Colors.BOLD))
    
    jobs_file = Path.home() / ".hermes" / "cron" / "jobs.json"
    if jobs_file.exists():
        import json
        try:
            with open(jobs_file) as f:
                data = json.load(f)
                jobs = data.get("jobs", [])
                enabled_jobs = [j for j in jobs if j.get("enabled", True)]
                print(f"  Jobs:         {len(enabled_jobs)} active, {len(jobs)} total")
        except:
            print(f"  Jobs:         (error reading jobs file)")
    else:
        print(f"  Jobs:         0")
    
    # =========================================================================
    # Sessions
    # =========================================================================
    print()
    print(color("◆ Sessions", Colors.CYAN, Colors.BOLD))
    
    sessions_file = Path.home() / ".hermes" / "sessions" / "sessions.json"
    if sessions_file.exists():
        import json
        try:
            with open(sessions_file) as f:
                data = json.load(f)
                print(f"  Active:       {len(data)} session(s)")
        except:
            print(f"  Active:       (error reading sessions file)")
    else:
        print(f"  Active:       0")
    
    # =========================================================================
    # Deep checks
    # =========================================================================
    if deep:
        print()
        print(color("◆ Deep Checks", Colors.CYAN, Colors.BOLD))
        
        # Check OpenRouter connectivity
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            try:
                import httpx
                response = httpx.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {openrouter_key}"},
                    timeout=10
                )
                ok = response.status_code == 200
                print(f"  OpenRouter:   {check_mark(ok)} {'reachable' if ok else f'error ({response.status_code})'}")
            except Exception as e:
                print(f"  OpenRouter:   {check_mark(False)} error: {e}")
        
        # Check gateway port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 18789))
            sock.close()
            # Port in use = gateway likely running
            port_in_use = result == 0
            # This is informational, not necessarily bad
            print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
        except:
            pass
    
    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  Run 'hermes doctor' for detailed diagnostics", Colors.DIM))
    print(color("  Run 'hermes setup' to configure", Colors.DIM))
    print()
