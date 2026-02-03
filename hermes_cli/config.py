"""
Configuration management for Hermes Agent.

Config files are stored in ~/.hermes/ for easy access:
- ~/.hermes/config.yaml  - All settings (model, toolsets, terminal, etc.)
- ~/.hermes/.env         - API keys and secrets

This module provides:
- hermes config          - Show current configuration
- hermes config edit     - Open config in editor
- hermes config set      - Set a specific value
- hermes config wizard   - Re-run setup wizard
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

def color(text: str, *codes) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + Colors.RESET


# =============================================================================
# Config paths
# =============================================================================

def get_hermes_home() -> Path:
    """Get the Hermes home directory (~/.hermes)."""
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

def get_config_path() -> Path:
    """Get the main config file path."""
    return get_hermes_home() / "config.yaml"

def get_env_path() -> Path:
    """Get the .env file path (for API keys)."""
    return get_hermes_home() / ".env"

def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()

def ensure_hermes_home():
    """Ensure ~/.hermes directory structure exists."""
    home = get_hermes_home()
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "sessions").mkdir(parents=True, exist_ok=True)
    (home / "logs").mkdir(parents=True, exist_ok=True)


# =============================================================================
# Config loading/saving
# =============================================================================

DEFAULT_CONFIG = {
    "model": "anthropic/claude-sonnet-4",
    "toolsets": ["hermes-cli"],
    "max_turns": 100,
    
    "terminal": {
        "backend": "local",
        "cwd": ".",  # Use current directory
        "timeout": 180,
        "docker_image": "python:3.11-slim",
    },
    
    "browser": {
        "inactivity_timeout": 120,
    },
    
    "compression": {
        "enabled": True,
        "threshold": 0.85,
        "summary_model": "google/gemini-2.0-flash-001",
    },
    
    "display": {
        "compact": False,
        "personality": "kawaii",
    },
}


def load_config() -> Dict[str, Any]:
    """Load configuration from ~/.hermes/config.yaml."""
    config_path = get_config_path()
    
    config = DEFAULT_CONFIG.copy()
    
    if config_path.exists():
        try:
            with open(config_path) as f:
                user_config = yaml.safe_load(f) or {}
            
            # Deep merge
            for key, value in user_config.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
    
    return config


def save_config(config: Dict[str, Any]):
    """Save configuration to ~/.hermes/config.yaml."""
    ensure_hermes_home()
    config_path = get_config_path()
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_env() -> Dict[str, str]:
    """Load environment variables from ~/.hermes/.env."""
    env_path = get_env_path()
    env_vars = {}
    
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    env_vars[key.strip()] = value.strip().strip('"\'')
    
    return env_vars


def save_env_value(key: str, value: str):
    """Save or update a value in ~/.hermes/.env."""
    ensure_hermes_home()
    env_path = get_env_path()
    
    # Load existing
    lines = []
    if env_path.exists():
        with open(env_path) as f:
            lines = f.readlines()
    
    # Find and update or append
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            found = True
            break
    
    if not found:
        lines.append(f"{key}={value}\n")
    
    with open(env_path, 'w') as f:
        f.writelines(lines)


def get_env_value(key: str) -> Optional[str]:
    """Get a value from ~/.hermes/.env or environment."""
    # Check environment first
    if key in os.environ:
        return os.environ[key]
    
    # Then check .env file
    env_vars = load_env()
    return env_vars.get(key)


# =============================================================================
# Config display
# =============================================================================

def redact_key(key: str) -> str:
    """Redact an API key for display."""
    if not key:
        return color("(not set)", Colors.DIM)
    if len(key) < 12:
        return "***"
    return key[:4] + "..." + key[-4:]


def show_config():
    """Display current configuration."""
    config = load_config()
    env_vars = load_env()
    
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│              🦋 Hermes Configuration                    │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))
    
    # Paths
    print()
    print(color("◆ Paths", Colors.CYAN, Colors.BOLD))
    print(f"  Config:       {get_config_path()}")
    print(f"  Secrets:      {get_env_path()}")
    print(f"  Install:      {get_project_root()}")
    
    # API Keys
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))
    
    keys = [
        ("OPENROUTER_API_KEY", "OpenRouter"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("OPENAI_API_KEY", "OpenAI"),
        ("FIRECRAWL_API_KEY", "Firecrawl"),
        ("BROWSERBASE_API_KEY", "Browserbase"),
        ("FAL_KEY", "FAL"),
    ]
    
    for env_key, name in keys:
        value = get_env_value(env_key)
        print(f"  {name:<14} {redact_key(value)}")
    
    # Model settings
    print()
    print(color("◆ Model", Colors.CYAN, Colors.BOLD))
    print(f"  Model:        {config.get('model', 'not set')}")
    print(f"  Max turns:    {config.get('max_turns', 100)}")
    print(f"  Toolsets:     {', '.join(config.get('toolsets', ['all']))}")
    
    # Terminal
    print()
    print(color("◆ Terminal", Colors.CYAN, Colors.BOLD))
    terminal = config.get('terminal', {})
    print(f"  Backend:      {terminal.get('backend', 'local')}")
    print(f"  Working dir:  {terminal.get('cwd', '.')}")
    print(f"  Timeout:      {terminal.get('timeout', 60)}s")
    
    if terminal.get('backend') == 'docker':
        print(f"  Docker image: {terminal.get('docker_image', 'python:3.11-slim')}")
    elif terminal.get('backend') == 'ssh':
        ssh_host = get_env_value('TERMINAL_SSH_HOST')
        ssh_user = get_env_value('TERMINAL_SSH_USER')
        print(f"  SSH host:     {ssh_host or '(not set)'}")
        print(f"  SSH user:     {ssh_user or '(not set)'}")
    
    # Compression
    print()
    print(color("◆ Context Compression", Colors.CYAN, Colors.BOLD))
    compression = config.get('compression', {})
    enabled = compression.get('enabled', True)
    print(f"  Enabled:      {'yes' if enabled else 'no'}")
    if enabled:
        print(f"  Threshold:    {compression.get('threshold', 0.85) * 100:.0f}%")
        print(f"  Model:        {compression.get('summary_model', 'google/gemini-2.0-flash-001')}")
    
    # Messaging
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))
    
    telegram_token = get_env_value('TELEGRAM_BOT_TOKEN')
    discord_token = get_env_value('DISCORD_BOT_TOKEN')
    
    print(f"  Telegram:     {'configured' if telegram_token else color('not configured', Colors.DIM)}")
    print(f"  Discord:      {'configured' if discord_token else color('not configured', Colors.DIM)}")
    
    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  hermes config edit     # Edit config file", Colors.DIM))
    print(color("  hermes config set KEY VALUE", Colors.DIM))
    print(color("  hermes setup           # Run setup wizard", Colors.DIM))
    print()


def edit_config():
    """Open config file in user's editor."""
    config_path = get_config_path()
    
    # Ensure config exists
    if not config_path.exists():
        save_config(DEFAULT_CONFIG)
        print(f"Created {config_path}")
    
    # Find editor
    editor = os.getenv('EDITOR') or os.getenv('VISUAL')
    
    if not editor:
        # Try common editors
        for cmd in ['nano', 'vim', 'vi', 'code', 'notepad']:
            import shutil
            if shutil.which(cmd):
                editor = cmd
                break
    
    if not editor:
        print(f"No editor found. Config file is at:")
        print(f"  {config_path}")
        return
    
    print(f"Opening {config_path} in {editor}...")
    subprocess.run([editor, str(config_path)])


def set_config_value(key: str, value: str):
    """Set a configuration value."""
    # Check if it's an API key (goes to .env)
    api_keys = [
        'OPENROUTER_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY',
        'FIRECRAWL_API_KEY', 'BROWSERBASE_API_KEY', 'BROWSERBASE_PROJECT_ID',
        'FAL_KEY', 'TELEGRAM_BOT_TOKEN', 'DISCORD_BOT_TOKEN',
        'TERMINAL_SSH_HOST', 'TERMINAL_SSH_USER', 'TERMINAL_SSH_KEY',
        'SUDO_PASSWORD'
    ]
    
    if key.upper() in api_keys or key.upper().startswith('TERMINAL_SSH'):
        save_env_value(key.upper(), value)
        print(f"✓ Set {key} in {get_env_path()}")
        return
    
    # Otherwise it goes to config.yaml
    config = load_config()
    
    # Handle nested keys (e.g., "terminal.backend")
    parts = key.split('.')
    current = config
    
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    
    # Convert value to appropriate type
    if value.lower() in ('true', 'yes', 'on'):
        value = True
    elif value.lower() in ('false', 'no', 'off'):
        value = False
    elif value.isdigit():
        value = int(value)
    elif value.replace('.', '', 1).isdigit():
        value = float(value)
    
    current[parts[-1]] = value
    save_config(config)
    print(f"✓ Set {key} = {value} in {get_config_path()}")


# =============================================================================
# Command handler
# =============================================================================

def config_command(args):
    """Handle config subcommands."""
    subcmd = getattr(args, 'config_command', None)
    
    if subcmd is None or subcmd == "show":
        show_config()
    
    elif subcmd == "edit":
        edit_config()
    
    elif subcmd == "set":
        key = getattr(args, 'key', None)
        value = getattr(args, 'value', None)
        if not key or not value:
            print("Usage: hermes config set KEY VALUE")
            print()
            print("Examples:")
            print("  hermes config set model anthropic/claude-sonnet-4")
            print("  hermes config set terminal.backend docker")
            print("  hermes config set OPENROUTER_API_KEY sk-or-...")
            sys.exit(1)
        set_config_value(key, value)
    
    elif subcmd == "path":
        print(get_config_path())
    
    elif subcmd == "env-path":
        print(get_env_path())
    
    else:
        print(f"Unknown config command: {subcmd}")
        sys.exit(1)
